"""APH grouped validation, native feature construction, and model dispatch."""

import csv
import hashlib

import numpy as np
import torch

from .features import AA_ORDER, amino_acid_properties, chiral_fingerprints
from .models import MLPFitness
from .prototypes import PrototypeMLP
from .training import fit


CONDITIONS = ("Kan", "Neo", "Paro", "Ribo")
PROCESSED = "data/processed/APH_variants.csv"
SPLIT = "data/splits/aph_assignments.csv"
WT = "data/reference/aph_wt.fasta"
FILES = {
    PROCESSED: "dc27f76cbe0773cc2183e23f72458004d778df32cb8683c238fc728a295eeb99",
    SPLIT: "ac9eadf6d13228f5e78e1653d867fd2ae32e71fae5dc10a20d1cd6ccae2596d4",
    WT: "60f596453a48b7140b2d106501891e0ce9ef8a22a1d8670639d4d91ff5fc476a",
}
FOLD_VALIDATION_COUNTS = (719, 718, 718, 718, 718)
EMBEDDING_WIDTH = 1152


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _load_inputs(root):
    for file, digest in FILES.items():
        if hashlib.sha256((root / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"APH input does not match the historical SHA-256 for {file}")
    wt = "".join(line.strip() for line in (root / WT).read_text(encoding="utf-8-sig").splitlines()
                 if not line.startswith(">"))
    rows = _read_csv(root / PROCESSED)
    assignments = _read_csv(root / SPLIT)
    mapping = {row["sequence_id"]: row for row in assignments}
    if len(wt) != 264 or len(rows) != 3990 or len(mapping) != len(assignments) or len(mapping) != len(rows):
        raise ValueError("APH cohort size or sequence identity mismatch")
    seen = set()
    for row in rows:
        sequence = row["sequence"]
        pos = int(row["position_1based"]) - 1
        differences = [(i, a, b) for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
        if (len(sequence) != len(wt) or differences != [(pos, row["WT"], row["MT"])]
                or row["mutation"] != f'{row["WT"]}{pos + 1}{row["MT"]}' or row["mutation_count"] != "1"):
            raise ValueError("APH sequence or substitution annotation changed")
        sequence_id = row["sequence_id"]
        if (sequence_id != "seq_" + hashlib.sha256(sequence.encode()).hexdigest()
                or sequence_id in seen or sequence_id not in mapping):
            raise ValueError("APH variant identifiers and sequence splits disagree")
        seen.add(sequence_id)
        split = mapping[sequence_id]
        if (split["variant_id"] != row["variant_id"] or split["partition"] not in {"test", "development"}
                or (split["partition"] == "test" and split["fold"] != "")
                or (split["partition"] == "development" and split["fold"] not in {"0", "1", "2", "3", "4"})):
            raise ValueError("Invalid APH variant partition")
        if pos == 0 or min(int(row["background_count_1"]), int(row["background_count_2"])) < 10:
            raise ValueError("APH variant does not pass the archived background-count rule")
        for condition in CONDITIONS:
            y = float(row["y_" + condition])
            raw = float(row["raw_" + condition])
            if not np.isfinite(y) or raw <= 0 or not np.isclose(y, np.log2(raw), atol=1e-12):
                raise ValueError("APH processed response does not match the archived log2 transform")
    return wt, rows, mapping


def _partitions(rows, assignments, fold):
    groups = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(rows):
        a = assignments[row["sequence_id"]]
        part = "test" if a["partition"] == "test" else ("validation" if int(a["fold"]) == fold else "train")
        groups[part].append(index)
    expected_val = FOLD_VALIDATION_COUNTS[fold]
    if len(groups["test"]) != 399 or len(groups["validation"]) != expected_val:
        raise ValueError("APH train/validation/test sizes changed")
    ids = [{rows[index]["sequence_id"] for index in group} for group in groups.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)) or sum(map(len, ids)) != len(rows):
        raise ValueError("APH sequence leakage or incomplete assignment")
    return groups


def verify_splits(root):
    _, rows, assignments = _load_inputs(root)
    folds = [{key: len(group) for key, group in _partitions(rows, assignments, fold).items()}
             for fold in range(5)]
    return {"variants": len(rows), "sequence_leakage": 0, "folds": folds, "split_sha256": FILES[SPLIT]}


def _conditions(config):
    mode = config["training_mode"]
    if mode == "joint":
        return CONDITIONS
    if mode == "independent" and config["condition"] in CONDITIONS:
        return (config["condition"],)
    raise ValueError("APH requires joint training or one named condition for independent training")


def _model(config):
    conditions = _conditions(config)
    features = {"id": "id", "fp": "chem"}[config["encoding"]]
    kwargs = dict(esm_dim=EMBEDDING_WIDTH, fp_dim=1024, hidden=config["hidden_dim"],
                  dropout=config["dropout"], n_conditions=len(conditions), id_dim=len(CONDITIONS),
                  joint=config["training_mode"] == "joint")
    if config["arm"] == "plain":
        return MLPFitness(ligand_kind=features, **kwargs)
    if config["arm"] == "proto" and tuple(config["slots"]) == (6, 8):
        experiment = dict(position="site", module="prototype", banks="intermediate",
                          activation_weight=0.0, diversity_weight=0.001, ligand_kind=features)
        return PrototypeMLP(experiment, **kwargs)
    raise ValueError("Unrecognized APH arm or slot counts")


def count_parameters(config):
    return sum(p.numel() for p in _model(config).parameters() if p.requires_grad)


class Inputs:
    def __init__(self, root, wt, rows, conditions, embedding, device, encoding):
        self.device = torch.device(device)
        embedding = torch.from_numpy(embedding)
        self.wt_mean = embedding.mean(dim=0).to(self.device)
        reference = root / "data/reference"
        positions = torch.tensor([int(r["position_1based"]) - 1 for r in rows])
        first = torch.tensor([AA_ORDER.index(r["WT"]) for r in rows])
        second = torch.tensor([AA_ORDER.index(r["MT"]) for r in rows])
        eye = torch.eye(20)
        properties = torch.from_numpy(amino_acid_properties(reference))
        features = torch.cat((embedding[positions], eye[first], eye[second] - eye[first],
                              properties[second] - properties[first],
                              (positions[:, None].float() + 1) / len(wt)), dim=-1)
        self.features = features[:, None].to(self.device)
        self.positions = positions[:, None].to(self.device)
        self.labels = torch.tensor([[float(row["y_" + c]) for c in conditions] for row in rows],
                                   dtype=torch.float32, device=self.device)
        panel_columns = torch.tensor([CONDITIONS.index(c) for c in conditions], device=self.device)
        self.onehot = torch.nn.functional.one_hot(panel_columns, num_classes=len(CONDITIONS)).float()
        self.condition_ids = torch.arange(len(conditions), dtype=torch.long, device=self.device)
        self.fingerprints = (torch.from_numpy(chiral_fingerprints(reference, conditions)).to(self.device)
                             if encoding == "fp" else None)

    def batch(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        result = dict(site_inputs=self.features[indices], wt_mean=self.wt_mean,
                      valid_mask=torch.ones((len(indices), 1), dtype=torch.bool, device=self.device),
                      site_positions=self.positions[indices], ligand_ids=self.condition_ids,
                      ligand_onehot=self.onehot)
        if self.fingerprints is not None:
            result["fingerprints"] = self.fingerprints
        return result


def run_fold(config, *, root, cache_dir, device, seed, fold, smoke):
    if (config["split_file"] != SPLIT or config["scaling"] != "train_fold_zscore"
            or config["val_selection"] != "val_standardized_macro_mse"):
        raise ValueError("APH requires its archived split and standardized validation score")
    wt, rows, assignments = _load_inputs(root)
    groups = _partitions(rows, assignments, fold)
    embedding = np.load(cache_dir / "aph.npy", allow_pickle=False)
    if (embedding.shape != (len(wt), EMBEDDING_WIDTH) or embedding.dtype != np.float32
            or not np.isfinite(embedding).all()):
        raise ValueError("APH requires finite float32 embeddings of the archived WT sequence")
    conditions = _conditions(config)
    inputs = Inputs(root, wt, rows, conditions, embedding, device, config["encoding"])
    return fit(config, lambda: _model(config), inputs, groups, conditions, seed, smoke)
