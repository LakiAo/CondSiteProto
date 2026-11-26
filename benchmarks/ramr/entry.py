"""Execute the archived RamR joint-model protocol from release inputs."""
from collections import defaultdict
from copy import deepcopy
import csv
from functools import lru_cache
from hashlib import sha256
from io import StringIO
import json
import math
from statistics import mean

import numpy as np
import torch
from torch.utils.data import Subset

from .data import MeasurementDataset, sequence_hash
from .mmoe import MMoEHeads
from .models import MLPFitness
from .prototypes import PrototypeMLP
from .schema import AA_ORDER, LIGANDS, TARGET_COLUMNS
from .training import _loader, _small_training_sample, model_inputs, predict, set_seed, to_device, train_epoch

IDS = (0, 1, 2)
PROCESSED = "data/processed/RamR_EP32_single_double_wt.csv"
PAIR_FIELDS = ("position_a", "position_b", "partition", "validation_fold", "n_records")
PAIR_ASSIGNMENTS_SHA256 = "b0fe1f45ff4dbd5fff161d760fbc7b6483493cf6a37cb7332fb9721f09493735"


def _model(config):
    hidden = int(config["hidden_dim"])
    dropout = float(config["dropout"])
    arm = config["arm"]
    if config["encoding"] != "id" or config["training_mode"] != "joint":
        raise ValueError("The RamR release requires the joint condition-ID protocol")
    if arm == "plain" and not config["slots"]:
        model = MLPFitness(1152, 1024, hidden, dropout, joint=True, ligand_kind="id")
    elif arm in ("proto", "adapter"):
        if config["slots"] != ([6, 8] if arm == "proto" else []):
            raise ValueError("RamR prototypes require six shared and eight condition-specific vectors")
        module = "prototype" if arm == "proto" else arm
        experiment = dict(module=module, position="site", banks="intermediate",
                          model_type="mlp_id", activation_weight=0.0,
                          diversity_weight=float(config["diversity_weight"]))
        model = PrototypeMLP(experiment, 1152, 1024, hidden, dropout)
    else:
        raise ValueError(f"Invalid RamR arm or allocation: {arm!r}")
    if config["backbone"] == "mmoe":
        model.heads = MMoEHeads(hidden, dropout)
    elif config["backbone"] != "mlp":
        raise ValueError(f"Invalid RamR backbone: {config['backbone']!r}")
    return model


def count_parameters(config):
    return sum(p.numel() for p in _model(config).parameters() if p.requires_grad)


@lru_cache(maxsize=1)
def _processed(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row["record_id"]: row for row in reader}
        return reader.fieldnames, rows


def _csv_sha256(fields, rows, *, lineterminator="\n"):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator=lineterminator)
    writer.writeheader()
    writer.writerows(rows)
    return sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def _pair(row):
    return tuple(sorted(int(mutation[1:-1]) for mutation in row["mutations"].split(":")))


def _load_split(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def _fold_rows(source, split, fold):
    return {
        key: [source[record_id] for record_id in identifiers]
        for key, identifiers in {
            "train": split["folds"][fold]["train"],
            "validation_single": split["folds"][fold]["validation_single"],
            "validation_double": split["folds"][fold]["validation_double"],
            "test_single": split["test_single"],
            "test_double": split["test_double"],
        }.items()
    }


def _additive(test_rows, train_rows, wt_rows):
    single_labels = defaultdict(list)
    for row in train_rows:
        if row["mutation_count"] == "1":
            single_labels[row["mutations"]].append(row)
    wt_y = [mean(float(row[c]) for row in wt_rows) for c in TARGET_COLUMNS]
    covered, truth, values = [], [], []
    for row in test_rows:
        a, b = row["mutations"].split(":")
        if a in single_labels and b in single_labels:
            covered.append(row["record_id"])
            truth.append([float(row[c]) for c in TARGET_COLUMNS])
            values.append([
                mean(float(item[col]) for item in single_labels[a])
                + mean(float(item[col]) for item in single_labels[b]) - wt_y[i]
                for i, col in enumerate(TARGET_COLUMNS)
            ])
    return covered, np.array(truth), np.array(values)


def verify_splits(root):
    fields, source = _processed(root / PROCESSED)
    for row in source.values():
        if row["sequence_id"] != "seq_" + sequence_hash(row["sequence"]):
            raise ValueError("A RamR sequence identity no longer matches its original sequence")
    summary = {}
    for protocol in ("regular", "unseen_pairs"):
        split = _load_split(root, f"data/splits/ramr_{protocol}.json")
        counts, additive_mse = [], []
        if protocol == "unseen_pairs":
            pair_hash = _csv_sha256(PAIR_FIELDS, split["pairs"], lineterminator="\r\n")
            if pair_hash != PAIR_ASSIGNMENTS_SHA256:
                raise ValueError("RamR position-pair assignments differ from the archived CSV")
            assignments = {(item["position_a"], item["position_b"]): item for item in split["pairs"]}
            n_test_pairs = sum(item["partition"] == "test" for item in assignments.values())
            if n_test_pairs != 1283 or len(split["test_double"]) != 3142:
                raise ValueError("RamR unseen test cohort changed")
        for fold in range(5):
            tables = _fold_rows(source, split, fold)
            partitions = [{row["sequence"] for row in rows} for rows in tables.values()]
            if sum(map(len, partitions)) != len(set.union(*partitions)):
                raise ValueError(f"RamR {protocol} fold {fold} has sequence leakage")
            counts.append({key: len(rows) for key, rows in tables.items()})
            if protocol == "regular":
                for name, rows in tables.items():
                    if _csv_sha256(fields, rows) != split["historical_csv_sha256"][fold][name]:
                        raise ValueError(f"RamR archived hash mismatch at fold {fold}, {name}")
            else:
                p_train = {_pair(row) for row in tables["train"] if row["mutation_count"] == "2"}
                p_val = {_pair(row) for row in tables["validation_double"]}
                p_test = {_pair(row) for row in tables["test_double"]}
                if len(p_test) != 1283 or sum(map(len, [p_train, p_val, p_test])) != len(p_train | p_val | p_test):
                    raise ValueError(f"RamR unseen fold {fold} has position-pair leakage")
                if (any(assignments[p]["validation_fold"] == fold or assignments[p]["partition"] == "test"
                        for p in p_train)
                    or any(assignments[p]["validation_fold"] != fold for p in p_val)
                    or any(assignments[p]["partition"] != "test" for p in p_test)):
                    raise ValueError(f"RamR unseen fold {fold} changed pair assignments")
                covered, truth, model = _additive(tables["test_double"], tables["train"],
                                                 [source[x] for x in split["wt_reference"]])
                if covered != split["folds"][fold]["additive_covered"]:
                    raise ValueError(f"RamR unseen fold {fold} changed additive coverage")
                counts[-1]["test_double_additive_covered"] = len(covered)
                additive_mse.append(float(np.square((model[:, 0] - model[:, 1]) - (truth[:, 0] - truth[:, 1])).mean()))
                expected = split["historical_counts"][fold]
                if (sum(row["mutation_count"] == "1" for row in tables["train"]) != expected["n_train_single"]
                    or sum(row["mutation_count"] == "2" for row in tables["train"]) != expected["n_train_double"]
                    or len(tables["validation_double"]) != expected["n_validation_double"]
                    or len(covered) != expected["n_additive"]):
                    raise ValueError(f"RamR unseen fold {fold} changed archived cohort counts")
        summary[protocol] = {"folds": counts, "sequence_leakage": 0}
        if protocol == "regular":
            summary[protocol]["matching_archived_csv_hashes"] = 25
        else:
            summary[protocol].update(test_pairs=n_test_pairs, test_doubles=len(split["test_double"]),
                                    pair_leakage=0, archived_pair_assignments_sha256=pair_hash,
                                    additive_delta_s_r_rmse=math.sqrt(mean(additive_mse)))
    return summary


class _Inputs:
    """Read the frozen WT residue matrix and collate the archived assay rows."""

    def __init__(self, root, cache_dir):
        fasta = (root / "data/reference/ramr_wt.fasta").read_text(encoding="utf-8-sig")
        self.wt_sequence = "".join(line.strip() for line in fasta.splitlines() if not line.startswith(">"))
        array = np.load(cache_dir / "ramr.npy", allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (len(self.wt_sequence), 1152) or not np.isfinite(array).all():
            raise ValueError("Expected a finite RamR float32 WT array with 1,152 features per residue")
        self.wt_esm = torch.from_numpy(np.ascontiguousarray(array))
        self.wt_aa = torch.eye(20)[torch.tensor([AA_ORDER.index(a) for a in self.wt_sequence])]

    def dataset(self, rows):
        return MeasurementDataset(rows, self.wt_sequence, IDS)

    def collate(self, items, ligand_ids):
        if any(item["ligand_ids"] != tuple(ligand_ids) for item in items):
            raise ValueError("Dataset/collator ligand selections differ")
        batch = {key: torch.stack([item[key] for item in items])
                 for key in ("mut_mask", "aa_delta", "prop_delta", "y", "errs")}
        batch.update(wt_esm=self.wt_esm, wt_aa=self.wt_aa, ligand_ids=torch.tensor(ligand_ids, dtype=torch.long),
                     record_ids=[x["record_id"] for x in items], sequence_ids=[x["sequence_id"] for x in items],
                     mutations=[x["mutations"] for x in items],
                     mutation_count=torch.tensor([x["mutation_count"] for x in items]))
        return batch


def _metrics(truth, prediction, partition, arm):
    result = []
    for c, condition in enumerate((*LIGANDS, "delta_s_r")):
        y = truth[:, c] if c < 3 else truth[:, 0] - truth[:, 1]
        yhat = prediction[:, c] if c < 3 else prediction[:, 0] - prediction[:, 1]
        pearson = float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 1 and y.std() > 0 and yhat.std() > 0 else None
        result.append(dict(partition=partition, condition=condition, n=len(y),
                           mse=float(np.square(yhat - y).mean()), pearson=pearson, arm=arm))
    return result


def _predictions(model, loader, device, experiment, fold, partition, record_ids):
    rows, _ = predict(model, loader, device, experiment=experiment, fold=fold, partition=partition)
    by_id = {(row["record_id"], row["ligand"]): row for row in rows if row["branch"] == "specific"}
    return tuple(np.array([[by_id[(record_id, ligand)][key] for ligand in LIGANDS] for record_id in record_ids])
                 for key in ("y_true", "y_pred"))


def run_fold(config, *, root, cache_dir, device, seed, fold, smoke):
    if config["double_budget"] != 20 or config["val_selection"] != "specific_double_macro_mse" or config["scaling"] != "raw":
        raise ValueError("RamR Tables 2–4 use 20% double training and unscaled specific-head double validation")
    if seed not in config["seeds"] or not 0 <= fold < 5:
        raise ValueError("Seed or fold falls outside the archived experiment")
    _, source = _processed(root / PROCESSED)
    split = _load_split(root, config["split_file"])
    tables = _fold_rows(source, split, fold)
    inputs = _Inputs(root, cache_dir)
    settings = dict(config, seed=seed)
    set_seed(seed, int(settings["cpu_threads"]), bool(settings["deterministic"]))
    datasets = {name: inputs.dataset(rows) for name, rows in tables.items()}
    if smoke:
        datasets["train"] = _small_training_sample(datasets["train"], 16)
        datasets["validation_double"] = Subset(datasets["validation_double"], range(8))
    loaders = {name: _loader(inputs, dataset, IDS, settings, shuffle=name == "train")
               for name, dataset in datasets.items()}
    model = _model(config).to(device)
    if config["arm"] == "proto":
        anchor_feed = _loader(inputs, datasets["train"], IDS, settings)
        model.refresh_training_anchors(
            (to_device(model_inputs(batch), device) for batch in anchor_feed), initialize=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    best_mse, wait = math.inf, 0
    for epoch in range(1, (1 if smoke else int(config["epochs"])) + 1):
        train_epoch(model, loaders["train"], optimizer, device, joint=True,
                    alpha_shared=float(config["alpha_shared"]),
                    gradient_clip=float(config["gradient_clip"]))
        _, val_mse = predict(model, loaders["validation_double"], device,
                             experiment=config["experiment"], fold=fold, partition="validation_double")
        if val_mse < best_mse:
            best_mse, wait, best_state = val_mse, 0, deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= config["patience"]:
            break
    model.load_state_dict(best_state, strict=True)
    metrics = []
    for partition in ("test_single", "test_double"):
        truth, predictions = _predictions(model, loaders[partition], device, config["experiment"],
                                          fold, partition, split[partition])
        metrics.extend(_metrics(truth, predictions, partition, config["arm"]))
    if "additive_covered" in split["folds"][fold]:
        covered, additive_truth, additive_values = _additive(
            tables["test_double"], tables["train"], [source[x] for x in split["wt_reference"]])
        if covered != split["folds"][fold]["additive_covered"]:
            raise ValueError("The fold-specific additive common subset changed")
        test_indices = {record_id: i for i, record_id in enumerate(split["test_double"])}
        common = [test_indices[record_id] for record_id in covered]
        partition = "test_double_additive_covered"
        metrics.extend(_metrics(truth[common], predictions[common], partition, config["arm"]))
        metrics.extend(_metrics(additive_truth, additive_values, partition, "additive"))
    return dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                metrics=metrics, epochs=epoch)
