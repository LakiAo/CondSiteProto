"""Load the frozen DAOx cohorts and construct the native site inputs."""

from collections import Counter
import csv
from hashlib import sha256

import numpy as np
import torch
from torch.nn import functional as F

from .models import ESM_WIDTH, SUBSTRATES


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
WT_LENGTH = 365
WT_SEQUENCE_SHA256 = "ab4034db06b2a0489bc612d4de672908804338199027a96c2379db4fa7530c9a"
PROCESSED = "data/processed/DAOx_variants.csv"
PROCESSED_SHA256 = "fec84b8481ff0fde803330335e33511837dd70e6ec86d18f60144e5481acc61a"
SPLITS = "data/splits/daox_assignments.csv"
SPLITS_SHA256 = "ec2b2a283187e24c4626a822561e4e27573053d9e70cb959b4005c3f361284b2"
FP_SHA256 = "4eebecb894dddfec4303a5b2495c265154bdbbb4e4bb6f9583a3c20343e60c37"
SPLIT_FIELDS = ("mutation", "partition", "validation_fold")


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_frozen(root, split_file):
    source = root / PROCESSED
    splits = root / split_file
    if sha256(source.read_bytes()).hexdigest() != PROCESSED_SHA256:
        raise ValueError("DAOx processed table does not match the original frozen archive")
    if sha256(splits.read_bytes()).hexdigest() != SPLITS_SHA256:
        raise ValueError("DAOx split projection checksum differs from the archived assignments")
    rows = _read_csv(source)
    assignments = _read_csv(splits)
    expected = [{key: row[key] for key in SPLIT_FIELDS} for row in rows]
    if assignments != expected:
        raise ValueError("DAOx split projection does not match the processed table")
    fasta = (root / "data/reference/daox_wt.fasta").read_text(encoding="utf-8")
    wt = "".join(line.strip() for line in fasta.splitlines() if not line.startswith(">"))
    if len(wt) != WT_LENGTH or sha256(wt.encode("ascii")).hexdigest() != WT_SEQUENCE_SHA256:
        raise ValueError("DAOx WT sequence differs from the archived assay construct")
    return wt, rows


def verify_splits(root):
    wt, rows = _load_frozen(root, SPLITS)
    by_fold = {fold: set() for fold in range(-1, 5)}
    seen = set()
    for row in rows:
        pos = int(row["position"])
        fold = int(row["validation_fold"])
        old, new = row["wt_aa"], row["mt_aa"]
        if (not 1 <= pos <= WT_LENGTH or wt[pos - 1] != old
                or new not in AA_ORDER or new == old or row["mutation"] != f"{old}{pos}{new}"):
            raise ValueError("Invalid DAOx assay substitution")
        if row["partition"] != ("test" if fold == -1 else "development"):
            raise ValueError("DAOx test/development assignments differ from the fold IDs")
        sequence = wt[:pos - 1] + new + wt[pos:]
        if sequence in seen:
            raise ValueError("DAOx variant sequence repeats within or across partitions")
        seen.add(sequence)
        by_fold[fold].add(sequence)
    counts = Counter({key: len(value) for key, value in by_fold.items()})
    expected = {-1: 580, **{fold: 1044 for fold in range(5)}}
    if counts != expected or len(rows) != 5800:
        raise ValueError("DAOx cohort sizes differ from the original holdout and validation folds")
    return {
        "variants": len(seen),
        "folds": [{"train": 4176, "validation": 1044, "test": 580} for _ in range(5)],
        "sequence_leakage": 0,
        "archived_processed_sha256": PROCESSED_SHA256,
        "split_sha256": SPLITS_SHA256,
    }


def _fingerprints(root):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    descriptors = {
        row["condition"]: row["canonical_isomeric_smiles"]
        for row in _read_csv(root / "data/reference/small_molecules_20.csv")
        if row["dataset"] == "daox"
    }
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
    matrix = np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(descriptors[name]))
                       for name in SUBSTRATES])
    if sha256(matrix.astype(np.uint8).tobytes()).hexdigest() != FP_SHA256:
        raise ValueError("DAOx fingerprints differ from the archived RDKit fingerprints")
    return matrix


class Inputs:
    def __init__(self, root, cache_dir, split_file, feature_kind, device):
        self.device = torch.device(device)
        _, rows = _load_frozen(root, split_file)
        embedding = np.load(cache_dir / "daox.npy", allow_pickle=False)
        if (embedding.shape != (WT_LENGTH, ESM_WIDTH) or embedding.dtype != np.float32
                or not np.isfinite(embedding).all()):
            raise ValueError("Invalid DAOx WT ESM-C tensor")
        wt_esm = torch.from_numpy(np.ascontiguousarray(embedding))

        position = torch.tensor([int(r["position"]) for r in rows], dtype=torch.long)
        wt_idx = torch.tensor([AA_ORDER.index(r["wt_aa"]) for r in rows])
        mt_idx = torch.tensor([AA_ORDER.index(r["mt_aa"]) for r in rows])
        wt_aa = F.one_hot(wt_idx, len(AA_ORDER)).float()
        mt_aa = F.one_hot(mt_idx, len(AA_ORDER)).float()
        with (root / "data/reference/aa_properties_kras_nora_daox.csv").open(
                encoding="utf-8", newline="") as handle:
            raw = csv.reader(handle)
            next(raw)
            raw_properties = {row[0]: tuple(float(value) for value in row[1:]) for row in raw}
        properties = torch.tensor([raw_properties[a] for a in AA_ORDER], dtype=torch.float32)
        properties = (properties - properties.mean(0)) / properties.std(0).clamp_min(1e-6)
        self.sites = torch.cat((
            wt_esm[position - 1], wt_aa, mt_aa - wt_aa,
            properties[mt_idx] - properties[wt_idx], position[:, None].float() / WT_LENGTH,
        ), dim=1).to(self.device)
        self.wt_summary = wt_esm.mean(dim=0).to(self.device)
        self.substrate_features = {"id": torch.eye(len(SUBSTRATES), device=self.device)}
        if feature_kind == "fp":
            self.substrate_features["fp"] = torch.from_numpy(_fingerprints(root)).to(
                self.device, dtype=torch.float32)
        self.targets = np.array([[float(row[f"y_{key}"]) for key in SUBSTRATES]
                                 for row in rows], dtype=np.float64)
        self.target_tensor = torch.from_numpy(self.targets).to(self.device, dtype=torch.float32)
        self.folds = np.array([int(row["validation_fold"]) for row in rows])

    def split(self, fold):
        return {
            "train": np.flatnonzero((self.folds >= 0) & (self.folds != fold)),
            "validation": np.flatnonzero(self.folds == fold),
            "test": np.flatnonzero(self.folds < 0),
        }
