"""Load the archived NorA modeling rows and construct native features."""

import csv
import math
from collections import Counter
from hashlib import sha256

import numpy as np
import torch
from torch.nn import functional as F

from .schema import AA_ORDER, CONDITIONS, ESM_WIDTH, WT_LENGTH


MASTER = "data/processed/NorA_master.csv"
MASTER_SHA256 = "91ea98b91ae43a3a0b2fa356f532e68598ecadce0fb50859ef967adde076c8af"
ASSIGNMENTS = "data/splits/nora_assignments.csv"
ASSIGNMENTS_SHA256 = "647c3059abcb766b1c5570e87120afc5b05693b57b490b6aaf5e6b777498c692"
WT_SHA256 = "d5c40b9291530c7bbbeb33a97ccd13038e61bc641068f6e8a4b7064251c1ca89"
FINGERPRINT_SHA256 = "66c67c19e0f90be2c6034bb3641df46f6ed3ac687867e6e77a1965bfa9fd8556"
FOLD_COUNTS = {-1: 703, 0: 1265, 1: 1265, 2: 1265, 3: 1265, 4: 1264}


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as rows:
        return list(csv.DictReader(rows))


def read_modeling_rows(root, split_file):
    master, assignment_file = root / MASTER, root / split_file
    for path, digest in ((master, MASTER_SHA256), (assignment_file, ASSIGNMENTS_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"NorA archive checksum mismatch for {path.name}")
    rows = _read_csv(master)
    assignments = _read_csv(assignment_file)
    keys = ("mutation", "sequence_id", "validation_fold")
    if assignments != [{key: row[key] for key in keys} for row in rows]:
        raise ValueError("NorA assignments disagree with the archived modeling rows")
    folds = np.array([int(row["validation_fold"]) for row in rows], dtype=np.int64)
    if (len(rows) != 7027 or len({row["sequence_id"] for row in rows}) != 7027
            or {row["mutation_count"] for row in rows} != {"1"} or Counter(folds) != FOLD_COUNTS):
        raise ValueError("NorA archive has unexpected variant or split counts")
    return rows, folds


def read_wt(root):
    with (root / "data/reference/nora_wt.fasta").open(encoding="utf-8-sig") as file:
        wt = "".join(line.strip() for line in file if not line.startswith(">"))
    if sha256(wt.encode("ascii")).hexdigest() != WT_SHA256 or len(wt) != WT_LENGTH:
        raise ValueError("Unexpected NorA WT sequence")
    return wt


def check_native_rows(rows, wt):
    for row in rows:
        pos = int(row["position"]) - 1
        before, after = row["wt_aa"], row["mt_aa"]
        if row["mutation"] != f"{before}{pos + 1}{after}" or wt[pos] != before:
            raise ValueError("NorA mutation and WT sequence disagree")
        sequence = wt[:pos] + after + wt[pos + 1:]
        if sha256(sequence.encode("ascii")).hexdigest() != row["sequence_id"]:
            raise ValueError("NorA mutation and sequence ID disagree")
        water = float(row["water_f_score_log2"])
        for condition in CONDITIONS:
            target = float(row[f"y_{condition}"])
            if not math.isfinite(target) or not math.isclose(
                    target, float(row[f"raw_{condition}"]) - water, abs_tol=1e-12):
                raise ValueError("NorA control-subtracted label arithmetic failed")


def fingerprint_panel(root):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    rows = _read_csv(root / "data/reference/small_molecules_20.csv")
    smiles = {row["condition"]: row["canonical_isomeric_smiles"] for row in rows if row["dataset"] == "nora"}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
    bits = np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles[name])) for name in CONDITIONS])
    if sha256(bits.astype(np.uint8).tobytes()).hexdigest() != FINGERPRINT_SHA256:
        raise ValueError("NorA Morgan fingerprints differ from the archived bit vectors")
    return bits


def fit_target_scaler(targets, train_indices, conditions):
    train = targets[np.asarray(train_indices)][:, conditions]
    mean, scale = train.mean(0), train.std(0)
    if not np.isfinite(train).all() or np.any(scale < 1e-8):
        raise ValueError("Invalid or constant training labels")
    return mean, scale


class Inputs:
    def __init__(self, root, split_file, cache_dir, device, encoding):
        self.device = torch.device(device)
        self.rows, self.folds = read_modeling_rows(root, split_file)
        check_native_rows(self.rows, read_wt(root))
        array = np.load(cache_dir / "nora.npy", allow_pickle=False)
        if (array.shape != (WT_LENGTH, ESM_WIDTH) or array.dtype != np.float32
                or not np.isfinite(array).all()):
            raise ValueError("Expected a finite 388×1152 float32 NorA WT embedding")
        embedding = torch.from_numpy(np.ascontiguousarray(array))
        positions = torch.tensor([int(row["position"]) for row in self.rows])
        before = torch.tensor([AA_ORDER.index(row["wt_aa"]) for row in self.rows])
        after = torch.tensor([AA_ORDER.index(row["mt_aa"]) for row in self.rows])
        wt_aa = F.one_hot(before, 20).float()
        mt_aa = F.one_hot(after, 20).float()
        with (root / "data/reference/aa_properties_kras_nora_daox.csv").open(
                newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            table = {row[0]: tuple(map(float, row[1:])) for row in reader}
        properties = torch.tensor([table[aa] for aa in AA_ORDER], dtype=torch.float32)
        properties = (properties - properties.mean(0)) / properties.std(0).clamp_min(1e-6)
        self.sites = torch.cat((
            embedding[positions - 1], wt_aa, mt_aa - wt_aa,
            properties[after] - properties[before], positions[:, None].float() / WT_LENGTH,
        ), 1).to(device)
        self.wt_summary = embedding.mean(0).to(device)
        self.features = {"id": torch.eye(len(CONDITIONS), device=device)}
        if encoding == "fp":
            self.features["fp"] = torch.tensor(fingerprint_panel(root), dtype=torch.float32, device=device)
        self.targets = np.array([[float(row[f"y_{name}"]) for name in CONDITIONS]
                                 for row in self.rows], dtype=np.float64)

    def split(self, fold):
        return {"train": np.flatnonzero((self.folds >= 0) & (self.folds != fold)),
                "validation": np.flatnonzero(self.folds == fold), "test": np.flatnonzero(self.folds < 0)}
