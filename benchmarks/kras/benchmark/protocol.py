"""Frozen Block 1 input identity and the KRAS experiment matrix."""

import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path


CONDITIONS = ("RAF1RBD", "PIK3CGRBD", "RALGDSRBD", "SOS1", "K27", "K55")
N_FOLDS = 5
SETTINGS = {"batch_size": 32, "learning_rate": 0.0006, "weight_decay": 0.0001,
            "epochs": 150, "patience": 20, "dropout": 0.2, "temperature": 0.25,
            "diversity_weight": 0.001, "shared_head_weight": 0.5, "gradient_clip": 5.0}
ALLOCATION = {"plain": (0, 0), "adapter": (0, 0), "s06_p08": (6, 8)}
INPUT_HASHES = {
    "kras_bindingpca_block1_all6.csv": "49eb96895c9f36843d45eaa5769b5ea31b8b42c4edcfb9a039c68b932528ee03",
    "P01116-2.fasta": "8bcf1d107ea750b5ddc8a0f80b60227d16a3e559a6744798aea46b65fc97663b",
}
AA = "ACDEFGHIKLMNPQRSTVWY"
MUTATION = re.compile(rf"([{AA}])(\d+)([{AA}])")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(root):
    table = root / "data/processed/kras_bindingpca_block1_all6.csv"
    fasta = root / "data/reference/kras_wt.fasta"
    for path, name in ((table, table.name), (fasta, "P01116-2.fasta")):
        if sha256(path) != INPUT_HASHES[name]:
            raise ValueError(f"Frozen input changed: {name}")
    sequence = "".join(fasta.read_text(encoding="utf-8").splitlines()[1:])
    if len(sequence) != 188 or set(sequence) - set(AA):
        raise ValueError("Invalid reference FASTA")
    with table.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    names, tokens, targets, errors = [], [], [], []
    for row in rows:
        mutations = row["mutations"].split(";")
        if len(mutations) != int(row["n_mutations"]) or len(mutations) not in (1, 2):
            raise ValueError("Unexpected substitution count")
        sites = []
        for mutation in mutations:
            match = MUTATION.fullmatch(mutation)
            if match is None:
                raise ValueError(f"Invalid substitution: {mutation}")
            ref, pos, alt = match.groups()
            pos = int(pos)
            if not 2 <= pos <= len(sequence) or sequence[pos - 1] != ref or ref == alt:
                raise ValueError(f"Substitution disagrees with WT: {mutation}")
            sites.append((pos, ref, alt))
        if [entry[0] for entry in sites] != sorted({entry[0] for entry in sites}):
            raise ValueError(f"Non-canonical variant: {row['mutations']}")
        scores = [float(row[f"{name}_score"]) for name in CONDITIONS]
        se = [float(row[f"{name}_se"]) for name in CONDITIONS]
        if not all(math.isfinite(v) for v in scores + se) or min(se) <= 0:
            raise ValueError(f"Invalid assay label or SE: {row['mutations']}")
        names.append(row["mutations"])
        tokens.append(tuple(sites))
        targets.append(scores)
        errors.append(se)
    if len(set(names)) != len(names) or Counter(map(len, tokens)) != {1: 1166, 2: 7917}:
        raise ValueError("Unexpected number of unique variants")
    return sequence, names, tokens, targets, errors


def validate_splits(names, sites, split):
    if split["input_sha256"] != INPUT_HASHES or len(split["folds"]) != N_FOLDS:
        raise ValueError("Split doesn't match the frozen input protocol")
    universe = {name: len(mutations) for name, mutations in zip(names, sites)}
    validations = []
    for fold in split["folds"]:
        panels = [(1, fold["train_single"]), (2, fold["double_rank"]),
                  (1, fold["val_single"]), (2, fold["val_double"]),
                  (1, split["test"]["single"]), (2, split["test"]["double"])]
        counts = Counter(name for _, panel in panels for name in panel)
        if set(counts) != universe.keys() or max(counts.values()) != 1:
            raise ValueError("Train, validation and test must partition complete genotypes")
        if any(universe[name] != order for order, panel in panels for name in panel):
            raise ValueError("Single/double variant partitions are inconsistent")
        validations.extend(fold["val_single"] + fold["val_double"])
    dev = universe.keys() - set(split["test"]["single"] + split["test"]["double"])
    if set(validations) != dev or len(validations) != len(dev):
        raise ValueError("Expected each development variant in exactly one validation fold")


def selected_names(fold, regime, budget):
    doubles = fold["double_rank"][:math.floor(len(fold["double_rank"]) * budget / 100 + 0.5)]
    singles = fold["train_single"] if regime == "mixed" else []
    return singles, doubles
