"""Repository entry points for the fixed six-partner KRAS Block 1 benchmark."""

import json
from hashlib import sha256 as hash_bytes
from math import isclose
from pathlib import Path

import numpy as np
import torch

from .alignment_v1.aligned import routed_model
from .alignment_v1.features import fasta_records, partner_features
from .benchmark.model import Inputs, parameters
from .benchmark.protocol import CONDITIONS, load_inputs, selected_names, sha256, validate_splits
from .benchmark.training import fit


SPLIT_FILE = "data/splits/kras_block1.json"
SPLIT_SHA256 = "c0ff9ca4eb5ef77ed377bf5ba79fa0e90a1eb322ced7911e09d98a87e5a514a4"


def _load_split(path):
    if sha256(path) != SPLIT_SHA256:
        raise ValueError("KRAS split file differs from the frozen archive")
    return json.loads(path.read_text(encoding="utf-8"))


def _partner_matrix(root):
    reference = root / "data/reference"
    sequences = fasta_records(reference / "kras_partner_sequences.fasta")
    provenance = json.loads((reference / "kras_partner_similarity.json").read_text(encoding="utf-8"))
    if list(CONDITIONS) != provenance["order"]:
        raise ValueError("The partner matrix condition order differs from the assay order")
    for name in CONDITIONS:
        if hash_bytes(sequences[name].encode()).hexdigest() != provenance["constructs"][name]["sequence_sha256"]:
            raise ValueError(f"KRAS partner sequence differs from the archived construct: {name}")
    features = np.array(partner_features(sequences, CONDITIONS), dtype=np.float64)
    expected = np.array(provenance["features"], dtype=np.float64)
    if expected.shape != (6, 6) or not all(
        isclose(x, y, abs_tol=1e-12) for x, y in zip(features.flat, expected.flat)
    ):
        raise ValueError("KRAS partner sequence matrix differs from the recorded input")
    matrix = torch.tensor(features, dtype=torch.float32)
    if not torch.allclose(torch.linalg.vector_norm(matrix, dim=1), torch.ones(6)):
        raise ValueError("KRAS partner sequence matrix is not normalized to unit row norm")
    if int(torch.linalg.matrix_rank(matrix)) != 6:
        raise ValueError("KRAS partner sequence matrix cannot distinguish the six known partners")
    return matrix


def _job(config, seed, fold):
    return {
        "width": config["hidden_dim"],
        "arm": {"plain": "plain", "proto": "s06_p08", "adapter": "adapter"}[config["arm"]],
        "condition": "all",
        "regime": "mixed",
        "budget": config["double_budget"],
        "seed": seed,
        "fold": fold,
    }


def count_parameters(config):
    """Count parameters of the exact backbone and bank allocation."""
    return parameters(routed_model(_job(config, seed=0, fold=0)))


def run_fold(config, *, root: Path, cache_dir: Path, device: str, seed: int, fold: int, smoke: bool):
    """Train one fit and score the frozen single- and double-mutant test panels."""
    split = _load_split(root / config["split_file"])
    embedding = np.load(cache_dir / "kras.npy", allow_pickle=False)
    data = Inputs(root, split, embedding, torch.device(device))
    if config["encoding"] == "seqsim":
        data.partner_ids = _partner_matrix(root).to(data.device)
    return fit(data, split, _job(config, seed, fold), smoke)


def verify_splits(root: Path):
    """Verify complete-genotype isolation, archive checksums, and partner features."""
    split = _load_split(root / SPLIT_FILE)
    sequence, names, tokens, _, _ = load_inputs(root)
    validate_splits(names, tokens, split)
    variants = set()
    for sites in tokens:
        variant = list(sequence)
        for pos, _, alternate in sites:
            variant[pos - 1] = alternate
        variants.add("".join(variant))
    if len(variants) != len(names):
        raise ValueError("Different KRAS identifiers encode the same variant sequence")
    test_single = len(split["test"]["single"])
    test_double = len(split["test"]["double"])
    train_counts = [list(map(len, selected_names(fold, "mixed", 20))) for fold in split["folds"]]
    mutable_positions = len({pos for sites in tokens for pos, _, _ in sites})
    if (test_single, test_double, mutable_positions) != (117, 792, 63):
        raise ValueError("KRAS test counts or mutable-position count differ from the frozen benchmark")
    if train_counts != [[839, 1140]] * 4 + [[840, 1140]]:
        raise ValueError("KRAS frozen 20% double-mutant allocation changed")
    _partner_matrix(root)
    return {
        "split_sha256": SPLIT_SHA256,
        "variants": len(variants),
        "single": sum(len(sites) == 1 for sites in tokens),
        "double": sum(len(sites) == 2 for sites in tokens),
        "mutable_positions": mutable_positions,
        "test_single": test_single,
        "test_double": test_double,
        "train_single_double_by_fold": train_counts,
        "folds": len(split["folds"]),
        "partner_matrix": "verified (6 x 6)",
    }
