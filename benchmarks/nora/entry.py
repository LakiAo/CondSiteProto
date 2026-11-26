"""NorA entry points for the fixed Table 2 protocol."""

import numpy as np

from .inputs import (ASSIGNMENTS, ASSIGNMENTS_SHA256, MASTER_SHA256, Inputs, check_native_rows,
                     read_modeling_rows, read_wt)
from .models import JointMLP
from .schema import CONDITIONS
from .training import fit_fold


def _model_settings(config):
    if (config["backbone"], config["scaling"], config["val_selection"]) != (
        "mlp", "fold_zscore", "specific_val_macro_mse_z"
    ):
        raise ValueError("NorA uses the MLP with fold-standardized validation MSE")
    return {
        "hidden": config["hidden_dim"],
        "dropout": config["dropout"],
        "prototype_shared": config["slots"][0],
        "prototype_private_per_substrate": config["slots"][1],
        "prototype_temperature": config["prototype_temperature"],
        "diversity_weight": config["diversity_weight"],
    }


def _panel_columns(config):
    if config["training_mode"] == "joint":
        return [list(range(len(CONDITIONS)))]
    if config["training_mode"] == "independent":
        return [[index] for index in range(len(CONDITIONS))]
    raise ValueError("NorA training_mode must be joint or independent")


def count_parameters(config):
    if config["arm"] not in ("plain", "proto"):
        raise ValueError("NorA supports Plain and Proto model arms")
    ligand_width = {"id": len(CONDITIONS), "fp": 1024}[config["encoding"]]
    model = JointMLP(ligand_width, _model_settings(config),
                     prototypes=config["arm"] == "proto", n_conditions=len(_panel_columns(config)[0]))
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def verify_splits(root):
    rows, folds = read_modeling_rows(root, ASSIGNMENTS)
    check_native_rows(rows, read_wt(root))
    sequences = np.array([row["sequence_id"] for row in rows])
    test_mask = folds == -1
    test = set(sequences[test_mask])
    for fold in range(5):
        train = set(sequences[(folds >= 0) & (folds != fold)])
        validation = set(sequences[folds == fold])
        if train & validation or train & test or validation & test:
            raise ValueError(f"NorA sequence leakage in fold {fold}")
    return {"sequence_leakage": 0, "n_test": len(test), "n_variants": len(rows),
            "n_validation": [int((folds == fold).sum()) for fold in range(5)],
            "archive_sha256": MASTER_SHA256, "split_sha256": ASSIGNMENTS_SHA256}


def run_fold(config, *, root, cache_dir, device, seed, fold, smoke):
    inputs = Inputs(root, config["split_file"], cache_dir, device, config["encoding"])
    results = [fit_fold(inputs, config, _model_settings(config), columns, seed, fold, smoke)
               for columns in _panel_columns(config)]
    if len(results) == 1:
        return results[0]
    return {
        "parameters": results[0]["parameters"],
        "metrics": [result["metrics"][0] for result in results],
        "epochs": max(result["epochs"] for result in results),
        "best_epoch_by_condition": {name: result["best_epoch"] for name, result in zip(CONDITIONS, results)},
    }
