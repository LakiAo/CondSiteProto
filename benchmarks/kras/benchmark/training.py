"""The KRAS training objective and single-mutant validation selection."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from ..alignment_v1.aligned import routed_model
from .model import parameters
from .protocol import CONDITIONS, SETTINGS, selected_names


@torch.no_grad()
def predict(model, data, panel):
    model.eval()
    specific = []
    for batch in panel.split(128):
        prediction, _, _ = model(data, batch)
        specific.append(prediction.cpu())
    return torch.cat(specific).numpy()


def partition_indices(data, split, job, smoke):
    fold = split["folds"][job["fold"]]
    singles, doubles = selected_names(fold, job["regime"], job["budget"])
    validation = fold["val_single"]
    if smoke:
        singles, doubles = singles[:80], doubles[:80]
        validation = validation[:40]
    return {
        "train": data.index(singles + doubles),
        "val_single": data.index(validation),
        "test_single": data.index(split["test"]["single"]),
        "test_double": data.index(split["test"]["double"]),
    }


def condition_scores(truth, prediction, conditions):
    prediction = prediction.astype(np.float64, copy=False)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Invalid KRAS predictions")
    rows = []
    for condition, actual, fitted in zip(conditions, truth.T, prediction.T):
        pearson = None
        if len(actual) > 1 and np.ptp(actual) != 0 and np.ptp(fitted) != 0:
            pearson = float(np.corrcoef(actual, fitted)[0, 1])
        rows.append({
            "condition": condition,
            "n": len(actual),
            "mse": float(np.mean((fitted - actual) ** 2)),
            "pearson": pearson,
        })
    return rows


def fit(data, split, job, smoke):
    parts = partition_indices(data, split, job, smoke)
    torch.manual_seed(job["seed"])
    model = routed_model(job).to(data.device)
    model.initialize(data, parts["train"])
    initial = model.prototype.vectors.detach().clone() if model.prototype is not None else None
    torch.manual_seed(job["seed"])
    rng = torch.Generator().manual_seed(job["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"],
    )
    columns = list(model.columns)
    conditions = tuple(CONDITIONS[i] for i in columns)
    targets = data.targets[:, columns]
    val_single_targets = data.values[parts["val_single"].cpu().numpy()][:, columns]
    best, best_epoch = float("inf"), 0
    for epoch in range(1, (1 if smoke else SETTINGS["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(parts["train"]), generator=rng).to(data.device)
        indices = parts["train"][permutation]
        for batch in indices.split(SETTINGS["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            specific, shared, penalty = model(data, batch, diversity=initial is not None)
            loss = F.mse_loss(specific, targets[batch])
            if shared is not None:
                alpha = SETTINGS["shared_head_weight"]
                loss = (1 - alpha) * loss + alpha * F.mse_loss(shared, targets[batch])
            if penalty is not None:
                loss = loss + SETTINGS["diversity_weight"] * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), SETTINGS["gradient_clip"], error_if_nonfinite=True,
            )
            optimizer.step()
        prediction = predict(model, data, parts["val_single"])
        rows = condition_scores(val_single_targets, prediction, conditions)
        mse = float(np.mean([row["mse"] for row in rows]))
        if mse < best:
            best, best_epoch = mse, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch - best_epoch >= SETTINGS["patience"]:
            break

    model.load_state_dict(best_state, strict=True)
    metrics = []
    for partition in ("test_single", "test_double"):
        panel = parts[partition]
        prediction = predict(model, data, panel)
        truth = data.values[panel.cpu().numpy()][:, columns]
        metrics.extend(dict(row, partition=partition) for row in condition_scores(truth, prediction, conditions))
    if initial is not None:
        movement = float((model.prototype.vectors.detach() - initial).norm())
        if not math.isfinite(movement) or movement <= 0:
            raise ValueError("KRAS prototype parameters did not receive usable training updates")
    return {"parameters": parameters(model), "metrics": metrics, "epochs": epoch, "best_epoch": best_epoch}
