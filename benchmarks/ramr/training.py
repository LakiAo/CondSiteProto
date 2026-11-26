"""Native optimizer updates, validation scoring, and deterministic loaders."""
from __future__ import annotations

from functools import partial
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .schema import LIGANDS


def to_device(value: Any, device: str | torch.device) -> Any:
    """Move all nested tensors without altering identifiers or metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def model_inputs(batch: dict) -> dict:
    """Labels and posterior errors are available for reporting, never as features."""
    excluded = {"y", "errs", "record_ids", "sequence_ids", "mutations", "mutation_count"}
    return {key: value for key, value in batch.items() if key not in excluded}


def compute_loss(outputs: dict, labels: torch.Tensor, *, joint: bool,
                 alpha_shared: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Equal condition means, with fixed shared/specific branch weights."""
    if labels.ndim != 2 or not torch.isfinite(labels).all():
        raise ValueError("Labels must be a finite [variants, conditions] tensor")
    if not 0 <= alpha_shared <= 1:
        raise ValueError("alpha_shared must be between 0 and 1")
    terms = {}
    for branch in (("specific", "shared") if joint else ("specific",)):
        if branch not in outputs or outputs[branch].shape != labels.shape:
            raise ValueError(f"Missing or invalid {branch} output shape")
        if not torch.isfinite(outputs[branch]).all():
            raise FloatingPointError(f"Nonfinite {branch} predictions")
        terms[branch] = (outputs[branch] - labels).square().mean(dim=0)
    specific = terms["specific"].mean()
    loss = ((1 - alpha_shared) * specific + alpha_shared * terms["shared"].mean()
            if joint else specific)
    detached = {f"{branch}_condition_{i}_mse": float(value.detach())
                for branch, values in terms.items() for i, value in enumerate(values)}
    return loss, detached


def set_seed(seed: int, threads: int = 4, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def train_epoch(model, loader, optimizer, device, *, joint: bool,
                alpha_shared: float = 0.5, gradient_clip: float = 5.0) -> dict:
    model.train()
    total, n, steps, detail = 0.0, 0, 0, {}
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        features = model_inputs(batch)
        if hasattr(model, "forward_training"):
            outputs, auxiliary = model.forward_training(features)
        else:
            outputs, auxiliary = model(features), {}
        loss, parts = compute_loss(outputs, batch["y"], joint=joint,
                                   alpha_shared=alpha_shared)
        if hasattr(model, "forward_training"):
            parts["prediction_loss"] = float(loss.detach())
        for name, term in auxiliary.items():
            if term.ndim != 0 or not term.requires_grad or not torch.isfinite(term):
                raise ValueError(f"Invalid differentiable auxiliary loss: {name}")
            loss = loss + term
            parts[f"weighted_{name}_loss"] = float(term.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        prototypes = getattr(model, "prototype_vectors", None)
        if prototypes is not None:
            if prototypes.grad is None:
                raise RuntimeError("Prototype vectors are disconnected from the training loss")
            parts["prototype_grad_norm"] = float(prototypes.grad.detach().norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                       error_if_nonfinite=True)
        optimizer.step()  # Exactly one update for all ligand conditions together.
        count = len(batch["y"])
        total += float(loss.detach()) * count
        n += count
        steps += 1
        for key, value in parts.items():
            detail[key] = detail.get(key, 0.0) + value * count
    if not n:
        raise ValueError("No training batches")
    return {"train_loss": total / n, "optimizer_steps": steps,
            **{f"train_{key}": value / n for key, value in detail.items()}}


def _list(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else list(value)


@torch.inference_mode()
def predict(model, loader, device, *, experiment: str, fold: int,
            partition: str) -> tuple[list[dict], float]:
    """Preserve record identity for every condition and every prediction branch."""
    model.eval()
    result, squared, total, expected_branches = [], None, 0, None
    seen = set()
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        outputs = model(model_inputs(batch))
        branches = set(outputs)
        if expected_branches is None:
            expected_branches = branches
        elif branches != expected_branches:
            raise ValueError("Prediction branches changed between batches")
        labels = batch["y"]
        if labels.ndim != 2 or not torch.isfinite(labels).all():
            raise ValueError("Prediction requires finite [B,C] labels")
        ids = _list(raw_batch["record_ids"])
        sequences = _list(raw_batch["sequence_ids"])
        mutations = _list(raw_batch["mutations"])
        orders = _list(raw_batch["mutation_count"])
        ligands = [LIGANDS[int(index)] for index in _list(batch["ligand_ids"])]
        truth = labels.detach().cpu().double()
        errs = batch["errs"].detach().cpu().double()
        if len(ids) != labels.shape[0] or len(ligands) != labels.shape[1]:
            raise ValueError("Prediction identities and tensor dimensions disagree")
        for branch, values in outputs.items():
            if branch not in {"specific", "shared", "mean"}:
                raise ValueError(f"Unexpected prediction branch: {branch}")
            if values.shape != labels.shape or not torch.isfinite(values).all():
                raise FloatingPointError(f"Invalid {branch} prediction tensor")
            values = values.detach().cpu().double()
            for i, record_id in enumerate(ids):
                for j, ligand in enumerate(ligands):
                    key = (str(record_id), ligand, branch)
                    if key in seen:
                        raise ValueError(f"Duplicated prediction identity: {key}")
                    seen.add(key)
                    result.append({"experiment": experiment, "fold": fold,
                                   "partition": partition, "branch": branch,
                                   "ligand": ligand, "record_id": str(record_id),
                                   "sequence_id": str(sequences[i]), "mutations": str(mutations[i]),
                                   "mutation_count": int(orders[i]),
                                   "y_true": float(truth[i, j]), "y_pred": float(values[i, j]),
                                   "err": float(errs[i, j])})
        if "specific" not in outputs:
            raise ValueError("Primary specific branch is absent")
        batch_squared = (outputs["specific"] - labels).square().sum(dim=0).cpu().double()
        squared = batch_squared if squared is None else squared + batch_squared
        total += labels.shape[0]
    if not total:
        raise ValueError("No prediction batches")
    if hasattr(loader, "dataset") and total != len(loader.dataset):
        raise ValueError("Prediction did not cover every dataset observation exactly once")
    return result, float((squared / total).mean())


def _loader(inputs, dataset, ligand_ids, settings, *, shuffle=False):
    generator = torch.Generator().manual_seed(int(settings.get("seed", 42)))
    return DataLoader(dataset, batch_size=int(settings.get("batch_size", 32)),
                      shuffle=shuffle, num_workers=0, generator=generator,
                      collate_fn=partial(inputs.collate, ligand_ids=list(ligand_ids)))


def _small_training_sample(dataset, limit=16):
    """Keep both mutation orders in a mixed technical test, if present."""
    rows = dataset.rows
    by_order = {order: [i for i, row in enumerate(rows) if int(row["mutation_count"]) == order]
                for order in sorted({int(row["mutation_count"]) for row in rows})}
    if limit < len(by_order):
        raise ValueError("Technical sample limit cannot cover every training order")
    chosen = []
    while len(chosen) < min(limit, len(rows)):
        for indices in by_order.values():
            if indices and len(chosen) < limit:
                chosen.append(indices.pop(0))
    return Subset(dataset, sorted(chosen))
