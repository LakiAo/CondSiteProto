"""Native APH standardized objective and held-out scoring."""

from copy import deepcopy
import math
import random

import numpy as np
import torch


def compute_loss(outputs, labels, joint, alpha_shared):
    if labels.ndim != 2 or not torch.isfinite(labels).all():
        raise ValueError("Invalid targets")
    if any(pred.shape != labels.shape or not torch.isfinite(pred).all() for pred in outputs.values()):
        raise ValueError("Invalid predictions")
    specific = (outputs["specific"] - labels).square().mean()
    if not joint:
        return specific
    return (1 - alpha_shared) * specific + alpha_shared * (outputs["shared"] - labels).square().mean()


@torch.no_grad()
def predict(model, inputs, indices, mean, scale, batch_size):
    model.eval()
    parts = []
    for offset in range(0, len(indices), batch_size):
        batch = inputs.batch(indices[offset:offset + batch_size])
        parts.append((model(batch)["specific"] * scale + mean).cpu())
    return torch.cat(parts).numpy()


def fit(config, make_model, inputs, partitions, conditions, seed, smoke):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    train, val, test = (partitions[k] for k in ("train", "validation", "test"))
    if smoke:
        train, val, test = train[:64], val[:32], test[:32]
    train_labels = inputs.labels[train]
    mean = train_labels.mean(dim=0)
    scale = train_labels.std(dim=0, correction=0).clamp_min(1e-6)
    model = make_model().to(inputs.device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    batch_size = config["batch_size"]
    prototype = config["arm"] == "proto"
    joint = config["training_mode"] == "joint"
    if prototype:
        model.refresh_training_anchors((inputs.batch(train[offset:offset + batch_size])
                                        for offset in range(0, len(train), batch_size)), initialize=True)
    torch.manual_seed(seed + 100000)
    best, best_epoch, wait, updates = math.inf, 0, 0, 0
    validation_truth = inputs.labels[val].cpu().numpy()
    for epoch in range(1, (1 if smoke else config["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        for offset in range(0, len(train), batch_size):
            indices = [train[i] for i in order[offset:offset + batch_size]]
            labels = (inputs.labels[indices] - mean) / scale
            batch = inputs.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            outputs, auxiliary = model.forward_training(batch) if prototype else (model(batch), {})
            loss = compute_loss(outputs, labels, joint, config["alpha_shared"])
            for value in auxiliary.values():
                if value.ndim != 0 or not value.requires_grad or not torch.isfinite(value):
                    raise ValueError("Disconnected or nonfinite prototype objective")
                loss = loss + value
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite APH training loss")
            loss.backward()
            if prototype:
                gradient = model.prototype_vectors.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise ValueError("Prototype bank is disconnected from training")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        val_predictions = predict(model, inputs, val, mean, scale, batch_size)
        selection = float(np.mean(((val_predictions - validation_truth) / scale.cpu().numpy()) ** 2))
        if selection < best - config["min_delta"]:
            best, best_epoch, wait = selection, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= config["patience"]:
            break
    if not best_epoch:
        raise RuntimeError("Validation never produced a finite APH score")
    model.load_state_dict(best_state)
    restored_validation = predict(model, inputs, val, mean, scale, batch_size)
    restored = float(np.mean(((restored_validation - validation_truth) / scale.cpu().numpy()) ** 2))
    if not math.isclose(restored, best, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Restored APH state does not reproduce the selected validation result")
    y_true = inputs.labels[test].cpu().numpy().astype(np.float64)
    y_pred = predict(model, inputs, test, mean, scale, batch_size).astype(np.float64)
    scores = []
    for col, condition in enumerate(conditions):
        y, p = y_true[:, col], y_pred[:, col]
        r = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0 else None
        scores.append(dict(partition="test_single", condition=condition, n=len(y),
                           mse=float(np.mean((p - y) ** 2)), pearson=r))
    return dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                epochs=epoch, best_epoch=best_epoch, optimizer_steps=updates, metrics=scores)
