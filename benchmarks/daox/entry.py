"""Expose the archive's DAOx model and training loop through the release CLI."""

import numpy as np
import torch
from torch.nn import functional as F

from .inputs import Inputs, verify_splits
from .models import JointMLP, SUBSTRATES


def _settings(config):
    if (config["backbone"] != "mlp" or config["arm"] not in ("plain", "proto")
            or config["training_mode"] not in ("joint", "independent")
            or config["encoding"] not in ("id", "fp") or config["scaling"] != "native"
            or config["val_selection"] != "val_single_macro_mse"):
        raise ValueError("DAOx requires its native MLP model, label scale and validation criterion")
    shared, private = config["slots"] if config["arm"] == "proto" else (0, 0)
    return {
        "hidden": config["hidden_dim"],
        "dropout": config["dropout"],
        "prototype_shared": shared,
        "prototype_private_per_substrate": private,
        "prototype_temperature": config["prototype_temperature"],
        "diversity_weight": config["diversity_weight"],
    }


def _build(config, condition_id):
    width = len(SUBSTRATES) if config["encoding"] == "id" else 1024
    return JointMLP(width, _settings(config), prototypes=config["arm"] == "proto",
                    active_condition_id=condition_id)


def count_parameters(config):
    condition_id = 0 if config["training_mode"] == "independent" else None
    return sum(param.numel() for param in _build(config, condition_id).parameters() if param.requires_grad)


def _forward(model, inputs, indices, feature_kind, *, with_diversity=False):
    return model(inputs.sites[indices], inputs.wt_summary, inputs.substrate_features[feature_kind],
                 with_diversity=with_diversity)


@torch.no_grad()
def _predict(model, inputs, indices, feature_kind, batch_size):
    model.eval()
    predictions = [_forward(model, inputs, batch, feature_kind)[0].cpu()
                   for batch in indices.split(batch_size)]
    return torch.cat(predictions).numpy()


def _mse(truth, prediction):
    error = np.asarray(prediction, dtype=np.float64) - truth
    return float(np.mean(error**2))


def _macro_mse(truth, prediction):
    return float(np.mean([_mse(truth[:, j], prediction[:, j]) for j in range(truth.shape[1])]))


def _fit_fold(config, inputs, seed, fold, smoke, condition_id):
    target_columns = slice(None) if condition_id is None else slice(condition_id, condition_id + 1)
    conditions = SUBSTRATES if condition_id is None else (SUBSTRATES[condition_id],)
    raw_indices = inputs.split(fold)
    if smoke:
        sizes = {"train": 96, "validation": 32, "test": 32}
        raw_indices = {name: indices[np.linspace(0, len(indices) - 1, sizes[name], dtype=int)]
                       for name, indices in raw_indices.items()}
    split = {name: torch.as_tensor(indices, device=inputs.device)
             for name, indices in raw_indices.items()}
    training = split["train"]
    validation_truth = inputs.targets[raw_indices["validation"], target_columns]
    feature_kind = config["encoding"]
    batch_size = int(config["batch_size"])
    use_prototypes = config["arm"] == "proto"
    torch.manual_seed(seed)
    model = _build(config, condition_id).to(inputs.device)
    if use_prototypes:
        model.initialize_prototypes(inputs.sites[training])
        initial_prototypes = model.prototypes.vectors.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    # The initialization must not perturb the native dropout or training sampler.
    torch.manual_seed(seed)
    sampler = torch.Generator().manual_seed(seed)
    best_mse, best_epoch, best_state = float("inf"), 0, None
    maximum_prototype_gradient = 0.0
    alpha = config["shared_head_weight"]
    for epoch in range(1, (1 if smoke else config["epochs"]) + 1):
        model.train()
        batches = training[torch.randperm(len(training), generator=sampler).to(inputs.device)]
        for ids in batches.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            specific, shared, diversity = _forward(model, inputs, ids, feature_kind,
                                                   with_diversity=use_prototypes)
            labels = inputs.target_tensor[ids, target_columns]
            supervision = (1 - alpha) * F.mse_loss(specific, labels) + alpha * F.mse_loss(shared, labels)
            regularizer = config["diversity_weight"] * diversity if use_prototypes else 0.0
            loss = supervision + regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite DAOx training objective")
            loss.backward()
            if use_prototypes:
                gradient = model.prototypes.vectors.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise ValueError("DAOx prototype gradient is missing or nonfinite")
                maximum_prototype_gradient = max(maximum_prototype_gradient, float(gradient.norm().detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        validation = _predict(model, inputs, split["validation"], feature_kind, batch_size)
        validation_mse = _macro_mse(validation_truth, validation)
        if validation_mse < best_mse:
            best_mse, best_epoch = validation_mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= config["patience"]:
            break

    model.load_state_dict(best_state)
    if use_prototypes:
        movement = float((model.prototypes.vectors - initial_prototypes).norm().detach())
        if maximum_prototype_gradient <= 0.0 or movement <= 0.0:
            raise ValueError("DAOx prototypes did not move from their training-fold initialization")
    validation = _predict(model, inputs, split["validation"], feature_kind, batch_size)
    if not np.isclose(_macro_mse(validation_truth, validation), best_mse, rtol=1e-6, atol=1e-9):
        raise ValueError("Restored DAOx model does not reproduce its selected validation score")
    prediction = _predict(model, inputs, split["test"], feature_kind, batch_size)
    truth = inputs.targets[raw_indices["test"], target_columns]
    metrics = []
    for j, condition in enumerate(conditions):
        y, yhat = truth[:, j], prediction[:, j].astype(np.float64)
        pearson = float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 1 and np.ptp(y) and np.ptp(yhat) else None
        metrics.append({"partition": "test_single", "condition": condition, "n": len(y),
                        "mse": _mse(y, yhat), "pearson": pearson})
    return metrics, epoch


def run_fold(config, *, root, cache_dir, device, seed, fold, smoke):
    if config["double_budget"] != 0 or fold not in range(5) or seed not in config["seeds"]:
        raise ValueError("DAOx uses the archived single-substitution panel and five folds")
    torch.set_num_threads(int(config["cpu_threads"]))
    inputs = Inputs(root, cache_dir, config["split_file"], config["encoding"], device)
    condition_ids = range(len(SUBSTRATES)) if config["training_mode"] == "independent" else (None,)
    runs = [_fit_fold(config, inputs, seed, fold, smoke, condition_id) for condition_id in condition_ids]
    return {"parameters": count_parameters(config),
            "metrics": [score for metrics, _ in runs for score in metrics],
            "epochs": max(epochs for _, epochs in runs)}
