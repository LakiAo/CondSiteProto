"""Native NorA fold-local normalized regression."""

import random

import numpy as np
import torch
from torch.nn import functional as F

from .inputs import fit_target_scaler
from .models import JointMLP
from .schema import CONDITIONS


def set_seed(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def supervised_loss(specific, shared, labels, shared_weight=.5):
    if specific.shape != labels.shape or not torch.isfinite(labels).all():
        raise ValueError("Selected condition labels do not match model output")
    specific_loss = F.mse_loss(specific, labels)
    if shared is None:
        return specific_loss
    if shared.shape != labels.shape:
        raise ValueError("Shared output shape mismatch")
    return (1 - shared_weight) * specific_loss + shared_weight * F.mse_loss(shared, labels)


@torch.inference_mode()
def predict(model, inputs, indices, condition_features, batch_size):
    model.eval()
    specific = []
    for ids in torch.as_tensor(indices, device=inputs.device).split(batch_size):
        output, _, _ = model(inputs.sites[ids], inputs.wt_summary, condition_features)
        specific.append(output.cpu())
    return torch.cat(specific).numpy()


def subset_for_smoke(indices, count):
    return indices[np.linspace(0, len(indices) - 1, min(count, len(indices)), dtype=int)]


def fit_fold(inputs, config, model_settings, columns, seed, fold, smoke):
    split = inputs.split(fold)
    if smoke:
        split = {key: subset_for_smoke(ids, 128 if key == "train" else 24) for key, ids in split.items()}
    train = split["train"]
    mean, scale = fit_target_scaler(inputs.targets, train, columns)
    raw_targets = inputs.targets[:, columns]
    normalized = torch.tensor((raw_targets - mean) / scale, dtype=torch.float32, device=inputs.device)
    condition_features = inputs.features[config["encoding"]][columns]
    use_prototypes = config["arm"] == "proto"
    set_seed(seed, config["cpu_threads"])

    def new_model():
        return JointMLP(condition_features.shape[1], model_settings, prototypes=use_prototypes,
                        n_conditions=len(columns)).to(inputs.device)

    model = new_model()
    if model.prototypes is not None:
        model.initialize_prototypes(inputs.sites[torch.as_tensor(train, device=inputs.device)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    # The archived runs used matched minibatch/dropout streams after prototype initialization.
    set_seed(seed, config["cpu_threads"])
    sampler = torch.Generator().manual_seed(seed)
    batch_size = config["batch_size"]
    epochs = 1 if smoke else config["epochs"]
    best, best_epoch, best_state = float("inf"), 0, None
    maximum_proto_gradient = 0.
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.as_tensor(train, dtype=torch.long)[torch.randperm(len(train), generator=sampler)]
        for batch in order.split(batch_size):
            batch = batch.to(inputs.device)
            optimizer.zero_grad(set_to_none=True)
            specific, shared, diversity = model(
                inputs.sites[batch], inputs.wt_summary, condition_features, with_diversity=use_prototypes
            )
            supervision = supervised_loss(specific, shared, normalized[batch], config["shared_head_weight"])
            penalty = model_settings["diversity_weight"] * diversity if use_prototypes else supervision.new_zeros(())
            loss = supervision + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite NorA training objective")
            loss.backward()
            if use_prototypes:
                gradient = model.prototypes.vectors.grad
                if gradient is None:
                    raise RuntimeError("Prototype parameters are disconnected from the objective")
                maximum_proto_gradient = max(maximum_proto_gradient, float(gradient.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        validation = predict(model, inputs, split["validation"], condition_features, batch_size)
        normalized_mse = float(np.mean((validation - (raw_targets[split["validation"]] - mean) / scale) ** 2))
        if normalized_mse < best:
            best, best_epoch = normalized_mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= config["patience"]:
            break
    if use_prototypes and maximum_proto_gradient <= 0:
        raise RuntimeError("Prototype parameters never received a nonzero training gradient")

    restored = new_model()
    restored.load_state_dict(best_state, strict=True)
    check = predict(restored, inputs, split["validation"], condition_features, batch_size)
    check_mse = float(np.mean((check - (raw_targets[split["validation"]] - mean) / scale) ** 2))
    if not np.isclose(check_mse, best, rtol=1e-6, atol=1e-8):
        raise RuntimeError("Restoring the selected NorA model changed its validation predictions")
    fitted = predict(restored, inputs, split["test"], condition_features, batch_size).astype(np.float64)
    predicted = fitted * scale + mean
    truth = raw_targets[split["test"]]
    metrics = []
    for offset, column in enumerate(columns):
        y, p = truth[:, offset], predicted[:, offset]
        pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 1e-12 and np.std(p) > 1e-12 else None
        metrics.append({"partition": "test_single", "condition": CONDITIONS[column], "n": len(y),
                        "mse": float(np.mean((y - p) ** 2)), "pearson": pearson})
    return {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "metrics": metrics, "epochs": epoch, "best_epoch": best_epoch}
