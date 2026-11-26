"""RamR distance retrieval and initialization used by aligned KRAS."""

import math

import torch
from torch.nn import functional as F


class PrototypeRouting:
    @staticmethod
    def _bank_weights(logits, available):
        available = available[None, :, None, :]
        minimum = torch.finfo(logits.dtype).min
        return logits.masked_fill(~available, minimum).softmax(dim=-1) * available

    def _match(self, ligand_ids, features, valid, context, exclude_prototypes=(),
               *, ignore_similarity=False):
        rows = torch.stack([self.prototype_indices(i) for i in ligand_ids.tolist()])
        prototypes = F.normalize(self.prototype_vectors[rows], dim=-1)
        squared_norms = (features.square().sum(dim=-1, keepdim=True)
                         + prototypes.square().sum(dim=-1)[None, :, None, :])
        inner_products = torch.einsum("bcmh,ckh->bcmk", features, prototypes)
        distances = (squared_norms - 2 * inner_products).clamp(0, 4)
        activations = torch.exp(-distances / self.temperature)
        attention = self.prototype_attention(context)
        attention = attention.gather(-1, rows.unsqueeze(0).expand(len(attention), -1, -1))
        # Audit-only counterfactual isolates distance matching from prototype
        # values and learned attention. Normal training always uses distances.
        distance_logits = torch.zeros_like(distances) if ignore_similarity else -distances / self.temperature
        logits = distance_logits + attention.unsqueeze(2)
        available = torch.ones_like(rows, dtype=torch.bool)
        for excluded in exclude_prototypes:
            available &= rows != excluded
        if self.shared_count and self.private_count:
            split = self.shared_count
            shared_available, private_available = available[:, :split], available[:, split:]
            shared_weights = self._bank_weights(logits[..., :split], shared_available)
            private_weights = self._bank_weights(logits[..., split:], private_available)
            shared_fraction = self.shared_gate(context).sigmoid().unsqueeze(2)
            shared_fraction = torch.where(~private_available.any(dim=-1)[None, :, None, None],
                                          1.0, shared_fraction)
            shared_fraction = torch.where(~shared_available.any(dim=-1)[None, :, None, None],
                                          0.0, shared_fraction)
            weights = torch.cat((shared_fraction * shared_weights,
                                 (1 - shared_fraction) * private_weights), dim=-1)
        else:
            weights = self._bank_weights(logits, available)
        mask = valid[:, None, :, None]
        weights = weights * mask
        retrieved = torch.einsum("bcmk,ckh->bcmh", weights, prototypes)
        return retrieved, {"activations": activations * mask, "weights": weights,
                           "distances": distances, "prototype_indices": rows}


def _spread_out_features(features, count):
    """Deterministic farthest-point selection, beginning near the feature mean."""
    if len(features) < count:
        # Very small technical smoke subsets can have fewer tokens than prototypes.
        features = features.repeat(math.ceil(count / len(features)), 1)
    distances = (features - features.mean(dim=0)).square().sum(dim=-1)
    norms = features.square().sum(dim=-1)
    selected = []
    for _ in range(count):
        index = distances.argmin() if not selected else distances.argmax()
        point = features[index]
        selected.append(point)
        to_point = norms + norms[index] - 2 * (features @ point)
        distances = to_point if len(selected) == 1 else torch.minimum(distances, to_point)
        distances[index] = -float("inf")
    return torch.stack(selected)


def _activation_overlap(responses):
    """Penalize highly similar prototype activations across distinct variants."""
    if len(responses) < 2 or responses.shape[1] < 2:
        return responses.sum() * 0
    normalized = F.normalize(responses, dim=0)
    similarity = normalized.T @ normalized
    rows, columns = torch.triu_indices(len(similarity), len(similarity), offset=1,
                                     device=similarity.device)
    return (similarity[rows, columns] - 0.6).clamp_min(0).square().mean()
