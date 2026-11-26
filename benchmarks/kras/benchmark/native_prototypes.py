"""Native prototype initialization retained for the aligned KRAS constructor."""
import torch
from torch import nn
from torch.nn import functional as F

def spread_out_features(features, count):
    if features.ndim != 2 or len(features) < count:
        raise ValueError("Not enough training-fold site features for prototype initialization")
    features = F.normalize(features, dim=-1)
    distances = (features - features.mean(0)).square().sum(-1)
    selected = []
    for _ in range(count):
        index = distances.argmin() if not selected else distances.argmax()
        point = features[index]
        selected.append(point)
        to_point = (features - point).square().sum(-1)
        distances = to_point if len(selected) == 1 else torch.minimum(distances, to_point)
        distances[index] = -torch.inf
    return torch.stack(selected)


def activation_overlap(responses):
    if responses.shape[0] < 2:
        return responses.sum() * 0
    normalized = F.normalize(responses, dim=0)
    similarity = normalized.T @ normalized
    i, j = torch.triu_indices(len(similarity), len(similarity), offset=1, device=similarity.device)
    return (similarity[i, j] - 0.6).clamp_min(0).square().mean()


class SitePrototypes(nn.Module):
    def __init__(self, hidden, settings, n_conditions):
        super().__init__()
        self.n_conditions = n_conditions
        self.shared = int(settings["prototype_shared"])
        self.private = int(settings["prototype_private_per_substrate"])
        self.temperature = float(settings["prototype_temperature"])
        if min(self.shared, self.private) < 2 or self.temperature <= 0:
            raise ValueError("Invalid prototype allocation/temperature")
        total = self.shared + n_conditions * self.private
        self.vectors = nn.Parameter(torch.randn(total, hidden))
        width = max(4, hidden // 4)
        self.context = nn.Sequential(nn.Linear(2 * hidden, width), nn.ReLU())
        self.attention = nn.Linear(width, total)
        self.shared_gate = nn.Linear(width, 1)
        self.residual = nn.Linear(hidden, hidden, bias=False)
        shared = torch.arange(self.shared).expand(n_conditions, -1)
        private = torch.arange(self.shared, total).reshape(n_conditions, self.private)
        self.register_buffer("accessible_rows", torch.cat((shared, private), dim=1))

    def forward(self, sites, ligands, *, with_diversity=False):
        if ligands.shape[0] != self.n_conditions:
            raise ValueError("Prototype condition panel changed")
        expanded = sites[:, None].expand(-1, self.n_conditions, -1)
        context = self.context(torch.cat((expanded, ligands[None].expand(len(sites), -1, -1)), -1))
        source = F.normalize(sites, dim=-1)
        vectors = F.normalize(self.vectors[self.accessible_rows], dim=-1)
        distance = (source.square().sum(-1)[:, None, None]
                    + vectors.square().sum(-1)[None]
                    - 2 * torch.einsum("bh,ckh->bck", source, vectors)).clamp(0, 4)
        rows = self.accessible_rows[None].expand(len(sites), -1, -1)
        logits = -distance / self.temperature + self.attention(context).gather(-1, rows)
        shared_fraction = self.shared_gate(context).sigmoid()
        weights = torch.cat((shared_fraction * logits[..., :self.shared].softmax(-1),
                             (1-shared_fraction) * logits[..., self.shared:].softmax(-1)), -1)
        retrieved = torch.einsum("bck,ckh->bch", weights, vectors)
        enhanced = expanded + self.residual(retrieved)
        if not with_diversity:
            return enhanced, None
        response = torch.exp(-distance / self.temperature)
        groups = [response[..., :self.shared].mean(1), *response[..., self.shared:].unbind(1)]
        return enhanced, torch.stack([activation_overlap(group) for group in groups]).mean()
