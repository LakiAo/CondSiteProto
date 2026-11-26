"""DAOx-style MLPs generalized to one, two, four or eight active conditions."""
import torch
from torch import nn
from torch.nn import functional as F

from .schema import ESM_WIDTH


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


class JointMLP(nn.Module):
    def __init__(self, ligand_width, settings, *, prototypes, n_conditions, esm_width=ESM_WIDTH):
        super().__init__()
        if n_conditions not in (1, 2, 4, 8):
            raise ValueError("NorA supports the approved one/two/four/eight-condition panels")
        self.n_conditions = n_conditions
        hidden, dropout = int(settings["hidden"]), float(settings["dropout"])
        self.site_encoder = nn.Sequential(nn.Linear(esm_width+49, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden), nn.ReLU())
        self.wt_encoder = nn.Sequential(nn.Linear(esm_width, hidden), nn.ReLU())
        self.ligand_encoder = nn.Sequential(nn.Linear(ligand_width, hidden), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(3*hidden, hidden), nn.ReLU(), nn.Dropout(dropout))

        def head():
            width = max(8, hidden//2)
            return nn.Sequential(nn.Linear(hidden, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))

        if n_conditions == 1:
            self.single_head = head()
        else:
            self.specific_heads = nn.ModuleList(head() for _ in range(n_conditions))
            self.shared_head = head()
        self.prototypes = SitePrototypes(hidden, settings, n_conditions) if prototypes else None

    @torch.no_grad()
    def initialize_prototypes(self, train_sites):
        features = self.site_encoder(train_sites)
        initial = spread_out_features(features, len(self.prototypes.vectors))
        self.prototypes.vectors.copy_(initial[torch.randperm(len(initial), device=initial.device)])

    def forward(self, site_inputs, wt_summary, substrate_inputs, *, with_diversity=False,
                disable_prototypes=False):
        site = self.site_encoder(site_inputs)
        ligand = self.ligand_encoder(substrate_inputs)
        if len(ligand) != self.n_conditions:
            raise ValueError("Incorrect condition feature panel")
        if self.prototypes is None or disable_prototypes:
            site, diversity = site[:, None].expand(-1, self.n_conditions, -1), None
        else:
            site, diversity = self.prototypes(site, ligand, with_diversity=with_diversity)
        wt = self.wt_encoder(wt_summary).expand(len(site), self.n_conditions, -1)
        fused = self.fusion(torch.cat((wt, site, ligand[None].expand(len(site), -1, -1)), -1))
        if self.n_conditions == 1:
            return self.single_head(fused).squeeze(-1), None, diversity
        specific = torch.stack([head(fused[:, i]).squeeze(-1)
                                for i, head in enumerate(self.specific_heads)], 1)
        return specific, self.shared_head(fused).squeeze(-1), diversity
