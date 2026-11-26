"""RamR-style joint MLPs with optional shared/private site prototypes."""
import torch
from torch import nn
from torch.nn import functional as F

ESM_WIDTH = 1152
SUBSTRATES = ("D-Ala", "D-Phe", "D-Met", "D-Asn", "D-Gln")


def spread_out_features(features, count):
    """Select well-spaced representatives from real training-fold site tokens."""
    if features.ndim != 2 or len(features) < count:
        raise ValueError("Prototype initialization needs enough training site tokens")
    features = F.normalize(features, dim=-1)
    distance = (features - features.mean(dim=0)).square().sum(dim=-1)
    selected = []
    for _ in range(count):
        index = distance.argmin() if not selected else distance.argmax()
        vector = features[index]
        selected.append(vector)
        to_vector = (features - vector).square().sum(dim=-1)
        distance = to_vector if len(selected) == 1 else torch.minimum(distance, to_vector)
        distance[index] = -torch.inf
    return torch.stack(selected)


def activation_overlap(responses):
    """Penalty for prototype pairs with highly similar responses across variants."""
    normalized = F.normalize(responses, dim=0)
    similarity = normalized.T @ normalized
    i, j = torch.triu_indices(len(similarity), len(similarity), offset=1, device=similarity.device)
    return (similarity[i, j] - 0.6).clamp_min(0).square().mean()


class SitePrototypes(nn.Module):
    def __init__(self, hidden, settings, *, n_conditions=len(SUBSTRATES)):
        super().__init__()
        self.shared = int(settings["prototype_shared"])
        self.private = int(settings["prototype_private_per_substrate"])
        self.temperature = float(settings["prototype_temperature"])
        self.n_conditions = int(n_conditions)
        if self.n_conditions < 1 or self.n_conditions > len(SUBSTRATES):
            raise ValueError("Invalid number of active substrate conditions")
        if min(self.shared, self.private) < 2 or self.temperature <= 0:
            raise ValueError("Expected shared and private prototype banks with positive temperature")
        total = self.shared + self.n_conditions * self.private
        self.vectors = nn.Parameter(torch.randn(total, hidden))
        width = max(4, hidden // 4)
        self.context = nn.Sequential(nn.Linear(2 * hidden, width), nn.ReLU())
        self.attention = nn.Linear(width, total)
        self.shared_gate = nn.Linear(width, 1)
        self.residual = nn.Linear(hidden, hidden, bias=False)

        shared = torch.arange(self.shared).expand(self.n_conditions, -1)
        private = torch.arange(self.shared, total).reshape(self.n_conditions, self.private)
        self.register_buffer("accessible_rows", torch.cat((shared, private), dim=1))

    def forward(self, sites, ligands, *, with_diversity=False):
        # Each mutation can retrieve the common bank plus only this substrate's bank.
        if len(ligands) != self.n_conditions:
            raise ValueError("Ligand contexts do not match the active prototype banks")
        batch_sites = sites[:, None, :].expand(-1, self.n_conditions, -1)
        ligand_context = ligands[None].expand(len(sites), -1, -1)
        context = self.context(torch.cat((batch_sites, ligand_context), dim=-1))
        source = F.normalize(sites, dim=-1)
        vectors = F.normalize(self.vectors[self.accessible_rows], dim=-1)
        distance = (
            source.square().sum(dim=-1)[:, None, None]
            + vectors.square().sum(dim=-1)[None]
            - 2 * torch.einsum("bh,ckh->bck", source, vectors)
        ).clamp(0, 4)
        rows = self.accessible_rows[None].expand(len(sites), -1, -1)
        logits = -distance / self.temperature + self.attention(context).gather(-1, rows)
        shared_fraction = self.shared_gate(context).sigmoid()
        weights = torch.cat((
            shared_fraction * logits[..., :self.shared].softmax(dim=-1),
            (1 - shared_fraction) * logits[..., self.shared:].softmax(dim=-1),
        ), dim=-1)
        retrieved = torch.einsum("bck,ckh->bch", weights, vectors)
        enhanced = batch_sites + self.residual(retrieved)
        if not with_diversity:
            return enhanced, None
        response = torch.exp(-distance / self.temperature)
        groups = [response[..., :self.shared].mean(dim=1), *response[..., self.shared:].unbind(dim=1)]
        diversity = torch.stack([activation_overlap(group) for group in groups]).mean()
        return enhanced, diversity


class JointMLP(nn.Module):
    def __init__(self, ligand_width, settings, *, prototypes, active_condition_id=None):
        super().__init__()
        if active_condition_id is not None and active_condition_id not in range(len(SUBSTRATES)):
            raise ValueError("Invalid independent substrate condition")
        self.active_condition_id = active_condition_id
        hidden, dropout = int(settings["hidden"]), float(settings["dropout"])
        self.site_encoder = nn.Sequential(
            nn.Linear(ESM_WIDTH + 49, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.wt_encoder = nn.Sequential(nn.Linear(ESM_WIDTH, hidden), nn.ReLU())
        self.ligand_encoder = nn.Sequential(nn.Linear(ligand_width, hidden), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout))

        def head():
            width = max(8, hidden // 2)
            return nn.Sequential(nn.Linear(hidden, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))

        self.specific_heads = nn.ModuleList(head() for _ in SUBSTRATES)
        self.shared_head = head()
        self.prototypes = (SitePrototypes(hidden, settings,
                           n_conditions=1 if active_condition_id is not None else len(SUBSTRATES))
                           if prototypes else None)

    @torch.no_grad()
    def initialize_prototypes(self, train_sites):
        vectors = self.prototypes.vectors
        features = self.site_encoder(train_sites)
        initial = spread_out_features(features, len(vectors))
        vectors.copy_(initial[torch.randperm(len(initial), device=initial.device)])

    def forward(self, site_inputs, wt_summary, substrate_inputs, *,
                with_diversity=False, condition_id=None):
        indices = range(len(SUBSTRATES))
        if self.active_condition_id is not None:
            if condition_id is not None and condition_id != self.active_condition_id:
                raise ValueError("The independent model cannot route to another condition")
            condition_id = self.active_condition_id
        if condition_id is not None:
            if condition_id not in indices:
                raise ValueError("Expected a valid independent substrate condition")
            if self.prototypes is not None and self.active_condition_id is None:
                raise ValueError("Independent prototypes must be constructed with an active condition")
            indices = (condition_id,)
            substrate_inputs = substrate_inputs[condition_id:condition_id + 1]
        site = self.site_encoder(site_inputs)
        wt = self.wt_encoder(wt_summary).expand(len(site), len(indices), -1)
        ligand = self.ligand_encoder(substrate_inputs)
        if self.prototypes is None:
            site = site[:, None].expand(-1, len(indices), -1)
            diversity = None
        else:
            site, diversity = self.prototypes(site, ligand, with_diversity=with_diversity)
        fused = self.fusion(torch.cat((wt, site, ligand[None].expand(len(site), -1, -1)), dim=-1))
        specific = torch.stack([
            self.specific_heads[head_id](fused[:, index]).squeeze(-1) for index, head_id in enumerate(indices)
        ], dim=1)
        shared = self.shared_head(fused).squeeze(-1)
        return specific, shared, diversity
