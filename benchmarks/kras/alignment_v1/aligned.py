"""Use the RamR retrieval kernel with the six-condition KRAS backbone."""

import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark.model import FitnessModel, ResidualControl
from ..benchmark.protocol import ALLOCATION
from .retrieval import PrototypeRouting, _activation_overlap, _spread_out_features


def variant_context(sites, partners):
    pooled = sites.sum(dim=1)[:, None].expand(-1, len(partners), -1)
    conditions = partners[None].expand(len(sites), -1, -1)
    return torch.cat((pooled, conditions), dim=-1)


class VariantPrototypes(PrototypeRouting, nn.Module):
    def __init__(self, hidden, settings, n_conditions):
        super().__init__()
        self.n_conditions = n_conditions
        self.shared_count, self.private_count = settings
        self.temperature = 0.25
        total = self.shared_count + self.n_conditions * self.private_count
        width = max(4, hidden // 4)
        self.context_encoder = nn.Sequential(nn.Linear(2 * hidden, width), nn.ReLU())
        self.prototype_attention = nn.Linear(width, total)
        self.shared_gate = nn.Linear(width, 1)
        self.residual_projection = nn.Linear(hidden, hidden, bias=False)
        self.prototype_vectors = nn.Parameter(torch.randn(total, hidden))

    def prototype_indices(self, condition):
        if not 0 <= condition < self.n_conditions:
            raise ValueError("Condition index outside this prototype bank")
        device = self.prototype_vectors.device
        private_start = self.shared_count + condition * self.private_count
        return torch.cat((torch.arange(self.shared_count, device=device),
                          torch.arange(private_start, private_start+self.private_count, device=device)))

    @property
    def vectors(self):
        # The native trainer uses this attribute to check parameter updates.
        return self.prototype_vectors

    def forward(self, sites, valid, partners, *, with_diversity=False):
        sites = sites * valid[..., None]
        tokens = sites[:, None].expand(-1, len(partners), -1, -1)
        context = self.context_encoder(variant_context(sites, partners))
        ids = torch.arange(self.n_conditions, device=sites.device)
        adjustment, info = self._match(ids, F.normalize(tokens, dim=-1), valid, context)
        enhanced = tokens + self.residual_projection(adjustment)
        if not with_diversity:
            return enhanced.sum(dim=2), None
        counts = valid.sum(dim=-1)
        response = info["activations"].sum(dim=2) / counts[:, None, None].clamp_min(1)
        response = response[counts > 0]
        groups = [response[..., :self.shared_count].mean(dim=1),
                  *response[..., self.shared_count:].unbind(dim=1)]
        return enhanced.sum(dim=2), torch.stack([_activation_overlap(group) for group in groups]).mean()


class VariantResidual(ResidualControl):
    def forward(self, sites, valid, partners):
        sites = sites * valid[..., None]
        gate = self.context(variant_context(sites, partners)).sigmoid()
        delta = gate[:, :, None] * F.normalize(sites, dim=-1)[:, None]
        return (sites[:, None] + self.project(delta)).sum(dim=2)


class AlignedFitnessModel(FitnessModel):
    def __init__(self, job):
        super().__init__(job)
        if self.prototype is not None:
            self.prototype = VariantPrototypes(job["width"], ALLOCATION[job["arm"]], len(self.columns))
        elif self.adapter is not None:
            self.adapter = VariantResidual(job["width"])
        else:
            raise ValueError("Use the original FitnessModel for the plain control")

    @torch.no_grad()
    def initialize(self, inputs, index):
        if self.prototype is None:
            return
        # RamR initializes from every valid training observation's site feature,
        # preserving repeated substitutions in distinct training variants.
        features = []
        for batch in index.split(128):
            tokens = inputs.tokens[batch]
            real = tokens[tokens != 0]
            features.append(F.normalize(self.site(inputs.bank[real]), dim=-1))
        vectors = _spread_out_features(torch.cat(features), len(self.prototype.vectors))
        self.prototype.vectors.copy_(vectors[torch.randperm(len(vectors), device=vectors.device)])

    def forward(self, inputs, index, *, diversity=False):
        tokens = inputs.tokens[index]
        valid = tokens != 0
        sites = self.site(inputs.bank[tokens])
        partner = self.partner(inputs.partner_ids[list(self.columns)])
        if self.prototype is not None:
            mutation, penalty = self.prototype(sites, valid, partner, with_diversity=diversity)
        else:
            mutation, penalty = self.adapter(sites, valid, partner), None
        wt = self.wt(inputs.wt).expand_as(mutation)
        fused = self.fusion(torch.cat((wt, mutation, partner[None].expand_as(mutation)), dim=-1))
        specific = torch.cat([head(fused[:, j]) for j, head in enumerate(self.specific)], dim=1)
        shared = self.shared(fused).squeeze(-1) if self.shared is not None else None
        return specific, shared, penalty


def routed_model(job):
    if job["arm"] == "plain":
        return FitnessModel(job)
    return AlignedFitnessModel(job)
