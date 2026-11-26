"""Frozen WT ESM-C features, site-wise prototype retrieval and partner heads."""

import csv

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native_prototypes import SitePrototypes, spread_out_features
from .protocol import AA, ALLOCATION, CONDITIONS, SETTINGS, load_inputs, validate_splits


class Inputs:
    def __init__(self, root, split, embedding, device):
        sequence, self.names, tokens, values, errors = load_inputs(root)
        validate_splits(self.names, tokens, split)
        esm = torch.as_tensor(embedding, dtype=torch.float32)
        if esm.shape != (len(sequence), 1152) or not torch.isfinite(esm).all():
            raise ValueError("KRAS ESM-C cache must contain one finite 1,152-dimensional vector per WT residue")
        distinct = sorted(set(token for variant in tokens for token in variant))
        remap = {token: index + 1 for index, token in enumerate(distinct)}
        positions, refs, alts = zip(*distinct)
        pos = torch.tensor(positions)
        ref = torch.tensor([AA.index(item) for item in refs])
        alt = torch.tensor([AA.index(item) for item in alts])
        aa = torch.eye(len(AA))
        with (root / "data/reference/aa_properties_kras_nora_daox.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        properties = {residue: [float(value) for value in values] for residue, *values in rows[1:]}
        prop = torch.tensor([properties[residue] for residue in AA]).float()
        prop = (prop - prop.mean(dim=0)) / prop.std(dim=0).clamp_min(1e-6)
        bank = torch.cat((esm[pos - 1], aa[ref], aa[alt] - aa[ref], prop[alt] - prop[ref],
                          pos[:, None].float() / len(sequence)), dim=1)
        self.bank = torch.cat((bank.new_zeros(1, bank.shape[1]), bank)).to(device)
        self.wt = esm.mean(dim=0).to(device)
        self.tokens = torch.tensor([[*[remap[t] for t in row], *[0] * (2 - len(row))] for row in tokens],
                                   device=device)
        self.partner_ids = torch.eye(len(CONDITIONS), device=device)
        self.values = np.array(values, dtype=np.float64)
        self.errors = np.array(errors, dtype=np.float64)
        self.targets = torch.as_tensor(self.values, dtype=torch.float32, device=device)
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.device = device

    def index(self, names):
        return torch.tensor([self.indices[name] for name in names], dtype=torch.long, device=self.device)


class ResidualControl(nn.Module):
    """The same insertion point and capacity as 6+8 with an ordinary learned gate."""

    def __init__(self, hidden):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(2 * hidden, 44), nn.ReLU(), nn.Linear(44, hidden))
        self.project = nn.Linear(hidden, hidden, bias=False)

    def forward(self, sites, partner):
        sites = sites[:, None].expand(-1, len(partner), -1)
        context = torch.cat((sites, partner[None].expand(len(sites), -1, -1)), dim=-1)
        return sites + self.project(self.context(context).sigmoid() * F.normalize(sites, dim=-1))


class FitnessModel(nn.Module):
    def __init__(self, job):
        super().__init__()
        hidden = job["width"]
        self.columns = tuple(range(len(CONDITIONS))) if job["condition"] == "all" else (
            CONDITIONS.index(job["condition"]),
        )
        self.site = nn.Sequential(nn.Linear(1201, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.wt = nn.Sequential(nn.Linear(1152, hidden), nn.ReLU())
        self.partner = nn.Sequential(nn.Linear(len(CONDITIONS), hidden), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(SETTINGS["dropout"]))

        def head():
            return nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(SETTINGS["dropout"]),
                                 nn.Linear(hidden // 2, 1))

        self.specific = nn.ModuleList(head() for _ in self.columns)
        self.shared = head() if len(self.columns) > 1 else None
        shared, private = ALLOCATION[job["arm"]]
        self.prototype = SitePrototypes(hidden, dict(prototype_shared=shared,
                                         prototype_private_per_substrate=private,
                                         prototype_temperature=SETTINGS["temperature"]), len(self.columns)) if shared else None
        self.adapter = ResidualControl(hidden) if job["arm"] == "adapter" else None

    @torch.no_grad()
    def initialize(self, inputs, index):
        if self.prototype is None:
            return
        tokens = inputs.tokens[index].unique()
        features = self.site(inputs.bank[tokens[tokens != 0]])
        vectors = spread_out_features(features, len(self.prototype.vectors))
        self.prototype.vectors.copy_(vectors[torch.randperm(len(vectors), device=vectors.device)])

    def forward(self, inputs, index, *, diversity=False):
        tokens = inputs.tokens[index]
        active = tokens != 0
        hidden = self.site(inputs.bank[tokens[active]])
        condition = self.partner(inputs.partner_ids[list(self.columns)])
        regularizer = None
        if self.prototype is not None:
            states, regularizer = self.prototype(hidden, condition, with_diversity=diversity)
        elif self.adapter is not None:
            states = self.adapter(hidden, condition)
        else:
            states = hidden[:, None].expand(-1, len(self.columns), -1)
        # Map real sites back to their variants. A missing second site contributes zero.
        flat = active.flatten()
        lookup = (flat.cumsum(dim=0) * flat).view_as(tokens)
        mutation = torch.cat((states.new_zeros(1, *states.shape[1:]), states), dim=0)[lookup].sum(dim=1)
        wt = self.wt(inputs.wt).expand_as(mutation)
        fused = self.fusion(torch.cat((wt, mutation, condition[None].expand_as(mutation)), dim=-1))
        specific = torch.cat([head(fused[:, j]) for j, head in enumerate(self.specific)], dim=1)
        shared = self.shared(fused).squeeze(-1) if self.shared is not None else None
        return specific, shared, regularizer


def parameters(model):
    return sum(value.numel() for value in model.parameters() if value.requires_grad)
