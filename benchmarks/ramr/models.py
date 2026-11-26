"""Wild-type-conditioned joint MLP and condition-specific prediction heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _ligand_ids(batch):
    ids = batch["ligand_ids"]
    if ids.ndim != 1 or not len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError("ligand_ids must be a nonempty vector of distinct condition IDs")
    if ids.dtype != torch.long or bool(((ids < 0) | (ids > 2)).any()):
        raise ValueError("ligand_ids must contain integer IDs 0=S, 1=R, 2=DHIQ")
    return ids


def _mutation_shape(batch):
    delta, mask = batch["aa_delta"], batch["mut_mask"]
    if delta.ndim != 3 or delta.shape[-1] != 20:
        raise ValueError("aa_delta must have shape [batch, residues, 20]")
    b, n, _ = delta.shape
    if mask.shape != (b, n, 1) or batch["prop_delta"].shape != (b, n, 8):
        raise ValueError("Mutation mask/property shapes do not match per-residue deltas")
    if batch["wt_esm"].ndim != 2 or batch["wt_esm"].shape[0] != n:
        raise ValueError("WT ESM must contain one row per residue")
    if batch["wt_aa"].shape != (n, 20):
        raise ValueError("WT amino-acid representation must have shape [residues, 20]")
    return b, n


class PredictionHeads(nn.Module):
    """Route each condition to its own head while retaining a general predictor."""

    def __init__(self, hidden, joint, dropout):
        super().__init__()
        self.joint = bool(joint)

        def head():
            return nn.Sequential(nn.Linear(hidden, max(hidden // 2, 8)), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(max(hidden // 2, 8), 1))

        if self.joint:
            self.shared_head = head()
            self.specific_heads = nn.ModuleList(head() for _ in range(3))
        else:
            self.single_head = head()

    def forward(self, z, ligand_ids):
        # z: batch x conditions x hidden; ligand_ids are GLOBAL S/R/D identities.
        if z.ndim != 3 or z.shape[1] != len(ligand_ids):
            raise ValueError("Condition representation and ligand IDs disagree")
        if not self.joint:
            if len(ligand_ids) != 1:
                raise ValueError("Independent training accepts exactly one ligand condition")
            return {"specific": self.single_head(z).squeeze(-1)}
        specific = torch.stack([
            self.specific_heads[int(ligand_id)](z[:, c]).squeeze(-1)
            for c, ligand_id in enumerate(ligand_ids.tolist())
        ], dim=1)
        shared = self.shared_head(z).squeeze(-1)
        return {"specific": specific, "shared": shared, "mean": (specific + shared) * 0.5}


class MLPFitness(nn.Module):
    def __init__(self, esm_dim, fp_dim, hidden=128, dropout=0.2,
                 joint=False, ligand_kind="chem"):
        super().__init__()
        if ligand_kind not in {"id", "chem"}:
            raise ValueError("Expected id or chem ligand representation")
        self.ligand_kind = ligand_kind
        self.site_encoder = nn.Sequential(
            nn.Linear(esm_dim + 49, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.wt_encoder = nn.Sequential(nn.Linear(esm_dim, hidden), nn.ReLU())
        self.ligand_encoder = nn.Sequential(
            nn.Linear(3 if ligand_kind == "id" else fp_dim, hidden), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.heads = PredictionHeads(hidden, joint, dropout)

    def encode(self, batch):
        b, n = _mutation_shape(batch)
        ids = _ligand_ids(batch)
        wt_esm = batch["wt_esm"].detach()
        wt_aa = batch["wt_aa"]
        position = torch.arange(1, n + 1, device=wt_esm.device,
                                dtype=wt_esm.dtype).view(1, n, 1) / n
        sites = torch.cat((wt_esm.unsqueeze(0).expand(b, -1, -1),
                           wt_aa.unsqueeze(0).expand(b, -1, -1),
                           batch["aa_delta"], batch["prop_delta"],
                           position.expand(b, -1, -1)), dim=-1)
        # The nonlinear encoder acts BEFORE summation. Site and replacement stay paired.
        mutation = (self.site_encoder(sites) * batch["mut_mask"]).sum(dim=1)
        global_wt = self.wt_encoder(wt_esm.mean(dim=0)).unsqueeze(0).expand(b, -1)
        if self.ligand_kind == "id":
            ligand_input = F.one_hot(ids, num_classes=3).to(dtype=wt_esm.dtype)
        else:
            ligand_input = batch["fingerprints"]
            if ligand_input.ndim != 2 or ligand_input.shape[0] != len(ids):
                raise ValueError("Fingerprints must correspond to selected ligand conditions")
        ligand = self.ligand_encoder(ligand_input)
        c = len(ids)
        return self.fusion(torch.cat((global_wt[:, None].expand(-1, c, -1),
                                     mutation[:, None].expand(-1, c, -1),
                                     ligand[None].expand(b, -1, -1)), dim=-1))

    def forward(self, batch):
        return self.heads(self.encode(batch), _ligand_ids(batch))
