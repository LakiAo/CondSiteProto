"""The archived APH site MLP and condition prediction heads."""
from __future__ import annotations
import torch
from torch import nn


def _ligand_ids(batch):
    ids = batch['ligand_ids']
    if ids.ndim != 1 or ids.dtype != torch.long or len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Expected distinct integer condition IDs')
    if bool((ids < 0).any()):
        raise ValueError('Negative condition ID')
    return ids


class PredictionHeads(nn.Module):
    def __init__(self, hidden, joint, dropout, n_conditions):
        super().__init__()
        self.joint = bool(joint)
        def head():
            return nn.Sequential(nn.Linear(hidden, max(hidden // 2, 8)), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(max(hidden // 2, 8), 1))
        if joint:
            self.shared_head = head()
            self.specific_heads = nn.ModuleList(head() for _ in range(n_conditions))
        else:
            self.single_head = head()

    def forward(self, z, ligand_ids):
        if z.ndim != 3 or z.shape[1] != len(ligand_ids):
            raise ValueError('Condition representation shape mismatch')
        if not self.joint:
            if len(ligand_ids) != 1:
                raise ValueError('Independent model must have one condition')
            return {'specific': self.single_head(z).squeeze(-1)}
        specific = torch.stack([self.specific_heads[i](z[:, c]).squeeze(-1)
                                for c, i in enumerate(ligand_ids.tolist())], dim=1)
        shared = self.shared_head(z).squeeze(-1)
        return {'specific': specific, 'shared': shared, 'mean': (specific + shared) * .5}


class MLPFitness(nn.Module):
    def __init__(self, esm_dim, fp_dim, hidden=128, dropout=.2, joint=False,
                 ligand_kind='chem', *, n_conditions=4, id_dim=None):
        super().__init__()
        self.ligand_kind = ligand_kind
        self.n_conditions = n_conditions
        self.id_dim = n_conditions if id_dim is None else id_dim
        if ligand_kind not in ('id', 'chem') or n_conditions < 1:
            raise ValueError('Invalid ligand representation / condition count')
        self.site_encoder = nn.Sequential(nn.Linear(esm_dim + 49, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden), nn.ReLU())
        self.wt_encoder = nn.Sequential(nn.Linear(esm_dim, hidden), nn.ReLU())
        self.ligand_encoder = nn.Sequential(nn.Linear(self.id_dim if ligand_kind == 'id' else fp_dim, hidden), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.heads = PredictionHeads(hidden, joint, dropout, n_conditions)

    def _features(self, batch):
        ids = _ligand_ids(batch)
        if bool((ids >= self.n_conditions).any()):
            raise ValueError('Condition outside this model panel')
        valid = batch['valid_mask']
        sites = self.site_encoder(batch['site_inputs']) * valid[..., None]
        mutation = sites.sum(dim=1)
        wt = self.wt_encoder(batch['wt_mean'].detach())
        ligand_input = batch['ligand_onehot'] if self.ligand_kind == 'id' else batch['fingerprints']
        ligand = self.ligand_encoder(ligand_input)
        return ids, sites, valid, batch['site_positions'], mutation, wt, ligand

    def _fuse(self, mutation, wt, ligand):
        b, c, _ = mutation.shape
        return self.fusion(torch.cat((wt.expand(b, c, -1), mutation,
                                     ligand.unsqueeze(0).expand(b, -1, -1)), dim=-1))

    def encode(self, batch):
        ids, _, _, _, mutation, wt, ligand = self._features(batch)
        return self._fuse(mutation[:, None].expand(-1, len(ids), -1), wt, ligand)

    def forward(self, batch):
        return self.heads(self.encode(batch), _ligand_ids(batch))


def build_model(job, inputs, config):
    kwargs = dict(esm_dim=inputs.esm_dim, fp_dim=1024, hidden=config['model']['hidden'],
                  dropout=config['model']['dropout'], n_conditions=len(job['conditions']),
                  id_dim=len(job['panel']), joint=job['joint'])
    if job['prototype']:
        from .prototypes import PrototypeMLP
        experiment = dict(config['prototype'], ligand_kind=job['kind'])
        return PrototypeMLP(experiment, **kwargs)
    return MLPFitness(ligand_kind=job['kind'], **kwargs)
