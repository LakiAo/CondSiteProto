"""A shared expert bank and one softmax gate per RamR assay condition."""
import torch
from torch import nn

from .models import PredictionHeads

N_EXPERTS = 4


class MMoEHeads(PredictionHeads):
    def __init__(self, hidden, dropout):
        super().__init__(hidden, joint=True, dropout=dropout)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()) for _ in range(N_EXPERTS)
        )
        self.condition_gates = nn.ModuleList(nn.Linear(hidden, N_EXPERTS) for _ in range(3))

    def forward(self, z, ligand_ids):
        ids = ligand_ids.tolist()
        if z.ndim != 3 or z.shape[1] != len(ids):
            raise ValueError("Condition representation and ligand IDs disagree")
        expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=2)
        gate_logits = torch.stack([
            self.condition_gates[ligand](z[:, column]) for column, ligand in enumerate(ids)
        ], dim=1)
        representation = (gate_logits.softmax(dim=-1).unsqueeze(-1) * expert_outputs).sum(dim=2)
        specific = torch.stack([
            self.specific_heads[ligand](representation[:, column]).squeeze(-1)
            for column, ligand in enumerate(ids)
        ], dim=1)
        shared = self.shared_head(z).squeeze(-1)
        return {"specific": specific, "shared": shared, "mean": (specific + shared) * 0.5}
