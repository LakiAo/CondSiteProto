"""Condition-aware residual adapters and prototypes for the RamR joint MLP."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .models import MLPFitness, _ligand_ids, _mutation_shape


class PrototypeRouting:
    """Shared retrieval, training anchors and losses for both encoders.

    Shared prototypes occupy the first rows. The remaining rows are consecutive
    equally sized banks ordered by global ligand ID. Anchor vectors are temporary
    training data, never part of the state required for prediction.
    """

    site_tokens_are_condition_independent = False

    def _init_prototype_routing(self, experiment, hidden):
        self.position = experiment["position"]
        self.module_kind = experiment["module"]
        self.activation_weight = float(experiment["activation_weight"])
        self.diversity_weight = float(experiment["diversity_weight"])
        self.temperature = 0.25
        context_width = max(4, hidden // 4)
        self.context_encoder = nn.Sequential(nn.Linear(2 * hidden, context_width), nn.ReLU())
        self.residual_projection = nn.Linear(hidden, hidden, bias=False)
        if self.module_kind == "prototype":
            self.shared_count, self.private_count = {
                "mixed": (12, 4), "shared": (24, 0), "private": (0, 8),
                "private_heavy": (6, 12),
                "balanced": (6, 6), "intermediate": (6, 8),
            }[experiment["banks"]]
            total = self.shared_count + 3 * self.private_count
            self.prototype_vectors = nn.Parameter(torch.randn(total, hidden))
            self.prototype_attention = nn.Linear(context_width, total)
            if self.shared_count and self.private_count:
                self.shared_gate = nn.Linear(context_width, 1)
            self.register_buffer("training_anchors", torch.empty(0, hidden), persistent=False)
        else:
            self.adapter_gate = nn.Linear(context_width, hidden)

    def prototype_indices(self, ligand_id):
        """Return the accessible prototype rows for one global ligand ID."""
        if ligand_id not in (0, 1, 2):
            raise ValueError(f"Unknown ligand ID {ligand_id}")
        device = self.prototype_vectors.device
        start = self.shared_count + ligand_id * self.private_count
        return torch.cat((torch.arange(self.shared_count, device=device),
                          torch.arange(start, start + self.private_count, device=device)))

    def anchor_candidates(self, batch):
        """Collect normalized matching features, token mask and zero-based positions."""
        _, tokens, valid, positions, *_ = self._tokens(batch)
        return F.normalize(tokens, dim=-1), valid, positions

    @torch.no_grad()
    def refresh_training_anchors(self, batches, *, initialize=False):
        """Find prototype representatives across the supplied training-fold batches.

        Initialization chooses spread-out actual training features. Subsequent
        refreshes cache the nearest detached feature for each prototype. The
        caller supplies only train-fold batches and refreshes at epoch boundaries.
        """
        if self.module_kind != "prototype" or (not initialize and not self.activation_weight):
            return
        was_training = self.training
        self.eval()
        candidates = {ligand_id: [] for ligand_id in range(3)}
        try:
            for batch in batches:
                features, valid, _ = self.anchor_candidates(batch)
                for column, ligand_id in enumerate(batch["ligand_ids"].tolist()):
                    candidates[ligand_id].append(features[:, column][valid])
        finally:
            self.train(was_training)
        if any(not parts or not any(len(part) for part in parts) for parts in candidates.values()):
            raise ValueError("Prototype anchors require training features for all three ligands")
        candidates = {ligand_id: torch.cat(parts) for ligand_id, parts in candidates.items()}
        if initialize and self.site_tokens_are_condition_independent:
            # Site features are ligand-independent. A single selection avoids
            # initializing all ligand banks with the same feature representatives.
            initial = _spread_out_features(candidates[0], len(self.prototype_vectors))
            order = torch.randperm(len(initial), device=initial.device)
            self.prototype_vectors.copy_(initial[order])
        groups = []
        if self.shared_count:
            shared_candidates = candidates[0] if self.site_tokens_are_condition_independent else torch.cat(list(candidates.values()))
            groups.append((self.prototype_vectors[:self.shared_count], shared_candidates))
        if self.private_count:
            for ligand_id in range(3):
                start = self.shared_count + ligand_id * self.private_count
                groups.append((self.prototype_vectors[start:start + self.private_count], candidates[ligand_id]))
        anchors = []
        for prototypes, features in groups:
            if initialize and not self.site_tokens_are_condition_independent:
                initial = _spread_out_features(features, len(prototypes))
                prototypes.copy_(initial)
            # The feature norm matters for degenerate all-zero representations.
            scores = 2 * (F.normalize(prototypes, dim=-1) @ features.T)
            nearest = (scores - features.square().sum(dim=-1)).argmax(dim=1)
            anchors.append(features[nearest])
        self.training_anchors = torch.cat(anchors)

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

    def encode(self, batch):
        encoded, _, _ = self._encode(batch)
        return encoded

    def forward(self, batch):
        encoded, ids, _ = self._encode(batch)
        return self.heads(encoded, ids)

    def forward_training(self, batch):
        encoded, ids, info = self._encode(batch)
        predictions = self.heads(encoded, ids)
        auxiliary = {}
        if self.activation_weight:
            if self.training_anchors.shape != self.prototype_vectors.shape:
                raise RuntimeError("Initialize prototype anchors from the training fold before training")
            counts = info["valid_mask"].sum(dim=-1)
            closest = info["distances"].amin(dim=-1)
            # Equal weight for each variant rather than each individual mutation.
            coverage = (closest * info["valid_mask"][:, None]).sum(dim=(1, 2))
            coverage = coverage / (len(ids) * counts.clamp_min(1))
            coverage = coverage.sum() / (counts > 0).sum().clamp_min(1)
            used_rows = info["prototype_indices"].unique()
            representatives = (F.normalize(self.prototype_vectors[used_rows], dim=-1)
                               - self.training_anchors[used_rows])
            alignment = representatives.square().sum(dim=-1).mean()
            auxiliary["activation"] = self.activation_weight * (coverage + alignment) / 2
        if self.diversity_weight:
            counts = info["valid_mask"].sum(dim=-1)
            sample_response = info["activations"].sum(dim=2) / counts[:, None, None].clamp_min(1)
            sample_response = sample_response[counts > 0]
            groups = []
            if self.shared_count:
                groups.append(sample_response[..., :self.shared_count].mean(dim=1))
            if self.private_count:
                groups.extend(sample_response[..., self.shared_count:].unbind(dim=1))
            diversity = torch.stack([_activation_overlap(group) for group in groups]).mean()
            auxiliary["diversity"] = self.diversity_weight * diversity
        return predictions, auxiliary


class PrototypeMLP(PrototypeRouting, MLPFitness):
    """Insert a condition-dependent residual into the original MLP."""

    def __init__(self, experiment, esm_dim, fp_dim, hidden, dropout):
        model_type = experiment.get("model_type", "mlp_id")
        ligand_kind = "chem" if model_type == "mlp_chem" else "id"
        super().__init__(esm_dim, fp_dim, hidden, dropout, joint=True, ligand_kind=ligand_kind)
        self._init_prototype_routing(experiment, hidden)
        self.site_tokens_are_condition_independent = self.position == "site"

    def _features(self, batch):
        batch_size, n_residues = _mutation_shape(batch)
        ligand_ids = _ligand_ids(batch)
        wt_esm, wt_aa = batch["wt_esm"].detach(), batch["wt_aa"]
        mutations = batch["mut_mask"].squeeze(-1).bool()
        # Gather only mutation sites. Padding sites are masked before summation.
        width = max(1, int(mutations.sum(dim=1).max().item()))
        positions = torch.arange(n_residues, device=mutations.device).expand(batch_size, -1)
        positions = positions.masked_fill(~mutations, n_residues)
        positions = positions.topk(width, dim=1, largest=False).values
        valid = positions < n_residues
        gather = positions.clamp_max(n_residues - 1)
        batch_indices = torch.arange(batch_size, device=gather.device)[:, None]
        site_inputs = torch.cat((wt_esm[gather], wt_aa[gather],
                                 batch["aa_delta"][batch_indices, gather],
                                 batch["prop_delta"][batch_indices, gather],
                                 (gather[..., None].to(wt_esm.dtype) + 1) / n_residues), dim=-1)
        sites = self.site_encoder(site_inputs) * valid[..., None]
        mutation = sites.sum(dim=1)
        wt = self.wt_encoder(wt_esm.mean(dim=0))
        ligand_input = (F.one_hot(ligand_ids, num_classes=3).to(wt_esm.dtype)
                        if self.ligand_kind == "id" else batch["fingerprints"])
        ligand = self.ligand_encoder(ligand_input)
        return ligand_ids, sites, valid, positions.masked_fill(~valid, -1), mutation, wt, ligand

    def _fuse(self, mutation, wt, ligand):
        batch_size, conditions, _ = mutation.shape
        return self.fusion(torch.cat((wt.expand(batch_size, conditions, -1),
                                      mutation,
                                      ligand.unsqueeze(0).expand(batch_size, -1, -1)), dim=-1))

    def _tokens(self, batch):
        ligand_ids, sites, valid, positions, mutation, wt, ligand = self._features(batch)
        ligand_context = ligand.unsqueeze(0).expand(sites.shape[0], -1, -1)
        if self.position == "site":
            tokens = sites[:, None].expand(-1, len(ligand_ids), -1, -1)
            context_input = mutation[:, None].expand(-1, len(ligand_ids), -1)
        else:
            context_input = self._fuse(mutation[:, None].expand(-1, len(ligand_ids), -1), wt, ligand)
            tokens = context_input.unsqueeze(2)
            valid = valid.new_ones((len(valid), 1))
            positions = positions.new_full((len(positions), 1), -1)
        context = self.context_encoder(torch.cat((context_input, ligand_context), dim=-1))
        return ligand_ids, tokens, valid, positions, wt, ligand, context

    def _encode(self, batch, exclude_prototypes=(), *, ignore_similarity=False):
        ligand_ids, tokens, valid, positions, wt, ligand, context = self._tokens(batch)
        features = F.normalize(tokens, dim=-1)
        if self.module_kind == "adapter":
            adjustment = self.adapter_gate(context).sigmoid().unsqueeze(2) * features
            info = {}
        else:
            adjustment, info = self._match(ligand_ids, features, valid, context, exclude_prototypes,
                                           ignore_similarity=ignore_similarity)
            info.update(valid_mask=valid, site_positions=positions)
        enhanced = tokens + self.residual_projection(adjustment)
        if self.position == "site":
            encoded = self._fuse((enhanced * valid[:, None, :, None]).sum(dim=2), wt, ligand)
        else:
            encoded = enhanced.squeeze(2)
        return encoded, ligand_ids, info


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
