"""Read wide measurement rows and assemble per-site, multi-condition tensors."""
from hashlib import sha256
import math

import torch
from torch.utils.data import Dataset

from .features import AA_PROPERTIES
from .schema import AA_ORDER, TARGET_COLUMNS, ERROR_COLUMNS


def sequence_hash(sequence):
    return sha256(sequence.encode("ascii")).hexdigest()


def mutation_tensors(wt, sequence):
    if len(sequence) != len(wt) or set(sequence) - set(AA_ORDER):
        raise ValueError("Variant sequence has invalid length or amino acids")
    mask = torch.zeros(len(wt), 1)
    delta = torch.zeros(len(wt), 20)
    prop = torch.zeros(len(wt), 8)
    mutations = []
    for i, (a, b) in enumerate(zip(wt, sequence)):
        if a != b:
            ia, ib = AA_ORDER.index(a), AA_ORDER.index(b)
            mask[i] = 1
            delta[i, ia], delta[i, ib] = -1, 1
            prop[i] = torch.from_numpy(AA_PROPERTIES[ib] - AA_PROPERTIES[ia])
            mutations.append(f"{a}{i+1}{b}")
    return mask, delta, prop, mutations


class MeasurementDataset(Dataset):
    def __init__(self, rows, wt, ligand_ids):
        self.rows = rows
        if not self.rows:
            raise ValueError("Empty measurement table")
        self.wt, self.ligand_ids = wt, list(ligand_ids)
        if not self.ligand_ids or len(set(self.ligand_ids)) != len(self.ligand_ids) or set(self.ligand_ids) - {0, 1, 2}:
            raise ValueError("Invalid ligand IDs")
        seen = set()
        for row in self.rows:
            if row["record_id"] in seen:
                raise ValueError("Duplicate record_id")
            seen.add(row["record_id"])
            sequence = row["sequence"]
            mutations = [f"{a}{i+1}{b}" for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
            if len(sequence) != len(wt) or set(sequence) - set(AA_ORDER):
                raise ValueError("Invalid variant sequence")
            if row["sequence_id"] != "seq_" + sequence_hash(sequence):
                raise ValueError("Sequence ID does not match actual sequence")
            if int(row["mutation_count"]) != len(mutations) or row["mutations"] != ":".join(mutations):
                raise ValueError("Mutation annotation does not match actual sequence")
            if row["library"] != "EP" or float(row["concentration_uM"]) != 32:
                raise ValueError("Only EP32 observations are supported in this protocol")
            for ycol, ecol in zip(TARGET_COLUMNS, ERROR_COLUMNS):
                y, err = float(row[ycol]), float(row[ecol])
                if not math.isfinite(y) or not math.isfinite(err) or not 0 <= err <= .3:
                    raise ValueError("Measurement violates the all-three err <= 0.3 protocol")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mask, delta, prop, _ = mutation_tensors(self.wt, row["sequence"])
        return {"mut_mask": mask, "aa_delta": delta, "prop_delta": prop,
                "ligand_ids": tuple(self.ligand_ids),
                "y": torch.tensor([float(row[TARGET_COLUMNS[i]]) for i in self.ligand_ids]),
                "errs": torch.tensor([float(row[ERROR_COLUMNS[i]]) for i in self.ligand_ids]),
                "record_id": row["record_id"], "sequence_id": row["sequence_id"],
                "mutations": row["mutations"], "mutation_count": int(row["mutation_count"])}
