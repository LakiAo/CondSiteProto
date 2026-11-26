"""APH reference features in the archived training order."""

import csv

import numpy as np


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


def amino_acid_properties(reference):
    with (reference / "aa_properties_ramr_aph.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        next(rows)
        raw = {code: values for code, *values in rows}
    if set(raw) != set(AA_ORDER) or any(len(values) != 8 for values in raw.values()):
        raise ValueError("APH requires the archived 20 by 8 amino-acid property table")
    matrix = np.array([raw[aa] for aa in AA_ORDER], dtype=np.float32)
    return (matrix - matrix.mean(0)) / matrix.std(0, ddof=1).clip(min=1e-6)


def chiral_fingerprints(reference, conditions):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    with (reference / "small_molecules_20.csv").open(newline="", encoding="utf-8-sig") as stream:
        chemistry = {row["condition"]: row for row in csv.DictReader(stream) if row["dataset"] == "aph"}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
    fingerprints = []
    for condition in conditions:
        molecule = Chem.MolFromSmiles(chemistry[condition]["canonical_isomeric_smiles"])
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f"Invalid APH parent SMILES for {condition}")
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        values = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), values)
        fingerprints.append(values)
    return np.stack(fingerprints)
