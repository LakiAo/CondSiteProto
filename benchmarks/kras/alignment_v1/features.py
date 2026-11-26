"""Fixed sequence-derived descriptors for the six known KRAS assay partners."""

from collections import Counter
from math import sqrt
from string import ascii_uppercase


AA = "ACDEFGHIKLMNPQRSTVWY"


def fasta_records(path):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:]
            if name in records:
                raise ValueError("Repeated FASTA record")
            records[name] = ""
        elif line.strip():
            if not records or any(c not in ascii_uppercase for c in line.strip()):
                raise ValueError("Invalid FASTA")
            records[name] += line.strip()
    return records


def normalized_counts(sequence, k):
    counts = Counter(sequence[i:i+k] for i in range(len(sequence)-k+1))
    norm = sqrt(sum(v*v for v in counts.values()))
    return {word: value/norm for word, value in counts.items()}


def partner_features(sequences, conditions):
    if set(sequences) != set(conditions) or len(set(sequences.values())) != len(conditions):
        raise ValueError("Expected six distinct sequence records in the assay panel")
    if any(len(seq)<2 or set(seq)-set(AA) for seq in sequences.values()):
        raise ValueError("A partner has an invalid amino-acid sequence")
    descriptors = [[normalized_counts(sequences[name], k) for k in (1, 2)] for name in conditions]
    # Cosine similarity contributes equally at each k. Normalize rows to unit
    # length so the six-input encoder receives the same norm as with one-hot ID.
    similarity = [[sum(sum(v * b.get(word, 0.0) for word, v in a.items())
                       for a, b in zip(left, right))/2 for right in descriptors] for left in descriptors]
    return [[v / sqrt(sum(x*x for x in row)) for v in row] for row in similarity]
