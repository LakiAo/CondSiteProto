"""Verify the processed data and prepare frozen WT residue embeddings."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ramr", "kras", "nora", "aph", "daox")
ARCHITECTURE = "esmc_600m_2024_12"
ESM_VERSION = "3.2.3"
WIDTH = 1152
WEIGHTS_SHA256 = "8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_processed():
    root = ROOT / "data/processed"
    checksums = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    entries = [line.split() for line in checksums if line and not line.startswith("#")]
    if len(entries) != len(DATASETS) or {parts[2] for parts in entries} != {p.name for p in root.glob("*.csv")}:
        raise ValueError("Processed file manifest does not cover the five modeling tables")
    for sha256, size, name in entries:
        path = root / name
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha256:
            raise ValueError(f"Processed table checksum or size mismatch for {name}")
        print(f"PASS {name} {size} bytes {sha256}", flush=True)


def read_wt(dataset):
    path = ROOT / "data/reference" / f"{dataset}_wt.fasta"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if sum(line.startswith(">") for line in lines) != 1:
        raise ValueError(f"Expected one WT sequence in {path.name}")
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def sequence_digest(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def validate_embedding(embedding, sequence):
    import numpy as np

    if embedding.shape != (len(sequence), WIDTH) or embedding.dtype != np.float32:
        raise ValueError(f"Expected float32 WT embedding of shape {(len(sequence), WIDTH)}")
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite values in WT embedding")


def reuse_cache(cache, dataset, sequence):
    import numpy as np

    path = cache / f"{dataset}.npy"
    manifest = cache / f"{dataset}.json"
    if not path.exists() and not manifest.exists():
        return False
    if not path.is_file() or not manifest.is_file():
        raise ValueError(f"Incomplete embedding cache for {dataset}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if (metadata["sequence_sha256"] != sequence_digest(sequence)
            or metadata["architecture"] != ARCHITECTURE
            or metadata.get("weights_sha256") != WEIGHTS_SHA256
            or metadata["embedding_sha256"] != digest(path)):
        raise ValueError(f"Embedding cache identity or checksum mismatch for {dataset}")
    validate_embedding(np.load(path, allow_pickle=False), sequence)
    return True


def save_cache(cache, dataset, sequence, embedding, metadata):
    import numpy as np

    validate_embedding(embedding, sequence)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{dataset}.npy"
    staged_array = path.with_suffix(".tmp.npy")
    np.save(staged_array, embedding, allow_pickle=False)
    provenance = {
        **metadata,
        "architecture": ARCHITECTURE,
        "sequence_sha256": sequence_digest(sequence),
        "embedding_sha256": digest(staged_array),
        "storage_dtype": "float32",
        "tokens": "BOS_EOS_removed",
    }
    manifest = path.with_suffix(".json")
    staged_manifest = manifest.with_suffix(".tmp.json")
    staged_manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    staged_array.replace(path)
    staged_manifest.replace(manifest)


def import_tensor(path, sequence):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("sequence") != sequence:
        raise ValueError("Legacy embedding must contain the exact WT sequence")
    metadata = {key: payload[key] for key in (
        "architecture", "weights_sha256", "esm_version", "inference_dtype", "torch_version"
    ) if key in payload}
    if metadata.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("Legacy embedding comes from a different architecture")
    if metadata.get("weights_sha256") != WEIGHTS_SHA256:
        raise ValueError("Legacy embedding must record the archived checkpoint checksum")
    return payload["embedding"].detach().float().contiguous().numpy(), metadata


def extract_embeddings(weights, device, dtype, dataset_sequences):
    import torch
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    version = importlib.metadata.version("esm")
    if version != ESM_VERSION:
        raise ValueError(f"Install esm=={ESM_VERSION}; found {version}")
    if dtype == "bfloat16" and torch.device(device).type != "cuda":
        raise ValueError("bfloat16 extraction requires CUDA")
    weights_hash = digest(weights)
    if weights_hash != WEIGHTS_SHA256:
        raise ValueError("Checkpoint checksum does not match the archived ESM-C 600M v0 weights")
    tokenizer = get_esmc_model_tokenizers()
    model = ESMC(d_model=WIDTH, n_heads=18, n_layers=36, tokenizer=tokenizer, use_flash_attn=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    del state
    model = model.to(device).eval().requires_grad_(False)
    if dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    metadata = {
        "weights_sha256": weights_hash,
        "esm_version": version,
        "torch_version": str(torch.__version__),
        "inference_dtype": dtype,
    }
    for dataset, sequence in dataset_sequences.items():
        ids = tokenizer.encode(sequence)
        if len(ids) != len(sequence) + 2:
            raise ValueError(f"Unexpected token/residue mapping for {dataset}")
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            embedding = model(sequence_tokens=tokens).embeddings[0, 1:-1].float().cpu().contiguous()
        yield dataset, sequence, embedding.numpy(), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--weights", type=Path, help="Local ESM-C 600M v0 checkpoint")
    parser.add_argument("--import-cache", action="append", default=[], metavar="DATASET=FILE")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    validate_processed()
    if args.check_only:
        return

    legacy = {}
    for item in args.import_cache:
        dataset, sep, path = item.partition("=")
        if not sep or dataset not in DATASETS or not path or dataset in legacy:
            parser.error(f"Invalid or duplicate --import-cache argument {item}")
        legacy[dataset] = Path(path)

    missing = {}
    for dataset in dict.fromkeys(args.dataset or DATASETS):
        sequence = read_wt(dataset)
        if reuse_cache(args.cache_dir, dataset, sequence):
            print(f"PASS {dataset}: reused verified WT cache")
        elif dataset in legacy:
            embedding, metadata = import_tensor(legacy[dataset], sequence)
            save_cache(args.cache_dir, dataset, sequence, embedding, metadata)
            print(f"PASS {dataset}: imported WT cache {embedding.shape}")
        else:
            missing[dataset] = sequence
    if missing and not args.weights:
        parser.error(f"missing verified caches for {', '.join(missing)}; pass --weights or --import-cache")
    if missing:
        for dataset, sequence, embedding, metadata in extract_embeddings(
                args.weights, args.device, args.dtype, missing):
            save_cache(args.cache_dir, dataset, sequence, embedding, metadata)
            print(f"PASS {dataset}: extracted WT embedding {embedding.shape}", flush=True)


if __name__ == "__main__":
    main()
