"""Run the frozen protocol for one experiment configuration."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_data import read_wt, reuse_cache


DATASETS = ("ramr", "kras", "nora", "aph", "daox")


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["dataset"] not in DATASETS:
        raise ValueError(f"Unknown benchmark {config['dataset']}")
    return config


def entry(config):
    return importlib.import_module(f"benchmarks.{config['dataset']}.entry")


def check_parameters(config):
    count = entry(config).count_parameters(config)
    if count != config["expected_parameters"]:
        raise ValueError(f"{config['experiment']}: got {count:,} trainable parameters, expected "
                         f"{config['expected_parameters']:,}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One training epoch using the given cache")
    parser.add_argument("--check-parameters", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model_parameters = check_parameters(config)
    print(f"{config['experiment']}: {model_parameters:,} trainable parameters", flush=True)
    if args.check_parameters:
        return

    if args.seed is not None and args.seed not in config["seeds"]:
        parser.error(f"--seed must be one of {config['seeds']}")
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    folds = [args.fold] if args.fold is not None else list(range(5))
    if args.smoke:
        seeds, folds = seeds[:1], folds[:1]

    output = args.output_dir or Path("smoke-runs" if args.smoke else "runs")
    dataset = config["dataset"]
    if args.smoke:
        cache_present = (args.cache_dir / f"{dataset}.npy").is_file()
    else:
        cache_present = reuse_cache(args.cache_dir, dataset, read_wt(dataset))
    if not cache_present:
        parser.error("missing WT embeddings; run prepare_data.py with this --cache-dir")
    benchmark = entry(config)
    for seed in seeds:
        for fold in folds:
            result = benchmark.run_fold(
                config, root=ROOT, cache_dir=args.cache_dir, device=args.device,
                seed=seed, fold=fold, smoke=args.smoke,
            )
            if result["parameters"] != model_parameters or not result["metrics"]:
                raise RuntimeError("Training results do not match the configured model")
            record = {
                "experiment": config["experiment"],
                "dataset": config["dataset"],
                "config": config,
                "seed": seed,
                "fold": fold,
                "smoke": args.smoke,
                **result,
            }
            path = output / config["experiment"] / f"seed_{seed}" / f"fold_{fold}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(path)
            print(f"seed={seed} fold={fold} epochs={result['epochs']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
