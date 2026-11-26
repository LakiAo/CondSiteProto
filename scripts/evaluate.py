"""Summarize the held-out metrics emitted by train.py."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def summarize(records, allow_partial=False):
    config = records[0]["config"]
    seen = set()
    by_seed = defaultdict(list)
    metric_panel = None
    for record in records:
        if record["config"] != config or record["smoke"] != records[0]["smoke"]:
            raise ValueError("Mixed configurations or run types in one experiment")
        seed, fold = record["seed"], record["fold"]
        if seed not in config["seeds"] or fold not in range(5):
            raise ValueError(f"Out-of-protocol seed/fold {(seed, fold)}")
        if (seed, fold) in seen:
            raise ValueError(f"Duplicate seed/fold {(seed, fold)}")
        seen.add((seed, fold))
        seen_metrics = set()
        for score in record["metrics"]:
            arm = score.get("arm", config["arm"])
            condition = score["condition"]
            metric = (score["partition"], arm, condition)
            if metric in seen_metrics:
                raise ValueError(f"Duplicate fold metric {metric}")
            mse, r = score["mse"], score["pearson"]
            if score["n"] < 1 or not math.isfinite(mse) or mse < 0 or (r is not None and not math.isfinite(r)):
                raise ValueError(f"Invalid fold score {metric}")
            seen_metrics.add(metric)
            by_seed[seed, *metric].append(score)
        if metric_panel is None:
            metric_panel = seen_metrics
        elif seen_metrics != metric_panel:
            raise ValueError(f"Inconsistent condition/partition coverage in seed {seed}, fold {fold}")

    expected = {(seed, fold) for seed in config["seeds"] for fold in range(5)}
    if not allow_partial and seen != expected:
        raise ValueError(f"{config['experiment']} is missing {len(expected - seen)} configured runs")

    seed_summary = {}
    for (seed, partition, arm, condition), scores in by_seed.items():
        if not allow_partial and len(scores) != 5:
            raise ValueError(f"Incomplete five-fold metric for {(seed, partition, arm, condition)}")
        key = seed, partition, arm
        group = "delta_s_r" if condition.lower() == "delta_s_r" else "conditions"
        seed_summary.setdefault((*key, group), []).extend(scores)

    per_seed = []
    result = defaultdict(list)
    for (seed, partition, arm, group), scores in sorted(seed_summary.items()):
        r = [row["pearson"] for row in scores if row["pearson"] is not None]
        row = {
            "seed": seed,
            "partition": partition,
            "arm": arm,
            "metric": "delta_s_r" if group == "delta_s_r" else "macro",
            "rmse": math.sqrt(mean(score["mse"] for score in scores)),
            "pearson": mean(r) if r else None,
        }
        per_seed.append(row)
        result[partition, arm, row["metric"]].append(row)

    summary = []
    for (partition, arm, metric), rows in sorted(result.items()):
        row = {"partition": partition, "arm": arm, "metric": metric, "seeds": len(rows)}
        for key in ("rmse", "pearson"):
            values = [seed_row[key] for seed_row in rows if seed_row[key] is not None]
            row[f"{key}_mean"] = mean(values) if values else None
            row[f"{key}_sd"] = stdev(values) if len(values) > 1 else None
        summary.append(row)
    return {"experiment": config["experiment"], "smoke": records[0]["smoke"], "n_runs": len(seen),
            "summary": summary, "per_seed": per_seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--experiment")
    parser.add_argument("--allow-partial", action="store_true", help="Allow a subset of seeds or folds")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON summary")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in sorted(args.runs_dir.glob("*/seed_*/fold_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if args.experiment and record["experiment"] != args.experiment:
            continue
        if record["smoke"] and not args.allow_smoke:
            parser.error("use --allow-smoke to evaluate technical runs")
        grouped[record["experiment"]].append(record)
    if not grouped:
        parser.error("no matching fold results")

    results = []
    for records in grouped.values():
        result = summarize(records, allow_partial=args.allow_partial or records[0]["smoke"])
        results.append(result)
        print(f"{result['experiment']}  ({result['n_runs']} fold records)")
        for row in result["summary"]:
            values = []
            for key in ("rmse", "pearson"):
                average, sd = row[f"{key}_mean"], row[f"{key}_sd"]
                value = "n/a" if average is None else f"{average:.6f}"
                if sd is not None:
                    value += f" ± {sd:.6f}"
                values.append(f"{key}={value}")
            print(f"  {row['partition']} {row['arm']} {row['metric']}  {'  '.join(values)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
