#!/usr/bin/env python3
"""Export official-evaluation JSON to a tidy row, CSV, and LaTeX.

The evaluator JSON remains the source of truth; this script never invents or
rounds missing metrics into the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ("R@1", "R@2", "R@3", "FID", "MM Dist", "Diversity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True, choices=("humanml3d", "kitml"))
    parser.add_argument("--nfe", required=True, type=int)
    parser.add_argument("--cfg", required=True, type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latex"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data.get("protocol") != "official_guo_t2m" or data.get("replication_times") != 20:
        raise ValueError("Expected official_guo_t2m with exactly 20 replications")
    missing = [key for key in METRICS if key not in data or f"{key}_ci" not in data]
    if missing:
        raise ValueError(f"Missing mean/CI fields: {missing}")

    row = {"dataset": args.dataset, "checkpoint": str(args.input), "name": args.name,
           "nfe": args.nfe, "cfg": args.cfg}
    for metric in METRICS:
        row[metric] = data[metric]
        row[f"{metric}_ci"] = data[f"{metric}_ci"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.name}_nfe{args.nfe}_cfg{args.cfg:g}".replace("+", "plus")
    (args.output_dir / f"{stem}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    values = [f"${data[key]:.3f}{{\\pm{data[f'{key}_ci']:.3f}}}$" for key in METRICS]
    latex = f"{args.name} & {args.nfe} & " + " & ".join(values) + " \\\\\n"
    (args.output_dir / f"{stem}.tex").write_text(latex, encoding="utf-8")
    print(latex.strip())


if __name__ == "__main__":
    main()
