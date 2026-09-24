#!/usr/bin/env python3
"""Pick screening weights. Do not use total loss; gate on FID/R@3 then physical metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    losses_path = path.with_name("last_epoch_losses.json")
    if not losses_path.exists():
        ckpt_losses = Path(str(path).replace("outputs/physical/screen", "checkpoints/physical_screen").replace("_val.json", "")) / "last_epoch_losses.json"
        if ckpt_losses.exists():
            data["train_losses"] = json.loads(ckpt_losses.read_text(encoding="utf-8"))
    return data


def quality_ok(run: dict, baseline: dict, fid_slack=0.05, r3_slack=0.015) -> bool:
    return (
        run["FID"] <= baseline["FID"] + fid_slack
        and run["R@3"] >= baseline["R@3"] - r3_slack
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("outputs/physical/screen"))
    parser.add_argument("--output", type=Path, default=Path("outputs/physical/screen/weight_choice.json"))
    args = parser.parse_args()
    runs = []
    for path in sorted(args.dir.glob("*_val.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        data["_name"] = path.stem.replace("_val", "")
        loss_path = Path("checkpoints/physical_screen") / data["_name"] / "last_epoch_losses.json"
        if loss_path.exists():
            data["train_losses"] = json.loads(loss_path.read_text(encoding="utf-8"))
        runs.append(data)
    if not runs:
        raise SystemExit(f"no screening evals in {args.dir}")
    by_name = {run["_name"]: run for run in runs}
    baseline = by_name.get("a3_s4000")
    if baseline is None:
        raise SystemExit("missing A3 control eval a3_s4000_val.json")

    def pick(prefix: str, metric: str, lower_better=True):
        candidates = [run for run in runs if run["_name"].startswith(prefix)]
        gated = [run for run in candidates if quality_ok(run, baseline)]
        pool = gated or candidates
        pool = sorted(pool, key=lambda run: run.get(metric, 1e9) if lower_better else -run.get(metric, 0))
        best = pool[0]
        return {
            "chosen": best["_name"],
            "metric": metric,
            "value": best.get(metric),
            "FID": best["FID"],
            "R@3": best["R@3"],
            "quality_gate_passed": best in gated if gated else False,
            "acc_over_flow": (best.get("train_losses") or {}).get("acc_over_flow"),
            "skate_over_flow": (best.get("train_losses") or {}).get("skate_over_flow"),
            "candidates": [
                {
                    "name": run["_name"],
                    "FID": run["FID"],
                    "R@3": run["R@3"],
                    "physical/acceleration": run.get("physical/acceleration"),
                    "physical/foot_skating": run.get("physical/foot_skating"),
                    "physical/acceleration_magnitude": run.get("physical/acceleration_magnitude"),
                    "gated": quality_ok(run, baseline),
                }
                for run in candidates
            ],
        }

    choice = {
        "baseline": {
            "name": baseline["_name"],
            "FID": baseline["FID"],
            "R@1": baseline["R@1"],
            "R@2": baseline["R@2"],
            "R@3": baseline["R@3"],
            "MM Dist": baseline["MM Dist"],
            "Diversity": baseline["Diversity"],
            "physical/foot_skating": baseline.get("physical/foot_skating"),
            "physical/acceleration": baseline.get("physical/acceleration"),
            "physical/acceleration_magnitude": baseline.get("physical/acceleration_magnitude"),
        },
        "acc": pick("acc_", "physical/acceleration"),
        "skate": pick("skate_", "physical/foot_skating"),
        "note": (
            "Main table uses paired physical/acceleration, not acceleration_magnitude. "
            "Skating is GT-contact-conditioned generated foot displacement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
