#!/usr/bin/env python
"""Generate quality-efficiency curve figure from saved efficiency and sweep data."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load efficiency data
eff = json.load(open("outputs/reproduction/table5/efficiency.json"))
sweep = json.load(open("outputs/reproduction/table5/cfg_steps_sweep.json"))

# Extract NFE -> latency mapping from efficiency
nfe_latency = {}
for s in eff["settings"]:
    nfe_latency[s["nfe"]] = {"latency_ms": s["latency_ms"], "fps": s["fps"]}

# Extract sweep data: use CFG=2.5 for 10/20 and CFG=2.0 for 50 (official protocol)
nfe_fid = {}
nfe_r3 = {}
for g in sweep["grid"]:
    nfe = g["steps"]
    cfg = g["guidance"]
    if (nfe == 10 and cfg == 2.5) or (nfe == 20 and cfg == 2.5) or (nfe == 50 and cfg == 2.0):
        nfe_fid[nfe] = g["FID"]
        nfe_r3[nfe] = g["R@3"]

# Also add main GALA test results for 20 NFE (distinct checkpoint)
gala_test_n20 = json.load(open("outputs/humanml3d/gala_distinct_test_n20_cfg2.5.json"))
gala_test_n50 = json.load(open("outputs/humanml3d/gala_distinct_test_n50_cfg2.0.json"))
# Use test-set numbers for the final efficiency table
test_fid = {20: gala_test_n20["FID"], 50: gala_test_n50["FID"]}
test_r3 = {20: gala_test_n20["R@3"], 50: gala_test_n50["R@3"]}

# Build figure with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

nfes = sorted(nfe_latency.keys())
latencies = [nfe_latency[n]["latency_ms"] for n in nfes]
fids = [nfe_fid.get(n, test_fid.get(n, None)) for n in nfes]
r3s = [nfe_r3.get(n, test_r3.get(n, None)) for n in nfes]

# Panel A: FID vs latency
ax1.plot(latencies, fids, "o-", color="steelblue", linewidth=2, markersize=8)
for i, n in enumerate(nfes):
    ax1.annotate(f"NFE={n}", (latencies[i], fids[i]), textcoords="offset points",
                 xytext=(8, 5), fontsize=9)
ax1.set_xlabel("Latency (ms)", fontsize=11)
ax1.set_ylabel("FID $\\downarrow$", fontsize=11)
ax1.set_title("(a) FID vs Latency", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Panel B: R@3 vs latency
ax2.plot(latencies, r3s, "s-", color="coral", linewidth=2, markersize=8)
for i, n in enumerate(nfes):
    ax2.annotate(f"NFE={n}", (latencies[i], r3s[i]), textcoords="offset points",
                 xytext=(8, 5), fontsize=9)
ax2.set_xlabel("Latency (ms)", fontsize=11)
ax2.set_ylabel("R@3 $\\uparrow$", fontsize=11)
ax2.set_title("(b) R@3 vs Latency", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.5, 0.85)

fig.suptitle("Quality-Efficiency Trade-off (RTX 4060, bf16, batch 1, T=196)", fontsize=12)
plt.tight_layout()

out_dir = Path("outputs/efficiency")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_efficiency_curve.pdf", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "fig_efficiency_curve.png", dpi=150, bbox_inches="tight")
print(f"Saved efficiency curve to {out_dir}")

# Save source data
source = {
    "nfe": nfes,
    "latency_ms": latencies,
    "fps": [nfe_latency[n]["fps"] for n in nfes],
    "fid": fids,
    "r3": r3s,
    "device": eff.get("device", "cuda"),
    "params": eff.get("params_m", 59.88),
    "batch_size": eff.get("batch_size", 1),
    "frames": eff.get("frames", 196),
}
(out_dir / "efficiency_curve.json").write_text(json.dumps(source, indent=2))
print(f"Saved source data to {out_dir / 'efficiency_curve.json'}")
