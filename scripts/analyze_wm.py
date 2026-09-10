#!/usr/bin/env python3
"""Turn eval JSON into a results note and fill the paper tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(block, key="mean"):
    if not isinstance(block, dict):
        return "—"
    if key in block and isinstance(block[key], dict):
        return f"{block[key]['mean']:.3f}"
    value = block.get(key)
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_analysis(report: dict, path: Path):
    rollout = report.get("rollout", {})
    swap = report.get("swap", {})
    cem = report.get("cem", {})
    lines = [
        "# GALA-WM 实验结果",
        "",
        f"- checkpoint: `{report.get('meta', {}).get('checkpoint', '')}`",
        f"- split: {report.get('meta', {}).get('split', '')}",
        f"- action: {report.get('meta', {}).get('action_type', '')}",
        "",
        "## 开环 rollout",
        "",
        "| 方法 | MPJPE@1 (mm) | MPJPE@8 (mm) | latent L2@1 | latent L2@8 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in ("gt", "copy_last", "zero", "shuffle"):
        block = rollout.get(method, {})
        lines.append(
            f"| {method} | {cell(block.get('1', {}), 'mpjpe_mm')} | {cell(block.get('8', {}), 'mpjpe_mm')} "
            f"| {cell(block.get('1', {}), 'latent_l2')} | {cell(block.get('8', {}), 'latent_l2')} |"
        )
    lines.extend([
        "",
        "## 换动作",
        "",
        f"- swap/noise 比: {cell(swap, 'ratio')}",
        f"- swap Δ: {cell(swap, 'swap_delta')}",
        f"- noise Δ: {cell(swap, 'noise_delta')}",
        "",
        "## CEM 规划",
        "",
        f"- 成功率: {cell(cem, 'success_rate')}",
        f"- MPJPE: {cell(cem.get('mpjpe_mm', {}), 'mean') if isinstance(cem.get('mpjpe_mm'), dict) else '—'}",
        f"- root XY (m): {cell(cem.get('root_xy_m', {}), 'mean') if isinstance(cem.get('root_xy_m'), dict) else '—'}",
        "",
        "## 解读",
        "",
    ])
    gt8 = rollout.get("gt", {}).get("8", {}).get("mpjpe_mm", {}).get("mean")
    copy8 = rollout.get("copy_last", {}).get("8", {}).get("mpjpe_mm", {}).get("mean")
    zero8 = rollout.get("zero", {}).get("8", {}).get("mpjpe_mm", {}).get("mean")
    ratio = swap.get("ratio", {}).get("mean") if isinstance(swap.get("ratio"), dict) else None
    success = cem.get("success_rate")
    if gt8 is not None and copy8 is not None:
        if gt8 < copy8:
            lines.append(f"- GT 动作 rollout 的 MPJPE@8 ({gt8:.1f}) 低于 copy-last ({copy8:.1f})，说明模型在用动作，而不是背序列。")
        else:
            lines.append("- GT 动作并未明显优于 copy-last：动力学要么欠拟合，要么动作通道信息不足。")
    if gt8 is not None and zero8 is not None:
        lines.append(f"- 置零动作 MPJPE@8={zero8:.1f}，与 GT 动作对比可衡量动作条件的贡献。")
    if ratio is not None:
        if ratio > 1.5:
            lines.append(f"- 换动作/微扰比={ratio:.2f}，同一初态下反向速度会产生可区分轨迹。")
        else:
            lines.append(f"- 换动作比={ratio:.2f}，可控性偏弱，需要加长训练或检查 root4 是否进条件。")
    if success is not None:
        lines.append(f"- CEM 成功率={success:.3f}。高于随机/copy-last 才说明世界模型能用于规划。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def fill_paper(report: dict, template: Path, output: Path):
    text = template.read_text(encoding="utf-8")
    rollout = report.get("rollout", {})
    swap = report.get("swap", {})
    cem = report.get("cem", {})

    def mpjpe(method, h):
        return cell(rollout.get(method, {}).get(str(h), {}), "mpjpe_mm")

    def lat(method, h):
        return cell(rollout.get(method, {}).get(str(h), {}), "latent_l2")

    replacements = {
        "{{MPJPE_GT_1}}": mpjpe("gt", 1),
        "{{MPJPE_GT_8}}": mpjpe("gt", 8),
        "{{MPJPE_COPY_1}}": mpjpe("copy_last", 1),
        "{{MPJPE_COPY_8}}": mpjpe("copy_last", 8),
        "{{MPJPE_ZERO_1}}": mpjpe("zero", 1),
        "{{MPJPE_ZERO_8}}": mpjpe("zero", 8),
        "{{MPJPE_SHUF_1}}": mpjpe("shuffle", 1),
        "{{MPJPE_SHUF_8}}": mpjpe("shuffle", 8),
        "{{LAT_GT_1}}": lat("gt", 1),
        "{{LAT_GT_8}}": lat("gt", 8),
        "{{SWAP_RATIO}}": cell(swap, "ratio"),
        "{{CEM_SUCCESS}}": cell(cem, "success_rate"),
        "{{CEM_MPJPE}}": cell(cem.get("mpjpe_mm", {}), "mean") if isinstance(cem.get("mpjpe_mm"), dict) else "—",
        "{{CEM_ROOT}}": cell(cem.get("root_xy_m", {}), "mean") if isinstance(cem.get("root_xy_m"), dict) else "—",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    output.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", default="/home/qinyang/桌面/project/checkpoints/gala_humanml3d_wm/eval.json")
    parser.add_argument("--analysis", default="/home/qinyang/桌面/project/paper/gala_wm_results.md")
    parser.add_argument("--template", default="/home/qinyang/桌面/project/paper/gala_wm_template.md")
    parser.add_argument("--paper", default="/home/qinyang/桌面/project/paper/gala_wm.md")
    args = parser.parse_args()
    report = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    Path(args.analysis).parent.mkdir(parents=True, exist_ok=True)
    write_analysis(report, Path(args.analysis))
    if Path(args.template).exists():
        fill_paper(report, Path(args.template), Path(args.paper))
    print(f"wrote {args.analysis}")
    print(f"wrote {args.paper}")


if __name__ == "__main__":
    main()
