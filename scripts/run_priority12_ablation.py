#!/usr/bin/env python3
"""Run ablation for Priority #1 and #2 improvements in Docker environment.

Variants (all with enrich disabled):
- baseline: no weighted ranker, no top1 gate
- weighted_only: weighted ranker ON, top1 gate OFF
- gate_only: weighted ranker OFF, top1 gate ON
- both: weighted ranker ON, top1 gate ON
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "data/processed/course_recommendation_metrics"


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_variant(name: str, extra_args: List[str]) -> Path:
    suffix = f"p12_{name}"
    inner = [
        "python3",
        "/app/src/service_api/scripts/evaluate_serving_designed_flow_groundtruth_gaps.py",
        "--disable-enrich",
        "--output-suffix",
        suffix,
    ]
    inner.extend(extra_args)

    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "data_factory",
        "sh",
        "-lc",
        " ".join(inner),
    ]
    run_cmd(cmd)

    return METRICS_DIR / f"designed_flow_groundtruth_gaps_metrics_{suffix}.json"


def load_summary(metrics_file: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    return data["summary"]


def main() -> None:
    variants = [
        ("baseline", ["--disable-weighted-ranker", "--disable-top1-gate"]),
        ("weighted_only", ["--disable-top1-gate"]),
        ("gate_only", ["--disable-weighted-ranker"]),
        ("both", []),
    ]

    outputs: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name, args in variants:
        metrics_file = run_variant(name, args)
        outputs[name] = load_summary(metrics_file)

    baseline = outputs["baseline"]
    report_rows = []

    for name in ["baseline", "weighted_only", "gate_only", "both"]:
        row = {
            "variant": name,
            "p@1": outputs[name]["@1"]["precision"],
            "nDCG@1": outputs[name]["@1"]["ndcg"],
            "p@10": outputs[name]["@10"]["precision"],
            "nDCG@10": outputs[name]["@10"]["ndcg"],
            "hit@10": outputs[name]["@10"]["hit_rate"],
            "delta_p@1_vs_baseline": outputs[name]["@1"]["precision"] - baseline["@1"]["precision"],
            "delta_nDCG@10_vs_baseline": outputs[name]["@10"]["ndcg"] - baseline["@10"]["ndcg"],
        }
        report_rows.append(row)

    out_json = METRICS_DIR / "priority12_ablation_report.json"
    out_json.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = METRICS_DIR / "priority12_ablation_report.csv"
    header = [
        "variant",
        "p@1",
        "nDCG@1",
        "p@10",
        "nDCG@10",
        "hit@10",
        "delta_p@1_vs_baseline",
        "delta_nDCG@10_vs_baseline",
    ]
    lines = [",".join(header)]
    for r in report_rows:
        lines.append(
            ",".join([
                r["variant"],
                f"{r['p@1']:.6f}",
                f"{r['nDCG@1']:.6f}",
                f"{r['p@10']:.6f}",
                f"{r['nDCG@10']:.6f}",
                f"{r['hit@10']:.6f}",
                f"{r['delta_p@1_vs_baseline']:.6f}",
                f"{r['delta_nDCG@10_vs_baseline']:.6f}",
            ])
        )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== Priority 1/2 Ablation Summary ===")
    for r in report_rows:
        print(
            f"{r['variant']:<14} P@1={r['p@1']:.4f} "
            f"nDCG@10={r['nDCG@10']:.4f} hit@10={r['hit@10']:.4f} "
            f"dP@1={r['delta_p@1_vs_baseline']:+.4f}"
        )

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
