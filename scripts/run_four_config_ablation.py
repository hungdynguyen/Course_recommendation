#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_CONTAINER = "vietcv_data_factory"
DEFAULT_SCRIPT = "/app/src/service_api/scripts/evaluate_serving_designed_flow_groundtruth_gaps.py"


def default_output_dir() -> Path:
    container_path = Path("/app/data/processed/course_recommendation_metrics")
    if container_path.exists():
        return container_path
    return Path("data/processed/course_recommendation_metrics")


DEFAULT_OUTPUT_DIR = default_output_dir()


def build_configs() -> List[Dict[str, object]]:
    return [
        {
            "name": "enrich_k10_top1_089",
            "env": {
                "SKILL_SEARCH_RAW_LIMIT": "10",
                "SKILL_TOP1_MIN_SCORE": "0.89",
            },
        },
        {
            "name": "enrich_k10_top1_090",
            "env": {
                "SKILL_SEARCH_RAW_LIMIT": "10",
                "SKILL_TOP1_MIN_SCORE": "0.90",
            },
        },
        {
            "name": "enrich_k10_margin_005",
            "env": {
                "SKILL_SEARCH_RAW_LIMIT": "10",
                "SKILL_SEARCH_SCORE_MARGIN": "0.05",
            },
        },
        {
            "name": "enrich_k10_top1_089_margin_005",
            "env": {
                "SKILL_SEARCH_RAW_LIMIT": "10",
                "SKILL_TOP1_MIN_SCORE": "0.89",
                "SKILL_SEARCH_SCORE_MARGIN": "0.05",
            },
        },
    ]


def run_config(container: str, script_path: str, cfg: Dict[str, object], exec_mode: str) -> None:
    name = str(cfg["name"])
    env = dict(cfg["env"])

    mode = exec_mode
    if mode == "auto":
        mode = "docker" if shutil.which("docker") else "local"

    if mode == "docker":
        cmd = ["docker", "exec", "-i"]
        cmd += ["-e", "EMBEDDING_DEVICE=cuda", "-e", "PYTHONUNBUFFERED=1"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [
            container,
            "python",
            "-u",
            script_path,
            "--enable-enrich",
            "--debug-trace",
            "--debug-miss-only",
            "--debug-rank-explain",
            "--output-suffix",
            name,
        ]
        run_env = None
    else:
        cmd = [
            "python",
            "-u",
            script_path,
            "--enable-enrich",
            "--debug-trace",
            "--debug-miss-only",
            "--debug-rank-explain",
            "--output-suffix",
            name,
        ]
        run_env = os.environ.copy()
        run_env.update({"EMBEDDING_DEVICE": "cuda", "PYTHONUNBUFFERED": "1"})
        run_env.update({str(k): str(v) for k, v in env.items()})

    print(f"\n=== Running: {name} ===", flush=True)
    print(f"Exec mode: {mode}", flush=True)
    print("Command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=run_env)


def read_metrics_json(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})

    def v(k: str, metric: str) -> float:
        return float(summary.get(k, {}).get(metric, 0.0) or 0.0)

    return {
        "hit@1": v("@1", "hit_rate"),
        "hit@3": v("@3", "hit_rate"),
        "hit@5": v("@5", "hit_rate"),
        "hit@10": v("@10", "hit_rate"),
        "recall@10": v("@10", "recall"),
        "precision@10": v("@10", "precision"),
        "mrr@10": v("@10", "mrr"),
        "map@10": v("@10", "map"),
        "ndcg@10": v("@10", "ndcg"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4 ablation configs and aggregate benchmark metrics.")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Docker container name")
    parser.add_argument("--script", default=DEFAULT_SCRIPT, help="Benchmark script path inside container")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Metrics output directory")
    parser.add_argument("--skip-run", action="store_true", help="Skip benchmark execution and only aggregate existing outputs")
    parser.add_argument(
        "--exec-mode",
        choices=["auto", "docker", "local"],
        default="auto",
        help="Execution mode: docker=run via docker exec, local=run in current environment, auto=detect by docker binary",
    )
    args = parser.parse_args()

    configs = build_configs()

    if not args.skip_run:
        for cfg in configs:
            run_config(args.container, args.script, cfg, args.exec_mode)

    rows = []
    for cfg in configs:
        name = str(cfg["name"])
        metric_file = args.output_dir / f"designed_flow_groundtruth_gaps_metrics_{name}.json"
        if not metric_file.exists():
            raise FileNotFoundError(f"Missing output file: {metric_file}")

        metrics = read_metrics_json(metric_file)
        row = {
            "config": name,
            "metric_file": str(metric_file),
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["hit@10", "recall@10", "hit@1"], ascending=[False, False, False]).reset_index(drop=True)

    out_csv = args.output_dir / "four_config_ablation_comparison.csv"
    out_xlsx = args.output_dir / "four_config_ablation_comparison.xlsx"
    out_json = args.output_dir / "four_config_ablation_comparison.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="comparison", index=False)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Ablation Comparison ===")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved XLSX: {out_xlsx}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
