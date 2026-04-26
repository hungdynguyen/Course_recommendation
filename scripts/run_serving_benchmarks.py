#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs" / "dataset_analysis"
METRICS_DIR = ROOT_DIR / "data" / "processed" / "course_recommendation_metrics"


@dataclass
class ModelConfig:
    label: str
    state_suffix: str
    model_name: str
    model_path: str


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_model_configs() -> List[ModelConfig]:
    before = ModelConfig(
        label="before",
        state_suffix="model_before",
        model_name=os.getenv("MODEL_BEFORE_NAME", "Qwen/Qwen3-Embedding-0.6B"),
        model_path=os.getenv("MODEL_BEFORE_PATH", ""),
    )
    after = ModelConfig(
        label="after",
        state_suffix="model_after",
        model_name=os.getenv("MODEL_AFTER_NAME", "Qwen/Qwen3-Embedding-0.6B"),
        model_path=os.getenv("MODEL_AFTER_PATH", "/app/models/qwen_embedding_finetuned"),
    )
    return [before, after]


def build_common_env_args(model: ModelConfig) -> List[str]:
    args = [
        "-e",
        "PYTHONPATH=/app/src",
        "-e",
        f"EMBEDDING_DEVICE={os.getenv('EMBEDDING_DEVICE', 'cuda')}",
        "-e",
        f"EMBEDDING_MODEL_NAME={model.model_name}",
        "-e",
        f"EMBEDDING_BATCH_SIZE={os.getenv('EMBEDDING_BATCH_SIZE', '1')}",
        "-e",
        "PYTHONUNBUFFERED=1",
    ]
    if model.model_path:
        args += ["-e", f"EMBEDDING_MODEL_PATH={model.model_path}"]
    return args


def run_and_tee(cmd: List[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(cmd))
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def copy_baseline_outputs(suffix: str) -> None:
    src_json = METRICS_DIR / "embedding_baseline_groundtruth_gaps_metrics.json"
    src_xlsx = METRICS_DIR / "embedding_baseline_groundtruth_gaps_comparison.xlsx"
    dst_json = METRICS_DIR / f"embedding_baseline_groundtruth_gaps_metrics_{suffix}.json"
    dst_xlsx = METRICS_DIR / f"embedding_baseline_groundtruth_gaps_comparison_{suffix}.xlsx"

    if not src_json.exists() or not src_xlsx.exists():
        raise FileNotFoundError("Baseline output files were not generated as expected")

    shutil.copy2(src_json, dst_json)
    shutil.copy2(src_xlsx, dst_xlsx)


def run_pair_for_model(model: ModelConfig, ts: str) -> None:
    use_enrich = env_bool("DESIGNED_ENABLE_ENRICH", True)
    use_weighted_ranker = not env_bool("DESIGNED_DISABLE_WEIGHTED_RANKER", False)
    use_top1_gate = not env_bool("DESIGNED_DISABLE_TOP1_GATE", False)
    debug_trace = env_bool("DESIGNED_DEBUG_TRACE", False)
    debug_miss_only = env_bool("DESIGNED_DEBUG_MISS_ONLY", False)
    debug_rank_explain = env_bool("DESIGNED_DEBUG_RANK_EXPLAIN", False)

    inline_python = "\n".join(
        [
            "from service_api.scripts.evaluate_serving_embedding_baseline_groundtruth_gaps import evaluate as evaluate_baseline",
            "from service_api.scripts.evaluate_serving_designed_flow_groundtruth_gaps import evaluate as evaluate_designed",
            "",
            "print('[embedded] Running baseline evaluation...')",
            "evaluate_baseline()",
            "",
            "print('[embedded] Running designed evaluation...')",
            (
                "evaluate_designed("
                f"use_enrich={use_enrich}, "
                f"use_weighted_ranker={use_weighted_ranker}, "
                f"use_top1_gate={use_top1_gate}, "
                f"debug_trace={debug_trace}, "
                f"debug_miss_only={debug_miss_only}, "
                f"debug_rank_explain={debug_rank_explain}, "
                f"output_suffix={model.state_suffix!r}"
                ")"
            ),
        ]
    )

    extra_env_args = [
        "-e",
        f"SKILL_SEARCH_RAW_LIMIT={os.getenv('SKILL_SEARCH_RAW_LIMIT', '10')}",
        "-e",
        f"SKILL_TOP1_MIN_SCORE={os.getenv('SKILL_TOP1_MIN_SCORE', '0.89')}",
        "-e",
        f"SKILL_SEARCH_SCORE_MARGIN={os.getenv('SKILL_SEARCH_SCORE_MARGIN', '0.05')}",
    ]
    cmd = [
        "docker",
        "exec",
        *build_common_env_args(model),
        *extra_env_args,
        "vietcv_data_factory",
        "python",
        "-c",
        inline_python,
    ]
    run_and_tee(cmd, LOG_DIR / f"serving_pair_{model.label}_{ts}.log")
    copy_baseline_outputs(model.state_suffix)


def collect_versions() -> List[Dict[str, Path | str]]:
    return [
        {
            "version": "baseline_before_train",
            "method": "baseline",
            "model_state": "before_train",
            "json": METRICS_DIR / "embedding_baseline_groundtruth_gaps_metrics_model_before.json",
            "xlsx": METRICS_DIR / "embedding_baseline_groundtruth_gaps_comparison_model_before.xlsx",
        },
        {
            "version": "designed_before_train",
            "method": "designed",
            "model_state": "before_train",
            "json": METRICS_DIR / "designed_flow_groundtruth_gaps_metrics_model_before.json",
            "xlsx": METRICS_DIR / "designed_flow_groundtruth_gaps_comparison_model_before.xlsx",
        },
        {
            "version": "baseline_after_train",
            "method": "baseline",
            "model_state": "after_train",
            "json": METRICS_DIR / "embedding_baseline_groundtruth_gaps_metrics_model_after.json",
            "xlsx": METRICS_DIR / "embedding_baseline_groundtruth_gaps_comparison_model_after.xlsx",
        },
        {
            "version": "designed_after_train",
            "method": "designed",
            "model_state": "after_train",
            "json": METRICS_DIR / "designed_flow_groundtruth_gaps_metrics_model_after.json",
            "xlsx": METRICS_DIR / "designed_flow_groundtruth_gaps_comparison_model_after.xlsx",
        },
    ]


def export_combined_workbooks() -> None:
    versions = collect_versions()

    summary_rows = []
    recommendation_frames = []

    for item in versions:
        metrics_path = item["json"]
        sample_path = item["xlsx"]
        if not isinstance(metrics_path, Path) or not isinstance(sample_path, Path):
            raise TypeError("Invalid version config")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing comparison file: {sample_path}")

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        for k, metrics in data.get("summary", {}).items():
            row = {
                "version": item["version"],
                "method": item["method"],
                "model_state": item["model_state"],
                "k": k,
            }
            row.update(metrics)
            summary_rows.append(row)

        per_sample = pd.read_excel(sample_path, sheet_name="per_sample")
        keep_cols = [
            col
            for col in [
                "pair_id",
                "jd_id",
                "jd_title",
                "gap_descriptions_used",
                "query_text",
                "truth_courses",
                "pred_top10",
                "pred_top10_titles",
                "overlap_top10",
                "hit@1",
                "hit@3",
                "hit@5",
                "hit@10",
            ]
            if col in per_sample.columns
        ]
        sample_df = per_sample[keep_cols].copy()
        sample_df.insert(0, "model_state", item["model_state"])
        sample_df.insert(0, "method", item["method"])
        sample_df.insert(0, "version", item["version"])
        recommendation_frames.append(sample_df)

    metrics_df = pd.DataFrame(summary_rows)
    metrics_pivot = (
        metrics_df.pivot_table(
            index=["k"],
            columns="version",
            values=["hit_rate", "ndcg", "mrr", "map", "precision", "recall"],
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    metrics_pivot.columns = [
        col if isinstance(col, str) else f"{col[0]}__{col[1]}" for col in metrics_pivot.columns
    ]

    recommendations_df = pd.concat(recommendation_frames, ignore_index=True)

    out_metrics = METRICS_DIR / "full_metrics_4_versions_latest.xlsx"
    out_reco = METRICS_DIR / "course_recommendations_4_versions_latest.xlsx"

    with pd.ExcelWriter(out_metrics, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="long_format", index=False)
        metrics_pivot.to_excel(writer, sheet_name="pivot", index=False)

    with pd.ExcelWriter(out_reco, engine="openpyxl") as writer:
        recommendations_df.to_excel(writer, sheet_name="all_methods", index=False)
        for version in recommendations_df["version"].drop_duplicates().tolist():
            sheet = version[:31]
            recommendations_df[recommendations_df["version"] == version].to_excel(
                writer, sheet_name=sheet, index=False
            )

    print(f"Metrics workbook: {out_metrics}")
    print(f"Recommendations workbook: {out_reco}")


def main() -> None:
    ts = os.getenv("TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Embedding batch size: {os.getenv('EMBEDDING_BATCH_SIZE', '1')}")
    for model in build_model_configs():
        model_path = model.model_path if model.model_path else "<none>"
        print(f"Model {model.label.upper()}: name='{model.model_name}' path='{model_path}'")

    models = build_model_configs()
    print("[1/4] Run baseline+designed with MODEL_BEFORE (shared embedding process)")
    run_pair_for_model(models[0], ts)

    print("[2/4] Run baseline+designed with MODEL_AFTER (shared embedding process)")
    run_pair_for_model(models[1], ts)

    print("[3/4] Exporting 4-version metric/recommendation workbooks")
    export_combined_workbooks()

    print("[4/4] Done")


if __name__ == "__main__":
    main()
