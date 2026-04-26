#!/usr/bin/env python3
"""Summarize rank-explain benchmark failures.

This script reads the latest groundtruth-gap benchmark outputs with the
`rank_explain` suffix and produces a compact statistics report answering:

- how many GT courses failed overall,
- why GT courses were lost,
- whether GT courses were inside the candidate pool,
- and which cases had the largest score deficit against the top-10 cutoff.

It is intentionally read-only with respect to the source benchmark outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_ROOT = Path("data/processed/course_recommendation_metrics")
DEFAULT_XLSX = DEFAULT_ROOT / "designed_flow_groundtruth_gaps_comparison_rank_explain.xlsx"
DEFAULT_JSON = DEFAULT_ROOT / "designed_flow_groundtruth_gaps_metrics_rank_explain.json"
DEFAULT_OUT_XLSX = DEFAULT_ROOT / "rank_explain_failure_summary.xlsx"
DEFAULT_OUT_JSON = DEFAULT_ROOT / "rank_explain_failure_summary.json"


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _maybe_load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"Input file not found: {path}")


def _top_rows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in ["pair_id", "jd_id", "course_id", "course_title", "rank", "weighted_score", "delta_score_vs_top10_cutoff", "dominant_loss_component_vs_top10"] if c in df.columns]
    return df.sort_values(["delta_score_vs_top10_cutoff", "pair_id", "course_id"], ascending=[True, True, True]).head(n)[cols]


def build_summary(xlsx_path: Path, json_path: Path) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    xlsx_path = _resolve_input(xlsx_path)
    json_data = _maybe_load_json(json_path)

    per_sample = pd.read_excel(xlsx_path, sheet_name="per_sample")
    rank_explain = pd.read_excel(xlsx_path, sheet_name="rank_explain_failures")

    fail_samples = per_sample[per_sample["hit@10"] == 0].copy()
    gt_rows = rank_explain[rank_explain["is_groundtruth"] == True].copy()  # noqa: E712

    total_samples = int(len(per_sample))
    fail_samples_count = int(len(fail_samples))
    gt_rows_count = int(len(gt_rows))

    # Pair-level view: a fail pair is considered "in-candidate" if any GT row exists with rank > 0.
    gt_by_pair = gt_rows.groupby("pair_id")
    pair_stats = gt_by_pair.agg(
        any_in_candidate=("rank", lambda s: bool((pd.to_numeric(s, errors="coerce") > 0).any())),
        any_not_in_candidate=("rank", lambda s: bool((pd.to_numeric(s, errors="coerce") == -1).any())),
        gt_rows=("course_id", "count"),
    ).reset_index()

    pair_in_candidate_count = int(pair_stats["any_in_candidate"].sum()) if not pair_stats.empty else 0
    pair_not_in_candidate_count = int((~pair_stats["any_in_candidate"]).sum()) if not pair_stats.empty else 0

    # GT-level breakdown of dominant reasons.
    dominant_counts = (
        gt_rows["dominant_loss_component_vs_top10"].fillna("unknown").value_counts(dropna=False)
        if not gt_rows.empty
        else pd.Series(dtype=int)
    )
    dominant_df = dominant_counts.rename_axis("dominant_loss_component_vs_top10").reset_index(name="count")
    if not dominant_df.empty:
        dominant_df["percentage"] = dominant_df["count"] / max(gt_rows_count, 1)
    else:
        dominant_df = pd.DataFrame(columns=["dominant_loss_component_vs_top10", "count", "percentage"])

    # Score deficit view.
    gt_ranked = gt_rows[pd.to_numeric(gt_rows["rank"], errors="coerce") > 0].copy()
    gt_in_pool_count = int(len(gt_ranked))
    gt_not_in_pool_count = int(len(gt_rows[pd.to_numeric(gt_rows["rank"], errors="coerce") == -1]))

    rank_metrics = {
        "mean_rank": _safe_float(gt_ranked["rank"].mean()) if not gt_ranked.empty else None,
        "median_rank": _safe_float(gt_ranked["rank"].median()) if not gt_ranked.empty else None,
        "mean_delta_score_vs_top10_cutoff": _safe_float(gt_ranked["delta_score_vs_top10_cutoff"].mean()) if not gt_ranked.empty else None,
        "mean_delta_core_vs_top10_cutoff": _safe_float(gt_ranked["delta_core_vs_top10_cutoff"].mean()) if not gt_ranked.empty else None,
        "mean_delta_coverage_vs_top10_cutoff": _safe_float(gt_ranked["delta_coverage_vs_top10_cutoff"].mean()) if not gt_ranked.empty else None,
        "mean_delta_bonus_vs_top10_cutoff": _safe_float(gt_ranked["delta_bonus_vs_top10_cutoff"].mean()) if not gt_ranked.empty else None,
    }

    top5_negative_delta = _top_rows(gt_rows, n=5)

    # Per-pair summary for the fail samples.
    fail_pair_ids = set(str(x) for x in fail_samples["pair_id"].tolist())
    pair_level = pair_stats.copy()
    pair_level["pair_id"] = pair_level["pair_id"].astype(str)
    pair_level["is_fail_pair"] = pair_level["pair_id"].isin(fail_pair_ids)
    fail_pair_level = pair_level[pair_level["is_fail_pair"]].copy()
    fail_pair_in_candidate = int(fail_pair_level["any_in_candidate"].sum()) if not fail_pair_level.empty else 0
    fail_pair_out_candidate = int((~fail_pair_level["any_in_candidate"]).sum()) if not fail_pair_level.empty else 0

    summary: Dict[str, object] = {
        "input_xlsx": str(xlsx_path),
        "input_json": str(json_path),
        "json_available": json_data is not None,
        "total_samples": total_samples,
        "fail_samples_hit_at_10": fail_samples_count,
        "fail_samples_hit_at_10_pct": fail_samples_count / max(total_samples, 1),
        "gt_rows": gt_rows_count,
        "gt_rows_not_in_candidate_pool": gt_not_in_pool_count,
        "gt_rows_not_in_candidate_pool_pct": gt_not_in_pool_count / max(gt_rows_count, 1),
        "gt_rows_in_candidate_pool": gt_in_pool_count,
        "gt_rows_in_candidate_pool_pct": gt_in_pool_count / max(gt_rows_count, 1),
        "pair_fail_count": int(len(fail_pair_level)),
        "pair_fail_in_candidate_count": fail_pair_in_candidate,
        "pair_fail_in_candidate_pct": fail_pair_in_candidate / max(len(fail_pair_level), 1),
        "pair_fail_out_candidate_count": fail_pair_out_candidate,
        "pair_fail_out_candidate_pct": fail_pair_out_candidate / max(len(fail_pair_level), 1),
        "rank_metrics": rank_metrics,
        "hit_rates": (json_data.get("summary") if json_data else None),
    }

    return summary, dominant_df, top5_negative_delta


def print_summary(summary: Dict[str, object], dominant_df: pd.DataFrame, top5_negative_delta: pd.DataFrame) -> None:
    print("=== Rank Explain Failure Summary ===")
    print(f"Input XLSX: {summary['input_xlsx']}")
    if summary.get("json_available"):
        print(f"Input JSON: {summary['input_json']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Fail@10 samples: {summary['fail_samples_hit_at_10']} ({_fmt_pct(summary['fail_samples_hit_at_10_pct'])})")
    print(f"GT rows: {summary['gt_rows']}")
    print(f"GT rows not in candidate pool: {summary['gt_rows_not_in_candidate_pool']} ({_fmt_pct(summary['gt_rows_not_in_candidate_pool_pct'])})")
    print(f"GT rows in candidate pool: {summary['gt_rows_in_candidate_pool']} ({_fmt_pct(summary['gt_rows_in_candidate_pool_pct'])})")
    print(f"Fail pairs: {summary['pair_fail_count']}")
    print(f"Fail pairs with GT in candidate pool: {summary['pair_fail_in_candidate_count']} ({_fmt_pct(summary['pair_fail_in_candidate_pct'])})")
    print(f"Fail pairs without GT in candidate pool: {summary['pair_fail_out_candidate_count']} ({_fmt_pct(summary['pair_fail_out_candidate_pct'])})")

    print("\n--- Dominant loss component (GT rows) ---")
    if dominant_df.empty:
        print("No GT rows found.")
    else:
        dominant_df = dominant_df.copy()
        dominant_df["percentage"] = dominant_df["percentage"].map(_fmt_pct)
        print(dominant_df.to_string(index=False))

    print("\n--- Rank metrics for GT rows inside candidate pool ---")
    rank_metrics = summary.get("rank_metrics", {}) or {}
    if not any(v is not None for v in rank_metrics.values()):
        print("No GT rows with rank > 0 were found.")
    else:
        for key, value in rank_metrics.items():
            print(f"{key}: {value}")

    print("\n--- Top 5 most negative delta_score_vs_top10_cutoff (GT rows) ---")
    if top5_negative_delta.empty:
        print("No GT rows found.")
    else:
        print(top5_negative_delta.to_string(index=False))


def save_outputs(summary: Dict[str, object], dominant_df: pd.DataFrame, top5_negative_delta: pd.DataFrame, out_xlsx: Path, out_json: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        dominant_df.to_excel(writer, sheet_name="dominant_components", index=False)
        top5_negative_delta.to_excel(writer, sheet_name="top5_negative_delta", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rank-explain benchmark failures.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Input rank-explain comparison XLSX.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Input rank-explain metrics JSON.")
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX, help="Output summary XLSX.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output summary JSON.")
    args = parser.parse_args()

    summary, dominant_df, top5_negative_delta = build_summary(args.xlsx, args.json)
    print_summary(summary, dominant_df, top5_negative_delta)
    save_outputs(summary, dominant_df, top5_negative_delta, args.out_xlsx, args.out_json)
    print(f"\nSaved summary XLSX: {args.out_xlsx}")
    print(f"Saved summary JSON: {args.out_json}")


if __name__ == "__main__":
    main()