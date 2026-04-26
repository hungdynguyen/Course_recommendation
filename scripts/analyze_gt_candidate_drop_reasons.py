#!/usr/bin/env python3
"""Analyze why GT courses are not in final scored candidate list.

This script classifies failed GT courses into three requested buckets:
1) canonical_skill_not_enough
2) canonical_present_but_neo4j_not_return_gt
3) gt_in_graph_but_cut_by_limit_or_coverage

Method:
- Read benchmark rank-explain XLSX (`per_sample`, `rank_explain_failures`).
- For each failed GT row (rank = -1), fetch GT taught skills from Neo4j.
- Compare GT taught skills with final canonical skill list used by the sample.
- Run a no-limit Neo4j retrieval per pair to recover GT position by coverage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from service_api.dependencies import close_all, get_neo4j  # noqa: E402


DEFAULT_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_rank_explain.xlsx")
DEFAULT_OUT_XLSX = Path("data/processed/course_recommendation_metrics/gt_candidate_drop_reasons.xlsx")
DEFAULT_OUT_JSON = Path("data/processed/course_recommendation_metrics/gt_candidate_drop_reasons.json")


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _norm_pair_id(value: object) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    return text


def _split_semicolon(text: object) -> List[str]:
    raw = _norm_text(text)
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return list(dict.fromkeys(parts))


def _normalize_course_id(course_id: object) -> str:
    return _norm_text(course_id).upper().replace(" ", "")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _norm_text(value).lower()
    return text in {"1", "true", "yes", "y"}


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _fetch_gt_taught_skills(neo4j, course_id: str) -> Tuple[bool, List[str]]:
    rows = neo4j.query(
        """
        MATCH (c:Course {course_id: $course_id})
        OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
        RETURN c.course_id AS course_id, collect(DISTINCT s.skill_id) AS taught_skill_ids
        """,
        {"course_id": course_id},
    )
    if not rows:
        return False, []
    taught = [str(x).strip() for x in rows[0].get("taught_skill_ids", []) if str(x).strip()]
    return True, list(dict.fromkeys(taught))


def _fetch_no_limit_candidate_ranking(neo4j, canonical_skill_ids: List[str]) -> Dict[str, int]:
    if not canonical_skill_ids:
        return {}
    rows = neo4j.query(
        """
        MATCH (c:Course)-[:TEACHES]->(s:Skill)
        WHERE s.skill_id IN $ids
        RETURN c.course_id AS course_id,
               size(collect(DISTINCT s.skill_id)) AS coverage_count
        ORDER BY coverage_count DESC, c.course_id ASC
        """,
        {"ids": canonical_skill_ids},
    )
    rank_by_course: Dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        cid = _normalize_course_id(row.get("course_id"))
        if cid and cid not in rank_by_course:
            rank_by_course[cid] = idx
    return rank_by_course


def classify_reason(
    *,
    gt_course_found_in_graph: bool,
    overlap_count: int,
    gt_rank_no_limit: Optional[int],
    candidate_limit: int,
) -> str:
    if not gt_course_found_in_graph:
        return "gt_course_not_found_in_graph"
    if overlap_count == 0:
        return "canonical_skill_not_enough"
    if gt_rank_no_limit is None:
        return "canonical_present_but_neo4j_not_return_gt"
    if gt_rank_no_limit > candidate_limit:
        return "gt_in_graph_but_cut_by_limit_or_coverage"
    return "gt_should_be_in_candidate_but_missing"


def build_analysis(input_xlsx: Path, candidate_limit: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_sample = pd.read_excel(input_xlsx, sheet_name="per_sample")
    rank_explain = pd.read_excel(input_xlsx, sheet_name="rank_explain_failures")

    # Target set: GT rows with rank = -1 (not in candidate pool in benchmark output).
    target = rank_explain[
        (rank_explain["is_groundtruth"].apply(_to_bool))
        & (pd.to_numeric(rank_explain["rank"], errors="coerce") == -1)
    ].copy()

    # pair_id -> canonical ids map from per_sample.
    canonical_by_pair: Dict[str, List[str]] = {}
    for _, row in per_sample.iterrows():
        pid = _norm_pair_id(row.get("pair_id"))
        if not pid:
            continue
        canonical_by_pair[pid] = _split_semicolon(row.get("canonical_skill_ids"))

    neo4j = get_neo4j()
    no_limit_rank_cache: Dict[str, Dict[str, int]] = {}
    detail_rows: List[Dict[str, object]] = []

    for _, row in target.iterrows():
        pid = _norm_pair_id(row.get("pair_id"))
        jd_id = _norm_text(row.get("jd_id"))
        course_id_raw = _norm_text(row.get("course_id"))
        course_id = _normalize_course_id(course_id_raw)

        canonical_ids = canonical_by_pair.get(pid, [])
        canonical_set: Set[str] = set(canonical_ids)

        found_in_graph, gt_taught_skills = _fetch_gt_taught_skills(neo4j, course_id_raw)
        gt_taught_set = set(gt_taught_skills)
        overlap_skills = sorted(gt_taught_set.intersection(canonical_set))
        overlap_count = len(overlap_skills)

        if pid not in no_limit_rank_cache:
            no_limit_rank_cache[pid] = _fetch_no_limit_candidate_ranking(neo4j, canonical_ids)
        gt_rank_no_limit = no_limit_rank_cache[pid].get(course_id)

        reason = classify_reason(
            gt_course_found_in_graph=found_in_graph,
            overlap_count=overlap_count,
            gt_rank_no_limit=gt_rank_no_limit,
            candidate_limit=candidate_limit,
        )

        detail_rows.append(
            {
                "pair_id": pid,
                "jd_id": jd_id,
                "course_id": course_id_raw,
                "candidate_limit": candidate_limit,
                "canonical_skill_count": len(canonical_ids),
                "gt_taught_skill_count": len(gt_taught_skills),
                "overlap_skill_count": overlap_count,
                "overlap_skill_ids": "; ".join(overlap_skills),
                "gt_course_found_in_graph": found_in_graph,
                "gt_rank_no_limit": gt_rank_no_limit if gt_rank_no_limit is not None else "",
                "reason": reason,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(
            [{"reason": "no_target_rows", "count": 0, "percentage": 0.0}]
        )
        return detail_df, summary_df

    counts = detail_df["reason"].value_counts(dropna=False)
    summary_df = counts.rename_axis("reason").reset_index(name="count")
    summary_df["percentage"] = summary_df["count"] / len(detail_df)

    # Pair-level summary: choose most severe reason per pair.
    priority = {
        "canonical_skill_not_enough": 1,
        "canonical_present_but_neo4j_not_return_gt": 2,
        "gt_in_graph_but_cut_by_limit_or_coverage": 3,
        "gt_should_be_in_candidate_but_missing": 4,
        "gt_course_not_found_in_graph": 5,
    }

    def pick_pair_reason(reasons: Iterable[str]) -> str:
        unique = sorted(set(str(r) for r in reasons), key=lambda x: priority.get(x, 999))
        return unique[0] if unique else "unknown"

    pair_reason = (
        detail_df.groupby("pair_id")["reason"].apply(pick_pair_reason).reset_index(name="pair_primary_reason")
    )
    pair_counts = pair_reason["pair_primary_reason"].value_counts(dropna=False)
    pair_summary_df = pair_counts.rename_axis("reason").reset_index(name="count")
    pair_summary_df["percentage"] = pair_summary_df["count"] / max(len(pair_reason), 1)
    pair_summary_df["level"] = "pair"

    summary_df["level"] = "course_row"
    summary_df = pd.concat([summary_df, pair_summary_df], ignore_index=True)
    return detail_df, summary_df


def save_outputs(detail_df: pd.DataFrame, summary_df: pd.DataFrame, out_xlsx: Path, out_json: Path, meta: Dict[str, object]) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        detail_df.to_excel(writer, sheet_name="detail", index=False)

    payload = {
        "meta": meta,
        "summary": summary_df.to_dict(orient="records"),
        "detail_count": int(len(detail_df)),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_console(summary_df: pd.DataFrame, detail_df: pd.DataFrame, meta: Dict[str, object]) -> None:
    print("=== GT Candidate Drop Reason Analysis ===")
    print(f"Input XLSX: {meta['input_xlsx']}")
    print(f"Candidate limit: {meta['candidate_limit']}")
    print(f"Analyzed GT fail rows (rank=-1): {len(detail_df)}")

    for level in ["course_row", "pair"]:
        sub = summary_df[summary_df["level"] == level].copy()
        if sub.empty:
            continue
        print(f"\n--- Summary ({level}) ---")
        printable = sub.copy()
        printable["percentage"] = printable["percentage"].map(lambda x: f"{x * 100.0:.2f}%")
        print(printable[["reason", "count", "percentage"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GT candidate drop reasons into A/B/C buckets.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_XLSX, help="Rank-explain comparison XLSX path.")
    parser.add_argument("--candidate-limit", type=int, default=50, help="Candidate list limit used in serving (default: 50).")
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX, help="Output XLSX summary path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output JSON summary path.")
    args = parser.parse_args()

    detail_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    try:
        detail_df, summary_df = build_analysis(args.input_xlsx, args.candidate_limit)
        meta = {
            "input_xlsx": str(args.input_xlsx),
            "candidate_limit": int(args.candidate_limit),
            "out_xlsx": str(args.out_xlsx),
            "out_json": str(args.out_json),
        }
        print_console(summary_df, detail_df, meta)
        save_outputs(detail_df, summary_df, args.out_xlsx, args.out_json, meta)
        print(f"\nSaved summary XLSX: {args.out_xlsx}")
        print(f"Saved summary JSON: {args.out_json}")
    finally:
        # Close singleton connections created by service_api.dependencies.
        close_all()


if __name__ == "__main__":
    main()