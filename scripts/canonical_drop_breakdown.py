#!/usr/bin/env python3
"""Break down GT skill drops at canonical stage.

Target split for GT failed rows (rank = -1):
1) no GT skill appears in raw top-5 search hits.
2) GT skill appears in raw top-5 hits but is dropped before final canonical list
   (gate / score margin / max-24 bound effects).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from service_api.config import settings  # noqa: E402
from service_api.dependencies import close_all, get_neo4j, get_skill_search_service  # noqa: E402


DEFAULT_INPUT_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_rank_explain.xlsx")
DEFAULT_OUT_XLSX = Path("data/processed/course_recommendation_metrics/canonical_drop_breakdown.xlsx")
DEFAULT_OUT_JSON = Path("data/processed/course_recommendation_metrics/canonical_drop_breakdown.json")


def norm_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def norm_pair_id(value: object) -> str:
    text = norm_text(value)
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    return text


def normalize_course_id(course_id: object) -> str:
    return norm_text(course_id).upper().replace(" ", "")


def split_semicolon(value: object) -> List[str]:
    raw = norm_text(value)
    if not raw:
        return []
    out = [x.strip() for x in raw.split(";") if x.strip()]
    return list(dict.fromkeys(out))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return norm_text(value).lower() in {"1", "true", "yes", "y"}


def parse_gap_query_map(gap_search_queries: object, identified_gaps: object) -> Dict[str, str]:
    """Parse serialized "gap -> query" map robustly even when query contains ';'."""
    text = norm_text(gap_search_queries)
    gaps = [g for g in split_semicolon(identified_gaps) if g]
    if not text or not gaps:
        return {}

    # Find all marker starts based on known gaps.
    markers: List[Tuple[int, str, str]] = []
    for gap in gaps:
        marker = f"{gap} -> "
        start = text.find(marker)
        if start >= 0:
            markers.append((start, gap, marker))
    if not markers:
        # Fallback: regex scan for first-level "X -> Y" chunks.
        out: Dict[str, str] = {}
        for part in split_semicolon(text):
            if "->" not in part:
                continue
            left, right = part.split("->", 1)
            l = left.strip()
            r = right.strip()
            if l and r:
                out[l] = r
        return out

    markers.sort(key=lambda x: x[0])
    out: Dict[str, str] = {}
    for idx, (start, gap, marker) in enumerate(markers):
        value_start = start + len(marker)
        value_end = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        query = text[value_start:value_end].strip()
        # Remove separator residue if present.
        query = query.strip(" ;")
        if gap and query:
            out[gap] = query
    return out


def fetch_course_taught_skills(neo4j, course_id: str) -> Tuple[bool, List[str]]:
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
    skills = [str(x).strip() for x in rows[0].get("taught_skill_ids", []) if str(x).strip()]
    return True, list(dict.fromkeys(skills))


def classify_row(
    *,
    gt_in_graph: bool,
    raw_overlap_count: int,
    canonical_overlap_count: int,
) -> str:
    if not gt_in_graph:
        return "gt_course_not_found_in_graph"
    if raw_overlap_count == 0:
        return "no_gt_skill_in_raw_top5_search"
    if raw_overlap_count > 0 and canonical_overlap_count == 0:
        return "gt_skill_in_raw_top5_but_dropped_by_filters"
    return "gt_skill_present_in_final_canonical_unexpected"


def build_breakdown(input_xlsx: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_sample = pd.read_excel(input_xlsx, sheet_name="per_sample")
    rank_explain = pd.read_excel(input_xlsx, sheet_name="rank_explain_failures")

    target = rank_explain[
        (rank_explain["is_groundtruth"].apply(to_bool))
        & (pd.to_numeric(rank_explain["rank"], errors="coerce") == -1)
    ].copy()

    per_by_pair: Dict[str, pd.Series] = {}
    for _, row in per_sample.iterrows():
        pid = norm_pair_id(row.get("pair_id"))
        if pid:
            per_by_pair[pid] = row

    skill_svc = get_skill_search_service()
    neo4j = get_neo4j()

    raw_union_by_pair: Dict[str, Set[str]] = {}
    raw_details_by_pair: Dict[str, List[Dict[str, object]]] = {}

    rows: List[Dict[str, object]] = []
    for _, row in target.iterrows():
        pid = norm_pair_id(row.get("pair_id"))
        jd_id = norm_text(row.get("jd_id"))
        course_id = norm_text(row.get("course_id"))
        course_norm = normalize_course_id(course_id)
        ps = per_by_pair.get(pid)

        canonical_ids = split_semicolon(ps.get("canonical_skill_ids") if ps is not None else "")
        canonical_set = set(canonical_ids)

        if pid not in raw_union_by_pair:
            identified = ps.get("identified_gaps") if ps is not None else ""
            query_blob = ps.get("gap_search_queries") if ps is not None else ""
            gap_query_map = parse_gap_query_map(query_blob, identified)

            raw_union: Set[str] = set()
            raw_gap_details: List[Dict[str, object]] = []
            for gap, query in gap_query_map.items():
                hits = skill_svc.search_by_text(query, limit=settings.SKILL_SEARCH_RAW_LIMIT)
                hit_ids: List[str] = []
                for h in hits:
                    sid = norm_text(h.get("skill_id"))
                    if sid:
                        raw_union.add(sid)
                        hit_ids.append(sid)
                raw_gap_details.append({"gap": gap, "query": query, "top5_skill_ids": hit_ids})

            raw_union_by_pair[pid] = raw_union
            raw_details_by_pair[pid] = raw_gap_details

        raw_union = raw_union_by_pair.get(pid, set())

        gt_in_graph, gt_taught_skills = fetch_course_taught_skills(neo4j, course_id)
        gt_taught_set = set(gt_taught_skills)

        raw_overlap = sorted(gt_taught_set.intersection(raw_union))
        canonical_overlap = sorted(gt_taught_set.intersection(canonical_set))

        reason = classify_row(
            gt_in_graph=gt_in_graph,
            raw_overlap_count=len(raw_overlap),
            canonical_overlap_count=len(canonical_overlap),
        )

        rows.append(
            {
                "pair_id": pid,
                "jd_id": jd_id,
                "course_id": course_id,
                "course_id_norm": course_norm,
                "gt_course_found_in_graph": gt_in_graph,
                "gt_taught_skill_count": len(gt_taught_skills),
                "canonical_skill_count": len(canonical_ids),
                "raw_top5_union_skill_count": len(raw_union),
                "raw_overlap_count": len(raw_overlap),
                "canonical_overlap_count": len(canonical_overlap),
                "raw_overlap_skill_ids": "; ".join(raw_overlap),
                "canonical_overlap_skill_ids": "; ".join(canonical_overlap),
                "reason": reason,
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        summary_df = pd.DataFrame([{"level": "course_row", "reason": "no_target_rows", "count": 0, "percentage": 0.0}])
        return detail_df, summary_df

    # Row-level summary
    c = detail_df["reason"].value_counts(dropna=False)
    summary_row = c.rename_axis("reason").reset_index(name="count")
    summary_row["percentage"] = summary_row["count"] / len(detail_df)
    summary_row["level"] = "course_row"

    # Pair-level summary
    priority = {
        "no_gt_skill_in_raw_top5_search": 1,
        "gt_skill_in_raw_top5_but_dropped_by_filters": 2,
        "gt_course_not_found_in_graph": 3,
        "gt_skill_present_in_final_canonical_unexpected": 4,
    }

    def choose_pair_reason(reasons: Iterable[str]) -> str:
        uniq = sorted(set(str(r) for r in reasons), key=lambda x: priority.get(x, 999))
        return uniq[0] if uniq else "unknown"

    pair_reason = detail_df.groupby("pair_id")["reason"].apply(choose_pair_reason).reset_index(name="reason")
    cp = pair_reason["reason"].value_counts(dropna=False)
    summary_pair = cp.rename_axis("reason").reset_index(name="count")
    summary_pair["percentage"] = summary_pair["count"] / max(len(pair_reason), 1)
    summary_pair["level"] = "pair"

    summary_df = pd.concat([summary_row, summary_pair], ignore_index=True)
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
    print("=== Canonical Drop Breakdown ===")
    print(f"Input XLSX: {meta['input_xlsx']}")
    print(f"Analyzed GT fail rows: {len(detail_df)}")
    for level in ["course_row", "pair"]:
        sub = summary_df[summary_df["level"] == level].copy()
        if sub.empty:
            continue
        print(f"\n--- Summary ({level}) ---")
        show = sub.copy()
        show["percentage"] = show["percentage"].map(lambda x: f"{x * 100.0:.2f}%")
        print(show[["reason", "count", "percentage"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down canonical drops for GT failed rows.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX, help="Rank-explain benchmark XLSX.")
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX, help="Output XLSX path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output JSON path.")
    args = parser.parse_args()

    detail_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    try:
        detail_df, summary_df = build_breakdown(args.input_xlsx)
        meta = {
            "input_xlsx": str(args.input_xlsx),
            "out_xlsx": str(args.out_xlsx),
            "out_json": str(args.out_json),
            "raw_topk_limit": int(settings.SKILL_SEARCH_RAW_LIMIT),
        }
        print_console(summary_df, detail_df, meta)
        save_outputs(detail_df, summary_df, args.out_xlsx, args.out_json, meta)
        print(f"\nSaved summary XLSX: {args.out_xlsx}")
        print(f"Saved summary JSON: {args.out_json}")
    finally:
        close_all()


if __name__ == "__main__":
    main()