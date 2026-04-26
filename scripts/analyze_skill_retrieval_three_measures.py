#!/usr/bin/env python3
"""Analyze skill retrieval failures with three core measurements.

Measurements:
1) GT-skill recall in union top-k retrieval (k=5/10/20)
2) Enrichment on vs off comparison for the same k values
3) Breakdown by gap groups (short/long/ambiguous/rare-domain)

This script focuses on GT rows that failed at course level (rank = -1) from the
rank-explain benchmark output, then inspects skill retrieval behavior.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from service_api.dependencies import close_all, get_neo4j, get_skill_search_service  # noqa: E402


DEFAULT_INPUT_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_rank_explain.xlsx")
DEFAULT_OUT_XLSX = Path("data/processed/course_recommendation_metrics/skill_retrieval_three_measures.xlsx")
DEFAULT_OUT_JSON = Path("data/processed/course_recommendation_metrics/skill_retrieval_three_measures.json")
DEFAULT_TOPK = (5, 10, 20)

GENERIC_GAP_TERMS = {
    "quan ly",
    "phan tich",
    "thiet ke",
    "van hanh",
    "giao tiep",
    "lap trinh",
    "quan tri",
    "phat trien",
    "toi uu",
    "bao cao",
    "thu nghiem",
}


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
    text = norm_text(value)
    if not text:
        return []
    out = [x.strip() for x in text.split(";") if x.strip()]
    return list(dict.fromkeys(out))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return norm_text(value).lower() in {"1", "true", "yes", "y"}


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", norm_text(text).lower())
    return [t for t in cleaned.split() if t]


def parse_gap_query_map(gap_search_queries: object, identified_gaps: object) -> Dict[str, str]:
    """Parse serialized "gap -> query" map robustly even when query contains ';'."""
    text = norm_text(gap_search_queries)
    gaps = [g for g in split_semicolon(identified_gaps) if g]
    if not text or not gaps:
        return {}

    markers: List[Tuple[int, str, str]] = []
    for gap in gaps:
        marker = f"{gap} -> "
        pos = text.find(marker)
        if pos >= 0:
            markers.append((pos, gap, marker))

    if not markers:
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
        query = text[value_start:value_end].strip().strip(" ;")
        if gap and query:
            out[gap] = query
    return out


def fetch_gt_taught_skills(neo4j, course_id: str) -> Tuple[bool, List[str]]:
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


def classify_gap_groups(gap: str, token_freq: Dict[str, int]) -> List[str]:
    toks = tokenize(gap)
    wc = len(toks)
    groups: List[str] = []

    if wc <= 2:
        groups.append("short")
    elif wc >= 6:
        groups.append("long")
    else:
        groups.append("medium")

    joined = " ".join(toks)
    if joined in GENERIC_GAP_TERMS or (toks and all(t in GENERIC_GAP_TERMS for t in toks)):
        groups.append("ambiguous")

    # Rare-domain heuristic: at least one token is very infrequent in fail-case gaps.
    if toks and min(token_freq.get(t, 0) for t in toks) <= 2:
        groups.append("rare_domain")

    return list(dict.fromkeys(groups))


def search_topk_union(skill_svc, queries: Sequence[str], k: int, cache: Dict[Tuple[str, int], List[str]]) -> Set[str]:
    union_ids: Set[str] = set()
    for q in queries:
        key = (q, k)
        if key not in cache:
            hits = skill_svc.search_by_text(q, limit=k)
            cache[key] = [norm_text(h.get("skill_id")) for h in hits if norm_text(h.get("skill_id"))]
        union_ids.update(cache[key])
    return union_ids


def build_analysis(input_xlsx: Path, topk_values: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_sample = pd.read_excel(input_xlsx, sheet_name="per_sample")
    rank_explain = pd.read_excel(input_xlsx, sheet_name="rank_explain_failures")

    target = rank_explain[
        (rank_explain["is_groundtruth"].apply(to_bool))
        & (pd.to_numeric(rank_explain["rank"], errors="coerce") == -1)
    ].copy()

    per_by_pair: Dict[str, pd.Series] = {}
    all_fail_gaps: List[str] = []
    for _, row in per_sample.iterrows():
        pid = norm_pair_id(row.get("pair_id"))
        if not pid:
            continue
        per_by_pair[pid] = row
    for _, row in target.iterrows():
        pid = norm_pair_id(row.get("pair_id"))
        ps = per_by_pair.get(pid)
        if ps is None:
            continue
        all_fail_gaps.extend(split_semicolon(ps.get("identified_gaps")))

    token_freq = Counter(t for g in all_fail_gaps for t in tokenize(g))

    skill_svc = get_skill_search_service()
    neo4j = get_neo4j()
    cache: Dict[Tuple[str, int], List[str]] = {}

    row_records: List[Dict[str, object]] = []
    gap_records: List[Dict[str, object]] = []

    for _, row in target.iterrows():
        pid = norm_pair_id(row.get("pair_id"))
        jd_id = norm_text(row.get("jd_id"))
        course_id = norm_text(row.get("course_id"))

        ps = per_by_pair.get(pid)
        if ps is None:
            continue

        identified_gaps = split_semicolon(ps.get("identified_gaps"))
        query_map = parse_gap_query_map(ps.get("gap_search_queries"), ps.get("identified_gaps"))
        enriched_queries = [query_map[g] for g in identified_gaps if g in query_map]
        plain_queries = identified_gaps

        gt_found, gt_skills = fetch_gt_taught_skills(neo4j, course_id)
        gt_set = set(gt_skills)

        rec: Dict[str, object] = {
            "pair_id": pid,
            "jd_id": jd_id,
            "course_id": course_id,
            "gt_course_found_in_graph": gt_found,
            "gt_skill_count": len(gt_skills),
            "gap_count": len(identified_gaps),
        }

        for k in topk_values:
            union_enriched = search_topk_union(skill_svc, enriched_queries, k, cache)
            union_plain = search_topk_union(skill_svc, plain_queries, k, cache)

            hit_enriched = int(bool(gt_set.intersection(union_enriched))) if gt_set else 0
            hit_plain = int(bool(gt_set.intersection(union_plain))) if gt_set else 0

            rec[f"hit_enriched_top{k}"] = hit_enriched
            rec[f"hit_plain_top{k}"] = hit_plain
            rec[f"recall_gain_enriched_vs_plain_top{k}"] = hit_enriched - hit_plain

        row_records.append(rec)

        # Gap-level diagnostics for group breakdown (per gap, per mode, per k).
        for gap in identified_gaps:
            groups = classify_gap_groups(gap, token_freq)
            eq = query_map.get(gap, gap)
            for mode, q in (("enriched", eq), ("plain", gap)):
                for k in topk_values:
                    key = (q, k)
                    if key not in cache:
                        hits = skill_svc.search_by_text(q, limit=k)
                        cache[key] = [norm_text(h.get("skill_id")) for h in hits if norm_text(h.get("skill_id"))]
                    top_ids = set(cache[key])
                    hit = int(bool(gt_set.intersection(top_ids))) if gt_set else 0
                    gap_records.append(
                        {
                            "pair_id": pid,
                            "jd_id": jd_id,
                            "course_id": course_id,
                            "gap": gap,
                            "mode": mode,
                            "k": k,
                            "hit": hit,
                            "groups": "; ".join(groups),
                        }
                    )

    row_df = pd.DataFrame(row_records)
    gap_df = pd.DataFrame(gap_records)

    if row_df.empty:
        summary_df = pd.DataFrame(
            [{"metric": "no_rows", "value": 0, "note": "No GT fail rows (rank=-1) found."}]
        )
        return row_df, gap_df, summary_df

    # Measurement 1 + 2 summary.
    summary_rows: List[Dict[str, object]] = []
    n = len(row_df)
    for k in topk_values:
        en_col = f"hit_enriched_top{k}"
        pl_col = f"hit_plain_top{k}"
        en_rate = float(row_df[en_col].mean()) if en_col in row_df.columns else 0.0
        pl_rate = float(row_df[pl_col].mean()) if pl_col in row_df.columns else 0.0
        summary_rows.append({"metric": f"recall_enriched_top{k}", "value": en_rate, "note": "GT-skill hit rate among failed GT rows"})
        summary_rows.append({"metric": f"recall_plain_top{k}", "value": pl_rate, "note": "GT-skill hit rate among failed GT rows"})
        summary_rows.append({"metric": f"recall_gain_enriched_vs_plain_top{k}", "value": en_rate - pl_rate, "note": "Positive means enrichment helps"})

    # Measurement 3 summary: by gap group.
    if not gap_df.empty:
        exploded = gap_df.copy()
        exploded["group"] = exploded["groups"].str.split("; ")
        exploded = exploded.explode("group")
        exploded = exploded[exploded["group"].notna() & (exploded["group"] != "")]
        if not exploded.empty:
            grp = (
                exploded.groupby(["group", "mode", "k"], as_index=False)["hit"].mean()
                .rename(columns={"hit": "value"})
            )
            for _, r in grp.iterrows():
                summary_rows.append(
                    {
                        "metric": f"group_recall_{r['group']}_{r['mode']}_top{int(r['k'])}",
                        "value": float(r["value"]),
                        "note": "Gap-level hit rate",
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    return row_df, gap_df, summary_df


def save_outputs(
    row_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_xlsx: Path,
    out_json: Path,
    meta: Dict[str, object],
) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        row_df.to_excel(writer, sheet_name="row_level", index=False)
        gap_df.to_excel(writer, sheet_name="gap_level", index=False)

    payload = {
        "meta": meta,
        "summary": summary_df.to_dict(orient="records"),
        "row_count": int(len(row_df)),
        "gap_count": int(len(gap_df)),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_console(summary_df: pd.DataFrame, row_df: pd.DataFrame, meta: Dict[str, object]) -> None:
    print("=== Skill Retrieval Three-Measure Diagnostics ===")
    print(f"Input XLSX: {meta['input_xlsx']}")
    print(f"Top-k values: {meta['topk_values']}")
    print(f"Analyzed failed GT rows: {len(row_df)}")

    base = summary_df[summary_df["metric"].str.startswith("recall_") | summary_df["metric"].str.startswith("recall_gain_")]
    if not base.empty:
        print("\n--- Core recall metrics ---")
        show = base.copy()
        show["value"] = show["value"].map(lambda x: f"{float(x) * 100.0:.2f}%" if "gain" not in str(x) else x)
        # Keep original numeric display for gain rows in a simple pass.
        print(base.to_string(index=False))

    # Short display for group recalls.
    group_rows = summary_df[summary_df["metric"].str.startswith("group_recall_")]
    if not group_rows.empty:
        print("\n--- Group recall metrics (sample) ---")
        print(group_rows.head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3 retrieval diagnostics for GT skill misses.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX, help="Rank-explain benchmark XLSX path.")
    parser.add_argument("--topk", default="5,10,20", help="Comma-separated top-k list, e.g. 5,10,20")
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX, help="Output XLSX path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output JSON path.")
    args = parser.parse_args()

    topk_values = tuple(int(x.strip()) for x in str(args.topk).split(",") if x.strip())
    if not topk_values:
        topk_values = DEFAULT_TOPK

    row_df = pd.DataFrame()
    gap_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    try:
        row_df, gap_df, summary_df = build_analysis(args.input_xlsx, topk_values)
        meta = {
            "input_xlsx": str(args.input_xlsx),
            "topk_values": list(topk_values),
            "out_xlsx": str(args.out_xlsx),
            "out_json": str(args.out_json),
        }
        print_console(summary_df, row_df, meta)
        save_outputs(row_df, gap_df, summary_df, args.out_xlsx, args.out_json, meta)
        print(f"\nSaved summary XLSX: {args.out_xlsx}")
        print(f"Saved summary JSON: {args.out_json}")
    finally:
        close_all()


if __name__ == "__main__":
    main()