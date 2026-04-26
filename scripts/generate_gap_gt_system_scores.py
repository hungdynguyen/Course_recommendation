#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

# Support both host execution (/root/courses_rec) and container execution (/app).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WORKSPACE_ROOT") or os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, f"{ROOT}/src")

from service_api.dependencies import get_skill_search_service  # noqa: E402

DESIGNED_XLSX = f"{ROOT}/data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison.xlsx"
BASELINE_XLSX = f"{ROOT}/data/processed/course_recommendation_metrics/embedding_baseline_groundtruth_gaps_comparison.xlsx"
COURSE_DIR = f"{ROOT}/data/Data_Courses_Filtered"
OUT_XLSX = f"{ROOT}/data/processed/course_recommendation_metrics/designed_failures_rootcause_full.xlsx"

SCORE_PAT = re.compile(r"^\s*(.*?)\s*->\s*([^:]+):(.*?)\s*\(([-+]?\d*\.?\d+)\)\s*$")
NOMATCH_PAT = re.compile(r"^\s*(.*?)\s*->\s*<no_match>\s*$", re.I)
QUERY_PAT = re.compile(r"^\s*(.*?)\s*->\s*(.*?)\s*$")

# ES hybrid_search: per_branch_limit=limit*3, vector_search num_candidates=per_branch_limit*5
# => num_candidates = limit*15 <= 10000 => limit <= 666
SAFE_QUERY_LIMIT = 600


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def split_semis(x: object) -> List[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    return [p.strip() for p in str(x).split(";") if str(p).strip()]


def norm_key(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def parse_top1_map(gap_to_canonical_top1: object) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for part in split_semis(gap_to_canonical_top1):
        nm = NOMATCH_PAT.match(part)
        if nm:
            gap = nm.group(1).strip()
            result[gap] = {
                "matched_skill_id": "",
                "matched_skill_name": "<no_match>",
                "matched_score_test": None,
            }
            continue
        m = SCORE_PAT.match(part)
        if not m:
            continue
        gap = m.group(1).strip()
        sid = m.group(2).strip()
        label = m.group(3).strip()
        score = float(m.group(4))
        result[gap] = {
            "matched_skill_id": sid,
            "matched_skill_name": label,
            "matched_score_test": score,
        }
    return result


def parse_gap_query_map(gap_search_queries: object) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in split_semis(gap_search_queries):
        m = QUERY_PAT.match(part)
        if not m:
            continue
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def rank_in_list(items: List[str], target: str) -> Optional[int]:
    for i, x in enumerate(items, start=1):
        if x == target:
            return i
    return None


def load_catalog(course_dir: str) -> Dict[str, dict]:
    catalog: Dict[str, dict] = {}
    for dp, _, fns in os.walk(course_dir):
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(dp, fn)
            try:
                d = json.load(open(fp, "r", encoding="utf-8"))
            except Exception:
                continue
            cid = str(d.get("course_id") or "").strip()
            if cid:
                catalog[cid] = d
    return catalog


def classify_gap_cause(
    matched_skill_id: str,
    matched_score_system: Optional[float],
    best_gt_score_system: Optional[float],
) -> str:
    if not matched_skill_id:
        return "no_top1_match"
    if matched_score_system is None and best_gt_score_system is not None:
        return "top1_not_in_runtime_pool"
    if matched_score_system is None and best_gt_score_system is None:
        return "both_unscored"
    if matched_score_system is not None and best_gt_score_system is None:
        return "gt_not_covered_by_runtime_pool"

    delta = float(best_gt_score_system) - float(matched_score_system)
    if delta >= 0.15:
        return "under_match_large"
    if delta >= 0.05:
        return "under_match_medium"
    if delta <= -0.05:
        return "matched_better"
    return "similar"


def primary_case_cause(causes: List[str]) -> str:
    if not causes:
        return "unknown"
    priority = [
        "under_match_large",
        "under_match_medium",
        "top1_not_in_runtime_pool",
        "no_top1_match",
        "gt_not_covered_by_runtime_pool",
        "both_unscored",
        "similar",
        "matched_better",
    ]
    cnt = Counter(causes)
    for p in priority:
        if cnt.get(p, 0) > 0:
            return p
    return cnt.most_common(1)[0][0]


def main() -> None:
    t0 = time.time()
    log(f"START workspace_root={ROOT}")
    log(f"Input designed={DESIGNED_XLSX}")
    log(f"Input baseline={BASELINE_XLSX}")
    log(f"Output xlsx={OUT_XLSX}")

    log("Loading designed/baseline benchmark outputs...")
    designed_per = pd.read_excel(DESIGNED_XLSX, sheet_name="per_sample")
    designed_ranked = pd.read_excel(DESIGNED_XLSX, sheet_name="ranked_predictions")
    baseline_per = pd.read_excel(BASELINE_XLSX, sheet_name="per_sample")
    baseline_ranked = pd.read_excel(BASELINE_XLSX, sheet_name="ranked_predictions")

    log(
        f"Loaded designed_per={len(designed_per)}, designed_ranked={len(designed_ranked)}, "
        f"baseline_per={len(baseline_per)}, baseline_ranked={len(baseline_ranked)}"
    )

    log("Loading course catalog JSON files...")
    catalog = load_catalog(COURSE_DIR)
    log(f"Loaded catalog courses={len(catalog)}")

    # Wrong cases: designed misses top-10 truth overlap.
    wrong = designed_per[designed_per["hit@10"] == 0].copy()
    wrong_pair_ids = set(str(x) for x in wrong["pair_id"].tolist())
    log(f"Wrong cases detected (designed hit@10 == 0): {len(wrong)}")

    b_per_by_pair = {str(r["pair_id"]): r for _, r in baseline_per.iterrows()}

    # Build lookup for baseline ranked score per (pair_id, course_id)
    baseline_rank_score: Dict[tuple, float] = {}
    for _, r in baseline_ranked.iterrows():
        pid = str(r.get("pair_id"))
        cid = str(r.get("course_id") or "").strip()
        if not cid:
            continue
        baseline_rank_score[(pid, cid)] = float(r.get("score") or 0.0)

    # Build lookup for designed ranked details per (pair_id, course_id)
    designed_rank_detail: Dict[tuple, dict] = {}
    for _, r in designed_ranked.iterrows():
        pid = str(r.get("pair_id"))
        cid = str(r.get("course_id") or "").strip()
        if not cid:
            continue
        designed_rank_detail[(pid, cid)] = {
            "coverage_count": int(r.get("coverage_count") or 0),
            "covered_gap_labels": str(r.get("covered_gap_labels") or ""),
        }

    log("Initializing SkillSearchService (may load embedding model)...")
    skill_svc = get_skill_search_service()
    log("SkillSearchService ready")

    # Prepare GT skill texts from wrong cases only.
    log("Collecting unique GT skill texts from wrong cases...")
    unique_gt_texts = set()
    case_gt_skills: Dict[str, List[dict]] = {}
    for _, r in wrong.iterrows():
        pid = str(r.get("pair_id"))
        gt_rows: List[dict] = []
        seen = set()
        for cid in split_semis(r.get("truth_courses", "")):
            c = catalog.get(cid, {})
            ctitle = str(c.get("title") or "")
            for so in (c.get("skill_outcomes") or []):
                if not isinstance(so, dict):
                    continue
                sn = str(so.get("skill_name") or "").strip()
                sd = str(so.get("outcome_description") or "").strip()
                if not sn:
                    continue
                text = f"{sn} ; {sd}" if sd else sn
                key = (cid, norm_key(text))
                if key in seen:
                    continue
                seen.add(key)
                unique_gt_texts.add(text)
                gt_rows.append(
                    {
                        "source_course_id": cid,
                        "source_course_title": ctitle,
                        "raw_skill_name": sn,
                        "raw_skill_description": sd,
                        "gt_skill_text": text,
                    }
                )
        case_gt_skills[pid] = gt_rows
    log(f"Collected unique GT skill texts={len(unique_gt_texts)}")

    # Phase 1: GT text -> canonical mapping
    gt_text_to_canonical: Dict[str, dict] = {}
    all_gt = list(unique_gt_texts)
    p1 = time.time()
    log("Phase 1/4: mapping GT skill text -> canonical skill (limit=1, batch mode)")
    for i, group in enumerate(chunked(all_gt, 64), start=1):
        try:
            batch_hits = skill_svc.search_batch(skill_names=group, limit_per_skill=1, min_score=0.0)
            for text in group:
                hits = batch_hits.get(text, [])
                if hits:
                    h = hits[0]
                    gt_text_to_canonical[text] = {
                        "canonical_skill_id": str(h.get("skill_id") or ""),
                        "canonical_skill_label": str(h.get("canonical_label") or ""),
                        "gt_to_canonical_score": float(h.get("score") or 0.0),
                    }
                else:
                    gt_text_to_canonical[text] = {
                        "canonical_skill_id": "",
                        "canonical_skill_label": "",
                        "gt_to_canonical_score": None,
                    }
        except Exception:
            for text in group:
                hits = skill_svc.search_by_text(text, limit=1)
                if hits:
                    h = hits[0]
                    gt_text_to_canonical[text] = {
                        "canonical_skill_id": str(h.get("skill_id") or ""),
                        "canonical_skill_label": str(h.get("canonical_label") or ""),
                        "gt_to_canonical_score": float(h.get("score") or 0.0),
                    }
                else:
                    gt_text_to_canonical[text] = {
                        "canonical_skill_id": "",
                        "canonical_skill_label": "",
                        "gt_to_canonical_score": None,
                    }

        done = min(i * 64, len(all_gt))
        if done % 128 == 0 or done == len(all_gt):
            elapsed = time.time() - p1
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(all_gt) - done) / rate if rate > 0 else 0.0
            log(f"mapped_gt_texts {done}/{len(all_gt)} | rate={rate:.2f}/s | eta={eta:.1f}s")

    # Collect unique gap queries from wrong cases only.
    log("Collecting unique gap queries from wrong cases...")
    unique_gap_queries = set()
    case_gap_query_map: Dict[str, Dict[str, str]] = {}
    case_top1_map: Dict[str, Dict[str, dict]] = {}
    for _, r in wrong.iterrows():
        pid = str(r.get("pair_id"))
        gmap = parse_gap_query_map(r.get("gap_search_queries", ""))
        case_gap_query_map[pid] = gmap
        case_top1_map[pid] = parse_top1_map(r.get("gap_to_canonical_top1", ""))
        for q in gmap.values():
            unique_gap_queries.add(q)
    log(f"Collected unique gap queries={len(unique_gap_queries)}")

    # Phase 2: Query -> score map
    query_to_score_map: Dict[str, Dict[str, float]] = {}
    all_queries = list(unique_gap_queries)
    p2 = time.time()
    log(f"Phase 2/4: scoring gap queries against canonical index (limit={SAFE_QUERY_LIMIT}, batch mode)")
    for i, group in enumerate(chunked(all_queries, 32), start=1):
        try:
            batch_hits = skill_svc.search_batch(skill_names=group, limit_per_skill=SAFE_QUERY_LIMIT, min_score=0.0)
            for q in group:
                hits = batch_hits.get(q, [])
                query_to_score_map[q] = {
                    str(h.get("skill_id") or ""): float(h.get("score") or 0.0)
                    for h in hits
                    if h.get("skill_id")
                }
        except Exception:
            for q in group:
                hits = skill_svc.search_by_text(q, limit=SAFE_QUERY_LIMIT)
                query_to_score_map[q] = {
                    str(h.get("skill_id") or ""): float(h.get("score") or 0.0)
                    for h in hits
                    if h.get("skill_id")
                }

        done = min(i * 32, len(all_queries))
        if done % 64 == 0 or done == len(all_queries):
            elapsed = time.time() - p2
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(all_queries) - done) / rate if rate > 0 else 0.0
            log(f"scored_gap_queries {done}/{len(all_queries)} | rate={rate:.2f}/s | eta={eta:.1f}s")

    # Phase 3: gap-level diagnostics on wrong cases
    log("Phase 3/4: building gap-level diagnostics...")
    gap_rows: List[dict] = []
    case_cause_counter: Dict[str, List[str]] = {}

    for idx, r in wrong.iterrows():
        pid = str(r.get("pair_id"))
        jd_id = str(r.get("jd_id"))
        jd_title = str(r.get("jd_title") or "")

        gap_query_map = case_gap_query_map.get(pid, {})
        top1_map = case_top1_map.get(pid, {})
        gt_skills = case_gt_skills.get(pid, [])

        case_causes: List[str] = []
        for gap_name in split_semis(r.get("identified_gaps", "")):
            q = gap_query_map.get(gap_name, gap_name)
            score_map = query_to_score_map.get(q, {})

            mapped = top1_map.get(
                gap_name,
                {
                    "matched_skill_id": "",
                    "matched_skill_name": "<missing_parse>",
                    "matched_score_test": None,
                },
            )

            m_sid = mapped["matched_skill_id"]
            m_name = mapped["matched_skill_name"]
            m_score_test = mapped["matched_score_test"]
            m_score_system = score_map.get(m_sid) if m_sid else None

            best_gt_score = None
            best_gt_name = None
            best_gt_course = None
            best_gt_sid = None
            best_gt_label = None

            for gts in gt_skills:
                gt_map = gt_text_to_canonical.get(
                    gts["gt_skill_text"],
                    {
                        "canonical_skill_id": "",
                        "canonical_skill_label": "",
                        "gt_to_canonical_score": None,
                    },
                )
                g_sid = gt_map["canonical_skill_id"]
                g_score = score_map.get(g_sid) if g_sid else None
                if g_score is not None and (best_gt_score is None or g_score > best_gt_score):
                    best_gt_score = g_score
                    best_gt_name = gts["raw_skill_name"]
                    best_gt_course = gts["source_course_title"]
                    best_gt_sid = g_sid
                    best_gt_label = gt_map.get("canonical_skill_label")

            cause = classify_gap_cause(m_sid, m_score_system, best_gt_score)
            case_causes.append(cause)

            delta = None
            if m_score_system is not None and best_gt_score is not None:
                delta = float(best_gt_score) - float(m_score_system)

            gap_rows.append(
                {
                    "pair_id": pid,
                    "jd_id": jd_id,
                    "jd_title": jd_title,
                    "gap": gap_name,
                    "gap_query_used": q,
                    "matched_skill_id": m_sid,
                    "matched_skill_label": m_name,
                    "matched_score_test_output": m_score_test,
                    "matched_score_system": m_score_system,
                    "best_gt_skill_name": best_gt_name,
                    "best_gt_course": best_gt_course,
                    "best_gt_canonical_id": best_gt_sid,
                    "best_gt_canonical_label": best_gt_label,
                    "best_gt_score_system": best_gt_score,
                    "delta_best_gt_minus_matched": delta,
                    "gap_cause": cause,
                }
            )

        case_cause_counter[pid] = case_causes
        done = len(case_cause_counter)
        if done % 5 == 0 or done == len(wrong):
            log(f"processed_wrong_cases_for_gaps {done}/{len(wrong)}")

    gap_df = pd.DataFrame(gap_rows)

    # Phase 4: case/course-level diagnostics + baseline comparison
    log("Phase 4/4: building case-level and missed-course diagnostics...")
    case_rows: List[dict] = []
    miss_course_rows: List[dict] = []

    gap_df_by_pair: Dict[str, pd.DataFrame] = {
        pid: sub for pid, sub in gap_df.groupby("pair_id")
    }

    for _, r in wrong.iterrows():
        pid = str(r.get("pair_id"))
        jd_id = str(r.get("jd_id"))
        jd_title = str(r.get("jd_title") or "")

        truth_ids = split_semis(r.get("truth_courses", ""))
        d_pred_ids = split_semis(r.get("pred_top10", ""))
        d_overlap = split_semis(r.get("overlap_top10", ""))
        b_row = b_per_by_pair.get(pid)

        b_hit = None
        b_pred_ids: List[str] = []
        b_overlap: List[str] = []
        if b_row is not None:
            b_hit = b_row.get("hit@10")
            b_pred_ids = split_semis(b_row.get("pred_top10", ""))
            b_overlap = split_semis(b_row.get("overlap_top10", ""))

        causes = case_cause_counter.get(pid, [])
        primary = primary_case_cause(causes)

        case_rows.append(
            {
                "pair_id": pid,
                "jd_id": jd_id,
                "jd_title": jd_title,
                "truth_count": len(truth_ids),
                "gap_count": int(r.get("gap_count") or 0),
                "canonical_skill_count": int(r.get("canonical_skill_count") or 0),
                "designed_hit@10": float(r.get("hit@10") or 0.0),
                "baseline_hit@10": None if b_hit is None else float(b_hit),
                "designed_overlap_top10_count": len(d_overlap),
                "baseline_overlap_top10_count": len(b_overlap),
                "primary_case_cause": primary,
                "case_cause_counts": "; ".join(f"{k}:{v}" for k, v in Counter(causes).most_common()),
                "truth_courses": "; ".join(truth_ids),
                "designed_pred_top10": "; ".join(d_pred_ids),
                "baseline_pred_top10": "; ".join(b_pred_ids),
            }
        )

        pair_gap = gap_df_by_pair.get(pid)
        for cid in truth_ids:
            d_rank = rank_in_list(d_pred_ids, cid)
            b_rank = rank_in_list(b_pred_ids, cid)
            d_detail = designed_rank_detail.get((pid, cid), {})

            reason = ""
            if d_rank is None and b_rank is not None:
                reason = "missed_by_designed_but_found_by_baseline"
            elif d_rank is None and b_rank is None:
                reason = "missed_by_both"
            else:
                reason = "found_by_designed"

            miss_course_rows.append(
                {
                    "pair_id": pid,
                    "jd_id": jd_id,
                    "jd_title": jd_title,
                    "gt_course_id": cid,
                    "gt_course_title": str(catalog.get(cid, {}).get("title") or ""),
                    "designed_rank@10": d_rank,
                    "baseline_rank@10": b_rank,
                    "baseline_score_if_found": baseline_rank_score.get((pid, cid)),
                    "designed_coverage_count_if_found": d_detail.get("coverage_count"),
                    "designed_covered_gap_labels_if_found": d_detail.get("covered_gap_labels"),
                    "course_miss_reason": reason,
                    "primary_case_cause": primary,
                    "gap_causes_in_case": (
                        "" if pair_gap is None else "; ".join(pair_gap["gap_cause"].fillna("unknown").astype(str).tolist())
                    ),
                }
            )

    case_df = pd.DataFrame(case_rows)
    miss_course_df = pd.DataFrame(miss_course_rows)

    # Summary sheet
    cause_rows = []
    for cause, cnt in Counter(gap_df.get("gap_cause", pd.Series(dtype=object)).dropna().tolist()).most_common():
        cause_rows.append({"level": "gap", "cause": cause, "count": cnt})
    for cause, cnt in Counter(case_df.get("primary_case_cause", pd.Series(dtype=object)).dropna().tolist()).most_common():
        cause_rows.append({"level": "case", "cause": cause, "count": cnt})

    rescued = 0
    total_missed_courses = 0
    if len(miss_course_df):
        missed = miss_course_df[miss_course_df["designed_rank@10"].isna()].copy()
        total_missed_courses = len(missed)
        rescued = int(missed["baseline_rank@10"].notna().sum())
    cause_rows.append(
        {
            "level": "meta",
            "cause": "missed_courses_rescued_by_baseline",
            "count": f"{rescued}/{total_missed_courses}",
        }
    )
    cause_rows.append({"level": "meta", "cause": "wrong_cases_total", "count": len(wrong)})

    cause_df = pd.DataFrame(cause_rows)

    # Sorting
    gap_df = gap_df.sort_values(["pair_id", "gap"])
    case_df = case_df.sort_values(["pair_id"])
    miss_course_df = miss_course_df.sort_values(["pair_id", "gt_course_id"])

    log("Writing Excel output...")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        case_df.to_excel(w, sheet_name="failed_cases_overview", index=False)
        gap_df.to_excel(w, sheet_name="gap_level_diagnostics", index=False)
        miss_course_df.to_excel(w, sheet_name="missed_gt_courses", index=False)
        cause_df.to_excel(w, sheet_name="cause_summary", index=False)

    log(f"WROTE {OUT_XLSX}")
    log(f"failed_cases_overview rows={len(case_df)}")
    log(f"gap_level_diagnostics rows={len(gap_df)}")
    log(f"missed_gt_courses rows={len(miss_course_df)}")
    log(f"cause_summary rows={len(cause_df)}")
    log(f"DONE total_elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
