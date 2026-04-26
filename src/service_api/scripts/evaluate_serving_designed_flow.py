#!/usr/bin/env python3
"""Evaluate designed serving flow and export detailed comparison artifacts.

Designed flow:
  JD/CV skills -> GapDetection -> SkillSearch -> KG CourseRecommendation

Outputs:
  - metrics JSON
  - detailed comparison Excel for manual inspection
"""
from __future__ import annotations

import json
import logging
import math
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service_api.config import settings
from service_api.dependencies import get_gap_detection_service, get_recommendation_service, get_skill_search_service


LOGGER = logging.getLogger("service_api.evaluate_designed_flow")

DEFAULT_LABELS = Path("data/processed/training_dataset/human_labeled_recommendations.json")
DEFAULT_SCENARIOS = Path("data/processed/course_recommendations/course_recommendations.json")
DEFAULT_OUTPUT_JSON = Path("data/processed/course_recommendation_metrics/designed_flow_metrics.json")
DEFAULT_OUTPUT_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_comparison.xlsx")
DEFAULT_K_VALUES = (1, 3, 5, 10)
EMBEDDING_BATCH_SIZE = 4
PLACEHOLDER_SKILL_TOKENS = {
    "not available",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not specified",
    "unspecified",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def is_placeholder_skill(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in PLACEHOLDER_SKILL_TOKENS


def split_skill_values(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items: Iterable[object] = value
    else:
        items = str(value).replace("\n", ";").split(";")
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip().strip(" ,\t")
        if text and not is_placeholder_skill(text):
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def normalize_course_id(course_id: str) -> str:
    return str(course_id).strip().upper().replace(" ", "")


def hit_rate_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    truth_set = {normalize_course_id(cid) for cid in truth}
    if not truth_set:
        return 0.0
    topk = [normalize_course_id(cid) for cid in predicted[:k]]
    return 1.0 if any(cid in truth_set for cid in topk) else 0.0


def precision_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    truth_set = {normalize_course_id(cid) for cid in truth}
    if k <= 0:
        return 0.0
    topk = [normalize_course_id(cid) for cid in predicted[:k]]
    if not topk:
        return 0.0
    hits = sum(1 for cid in topk if cid in truth_set)
    return hits / k


def recall_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    truth_set = {normalize_course_id(cid) for cid in truth}
    if not truth_set:
        return 0.0
    topk = [normalize_course_id(cid) for cid in predicted[:k]]
    hits = sum(1 for cid in topk if cid in truth_set)
    return hits / len(truth_set)


def ndcg_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    truth_set = {normalize_course_id(cid) for cid in truth}
    if not truth_set:
        return 0.0
    dcg = 0.0
    for idx, course_id in enumerate(predicted[:k], start=1):
        if normalize_course_id(course_id) in truth_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(truth_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    truth_set = {normalize_course_id(cid) for cid in truth}
    if not truth_set:
        return 0.0
    for idx, course_id in enumerate(predicted[:k], start=1):
        if normalize_course_id(course_id) in truth_set:
            return 1.0 / idx
    return 0.0


def map_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    truth_set = {normalize_course_id(cid) for cid in truth}
    if not truth_set:
        return 0.0
    topk = [normalize_course_id(cid) for cid in predicted[:k]]
    if not topk:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for idx, cid in enumerate(topk, start=1):
        if cid in truth_set:
            hit_count += 1
            precision_sum += hit_count / idx

    denom = min(len(truth_set), k)
    return precision_sum / denom if denom > 0 else 0.0


def aggregate_metrics(per_sample: List[Dict[str, Dict[str, float]]], top_k: Sequence[int]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    n = max(len(per_sample), 1)
    for k in top_k:
        key = f"@{k}"
        summary[key] = {
            metric: round(sum(sample[key][metric] for sample in per_sample) / n, 6)
            for metric in ("precision", "recall", "hit_rate", "ndcg", "mrr", "map")
        }
    return summary


def join_labels_with_scenarios(labels: List[dict], scenarios: List[dict]) -> List[Tuple[dict, dict]]:
    by_pair = {int(s.get("pair_id")): s for s in scenarios if s.get("pair_id") is not None}
    by_jd = {str(s.get("jd_id")): s for s in scenarios if s.get("jd_id")}

    joined: List[Tuple[dict, dict]] = []
    for item in labels:
        pair_id = item.get("pair_id")
        scenario = by_pair.get(int(pair_id)) if pair_id is not None and str(pair_id).isdigit() else None
        if scenario is None and item.get("jd_id"):
            scenario = by_jd.get(str(item.get("jd_id")))
        if scenario is not None:
            joined.append((item, scenario))
    return joined


def get_truth_course_ids(label_item: dict) -> List[str]:
    selected = label_item.get("selected_courses", [])
    result = []
    for item in selected:
        if isinstance(item, dict) and item.get("course_id"):
            result.append(str(item["course_id"]))
    return list(dict.fromkeys(result))


def extract_jd_skills(scenario: dict) -> List[str]:
    jd_info = scenario.get("jd_info", {}) if isinstance(scenario.get("jd_info"), dict) else {}
    skills = []
    skills.extend(split_skill_values(jd_info.get("technical_skills")))
    return list(dict.fromkeys(skills))


def extract_cv_skills(scenario: dict) -> List[str]:
    cv_info = scenario.get("cv_info", {}) if isinstance(scenario.get("cv_info"), dict) else {}
    skills = []
    skills.extend(split_skill_values(cv_info.get("technical_skills")))
    return list(dict.fromkeys(skills))


def format_gap_match_details(match_details: List[Dict[str, object]]) -> str:
    parts: List[str] = []
    for item in match_details:
        jd_skill = str(item.get("jd_skill", "")).strip()
        best_cv_skill = str(item.get("best_cv_skill", "")).strip() or "<none>"
        score = float(item.get("best_similarity", 0.0) or 0.0)
        flag = "GAP" if bool(item.get("is_gap", False)) else "MATCH"
        if jd_skill:
            parts.append(f"{jd_skill} -> {best_cv_skill} ({score:.4f}, {flag})")
    return " | ".join(parts)


def format_missed_gap_confusions(
    groundtruth_technical_gaps: List[str],
    identified_gaps: List[str],
    match_details: List[Dict[str, object]],
) -> str:
    identified_set = {str(x).strip().lower() for x in identified_gaps if str(x).strip()}
    detail_by_jd = {
        str(item.get("jd_skill", "")).strip().lower(): item
        for item in match_details
        if str(item.get("jd_skill", "")).strip()
    }

    parts: List[str] = []
    for skill in groundtruth_technical_gaps:
        s = str(skill).strip()
        if not s:
            continue
        key = s.lower()
        if key in identified_set:
            continue
        detail = detail_by_jd.get(key)
        if detail is None:
            parts.append(f"{s} -> <no_match_detail>")
            continue
        best_cv_skill = str(detail.get("best_cv_skill", "")).strip() or "<none>"
        score = float(detail.get("best_similarity", 0.0) or 0.0)
        parts.append(f"{s} -> {best_cv_skill} ({score:.4f})")
    return " | ".join(parts)


def export_excel(
    out_path: Path,
    summary: Dict[str, Dict[str, float]],
    per_sample_rows: List[Dict[str, object]],
    ranked_rows: List[Dict[str, object]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_records = []
    for k, metrics in summary.items():
        rec = {"k": k}
        rec.update(metrics)
        summary_records.append(rec)

    if importlib.util.find_spec("openpyxl") is not None:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(summary_records).to_excel(writer, sheet_name="summary", index=False)
            pd.DataFrame(per_sample_rows).to_excel(writer, sheet_name="per_sample", index=False)
            pd.DataFrame(ranked_rows).to_excel(writer, sheet_name="ranked_predictions", index=False)
        return

    base = out_path.with_suffix("")
    pd.DataFrame(summary_records).to_csv(f"{base}_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_sample_rows).to_csv(f"{base}_per_sample.csv", index=False, encoding="utf-8")
    pd.DataFrame(ranked_rows).to_csv(f"{base}_ranked_predictions.csv", index=False, encoding="utf-8")
    LOGGER.warning("openpyxl is not installed. Exported CSV files instead of XLSX.")


def evaluate() -> None:
    setup_logging()

    labels = json.loads(DEFAULT_LABELS.read_text(encoding="utf-8"))
    scenarios = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))
    joined = join_labels_with_scenarios(labels, scenarios)

    LOGGER.info("Loaded labels=%d, scenarios=%d, matched=%d", len(labels), len(scenarios), len(joined))

    if not joined:
        raise RuntimeError("No matched samples between labels and scenarios. Check pair_id/jd_id alignment.")

    gap_svc = get_gap_detection_service()
    skill_svc = get_skill_search_service()
    rec_svc = get_recommendation_service()

    k_values = DEFAULT_K_VALUES
    max_k = max(k_values)

    per_sample_metrics: List[Dict[str, Dict[str, float]]] = []
    per_sample_rows: List[Dict[str, object]] = []
    ranked_rows: List[Dict[str, object]] = []

    for idx, (label_item, scenario) in enumerate(joined, start=1):
        jd_skills = extract_jd_skills(scenario)
        cv_skills = extract_cv_skills(scenario)
        truth = get_truth_course_ids(label_item)
        scenario_skill_gaps = scenario.get("skill_gaps", {}) if isinstance(scenario.get("skill_gaps"), dict) else {}
        groundtruth_technical_gaps = split_skill_values(scenario_skill_gaps.get("missing_technical_skills"))

        gaps, gap_match_details = gap_svc.find_gaps_with_details(jd_skills=jd_skills, cv_skills=cv_skills)
        missed_gap_confusions = format_missed_gap_confusions(
            groundtruth_technical_gaps=groundtruth_technical_gaps,
            identified_gaps=gaps,
            match_details=gap_match_details,
        )
        search_results = skill_svc.search_batch(skill_names=gaps, limit_per_skill=settings.SKILL_SEARCH_LIMIT) if gaps else {}

        canonical_ids: List[str] = []
        seen_ids = set()
        for hits in search_results.values():
            for hit in hits:
                sid = hit.get("skill_id")
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    canonical_ids.append(sid)

        rec_result = rec_svc.recommend_for_gaps(gap_skill_ids=canonical_ids, max_courses=max_k) if canonical_ids else {"recommended_courses": []}
        pred_entries = rec_result.get("recommended_courses", [])

        predicted_ids = [str(c["course_id"]) for c in pred_entries if c.get("course_id")]
        predicted_titles = [str(c.get("course_title", "")) for c in pred_entries if c.get("course_id")]

        m: Dict[str, Dict[str, float]] = {}
        for k in k_values:
            m[f"@{k}"] = {
                "precision": precision_at_k(predicted_ids, truth, k),
                "recall": recall_at_k(predicted_ids, truth, k),
                "hit_rate": hit_rate_at_k(predicted_ids, truth, k),
                "ndcg": ndcg_at_k(predicted_ids, truth, k),
                "mrr": mrr_at_k(predicted_ids, truth, k),
                "map": map_at_k(predicted_ids, truth, k),
            }
        per_sample_metrics.append(m)

        truth_set = {normalize_course_id(x) for x in truth}
        overlap_top10 = [cid for cid in predicted_ids[:10] if normalize_course_id(cid) in truth_set]

        for rank, entry in enumerate(pred_entries[:max_k], start=1):
            cid = str(entry.get("course_id", ""))
            ranked_rows.append(
                {
                    "pair_id": label_item.get("pair_id"),
                    "jd_id": label_item.get("jd_id"),
                    "rank": rank,
                    "course_id": cid,
                    "course_title": str(entry.get("course_title", "")),
                    "coverage_count": int(entry.get("coverage_count", 0) or 0),
                    "is_relevant": normalize_course_id(cid) in truth_set,
                }
            )

        per_sample_rows.append(
            {
                "pair_id": label_item.get("pair_id"),
                "jd_id": label_item.get("jd_id"),
                "jd_skill_count": len(jd_skills),
                "cv_skill_count": len(cv_skills),
                "gap_count": len(gaps),
                "identified_gaps": "; ".join(gaps),
                "groundtruth_technical_gaps": "; ".join(groundtruth_technical_gaps),
                "gap_match_details": format_gap_match_details(gap_match_details),
                "missed_gap_confusions": missed_gap_confusions,
                "canonical_skill_count": len(canonical_ids),
                "truth_count": len(truth),
                "truth_courses": "; ".join(truth),
                "pred_top10": "; ".join(predicted_ids[:10]),
                "pred_top10_titles": "; ".join(predicted_titles[:10]),
                "overlap_top10": "; ".join(overlap_top10),
                "hit@1": m["@1"]["hit_rate"],
                "hit@3": m["@3"]["hit_rate"],
                "hit@5": m["@5"]["hit_rate"],
                "hit@10": m["@10"]["hit_rate"],
            }
        )

        if idx % 10 == 0 or idx == len(joined):
            LOGGER.info("Processed %d/%d samples", idx, len(joined))

    summary = aggregate_metrics(per_sample_metrics, k_values)
    report = {
        "mode": "designed_flow",
        "labels_file": str(DEFAULT_LABELS),
        "scenario_file": str(DEFAULT_SCENARIOS),
        "total_labeled_samples": len(labels),
        "matched_samples": len(joined),
        "k_values": list(k_values),
        "summary": summary,
        "excel_output": str(DEFAULT_OUTPUT_XLSX),
    }

    DEFAULT_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    export_excel(DEFAULT_OUTPUT_XLSX, summary, per_sample_rows, ranked_rows)

    LOGGER.info("Saved metrics JSON to %s", DEFAULT_OUTPUT_JSON)
    LOGGER.info("Saved comparison Excel to %s", DEFAULT_OUTPUT_XLSX)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
