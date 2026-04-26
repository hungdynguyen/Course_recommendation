#!/usr/bin/env python3
"""Evaluate embedding-only baseline using groundtruth gaps from dataset.

Flow:
  groundtruth missing technical skills -> embedding query -> cosine search over embedded course texts

This script is isolated from existing benchmark scripts.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service_api.dependencies import get_embedding

LOGGER = logging.getLogger("service_api.evaluate_embedding_baseline_groundtruth_gaps")

DEFAULT_LABELS = Path("data/processed/training_dataset/human_labeled_recommendations.json")
DEFAULT_SCENARIOS = Path("data/processed/course_recommendations/course_recommendations.json")
DEFAULT_COURSE_DIR = Path("data/Data_Courses_Filtered")
DEFAULT_OUTPUT_JSON = Path("data/processed/course_recommendation_metrics/embedding_baseline_groundtruth_gaps_metrics.json")
DEFAULT_OUTPUT_XLSX = Path("data/processed/course_recommendation_metrics/embedding_baseline_groundtruth_gaps_comparison.xlsx")
DEFAULT_K_VALUES = (1, 3, 5, 10)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "2"))
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


def normalize_course_id(course_id: str) -> str:
    return str(course_id).strip().upper().replace(" ", "")


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


def load_course_catalog(course_dir: Path) -> List[Dict[str, str]]:
    courses: List[Dict[str, str]] = []
    for p in sorted(course_dir.rglob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        course_id = str(payload.get("course_id") or payload.get("courseCode") or p.stem)
        title = str(payload.get("title") or payload.get("courseTitle") or p.stem)
        skill_texts: List[str] = []
        for skill in payload.get("skill_outcomes", []):
            if not isinstance(skill, dict):
                continue
            name = str(skill.get("skill_name") or "").strip()
            desc = str(skill.get("outcome_description") or "").strip()
            if name:
                skill_texts.append(name)
            if desc:
                skill_texts.append(desc)
        text = f"{title}. {' '.join(skill_texts)}".strip()
        courses.append({"course_id": course_id, "title": title, "text": text})
    return courses


def get_truth_course_ids(label_item: dict) -> List[str]:
    selected = label_item.get("selected_courses", [])
    result = []
    for item in selected:
        if isinstance(item, dict) and item.get("course_id"):
            result.append(str(item["course_id"]))
    return list(dict.fromkeys(result))


def get_groundtruth_technical_gaps(scenario: dict) -> List[str]:
    skill_gaps = scenario.get("skill_gaps", {}) if isinstance(scenario.get("skill_gaps"), dict) else {}
    return split_skill_values(skill_gaps.get("missing_technical_skills"))


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

    courses = load_course_catalog(DEFAULT_COURSE_DIR)
    if not courses:
        raise RuntimeError(f"No course files found in {DEFAULT_COURSE_DIR}")

    embedding = get_embedding()

    course_texts = [c["text"] for c in courses]
    course_vecs = embedding.encode(course_texts, batch_size=EMBEDDING_BATCH_SIZE)
    course_norms = np.linalg.norm(course_vecs, axis=1, keepdims=True)
    course_vecs = course_vecs / np.where(course_norms == 0, 1.0, course_norms)

    k_values = DEFAULT_K_VALUES
    max_k = max(k_values)

    per_sample_metrics: List[Dict[str, Dict[str, float]]] = []
    per_sample_rows: List[Dict[str, object]] = []
    ranked_rows: List[Dict[str, object]] = []

    for idx, (label_item, scenario) in enumerate(joined, start=1):
        groundtruth_gaps = get_groundtruth_technical_gaps(scenario)
        query_text = ", ".join(groundtruth_gaps)

        truth = get_truth_course_ids(label_item)
        truth_set = {normalize_course_id(x) for x in truth}
        predicted_ids: List[str] = []
        predicted_titles: List[str] = []

        if query_text:
            q = embedding.encode([query_text], batch_size=1)
            q = q / np.where(np.linalg.norm(q, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(q, axis=1, keepdims=True))
            sims = (q @ course_vecs.T)[0]
            top_idx = np.argsort(-sims)[:max_k]
            for rank, i in enumerate(top_idx, start=1):
                c = courses[int(i)]
                cid = str(c["course_id"])
                predicted_ids.append(cid)
                predicted_titles.append(str(c["title"]))
                ranked_rows.append(
                    {
                        "pair_id": label_item.get("pair_id"),
                        "jd_id": label_item.get("jd_id"),
                        "rank": rank,
                        "course_id": cid,
                        "course_title": str(c["title"]),
                        "score": float(sims[int(i)]),
                        "is_relevant": normalize_course_id(cid) in truth_set,
                    }
                )

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

        overlap_top10 = [cid for cid in predicted_ids[:10] if normalize_course_id(cid) in truth_set]
        per_sample_rows.append(
            {
                "pair_id": label_item.get("pair_id"),
                "jd_id": label_item.get("jd_id"),
                "gap_source": "groundtruth_dataset",
                "gap_count": len(groundtruth_gaps),
                "identified_gaps": "; ".join(groundtruth_gaps),
                "groundtruth_technical_gaps": "; ".join(groundtruth_gaps),
                "gap_match_details": "<groundtruth_gap_source>",
                "missed_gap_confusions": "",
                "query_text": query_text,
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
        "mode": "embedding_baseline_groundtruth_gaps",
        "description": "Vector-only retrieval using groundtruth missing technical skills from dataset",
        "labels_file": str(DEFAULT_LABELS),
        "scenario_file": str(DEFAULT_SCENARIOS),
        "course_catalog_dir": str(DEFAULT_COURSE_DIR),
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
