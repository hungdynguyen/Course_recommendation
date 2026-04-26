#!/usr/bin/env python3
"""Evaluate designed serving flow using groundtruth gaps from dataset.

Flow:
  groundtruth missing technical skills -> SkillSearch -> KG CourseRecommendation

This script is isolated from existing benchmark scripts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service_api.config import settings
from service_api.dependencies import get_recommendation_service, get_skill_search_service

LOGGER = logging.getLogger("service_api.evaluate_designed_flow_groundtruth_gaps")

DEFAULT_LABELS = Path("data/processed/training_dataset/human_labeled_recommendations.json")
DEFAULT_SCENARIOS = Path("data/processed/course_recommendations/course_recommendations.json")
DEFAULT_GAP_DESCRIPTIONS = Path(
    os.getenv(
        "GAP_DESCRIPTIONS_FILE",
        "data/processed/course_recommendations/labeled_skill_gap_descriptions.json",
    )
)
DEFAULT_OUTPUT_JSON = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_metrics.json")
DEFAULT_OUTPUT_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison.xlsx")
DEFAULT_K_VALUES = (1, 3, 5, 10)
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

GAP_ALIAS_HINTS = {
    "aws": "amazon web services cloud infrastructure",
    "ec2": "amazon ec2 compute instances",
    "ecs/eks": "container orchestration on aws kubernetes",
    "ecs": "container orchestration on aws",
    "eks": "managed kubernetes service",
    "kubernetes": "container orchestration and scaling",
    "docker": "containerization and deployment",
    "bigquery": "google cloud data warehouse",
    "redshift": "amazon cloud data warehouse",
    "sql": "structured query language database querying",
    "mongodb": "nosql document database",
    "python": "python programming and scripting",
    "java": "java programming and application development",
    "gcp": "google cloud platform cloud infrastructure",
    "azure": "microsoft azure cloud infrastructure",
    "cloud": "cloud infrastructure and services",
    "api": "application programming interface integration",
    "can": "controller area network embedded communication",
    "ethernet": "ethernet networking and socket communication",
    "fpga": "field programmable gate array hardware design",
    "ipcore": "ip core hardware design",
}

DOMAIN_ALIAS_HINTS = {
    "cloud": (
        "cloud infrastructure and deployment",
        {"aws", "azure", "gcp", "ec2", "ecs", "eks", "kubernetes", "docker", "cloud", "redshift", "bigquery"},
    ),
    "data": (
        "data engineering and database systems",
        {"sql", "database", "dbms", "mongodb", "warehouse", "etl", "query", "analytics", "data"},
    ),
    "programming": (
        "software engineering and programming",
        {"python", "java", "golang", "go", "oop", "algorithm", "multithreading", "parallel", "fpga", "ethernet", "can", "ipcore"},
    ),
    "ai": (
        "artificial intelligence and machine learning",
        {"ai", "ml", "nlp", "llm", "chatbot", "chatbots", "model"},
    ),
}


def _normalize_gap_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    # Remove common deficit wording so search focuses on capability semantics.
    patterns = [
        r"^missing\s+(proficiency|knowledge|experience|ability)\s+in\s+",
        r"^lacks\s+(the\s+ability\s+to|experience\s+in|knowledge\s+of)\s+",
        r"^absence\s+of\s+proficiency\s+in\s+",
        r"^inadequate\s+experience\s+in\s+",
        r"^requires\s+(technical\s+competency\s+in|knowledge\s+of)\s+",
        r"^need\s+to\s+",
    ]
    lowered = normalized.lower()
    for p in patterns:
        lowered = re.sub(p, "", lowered, flags=re.IGNORECASE)
    return lowered.strip(" ;,.-") or normalized


def _collect_alias_hints(gap: str) -> List[str]:
    gap_text = str(gap or "").strip().lower()
    if not gap_text:
        return []

    hints: List[str] = []
    if gap_text in GAP_ALIAS_HINTS:
        hints.append(GAP_ALIAS_HINTS[gap_text])

    tokens = [t for t in re.split(r"[^a-z0-9\+\./-]+", gap_text) if t]
    for token in tokens:
        if token in GAP_ALIAS_HINTS:
            hints.append(GAP_ALIAS_HINTS[token])

    token_set = set(tokens)
    for _domain, (hint, keywords) in DOMAIN_ALIAS_HINTS.items():
        if token_set & keywords or any(k in gap_text for k in keywords):
            hints.append(hint)

    # Deduplicate while preserving order and keep query compact.
    return list(dict.fromkeys([h for h in hints if h]))[:3]


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


def format_skill_values(values: List[str]) -> str:
    return "; ".join([v for v in values if str(v).strip()])


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


def get_groundtruth_technical_gaps(scenario: dict) -> List[str]:
    skill_gaps = scenario.get("skill_gaps", {}) if isinstance(scenario.get("skill_gaps"), dict) else {}
    return split_skill_values(skill_gaps.get("missing_technical_skills"))


def extract_jd_title(scenario: dict) -> str:
    return str(scenario.get("jd_title") or "").strip()


def extract_jd_keywords(scenario: dict) -> List[str]:
    jd_info = scenario.get("jd_info", {}) if isinstance(scenario.get("jd_info"), dict) else {}
    return split_skill_values(jd_info.get("keywords"))


def build_enriched_gap_query(gap: str, jd_title: str, jd_keywords: List[str], gap_description: str = "") -> str:
    raw_gap = str(gap).strip()
    normalized_gap = _normalize_gap_text(raw_gap)
    parts = [raw_gap]
    if normalized_gap and normalized_gap.lower() != raw_gap.lower():
        parts.append(f"capability {normalized_gap}")

    for hint in _collect_alias_hints(raw_gap):
        parts.append(f"related {hint}")

    if str(gap_description).strip():
        normalized_desc = _normalize_gap_text(str(gap_description).strip())
        parts.append(f"context {normalized_desc}")
    if jd_title:
        parts.append(f"role {jd_title}")
    if jd_keywords:
        parts.append(" ".join(jd_keywords[:8]))
    return " ; ".join([p for p in parts if p])


def _normalize_skill_key(skill: str) -> str:
    return " ".join(str(skill or "").strip().lower().split())


def load_gap_descriptions(path: Path) -> Dict[str, Dict[str, str]]:
    """Load generated gap descriptions into lookup maps by pair_id / jd_id.

    Returns structure:
      {
        "by_pair_id": {"<pair_id>": {"skill_key": "desc"}},
        "by_jd_id": {"<jd_id>": {"skill_key": "desc"}},
      }
    """
    if not path.exists():
        return {"by_pair_id": {}, "by_jd_id": {}}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("Failed to parse gap descriptions file: %s", path)
        return {"by_pair_id": {}, "by_jd_id": {}}

    if not isinstance(payload, list):
        return {"by_pair_id": {}, "by_jd_id": {}}

    by_pair_id: Dict[str, Dict[str, str]] = {}
    by_jd_id: Dict[str, Dict[str, str]] = {}
    for item in payload:
        if not isinstance(item, dict) or not item.get("success"):
            continue

        skill_map: Dict[str, str] = {}
        for rec in item.get("gap_descriptions", []):
            if not isinstance(rec, dict):
                continue
            key = _normalize_skill_key(rec.get("skill", ""))
            desc = str(rec.get("description") or "").strip()
            if key and desc:
                skill_map[key] = desc

        if not skill_map:
            continue

        pair_id = item.get("pair_id")
        jd_id = item.get("jd_id")
        if pair_id is not None:
            by_pair_id[str(pair_id)] = skill_map
        if jd_id is not None and str(jd_id).strip():
            by_jd_id[str(jd_id)] = skill_map

    return {"by_pair_id": by_pair_id, "by_jd_id": by_jd_id}


def dynamic_top1_threshold(gap_name: str, base_threshold: float, short_gap_bonus: float) -> float:
    # Uniform threshold for all gaps (no short-gap penalty)
    return base_threshold


def select_canonical_candidates(
    search_results: Dict[str, List[Dict]],
    min_total: int,
    max_total: int,
    min_score: float,
    score_margin: float,
    top1_gate_enabled: bool,
    top1_min_score: float,
    top1_short_gap_bonus: float,
) -> Tuple[List[Dict], List[str]]:
    mandatory: List[Dict] = []
    optional: List[Dict] = []
    deferred: List[Dict] = []

    for _gap_name, hits in search_results.items():
        if not hits:
            continue

        top = hits[0]
        top_score = float(top.get("score") or 0.0)

        if top1_gate_enabled:
            threshold = dynamic_top1_threshold(
                gap_name=_gap_name,
                base_threshold=top1_min_score,
                short_gap_bonus=top1_short_gap_bonus,
            )
            if top_score < threshold:
                continue

        mandatory.append(top)

        for h in hits[1:]:
            score = float(h.get("score") or 0.0)
            if score >= min_score and (top_score - score) <= score_margin:
                optional.append(h)
            else:
                deferred.append(h)

    optional.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    deferred.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    selected: List[Dict] = []
    seen_ids = set()

    def add_hit(hit: Dict) -> bool:
        sid = hit.get("skill_id")
        if not sid or sid in seen_ids:
            return False
        seen_ids.add(sid)
        selected.append(hit)
        return True

    for hit in mandatory:
        add_hit(hit)
    for hit in optional:
        if len(selected) >= max_total:
            break
        add_hit(hit)

    if len(selected) < min_total:
        for hit in deferred:
            if len(selected) >= min_total or len(selected) >= max_total:
                break
            add_hit(hit)

    canonical_ids = [h["skill_id"] for h in selected if h.get("skill_id")]
    return selected, canonical_ids


def export_excel(
    out_path: Path,
    summary: Dict[str, Dict[str, float]],
    per_sample_rows: List[Dict[str, object]],
    ranked_rows: List[Dict[str, object]],
    rank_explain_rows: List[Dict[str, object]],
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
            if rank_explain_rows:
                pd.DataFrame(rank_explain_rows).to_excel(writer, sheet_name="rank_explain_failures", index=False)
        return

    base = out_path.with_suffix("")
    pd.DataFrame(summary_records).to_csv(f"{base}_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_sample_rows).to_csv(f"{base}_per_sample.csv", index=False, encoding="utf-8")
    pd.DataFrame(ranked_rows).to_csv(f"{base}_ranked_predictions.csv", index=False, encoding="utf-8")
    if rank_explain_rows:
        pd.DataFrame(rank_explain_rows).to_csv(f"{base}_rank_explain_failures.csv", index=False, encoding="utf-8")
    LOGGER.warning("openpyxl is not installed. Exported CSV files instead of XLSX.")


def resolve_outputs(suffix: str) -> Tuple[Path, Path]:
    if not suffix:
        return DEFAULT_OUTPUT_JSON, DEFAULT_OUTPUT_XLSX

    suffix = suffix.strip()
    if not suffix:
        return DEFAULT_OUTPUT_JSON, DEFAULT_OUTPUT_XLSX
    if not suffix.startswith("_"):
        suffix = f"_{suffix}"

    out_json = DEFAULT_OUTPUT_JSON.with_name(f"{DEFAULT_OUTPUT_JSON.stem}{suffix}{DEFAULT_OUTPUT_JSON.suffix}")
    out_xlsx = DEFAULT_OUTPUT_XLSX.with_name(f"{DEFAULT_OUTPUT_XLSX.stem}{suffix}{DEFAULT_OUTPUT_XLSX.suffix}")
    return out_json, out_xlsx


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _dominant_loss_component(delta_core: float, delta_coverage: float, delta_bonus: float) -> str:
    components = {
        "weighted_core": delta_core,
        "coverage_count": delta_coverage,
        "coverage_bonus": delta_bonus,
    }
    worst = min(components.items(), key=lambda x: x[1])
    return worst[0] if worst[1] < 0 else "none"


def evaluate(
    *,
    use_enrich: bool = True,
    use_weighted_ranker: bool = True,
    use_top1_gate: bool = True,
    debug_trace: bool = False,
    debug_miss_only: bool = False,
    debug_rank_explain: bool = False,
    output_suffix: str = "",
) -> None:
    setup_logging()

    labels = json.loads(DEFAULT_LABELS.read_text(encoding="utf-8"))
    scenarios = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))
    joined = join_labels_with_scenarios(labels, scenarios)
    gap_desc_lookup = load_gap_descriptions(DEFAULT_GAP_DESCRIPTIONS)

    LOGGER.info("Loaded labels=%d, scenarios=%d, matched=%d", len(labels), len(scenarios), len(joined))
    LOGGER.info(
        "Loaded gap descriptions: by_pair_id=%d, by_jd_id=%d",
        len(gap_desc_lookup.get("by_pair_id", {})),
        len(gap_desc_lookup.get("by_jd_id", {})),
    )

    if not joined:
        raise RuntimeError("No matched samples between labels and scenarios. Check pair_id/jd_id alignment.")

    skill_svc = get_skill_search_service()
    rec_svc = get_recommendation_service()
    output_json, output_xlsx = resolve_outputs(output_suffix)

    k_values = DEFAULT_K_VALUES
    max_k = max(k_values)

    per_sample_metrics: List[Dict[str, Dict[str, float]]] = []
    per_sample_rows: List[Dict[str, object]] = []
    ranked_rows: List[Dict[str, object]] = []
    rank_explain_rows: List[Dict[str, object]] = []

    for idx, (label_item, scenario) in enumerate(joined, start=1):
        groundtruth_gaps = get_groundtruth_technical_gaps(scenario)
        jd_title = extract_jd_title(scenario)
        jd_keywords = extract_jd_keywords(scenario)
        truth = get_truth_course_ids(label_item)

        pair_desc = gap_desc_lookup.get("by_pair_id", {}).get(str(label_item.get("pair_id")), {})
        if not pair_desc:
            pair_desc = gap_desc_lookup.get("by_jd_id", {}).get(str(label_item.get("jd_id")), {})

        search_results: Dict[str, List[Dict]] = {}
        query_by_gap: Dict[str, str] = {}
        desc_by_gap: Dict[str, str] = {}
        for gap in groundtruth_gaps:
            gap_desc = pair_desc.get(_normalize_skill_key(gap), "")
            desc_by_gap[gap] = gap_desc
            query_text = (
                build_enriched_gap_query(
                    gap=gap,
                    jd_title=jd_title,
                    jd_keywords=jd_keywords,
                    gap_description=gap_desc,
                )
                if use_enrich
                else gap
            )
            query_by_gap[gap] = query_text
            search_results[gap] = skill_svc.search_by_text(
                skill_name=query_text,
                limit=settings.SKILL_SEARCH_RAW_LIMIT,
            )

        bounded_min = min(settings.SKILL_CANDIDATE_MIN_TOTAL, settings.SKILL_CANDIDATE_MAX_TOTAL)
        bounded_max = max(settings.SKILL_CANDIDATE_MIN_TOTAL, settings.SKILL_CANDIDATE_MAX_TOTAL)
        target_min = min(bounded_max, max(len(groundtruth_gaps), min(max_k, bounded_min)))

        _selected_hits, canonical_ids = select_canonical_candidates(
            search_results=search_results,
            min_total=target_min,
            max_total=bounded_max,
            min_score=settings.SKILL_SEARCH_MIN_SCORE,
            score_margin=settings.SKILL_SEARCH_SCORE_MARGIN,
            top1_gate_enabled=use_top1_gate,
            top1_min_score=settings.SKILL_TOP1_MIN_SCORE,
            top1_short_gap_bonus=settings.SKILL_TOP1_SHORT_GAP_BONUS,
        )

        top1_mappings: List[str] = []
        for raw_gap in groundtruth_gaps:
            hits = search_results.get(raw_gap, [])
            if not hits:
                top1_mappings.append(f"{raw_gap} -> <no_match>")
                continue
            best = hits[0]
            sid = str(best.get("skill_id") or "")
            label = str(best.get("canonical_label") or sid)
            score = best.get("score")
            if score is None:
                top1_mappings.append(f"{raw_gap} -> {sid}:{label}")
            else:
                top1_mappings.append(f"{raw_gap} -> {sid}:{label} ({float(score):.4f})")

        skill_match_scores = {
            str(h.get("skill_id")): float(h.get("score") or 1.0)
            for h in _selected_hits
            if h.get("skill_id")
        }

        rec_result = (
            rec_svc.recommend_for_gaps(
                gap_skill_ids=canonical_ids,
                max_courses=max_k,
                skill_match_scores=skill_match_scores,
                use_weighted_ranker=use_weighted_ranker,
                debug_rank_explain=debug_rank_explain,
            )
            if canonical_ids
            else {"recommended_courses": []}
        )
        pred_entries = rec_result.get("recommended_courses", [])
        all_ranked_entries = rec_result.get("all_ranked_courses", pred_entries)
        skill_id_to_label = {
            str(item.get("skill_id")): str(item.get("label") or item.get("canonical_label") or item.get("skill_id") or "")
            for item in rec_result.get("gap_skills", [])
            if item.get("skill_id")
        }

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
            covered_gap_ids = [str(x) for x in entry.get("covered_gaps", []) if str(x).strip()]
            covered_gap_labels = [skill_id_to_label.get(x, x) for x in covered_gap_ids]
            ranked_rows.append(
                {
                    "pair_id": label_item.get("pair_id"),
                    "jd_id": label_item.get("jd_id"),
                    "rank": rank,
                    "course_id": cid,
                    "course_title": str(entry.get("course_title", "")),
                    "coverage_count": int(entry.get("coverage_count", 0) or 0),
                    "covered_gap_ids": format_skill_values(covered_gap_ids),
                    "covered_gap_labels": format_skill_values(covered_gap_labels),
                    "is_relevant": normalize_course_id(cid) in truth_set,
                }
            )

        per_sample_rows.append(
            {
                "pair_id": label_item.get("pair_id"),
                "jd_id": label_item.get("jd_id"),
                "gap_source": "groundtruth_dataset",
                "jd_title": jd_title,
                "jd_keywords": format_skill_values(jd_keywords),
                "gap_count": len(groundtruth_gaps),
                "identified_gaps": "; ".join(groundtruth_gaps),
                "groundtruth_technical_gaps": "; ".join(groundtruth_gaps),
                "gap_match_details": "<groundtruth_gap_source>",
                "missed_gap_confusions": "",
                "canonical_skill_count": len(canonical_ids),
                "canonical_skill_ids": format_skill_values(canonical_ids),
                "canonical_skill_labels": format_skill_values([skill_id_to_label.get(sid, sid) for sid in canonical_ids]),
                "gap_to_canonical_top1": format_skill_values(top1_mappings),
                "gap_descriptions_used": format_skill_values([f"{g} -> {desc_by_gap.get(g, '')}" for g in groundtruth_gaps if str(desc_by_gap.get(g, '')).strip()]),
                "gap_search_queries": format_skill_values([f"{g} -> {query_by_gap.get(g, g)}" for g in groundtruth_gaps]),
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

        # Optional explain export: capture score components for fail@10 cases.
        if debug_rank_explain and m["@10"]["hit_rate"] == 0.0:
            pair_id_dbg = label_item.get("pair_id")
            jd_id_dbg = label_item.get("jd_id")
            ranked_by_id = {
                normalize_course_id(str(c.get("course_id", ""))): c
                for c in all_ranked_entries
                if str(c.get("course_id", "")).strip()
            }

            top10_cutoff = all_ranked_entries[9] if len(all_ranked_entries) >= 10 else (all_ranked_entries[-1] if all_ranked_entries else None)
            top10_cutoff_score = _to_float(top10_cutoff.get("weighted_score"), 0.0) if top10_cutoff else 0.0
            top10_cutoff_coverage = _to_float(top10_cutoff.get("coverage_count"), 0.0) if top10_cutoff else 0.0
            top10_cutoff_core = _to_float(top10_cutoff.get("weighted_core"), 0.0) if top10_cutoff else 0.0
            top10_cutoff_bonus = _to_float(top10_cutoff.get("coverage_bonus"), 0.0) if top10_cutoff else 0.0

            for entry in all_ranked_entries:
                cid = str(entry.get("course_id", "")).strip()
                if not cid:
                    continue
                norm_cid = normalize_course_id(cid)
                is_gt = norm_cid in truth_set
                score = _to_float(entry.get("weighted_score"), 0.0)
                cov = _to_float(entry.get("coverage_count"), 0.0)
                core = _to_float(entry.get("weighted_core"), 0.0)
                bonus = _to_float(entry.get("coverage_bonus"), 0.0)
                d_score = score - top10_cutoff_score
                d_cov = cov - top10_cutoff_coverage
                d_core = core - top10_cutoff_core
                d_bonus = bonus - top10_cutoff_bonus
                rank_explain_rows.append(
                    {
                        "pair_id": pair_id_dbg,
                        "jd_id": jd_id_dbg,
                        "is_fail_at_10": True,
                        "course_id": cid,
                        "course_title": str(entry.get("course_title", "")),
                        "rank": int(entry.get("rank") or 0),
                        "is_groundtruth": is_gt,
                        "coverage_count": cov,
                        "weighted_core": core,
                        "coverage_bonus": bonus,
                        "weighted_score": score,
                        "delta_score_vs_top10_cutoff": round(d_score, 6),
                        "delta_coverage_vs_top10_cutoff": round(d_cov, 6),
                        "delta_core_vs_top10_cutoff": round(d_core, 6),
                        "delta_bonus_vs_top10_cutoff": round(d_bonus, 6),
                        "dominant_loss_component_vs_top10": _dominant_loss_component(d_core, d_cov, d_bonus) if is_gt else "",
                        "covered_gaps": format_skill_values([str(x) for x in entry.get("covered_gaps", []) if str(x).strip()]),
                        "skill_contributions_json": json.dumps(entry.get("skill_contributions", []), ensure_ascii=False),
                    }
                )

            for gt_course in truth:
                norm_gt = normalize_course_id(gt_course)
                if norm_gt in ranked_by_id:
                    continue
                rank_explain_rows.append(
                    {
                        "pair_id": pair_id_dbg,
                        "jd_id": jd_id_dbg,
                        "is_fail_at_10": True,
                        "course_id": str(gt_course),
                        "course_title": "<not_in_candidate_pool>",
                        "rank": -1,
                        "is_groundtruth": True,
                        "coverage_count": 0.0,
                        "weighted_core": 0.0,
                        "coverage_bonus": 0.0,
                        "weighted_score": 0.0,
                        "delta_score_vs_top10_cutoff": round(-top10_cutoff_score, 6),
                        "delta_coverage_vs_top10_cutoff": round(-top10_cutoff_coverage, 6),
                        "delta_core_vs_top10_cutoff": round(-top10_cutoff_core, 6),
                        "delta_bonus_vs_top10_cutoff": round(-top10_cutoff_bonus, 6),
                        "dominant_loss_component_vs_top10": "not_in_candidate_pool",
                        "covered_gaps": "",
                        "skill_contributions_json": "[]",
                    }
                )

        # Optional debug trace for root-cause analysis. This block is read-only and
        # does not alter candidate selection or ranking logic.
        should_debug = debug_trace and ((not debug_miss_only) or (m["@10"]["hit_rate"] == 0.0))
        if should_debug:
            pair_id_dbg = label_item.get("pair_id")
            jd_id_dbg = label_item.get("jd_id")
            LOGGER.info(
                "DEBUG_SAMPLE pair_id=%s jd_id=%s title=%s gaps=%d truth=%d canonical=%d pred=%d hit@10=%.1f",
                pair_id_dbg,
                jd_id_dbg,
                jd_title,
                len(groundtruth_gaps),
                len(truth),
                len(canonical_ids),
                len(predicted_ids[:10]),
                m["@10"]["hit_rate"],
            )

            for raw_gap in groundtruth_gaps:
                query_text = query_by_gap.get(raw_gap, raw_gap)
                hits = search_results.get(raw_gap, [])
                top = hits[0] if hits else None
                top_score = float(top.get("score") or 0.0) if top else None
                threshold = dynamic_top1_threshold(
                    gap_name=raw_gap,
                    base_threshold=settings.SKILL_TOP1_MIN_SCORE,
                    short_gap_bonus=settings.SKILL_TOP1_SHORT_GAP_BONUS,
                )
                gate_pass = (top_score is not None) and ((not use_top1_gate) or (top_score >= threshold))
                top_sid = str(top.get("skill_id") or "") if top else ""
                top_lbl = str(top.get("canonical_label") or top_sid) if top else "<no_match>"

                hit_parts = []
                for h in hits[:5]:
                    hs = float(h.get("score") or 0.0)
                    hsid = str(h.get("skill_id") or "")
                    hlbl = str(h.get("canonical_label") or hsid)
                    hit_parts.append(f"{hsid}:{hlbl} ({hs:.4f})")
                top5_text = " | ".join(hit_parts) if hit_parts else "<none>"

                LOGGER.info(
                    "DEBUG_GAP pair_id=%s gap=%s query=%s top1=%s:%s top1_score=%s threshold=%.4f gate_pass=%s top5=%s",
                    pair_id_dbg,
                    raw_gap,
                    query_text,
                    top_sid,
                    top_lbl,
                    "None" if top_score is None else f"{top_score:.4f}",
                    threshold,
                    gate_pass,
                    top5_text,
                )

            LOGGER.info(
                "DEBUG_CANONICAL pair_id=%s canonical_ids=%s",
                pair_id_dbg,
                format_skill_values(canonical_ids),
            )
            LOGGER.info(
                "DEBUG_TRUTH pair_id=%s truth=%s",
                pair_id_dbg,
                format_skill_values(truth),
            )
            LOGGER.info(
                "DEBUG_PRED_TOP10 pair_id=%s pred=%s overlap=%s",
                pair_id_dbg,
                format_skill_values(predicted_ids[:10]),
                format_skill_values(overlap_top10),
            )

        if idx % 10 == 0 or idx == len(joined):
            LOGGER.info("Processed %d/%d samples", idx, len(joined))

    summary = aggregate_metrics(per_sample_metrics, k_values)
    report = {
        "mode": "designed_flow_groundtruth_gaps",
        "query_enrichment": use_enrich,
        "gap_descriptions_file": str(DEFAULT_GAP_DESCRIPTIONS),
        "gap_descriptions_enabled": DEFAULT_GAP_DESCRIPTIONS.exists(),
        "weighted_ranker": use_weighted_ranker,
        "top1_confidence_gate": use_top1_gate,
        "rank_explain_failures": debug_rank_explain,
        "labels_file": str(DEFAULT_LABELS),
        "scenario_file": str(DEFAULT_SCENARIOS),
        "total_labeled_samples": len(labels),
        "matched_samples": len(joined),
        "k_values": list(k_values),
        "summary": summary,
        "excel_output": str(output_xlsx),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    export_excel(output_xlsx, summary, per_sample_rows, ranked_rows, rank_explain_rows)

    LOGGER.info("Saved metrics JSON to %s", output_json)
    LOGGER.info("Saved comparison Excel to %s", output_xlsx)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate designed flow on groundtruth gaps")
    parser.add_argument("--enable-enrich", action="store_true", help="Enable JD title/keywords query enrichment (default: disabled)")
    parser.add_argument("--disable-enrich", action="store_true", help="Force disable JD title/keywords query enrichment")
    parser.add_argument("--disable-weighted-ranker", action="store_true", help="Disable weighted course reranker")
    parser.add_argument("--disable-top1-gate", action="store_true", help="Disable top-1 confidence gate for canonical selection")
    parser.add_argument("--debug-trace", action="store_true", help="Enable per-sample debug logs for retrieval/gate/canonical/predictions")
    parser.add_argument("--debug-miss-only", action="store_true", help="When debug enabled, only log samples with hit@10=0")
    parser.add_argument("--debug-rank-explain", action="store_true", help="Export weighted-score components for fail@10 analysis")
    parser.add_argument("--output-suffix", default="", help="Suffix appended to output files (e.g. no_enrich)")
    args = parser.parse_args()

    if args.enable_enrich and args.disable_enrich:
        raise SystemExit("Conflicting flags: use either --enable-enrich or --disable-enrich, not both.")

    use_enrich = args.enable_enrich and (not args.disable_enrich)

    evaluate(
        use_enrich=use_enrich,
        use_weighted_ranker=not args.disable_weighted_ranker,
        use_top1_gate=not args.disable_top1_gate,
        debug_trace=args.debug_trace,
        debug_miss_only=args.debug_miss_only,
        debug_rank_explain=args.debug_rank_explain,
        output_suffix=args.output_suffix,
    )
