#!/usr/bin/env python3
"""Build pairwise workbook for groundtruth-gap benchmark scripts."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

LOGGER = logging.getLogger("service_api.export_pairwise_report_groundtruth_gaps")

DEFAULT_LABELS = Path("data/processed/training_dataset/human_labeled_recommendations.json")
DEFAULT_SCENARIOS = Path("data/processed/course_recommendations/course_recommendations.json")
DEFAULT_BASELINE_XLSX = Path("data/processed/course_recommendation_metrics/embedding_baseline_groundtruth_gaps_comparison.xlsx")
DEFAULT_DESIGNED_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison.xlsx")
DEFAULT_BASELINE_JSON = Path("data/processed/course_recommendation_metrics/embedding_baseline_groundtruth_gaps_metrics.json")
DEFAULT_DESIGNED_JSON = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_metrics.json")
DEFAULT_OUTPUT_XLSX = Path("data/processed/course_recommendation_metrics/baseline_vs_designed_groundtruth_gaps_pairwise.xlsx")
DEFAULT_COURSE_DIR = Path("data/Data_Courses_Filtered")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def normalize_course_id(course_id: str) -> str:
    return str(course_id).strip().upper().replace(" ", "")


def normalize_skill(skill: str) -> str:
    return str(skill).strip().lower()


def format_skill_list(skills: List[str]) -> str:
    return "; ".join(skills)


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def normalize_course_ids(values: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in values:
        cid = normalize_course_id(item)
        if cid and cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def compute_overlap_courses(predicted: List[str], truth: List[str]) -> List[str]:
    truth_set = {normalize_course_id(x) for x in truth if str(x).strip()}
    return [cid for cid in predicted if normalize_course_id(cid) in truth_set]


def compute_extra_courses(predicted: List[str], truth: List[str]) -> List[str]:
    truth_set = {normalize_course_id(x) for x in truth if str(x).strip()}
    return [cid for cid in predicted if normalize_course_id(cid) not in truth_set]


def to_list(value: object) -> List[str]:
    if pd.isna(value) or value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def normalize_skill_list(value: object) -> List[str]:
    items = to_list(value)
    return sorted({normalize_skill(item) for item in items if normalize_skill(item)})


def compute_gap_deltas(groundtruth: List[str], identified: List[str]) -> Dict[str, str]:
    gt_ordered = list(dict.fromkeys(x for x in groundtruth if str(x).strip()))
    identified_ordered = list(dict.fromkeys(x for x in identified if str(x).strip()))

    gt_lookup = {normalize_skill(x): x for x in gt_ordered}
    identified_lookup = {normalize_skill(x): x for x in identified_ordered}

    matched_keys = set(gt_lookup) & set(identified_lookup)
    missed = [skill for skill in gt_ordered if normalize_skill(skill) not in matched_keys]
    extra = [skill for skill in identified_ordered if normalize_skill(skill) not in matched_keys]

    return {
        "missed_gaps": format_skill_list(missed),
        "extra_gaps": format_skill_list(extra),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_truth_course_ids(label_item: dict) -> List[str]:
    selected = label_item.get("selected_courses", [])
    result = []
    for item in selected:
        if isinstance(item, dict) and item.get("course_id"):
            result.append(str(item["course_id"]))
    return list(dict.fromkeys(result))


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


def extract_jd_skills(scenario: dict) -> List[str]:
    jd_info = scenario.get("jd_info", {}) if isinstance(scenario.get("jd_info"), dict) else {}
    return split_skill_values(jd_info.get("technical_skills"))


def extract_cv_skills(scenario: dict) -> List[str]:
    cv_info = scenario.get("cv_info", {}) if isinstance(scenario.get("cv_info"), dict) else {}
    return split_skill_values(cv_info.get("technical_skills"))


def load_benchmark_per_sample(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="per_sample")


def load_benchmark_ranked(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="ranked_predictions")


def load_course_catalog_details(course_dir: Path) -> Dict[str, Dict[str, object]]:
    details: Dict[str, Dict[str, object]] = {}
    for p in sorted(course_dir.rglob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        course_id = str(payload.get("course_id") or payload.get("courseCode") or p.stem)
        title = str(payload.get("title") or payload.get("courseTitle") or p.stem)
        description = str(payload.get("description") or payload.get("course_description") or "")

        skill_outcomes: List[str] = []
        for skill in payload.get("skill_outcomes", []):
            if not isinstance(skill, dict):
                continue
            name = str(skill.get("skill_name") or "").strip()
            desc = str(skill.get("outcome_description") or "").strip()
            if name:
                skill_outcomes.append(name)
            if desc:
                skill_outcomes.append(desc)

        entry_requirements: List[str] = []
        entry_section = payload.get("entry_requirements", {}) if isinstance(payload.get("entry_requirements"), dict) else {}
        for item in entry_section.get("minimum_entry_skills", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("skill_name") or "").strip()
            desc = str(item.get("description") or item.get("entry_description") or "").strip()
            if name:
                entry_requirements.append(name)
            if desc:
                entry_requirements.append(desc)

        details[normalize_course_id(course_id)] = {
            "course_id": course_id,
            "course_title": title,
            "category": str(payload.get("category") or ""),
            "source_file": str(p),
            "description": description,
            "skill_outcomes": "; ".join(dict.fromkeys(skill_outcomes)),
            "entry_requirements": "; ".join(dict.fromkeys(entry_requirements)),
        }
    return details


def build_summary_sheet() -> pd.DataFrame:
    baseline = load_json(DEFAULT_BASELINE_JSON)
    designed = load_json(DEFAULT_DESIGNED_JSON)

    rows = []
    for metric in ["@1", "@3", "@5", "@10"]:
        row = {"k": metric}
        for method_name, report in [("baseline", baseline), ("designed", designed)]:
            metric_values = report["summary"][metric]
            row[f"{method_name}_precision"] = metric_values["precision"]
            row[f"{method_name}_recall"] = metric_values["recall"]
            row[f"{method_name}_ndcg"] = metric_values["ndcg"]
            if "mrr" in metric_values:
                row[f"{method_name}_mrr"] = metric_values["mrr"]
            if "map" in metric_values:
                row[f"{method_name}_map"] = metric_values["map"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_sheet() -> pd.DataFrame:
    labels = load_json(DEFAULT_LABELS)
    scenarios = load_json(DEFAULT_SCENARIOS)
    joined = join_labels_with_scenarios(labels, scenarios)

    baseline_per = load_benchmark_per_sample(DEFAULT_BASELINE_XLSX)
    designed_per = load_benchmark_per_sample(DEFAULT_DESIGNED_XLSX)

    baseline_map = baseline_per.set_index("pair_id").to_dict(orient="index")
    designed_map = designed_per.set_index("pair_id").to_dict(orient="index")

    rows: List[Dict[str, object]] = []
    for label_item, scenario in joined:
        pair_id = label_item.get("pair_id")
        truth = get_truth_course_ids(label_item)
        jd_skills = extract_jd_skills(scenario)
        cv_skills = extract_cv_skills(scenario)

        b = baseline_map.get(pair_id, {})
        d = designed_map.get(pair_id, {})
        shared_identified_gaps = b.get("identified_gaps") or d.get("identified_gaps") or ""

        baseline_pred = to_list(b.get("pred_top10", ""))
        designed_pred = to_list(d.get("pred_top10", ""))
        baseline_overlap = compute_overlap_courses(baseline_pred, truth)
        designed_overlap = compute_overlap_courses(designed_pred, truth)
        baseline_extra = compute_extra_courses(baseline_pred, truth)
        designed_extra = compute_extra_courses(designed_pred, truth)

        rows.append(
            {
                "pair_id": pair_id,
                "jd_id": label_item.get("jd_id"),
                "jd_title": scenario.get("jd_title", ""),
                "job_description": (scenario.get("jd_info", {}) or {}).get("description", ""),
                "cv_experience": scenario.get("cv_experience", ""),
                "cv_degree": scenario.get("cv_degree", ""),
                "jd_skills": "; ".join(jd_skills),
                "cv_skills": "; ".join(cv_skills),
                "identified_gaps": shared_identified_gaps,
                "truth_courses": "; ".join(truth),
                "truth_count": len(truth),
                "baseline_pred_top10": "; ".join(baseline_pred),
                "baseline_overlap_top10": "; ".join(baseline_overlap),
                "baseline_extra_top10": "; ".join(baseline_extra),
                "baseline_extra_count": len(baseline_extra),
                "designed_pred_top10": "; ".join(designed_pred),
                "designed_overlap_top10": "; ".join(designed_overlap),
                "designed_extra_top10": "; ".join(designed_extra),
                "designed_extra_count": len(designed_extra),
            }
        )

    pairwise_df = pd.DataFrame(rows)
    return pairwise_df.sort_values(by=["pair_id"])


def build_gap_analysis_sheet() -> pd.DataFrame:
    labels = load_json(DEFAULT_LABELS)
    scenarios = load_json(DEFAULT_SCENARIOS)
    joined = join_labels_with_scenarios(labels, scenarios)

    baseline_per = load_benchmark_per_sample(DEFAULT_BASELINE_XLSX)
    designed_per = load_benchmark_per_sample(DEFAULT_DESIGNED_XLSX)

    baseline_map = baseline_per.set_index("pair_id").to_dict(orient="index")
    designed_map = designed_per.set_index("pair_id").to_dict(orient="index")

    rows: List[Dict[str, object]] = []
    for label_item, scenario in joined:
        pair_id = label_item.get("pair_id")
        skill_gaps = scenario.get("skill_gaps", {}) if isinstance(scenario.get("skill_gaps"), dict) else {}
        groundtruth_tech = split_skill_values(skill_gaps.get("missing_technical_skills"))
        groundtruth_soft = split_skill_values(skill_gaps.get("missing_soft_skills"))
        groundtruth_all = list(dict.fromkeys(groundtruth_tech + groundtruth_soft))

        b = baseline_map.get(pair_id, {})
        d = designed_map.get(pair_id, {})
        baseline_gaps = to_list(b.get("identified_gaps", ""))
        designed_gaps = to_list(d.get("identified_gaps", ""))
        baseline_deltas = compute_gap_deltas(groundtruth_tech, baseline_gaps)
        designed_deltas = compute_gap_deltas(groundtruth_tech, designed_gaps)

        rows.append(
            {
                "pair_id": pair_id,
                "jd_id": label_item.get("jd_id"),
                "groundtruth_technical_gaps": format_skill_list(groundtruth_tech),
                "groundtruth_soft_gaps": format_skill_list(groundtruth_soft),
                "groundtruth_all_gaps": format_skill_list(groundtruth_all),
                "experience_gap": skill_gaps.get("experience_gap", ""),
                "baseline_identified_gaps": b.get("identified_gaps", ""),
                "baseline_missed_gaps": baseline_deltas["missed_gaps"],
                "baseline_extra_gaps": baseline_deltas["extra_gaps"],
                "designed_identified_gaps": d.get("identified_gaps", ""),
                "designed_missed_gaps": designed_deltas["missed_gaps"],
                "designed_extra_gaps": designed_deltas["extra_gaps"],
            }
        )

    return pd.DataFrame(rows).sort_values(by=["pair_id"])


def build_course_detail_sheet(course_details: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    baseline_ranked = load_benchmark_ranked(DEFAULT_BASELINE_XLSX)
    designed_ranked = load_benchmark_ranked(DEFAULT_DESIGNED_XLSX)

    rows: List[Dict[str, object]] = []
    for method_name, ranked in [("baseline", baseline_ranked), ("designed", designed_ranked)]:
        for _, row in ranked.iterrows():
            cid = normalize_course_id(str(row.get("course_id", "")))
            info = course_details.get(cid, {})
            rows.append(
                {
                    "pair_id": row.get("pair_id"),
                    "jd_id": row.get("jd_id"),
                    "method": method_name,
                    "rank": row.get("rank"),
                    "course_id": info.get("course_id", row.get("course_id", "")),
                    "course_title": info.get("course_title", row.get("course_title", "")),
                    "category": info.get("category", ""),
                    "source_file": info.get("source_file", ""),
                    "description": info.get("description", ""),
                    "skill_outcomes": info.get("skill_outcomes", ""),
                    "entry_requirements": info.get("entry_requirements", ""),
                    "is_relevant": row.get("is_relevant", False),
                }
            )

    return pd.DataFrame(rows)


def build_recommendation_reasoning_sheet(course_details: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    baseline_per = load_benchmark_per_sample(DEFAULT_BASELINE_XLSX)
    designed_per = load_benchmark_per_sample(DEFAULT_DESIGNED_XLSX)
    baseline_ranked = load_benchmark_ranked(DEFAULT_BASELINE_XLSX)
    designed_ranked = load_benchmark_ranked(DEFAULT_DESIGNED_XLSX)

    baseline_per_map = baseline_per.set_index("pair_id").to_dict(orient="index")
    designed_per_map = designed_per.set_index("pair_id").to_dict(orient="index")

    rows: List[Dict[str, object]] = []

    for method_name, ranked_df, per_map in [
        ("baseline", baseline_ranked, baseline_per_map),
        ("designed", designed_ranked, designed_per_map),
    ]:
        for _, row in ranked_df.iterrows():
            pair_id = row.get("pair_id")
            per = per_map.get(pair_id, {})

            truth_courses = to_list(per.get("truth_courses", ""))
            truth_set = {normalize_course_id(x) for x in truth_courses}
            identified_gaps = to_list(per.get("identified_gaps", ""))

            course_id_raw = safe_text(row.get("course_id", ""))
            course_id_norm = normalize_course_id(course_id_raw)
            course_info = course_details.get(course_id_norm, {})

            matched_gap_labels = to_list(row.get("covered_gap_labels", ""))
            canonical_skill_labels = to_list(per.get("canonical_skill_labels", ""))
            unmatched_labels = [
                label for label in canonical_skill_labels if normalize_skill(label) not in {normalize_skill(x) for x in matched_gap_labels}
            ]

            if method_name == "designed":
                retrieval_logic = "gap text -> vector skill search (ES) -> canonical skill_ids"
                travel_logic = "Neo4j graph: MATCH (Course)-[:TEACHES]->(Skill in canonical ids), rank by coverage_count"
                reason_summary = f"Course covers {len(matched_gap_labels)} canonical gap skills"
            else:
                retrieval_logic = "gap text -> embedding query over course text"
                travel_logic = "Vector cosine similarity ranking"
                score = row.get("score")
                score_text = "" if pd.isna(score) else f", score={float(score):.4f}"
                reason_summary = f"Retrieved by semantic similarity{score_text}"

            rows.append(
                {
                    "pair_id": pair_id,
                    "jd_id": row.get("jd_id"),
                    "method": method_name,
                    "rank": row.get("rank"),
                    "course_id": course_info.get("course_id", course_id_raw),
                    "course_title": course_info.get("course_title", safe_text(row.get("course_title", ""))),
                    "is_relevant": bool(row.get("is_relevant", False)),
                    "groundtruth_gaps": format_skill_list(identified_gaps),
                    "canonical_skill_labels": format_skill_list(canonical_skill_labels),
                    "matched_skill_labels_for_course": format_skill_list(matched_gap_labels),
                    "unmatched_canonical_skills": format_skill_list(unmatched_labels),
                    "gap_to_canonical_top1": safe_text(per.get("gap_to_canonical_top1", "")),
                    "course_skill_outcomes": safe_text(course_info.get("skill_outcomes", "")),
                    "reason_summary": reason_summary,
                    "retrieval_logic": retrieval_logic,
                    "travel_logic": travel_logic,
                }
            )

    reasoning_df = pd.DataFrame(rows)
    if reasoning_df.empty:
        return reasoning_df
    return reasoning_df.sort_values(by=["pair_id", "method", "rank"])


def export_workbook() -> None:
    setup_logging()

    if not DEFAULT_BASELINE_XLSX.exists():
        raise FileNotFoundError(f"Missing baseline workbook: {DEFAULT_BASELINE_XLSX}")
    if not DEFAULT_DESIGNED_XLSX.exists():
        raise FileNotFoundError(f"Missing designed workbook: {DEFAULT_DESIGNED_XLSX}")

    summary_df = build_summary_sheet()
    pairwise_df = build_pairwise_sheet()
    gap_df = build_gap_analysis_sheet()
    course_details = load_course_catalog_details(DEFAULT_COURSE_DIR)
    detail_df = build_course_detail_sheet(course_details)
    reasoning_df = build_recommendation_reasoning_sheet(course_details)

    mismatch_mask = gap_df.apply(
        lambda row: normalize_skill_list(row.get("baseline_identified_gaps"))
        != normalize_skill_list(row.get("designed_identified_gaps")),
        axis=1,
    )
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count > 0:
        LOGGER.warning("Found %d pair(s) with different baseline/designed identified_gaps", mismatch_count)
    else:
        LOGGER.info("Groundtruth gap consistency check passed: baseline/designed identified_gaps are identical for all pairs")

    DEFAULT_OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DEFAULT_OUTPUT_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        pairwise_df.to_excel(writer, sheet_name="pairwise", index=False)
        gap_df.to_excel(writer, sheet_name="gap_analysis", index=False)
        detail_df.to_excel(writer, sheet_name="course_details", index=False)
        reasoning_df.to_excel(writer, sheet_name="recommendation_reasoning", index=False)

    LOGGER.info("Saved detailed pairwise workbook to %s", DEFAULT_OUTPUT_XLSX)
    print(DEFAULT_OUTPUT_XLSX)


if __name__ == "__main__":
    export_workbook()
