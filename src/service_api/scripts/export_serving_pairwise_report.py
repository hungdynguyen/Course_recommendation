#!/usr/bin/env python3
"""Build a detailed pairwise Excel report for both serving benchmarks.

This report joins:
  - human-labeled ground truth
  - JD/CV scenario details
  - baseline vector-only predictions
  - designed-flow predictions

Output workbook sheets:
  - summary
  - pairwise
    - gap_analysis
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


LOGGER = logging.getLogger("service_api.export_pairwise_report")

DEFAULT_LABELS = Path("data/processed/training_dataset/human_labeled_recommendations.json")
DEFAULT_SCENARIOS = Path("data/processed/course_recommendations/course_recommendations.json")
DEFAULT_BASELINE_XLSX = Path("data/processed/course_recommendation_metrics/embedding_baseline_comparison.xlsx")
DEFAULT_DESIGNED_XLSX = Path("data/processed/course_recommendation_metrics/designed_flow_comparison.xlsx")
DEFAULT_BASELINE_JSON = Path("data/processed/course_recommendation_metrics/embedding_baseline_metrics.json")
DEFAULT_DESIGNED_JSON = Path("data/processed/course_recommendation_metrics/designed_flow_metrics.json")
DEFAULT_OUTPUT_XLSX = Path("data/processed/course_recommendation_metrics/baseline_vs_designed_pairwise.xlsx")
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
    skills = []
    skills.extend(split_skill_values(jd_info.get("technical_skills")))
    return list(dict.fromkeys(skills))


def extract_cv_skills(scenario: dict) -> List[str]:
    cv_info = scenario.get("cv_info", {}) if isinstance(scenario.get("cv_info"), dict) else {}
    skills = []
    skills.extend(split_skill_values(cv_info.get("technical_skills")))
    return list(dict.fromkeys(skills))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_list(value: object) -> List[str]:
    if pd.isna(value) or value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def normalize_skill(skill: str) -> str:
    return str(skill).strip().lower()


def format_skill_list(skills: List[str]) -> str:
    return "; ".join(skills)


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


def normalize_skill_list(value: object) -> List[str]:
    items = to_list(value)
    return sorted({normalize_skill(item) for item in items if normalize_skill(item)})


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
            row[f"{method_name}_precision"] = report["summary"][metric]["precision"]
            row[f"{method_name}_recall"] = report["summary"][metric]["recall"]
            row[f"{method_name}_hit_rate"] = report["summary"][metric]["hit_rate"]
            row[f"{method_name}_ndcg"] = report["summary"][metric]["ndcg"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_sheet() -> pd.DataFrame:
    labels = load_json(DEFAULT_LABELS)
    scenarios = load_json(DEFAULT_SCENARIOS)
    joined = join_labels_with_scenarios(labels, scenarios)

    baseline_per = load_benchmark_per_sample(DEFAULT_BASELINE_XLSX)
    designed_per = load_benchmark_per_sample(DEFAULT_DESIGNED_XLSX)

    baseline_ranked = load_benchmark_ranked(DEFAULT_BASELINE_XLSX)
    designed_ranked = load_benchmark_ranked(DEFAULT_DESIGNED_XLSX)

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
                "baseline_pred_top10": b.get("pred_top10", ""),
                "baseline_pred_top10_titles": b.get("pred_top10_titles", ""),
                "baseline_overlap_top10": b.get("overlap_top10", ""),
                "baseline_hit@1": b.get("hit@1", 0),
                "baseline_hit@3": b.get("hit@3", 0),
                "baseline_hit@5": b.get("hit@5", 0),
                "baseline_hit@10": b.get("hit@10", 0),
                "designed_pred_top10": d.get("pred_top10", ""),
                "designed_pred_top10_titles": d.get("pred_top10_titles", ""),
                "designed_overlap_top10": d.get("overlap_top10", ""),
                "designed_hit@1": d.get("hit@1", 0),
                "designed_hit@3": d.get("hit@3", 0),
                "designed_hit@5": d.get("hit@5", 0),
                "designed_hit@10": d.get("hit@10", 0),
                "winner@10": "baseline"
                if float(b.get("hit@10", 0) or 0) > float(d.get("hit@10", 0) or 0)
                else "designed"
                if float(d.get("hit@10", 0) or 0) > float(b.get("hit@10", 0) or 0)
                else "tie",
            }
        )

    pairwise_df = pd.DataFrame(rows)
    pairwise_df = pairwise_df.sort_values(by=["pair_id"])
    return pairwise_df


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
        experience_gap = skill_gaps.get("experience_gap", "")

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
                "experience_gap": experience_gap,
                "baseline_identified_gaps": b.get("identified_gaps", ""),
                "baseline_gap_match_details": b.get("gap_match_details", ""),
                "baseline_missed_gap_confusions": b.get("missed_gap_confusions", ""),
                "baseline_missed_gaps": baseline_deltas["missed_gaps"],
                "baseline_extra_gaps": baseline_deltas["extra_gaps"],
                "designed_identified_gaps": d.get("identified_gaps", ""),
                "designed_gap_match_details": d.get("gap_match_details", ""),
                "designed_missed_gap_confusions": d.get("missed_gap_confusions", ""),
                "designed_missed_gaps": designed_deltas["missed_gaps"],
                "designed_extra_gaps": designed_deltas["extra_gaps"],
            }
        )

    gap_df = pd.DataFrame(rows)
    gap_df = gap_df.sort_values(by=["pair_id"])
    return gap_df


def build_course_detail_sheet() -> pd.DataFrame:
    course_details = load_course_catalog_details(DEFAULT_COURSE_DIR)
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


def export_workbook() -> None:
    setup_logging()

    if not DEFAULT_BASELINE_XLSX.exists():
        raise FileNotFoundError(f"Missing baseline workbook: {DEFAULT_BASELINE_XLSX}")
    if not DEFAULT_DESIGNED_XLSX.exists():
        raise FileNotFoundError(f"Missing designed workbook: {DEFAULT_DESIGNED_XLSX}")

    summary_df = build_summary_sheet()
    pairwise_df = build_pairwise_sheet()
    gap_df = build_gap_analysis_sheet()
    detail_df = build_course_detail_sheet()

    mismatch_mask = gap_df.apply(
        lambda row: normalize_skill_list(row.get("baseline_identified_gaps"))
        != normalize_skill_list(row.get("designed_identified_gaps")),
        axis=1,
    )
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count > 0:
        LOGGER.warning("Found %d pair(s) with different baseline/designed identified_gaps", mismatch_count)
    else:
        LOGGER.info("Gap detection consistency check passed: baseline/designed identified_gaps are identical for all pairs")

    DEFAULT_OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DEFAULT_OUTPUT_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        pairwise_df.to_excel(writer, sheet_name="pairwise", index=False)
        gap_df.to_excel(writer, sheet_name="gap_analysis", index=False)
        detail_df.to_excel(writer, sheet_name="course_details", index=False)

    LOGGER.info("Saved detailed pairwise workbook to %s", DEFAULT_OUTPUT_XLSX)
    print(DEFAULT_OUTPUT_XLSX)


if __name__ == "__main__":
    export_workbook()