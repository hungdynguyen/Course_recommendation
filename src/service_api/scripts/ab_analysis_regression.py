"""
A/B Analysis: Isolate impact of enrich query vs bounded selection

Runs 4 variants:
1. Baseline: search_batch(gaps), flatten all, no bounded
2. Enrich Only: search_by_text(enrich), flatten all, no bounded
3. Bounded Only: search_batch(gaps), apply bounded selection
4. Both: search_by_text(enrich), apply bounded selection
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.skill_search import SkillSearchService
from services.course_recommendation import CourseRecommendationService
from db.neo4j_client import Neo4jClient


def extract_jd_title(scenario):
    """Extract JD title from scenario"""
    jd_info = scenario.get("jd_info", {})
    if isinstance(jd_info, dict):
        return jd_info.get("title", "") or ""
    return ""

def extract_jd_keywords(scenario):
    """Extract JD keywords from scenario"""
    jd_info = scenario.get("jd_info", {})
    if isinstance(jd_info, dict):
        keywords = jd_info.get("keywords", [])
        if isinstance(keywords, str):
            return [k.strip() for k in keywords.split(";") if k.strip()]
        elif isinstance(keywords, list):
            return keywords
    return []

def _normalize_keywords(keywords: List[str]) -> List[str]:
    """Normalize and deduplicate keywords"""
    normalized = set()
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if kw_clean:
            normalized.add(kw_clean)
    return sorted(list(normalized))[:8]

def build_enriched_gap_query(gap: str, jd_title: str, jd_keywords: List[str]) -> str:
    """Build enriched gap query"""
    parts = [gap]
    if jd_title:
        parts.append(f"role {jd_title}")
    keywords_norm = _normalize_keywords(jd_keywords)
    if keywords_norm:
        parts.append(" ".join(keywords_norm))
    return " ; ".join(parts)

def select_canonical_candidates_bounded(results, min_total=10, max_total=24, min_score=0.80, margin=0.08):
    """Bounded candidate selection"""
    if not results:
        return [], []
    
    selected, deferred = [], []
    
    # Keep top-1 mandatory
    if results:
        selected.append(results[0])
    
    # Add optional close to top-1
    if selected and len(selected) < min_total:
        top_score = selected[0]["score"]
        min_optional = top_score - margin
        for r in results[1:]:
            if r["score"] >= min_optional and len(selected) < max_total:
                selected.append(r)
            elif r["score"] < min_optional:
                deferred.append(r)
    
    # Backfill deferred if too short
    if len(selected) < min_total and deferred:
        selected.extend(deferred[:min_total - len(selected)])
    
    # Cap at max
    selected = selected[:max_total]
    
    return selected, deferred


async def run_variant(variant_name: str, gaps: List[str], jd_title: str, jd_keywords: List[str],
                     use_enrich: bool, use_bounded: bool, 
                     skill_service: SkillSearchService) -> Tuple[List[str], Dict]:
    """Run single variant and return canonical skills + trace"""
    
    canonical_ids = []
    trace = {
        "variant": variant_name,
        "use_enrich": use_enrich,
        "use_bounded": use_bounded,
        "gap_queries": [],
        "canonical_counts": []
    }
    
    for gap in gaps:
        if use_enrich:
            # Enrich query mode
            query = build_enriched_gap_query(gap, jd_title, jd_keywords)
            trace["gap_queries"].append(query)
            
            results = await skill_service.search_by_text(query, top_k=5)
        else:
            # Batch mode (original)
            trace["gap_queries"].append(gap)
            results = await skill_service.search_by_text(gap, top_k=5)
        
        if use_bounded:
            selected, _ = select_canonical_candidates_bounded(results)
            results = selected
        
        canonical_ids.extend([r["skill_id"] for r in results])
        trace["canonical_counts"].append(len(results))
    
    trace["total_canonical"] = len(canonical_ids)
    canonical_ids = list(set(canonical_ids))  # Deduplicate
    trace["unique_canonical"] = len(canonical_ids)
    
    return canonical_ids, trace


async def evaluate_variant(variant_name: str, gaps: List[str], groundtruth_courses: List[str],
                          jd_title: str, jd_keywords: List[str],
                          use_enrich: bool, use_bounded: bool,
                          skill_service: SkillSearchService, 
                          course_service: CourseRecommendationService) -> Dict:
    """Evaluate single variant"""
    
    canonical_ids, trace = await run_variant(
        variant_name, gaps, jd_title, jd_keywords,
        use_enrich, use_bounded, skill_service
    )
    
    if not canonical_ids:
        return {"hit": 0, "offset": None, "trace": trace}
    
    # Get courses for canonical skills
    pred_courses = await course_service.get_courses_by_skills(canonical_ids, top_k=10)
    pred_ids = [c["course_id"] for c in pred_courses]
    
    # Compute hit@1
    hit = 1 if pred_ids and pred_ids[0] in groundtruth_courses else 0
    offset = pred_ids.index(groundtruth_courses[0]) if groundtruth_courses and groundtruth_courses[0] in pred_ids else None
    
    return {
        "hit": hit,
        "offset": offset,
        "pred_top1": pred_ids[0] if pred_ids else None,
        "pred_top3": pred_ids[:3] if pred_ids else [],
        "trace": trace
    }


def main():
    """Run A/B analysis on 68 benchmark samples"""
    
    # Load benchmark data
    labels_file = Path("/root/courses_rec/data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_labels.json")
    with open(labels_file) as f:
        labels = json.load(f)
    
    scenarios_file = Path("/root/courses_rec/data/raw/scenarios.json")
    with open(scenarios_file) as f:
        scenarios = json.load(f)
    
    # Initialize services
    skill_service = SkillSearchService()
    course_service = CourseRecommendationService()
    neo4j = Neo4jClient()
    
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(labels),
        "variants": {}
    }
    
    per_sample_results = []
    
    print(f"Running A/B analysis on {len(labels)} samples...")
    print("Variants: baseline, enrich_only, bounded_only, both")
    
    for idx, label in enumerate(labels):
        pair_id = label["pair_id"]
        groundtruth_courses = label["groundtruth_technical_gaps"]
        
        scenario = scenarios[pair_id]
        jd_id = scenario["jd_id"]
        cv_id = scenario["cv_id"]
        gaps = label["identified_gaps"].split("; ")
        jd_title = extract_jd_title(scenario)
        jd_keywords = extract_jd_keywords(scenario)
        
        # Run 4 variants
        import asyncio
        
        baseline_result = asyncio.run(evaluate_variant(
            "baseline", gaps, groundtruth_courses, jd_title, jd_keywords,
            use_enrich=False, use_bounded=False, 
            skill_service=skill_service, course_service=course_service
        ))
        
        enrich_result = asyncio.run(evaluate_variant(
            "enrich_only", gaps, groundtruth_courses, jd_title, jd_keywords,
            use_enrich=True, use_bounded=False,
            skill_service=skill_service, course_service=course_service
        ))
        
        bounded_result = asyncio.run(evaluate_variant(
            "bounded_only", gaps, groundtruth_courses, jd_title, jd_keywords,
            use_enrich=False, use_bounded=True,
            skill_service=skill_service, course_service=course_service
        ))
        
        both_result = asyncio.run(evaluate_variant(
            "both", gaps, groundtruth_courses, jd_title, jd_keywords,
            use_enrich=True, use_bounded=True,
            skill_service=skill_service, course_service=course_service
        ))
        
        per_sample_results.append({
            "pair_id": pair_id,
            "jd_id": jd_id,
            "cv_id": cv_id,
            "gap_count": len(gaps),
            "groundtruth_course": groundtruth_courses[0] if groundtruth_courses else None,
            "baseline": baseline_result,
            "enrich_only": enrich_result,
            "bounded_only": bounded_result,
            "both": both_result
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(labels)} samples")
    
    # Compute aggregate metrics
    for variant_key in ["baseline", "enrich_only", "bounded_only", "both"]:
        hits = sum(1 for s in per_sample_results if s[variant_key]["hit"] == 1)
        results_summary["variants"][variant_key] = {
            "hit@1": hits / len(per_sample_results),
            "hit_count": hits
        }
    
    # Save results
    output_dir = Path("/root/courses_rec/data/processed/ab_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ab_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    with open(output_dir / "ab_per_sample.json", "w") as f:
        json.dump(per_sample_results, f, indent=2)
    
    print("\n=== A/B Analysis Summary ===")
    for variant, metrics in results_summary["variants"].items():
        print(f"{variant}: hit@1 = {metrics['hit@1']:.4f} ({metrics['hit_count']}/{len(per_sample_results)})")
    
    print(f"\nDetailed results saved to: {output_dir / 'ab_summary.json'}")
    print(f"Per-sample results saved to: {output_dir / 'ab_per_sample.json'}")
    
    neo4j.close()


if __name__ == "__main__":
    main()
