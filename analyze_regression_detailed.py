"""Detailed regression analysis: identify why new changes made metrics worse"""

import openpyxl
import json
from pathlib import Path

xlsx_path = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison.xlsx")
wb = openpyxl.load_workbook(xlsx_path, data_only=True)
ws = wb["per_sample"]

# Get header indices
headers = [cell.value for cell in ws[1]]
print("Headers:", headers[:20])

col_pair_id = headers.index("pair_id")
col_hit1 = headers.index("hit@1")
col_gap_search_queries = headers.index("gap_search_queries")
col_identified_gaps = headers.index("identified_gaps")
col_jd_title = headers.index("jd_title")
col_jd_keywords = headers.index("jd_keywords")  
col_canonical_count = headers.index("canonical_skill_count")
col_pred_top10 = headers.index("pred_top10")
col_truth_courses = headers.index("truth_courses")
col_gap_to_canonical = headers.index("gap_to_canonical_top1")

# Read metrics from JSON file (new run)
new_metrics_file = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_metrics.json")
with open(new_metrics_file) as f:
    new_metrics_json = json.load(f)

new_p1 = new_metrics_json["summary"]["@1"]["precision"]

# Read old metrics from CSV
old_metrics_file = Path("data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_summary.csv")
with open(old_metrics_file) as f:
    lines = f.readlines()
    old_p1_str = lines[1].split(",")[1]
    old_p1 = float(old_p1_str)

print(f"\n===== REGRESSION ANALYSIS =====")
print(f"Old P@1: {old_p1:.4f}")
print(f"New P@1: {new_p1:.4f}")
print(f"Delta: {new_p1 - old_p1:.4f} ({100*(new_p1-old_p1)/old_p1:.1f}%)\n")

# Count samples
total_samples = ws.max_row - 1
expected_old_hits = int(old_p1 * total_samples)

# Actual hits in current XLSX
actual_new_hits = 0
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=col_hit1+1, max_col=col_hit1+1):
    val = row[0].value
    if val == 1 or val == 1.0:
        actual_new_hits += 1

print(f"Expected old hits (P@1={old_p1:.4f}): {expected_old_hits}/{total_samples}")
print(f"Actual new hits: {actual_new_hits}/{total_samples}")
print(f"Hits lost: {expected_old_hits - actual_new_hits}\n")

# Analyze regressed samples
print("===== TOP 10 REGRESSED SAMPLES (hit@1=0) =====")
regressed_samples = []

for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=False), start=2):
    hit = row[col_hit1].value
    if hit == 0 or hit == 0.0:
        pair_id = row[col_pair_id].value
        gaps = row[col_identified_gaps].value
        gap_queries = row[col_gap_search_queries].value
        jd_title = row[col_jd_title].value
        jd_keywords = row[col_jd_keywords].value
        canonical_count = row[col_canonical_count].value
        pred_top10 = row[col_pred_top10].value
        truth_courses = row[col_truth_courses].value
        gap_to_canonical = row[col_gap_to_canonical].value
        
        regressed_samples.append({
            "pair_id": pair_id,
            "gaps": gaps,
            "jd_title": jd_title,
            "jd_keywords": jd_keywords,
            "gap_queries": gap_queries,
            "canonical_count": canonical_count,
            "pred_top10": str(pred_top10)[:60] if pred_top10 else "",
            "truth": truth_courses,
            "gap_to_canonical": gap_to_canonical
        })

# Display top 10
for i, sample in enumerate(regressed_samples[:10], 1):
    print(f"\n{i}. Pair ID: {sample['pair_id']}")
    print(f"   Gaps: {sample['gaps'][:70]}")
    print(f"   JD Title: {sample['jd_title']}")
    print(f"   JD Keywords: {sample['jd_keywords'][:60]}")
    print(f"   Query: {str(sample['gap_queries'])[:100]}")
    print(f"   Canonical Count: {sample['canonical_count']}")
    print(f"   Pred Top1: {sample['pred_top10']}")
    print(f"   Truth: {sample['truth']}")

print(f"\n\nTotal misses (hit@1=0): {len(regressed_samples)}/{total_samples}")

# Analyze query enrichment
print("\n===== QUERY ENRICHMENT PATTERN =====")
samples_with_role_enrichment = 0
for s in regressed_samples:
    queries_str = str(s.get("gap_queries", ""))
    if "role" in queries_str or ";" in queries_str:
        samples_with_role_enrichment += 1

print(f"Samples with enriched query (role/keywords): {samples_with_role_enrichment}/{len(regressed_samples)}")

# Check canonical count distribution
canonical_counts = [s["canonical_count"] for s in regressed_samples if s["canonical_count"]]
if canonical_counts:
    print(f"\nCanonical skill count distribution (in misses):")
    print(f"  Range: {min(canonical_counts)}-{max(canonical_counts)}")
    print(f"  Avg: {sum(canonical_counts)/len(canonical_counts):.1f}")
    print(f"  <10: {sum(1 for c in canonical_counts if c < 10)}")
    print(f"  10-24: {sum(1 for c in canonical_counts if 10 <= c <= 24)}")
    print(f"  >24: {sum(1 for c in canonical_counts if c > 24)}")

print("\n✓ Analysis complete")
