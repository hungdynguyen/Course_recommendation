"""
Analyze ESCO skill mapping quality and score distribution
Usage: python testing/analyze_esco_mapping.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
MAPPING_FILE = BASE_DIR / "data" / "processed" / "jd_cv_pairs" / "skill_to_esco_mapping.json"


def analyze_score_distribution(mapping: dict):
    """Analyze similarity score distribution"""
    all_scores = []
    for skill, matches in mapping.items():
        if matches:
            all_scores.append(matches[0]['similarity'])

    arr = np.array(all_scores)

    print(f"\n{'='*60}")
    print(f"SCORE DISTRIBUTION (best match per skill)")
    print(f"{'='*60}")
    print(f"Total skills: {len(all_scores)}")
    print(f"\n--- Stats ---")
    print(f"  Min:    {arr.min():.3f}")
    print(f"  Max:    {arr.max():.3f}")
    print(f"  Mean:   {arr.mean():.3f}")
    print(f"  Median: {np.median(arr):.3f}")
    print(f"  Std:    {arr.std():.3f}")

    print(f"\n--- Percentiles ---")
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        val = np.percentile(arr, p)
        print(f"  P{p:2d}: {val:.3f}")

    print(f"\n--- Count by score range ---")
    ranges = [
        (0.3, 0.4, "🔴 Very low"),
        (0.4, 0.5, "🟠 Low"),
        (0.5, 0.6, "🟡 Medium"),
        (0.6, 0.7, "🟢 Good"),
        (0.7, 0.8, "🟢 Very good"),
        (0.8, 0.9, "🔵 High"),
        (0.9, 1.0, "🔵 Very high"),
    ]
    for lo, hi, label in ranges:
        count = sum(1 for s in all_scores if lo <= s < hi)
        pct = count / len(all_scores) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{lo:.1f}-{hi:.1f}) {label:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Impact on different thresholds
    print(f"\n--- Impact of different SIMILARITY_THRESHOLD values ---")
    print(f"  (skills that would be EXCLUDED below threshold)")
    print(f"  {'Threshold':>12} | {'Kept':>8} | {'Excluded':>10} | {'Keep %':>8}")
    print(f"  {'-'*50}")
    for thresh in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        kept = sum(1 for s in all_scores if s >= thresh)
        excluded = len(all_scores) - kept
        pct = kept / len(all_scores) * 100
        marker = " ← current" if thresh == 0.3 else ""
        print(f"  {thresh:>12.2f} | {kept:>8d} | {excluded:>10d} | {pct:>7.1f}%{marker}")

    return arr


def show_sample_mappings(mapping: dict, threshold_lo: float, threshold_hi: float, n: int = 10):
    """Show sample mappings in a given score range"""
    in_range = {
        skill: matches for skill, matches in mapping.items()
        if matches and threshold_lo <= matches[0]['similarity'] < threshold_hi
    }

    items = list(in_range.items())
    np.random.shuffle(items)

    print(f"\n--- Sample mappings [{threshold_lo:.1f}-{threshold_hi:.1f}) ---")
    for skill, matches in items[:n]:
        best = matches[0]
        sim = best['similarity']
        label = best['label']
        correct = "✅" if _looks_correct(skill, label) else "❌"
        print(f"  {correct} '{skill}'")
        print(f"     → '{label}' ({sim:.3f})")


def _looks_correct(raw: str, esco_label: str) -> bool:
    """Heuristic: check if mapping looks correct (keyword overlap)"""
    raw_words = set(raw.lower().replace('-', ' ').replace('/', ' ').split())
    label_words = set(esco_label.lower().replace('-', ' ').replace('/', ' ').split())
    overlap = raw_words & label_words
    # Filter common stop words
    stop = {'of', 'and', 'the', 'in', 'to', 'a', 'for', 'with', 'use', 'using'}
    overlap -= stop
    return len(overlap) >= 1


def analyze_mapping_accuracy(mapping: dict):
    """Heuristic analysis of mapping accuracy by score range"""
    print(f"\n{'='*60}")
    print(f"MAPPING ACCURACY ANALYSIS (heuristic keyword overlap)")
    print(f"{'='*60}")

    ranges = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for lo, hi in ranges:
        in_range = [
            (skill, matches[0])
            for skill, matches in mapping.items()
            if matches and lo <= matches[0]['similarity'] < hi
        ]
        if not in_range:
            continue

        correct = sum(1 for skill, m in in_range if _looks_correct(skill, m['label']))
        pct = correct / len(in_range) * 100
        bar = '█' * int(pct / 5)
        print(f"  [{lo:.1f}-{hi:.1f}): {correct:4d}/{len(in_range):4d} likely correct ({pct:.1f}%) {bar}")


def find_bad_mappings(mapping: dict, n: int = 15):
    """Find likely incorrect mappings (high confidence wrong)"""
    print(f"\n{'='*60}")
    print(f"LIKELY INCORRECT MAPPINGS (score 0.5-0.7, no keyword overlap)")
    print(f"{'='*60}")

    bad = []
    for skill, matches in mapping.items():
        if not matches:
            continue
        best = matches[0]
        sim = best['similarity']
        if 0.5 <= sim < 0.7 and not _looks_correct(skill, best['label']):
            bad.append((skill, best['label'], sim))

    bad.sort(key=lambda x: x[2])

    for skill, label, sim in bad[:n]:
        print(f"  ❌ '{skill}' ({sim:.3f})")
        print(f"     → '{label}'")


def find_good_mappings(mapping: dict, n: int = 15):
    """Find clearly correct mappings"""
    print(f"\n{'='*60}")
    print(f"CLEARLY CORRECT MAPPINGS (score ≥ 0.7, keyword overlap)")
    print(f"{'='*60}")

    good = []
    for skill, matches in mapping.items():
        if not matches:
            continue
        best = matches[0]
        sim = best['similarity']
        if sim >= 0.7 and _looks_correct(skill, best['label']):
            good.append((skill, best['label'], sim))

    good.sort(key=lambda x: -x[2])

    for skill, label, sim in good[:n]:
        print(f"  ✅ '{skill}' ({sim:.3f})")
        print(f"     → '{label}'")


def main():
    print(f"Loading ESCO mapping from {MAPPING_FILE}...")
    if not MAPPING_FILE.exists():
        print(f"❌ File not found! Run map_jdcv_skills_to_esco.py first.")
        return

    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    print(f"✓ Loaded {len(mapping)} skill mappings")

    # Score distribution
    analyze_score_distribution(mapping)

    # Accuracy by range (heuristic)
    analyze_mapping_accuracy(mapping)

    # Show bad mappings
    find_bad_mappings(mapping)

    # Show good mappings
    find_good_mappings(mapping)

    # Samples by range
    print(f"\n{'='*60}")
    print(f"RANDOM SAMPLES BY SCORE RANGE")
    print(f"{'='*60}")
    show_sample_mappings(mapping, 0.3, 0.5, n=5)
    show_sample_mappings(mapping, 0.5, 0.7, n=5)
    show_sample_mappings(mapping, 0.7, 1.0, n=5)

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    print(f"  - Current threshold: 0.3 (too low, includes many wrong mappings)")
    print(f"  - Recommended threshold: 0.5-0.6 (better precision)")
    print(f"  - To update: change SIMILARITY_THRESHOLD in map_jdcv_skills_to_esco.py")
    print(f"  - Then re-run: python data/data_generation/map_jdcv_skills_to_esco.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
