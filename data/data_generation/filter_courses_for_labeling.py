"""
Filter Courses for Labeling Tool
=================================
Reads all extracted course JSONs from ata_Courses_Output/ and removes courses
that are not suitable for external learners:

  - Chuyên đề thực tế  (practical field study)
  - Khóa luận / Khóa luận tốt nghiệp  (thesis / graduation project)
  - Thực tập  (internship)
  - Đồ án  (capstone / project course)

Saves the filtered courses to data/Data_Courses_Filtered/, preserving the
department sub-folder structure so the labeling tool can consume it directly.
"""

import json
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_DIR  = DATA_DIR / "ata_Courses_Output"
OUTPUT_DIR = DATA_DIR / "Data_Courses_Filtered"

# ---------------------------------------------------------------------------
# Patterns to EXCLUDE (matched against the Vietnamese course name in the
# filename, case-insensitive).
# ---------------------------------------------------------------------------
EXCLUDE_PATTERNS = [
    re.compile(r"^chuyên đề thực tế", re.IGNORECASE),
    re.compile(r"^khóa luận",          re.IGNORECASE),
    re.compile(r"thực tập",            re.IGNORECASE),
    re.compile(r"^đồ án",              re.IGNORECASE),
]


def viet_name_from_filename(filename: str) -> str:
    """Extract the Vietnamese course name from a filename like 'Course name_COURSEID.json'."""
    stem = Path(filename).stem          # strip .json
    # Remove trailing _COURSEID (last underscore + uppercase+digits segment)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^[A-ZÀÁẠẶ]{2,}[0-9]{4,}", parts[1]):
        return parts[0].strip()
    return stem.strip()


def should_exclude(filename: str) -> bool:
    viet_name = viet_name_from_filename(filename)
    return any(p.search(viet_name) for p in EXCLUDE_PATTERNS)


def run():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return

    # Collect all JSON files (ignore macOS Zone.Identifier sidecars)
    all_files = sorted(
        p for p in INPUT_DIR.rglob("*.json")
        if not p.name.endswith(".Zone.Identifier")
    )

    total     = len(all_files)
    kept      = 0
    excluded  = 0
    skipped   = 0

    excluded_names: list[str] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for src in all_files:
        # Preserve department sub-folder
        rel       = src.relative_to(INPUT_DIR)   # e.g. Khoa CNTT/Course_ID.json
        dept_dir  = rel.parent
        filename  = src.name

        if should_exclude(filename):
            excluded += 1
            excluded_names.append(str(rel))
            continue

        # Validate JSON before copying
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [SKIP] Invalid JSON in {rel}: {e}")
            skipped += 1
            continue

        dest_dir = OUTPUT_DIR / dept_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / filename)
        kept += 1

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("Filter Courses — Summary")
    print("=" * 60)
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print()
    print(f"  Total scanned : {total}")
    print(f"  Kept          : {kept}")
    print(f"  Excluded      : {excluded}")
    if skipped:
        print(f"  Skipped (bad JSON): {skipped}")
    print()

    if excluded_names:
        print("Excluded courses:")
        for name in sorted(excluded_names):
            print(f"  - {name}")

    print()
    print(f"[DONE] Filtered courses saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
