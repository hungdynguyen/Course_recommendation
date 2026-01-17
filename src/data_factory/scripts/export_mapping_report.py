from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_factory.src.settings import Settings, get_settings

LOGGER = logging.getLogger("data_factory.mapping_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export course-skill mappings to an Excel file for manual review",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to settings YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output Excel file (default: <course_skill_mappings_dir>/mapping_review.xlsx)",
    )
    return parser.parse_args()


def load_settings(config_path: str | None) -> Settings:
    if config_path:
        return Settings.load(Path(config_path))
    return get_settings()


def read_mappings(mapping_path: Path) -> pd.DataFrame:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    df = pd.read_json(mapping_path, lines=True)
    expected_columns = [
        "course_id",
        "course_title",
        "skill_name",
        "skill_type",
        "description",
        "category",
        "proficiency_level",
        "bloom_taxonomy_level",
        "source_file",
        "esco_skill_id",
        "esco_preferred_label",
        "esco_description",
        "similarity_score",
    ]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in mapping file: {missing}")
    # Order columns to keep review consistent.
    df = df[expected_columns]
    # Sort by similarity descending for quicker inspection.
    df = df.sort_values(by="similarity_score", ascending=False)
    return df


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    mapping_dir = Path(settings.paths.course_skill_mappings_dir)
    mapping_file = mapping_dir / "course_skill_mappings.jsonl"
    output_file = Path(args.output) if args.output else mapping_dir / "mapping_review.xlsx"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading mappings from %s", mapping_file)
    df = read_mappings(mapping_file)

    LOGGER.info("Writing review Excel to %s", output_file)
    df.to_excel(output_file, index=False)
    LOGGER.info("Done. Rows exported: %s", len(df))


if __name__ == "__main__":
    main()
