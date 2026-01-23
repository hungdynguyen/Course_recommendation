from __future__ import annotations

import logging.config
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logging(config_path: Path | str) -> None:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Logging config not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as stream:
        config_dict: Dict[str, Any] = yaml.safe_load(stream)

    log_dir = config_dict.get("handlers", {}).get("file", {}).get("filename")
    if log_dir:
        Path(log_dir).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config_dict)
