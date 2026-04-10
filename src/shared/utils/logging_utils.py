"""Logging utilities."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(config_path: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure logging from YAML file or fall back to basicConfig."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
