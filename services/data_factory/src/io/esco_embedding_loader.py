from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_esco_embeddings(directory: Path) -> Tuple[List[dict], np.ndarray]:
    metadata_path = directory / "skills_metadata.jsonl"
    embeddings_path = directory / "skills_embeddings.npy"

    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "Missing ESCO embedding artifacts. Run the skill embedding pipeline first."
        )

    metadata: List[dict] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                metadata.append(json.loads(line))

    embeddings = np.load(embeddings_path)
    return metadata, embeddings
