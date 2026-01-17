from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from ..settings import EmbeddingConfig

LOGGER = logging.getLogger("data_factory.pipeline")

class EmbeddingService:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        cache_folder = Path.home() / ".cache" if self._config.provider == "sentence_transformers" else None
        self._model = SentenceTransformer(
            model_name_or_path=self._config.model_name,
            device=self._config.device,
            cache_folder=str(cache_folder) if cache_folder else None,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        LOGGER.info("Generating embeddings using model: %s", self._config.model_name)
        sentences: List[str] = list(texts)
        
        embeddings = self._model.encode(
            sentences,
            batch_size=self._config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self._config.normalize,
        )
        return np.array(embeddings)

    @property
    def embedding_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
