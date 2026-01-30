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
        
        # Ưu tiên dùng model_path nếu có (local trained model)
        # Nếu không có thì dùng model_name (HuggingFace model)
        if self._config.model_path:
            model_source = self._config.model_path
            cache_folder = None  # Không cần cache cho local model
            LOGGER.info(f"Loading custom trained model from: {model_source}")
        else:
            model_source = self._config.model_name
            cache_folder = Path.home() / ".cache" if self._config.provider == "sentence_transformers" else None
            LOGGER.info(f"Loading model from HuggingFace: {model_source}")
        
        self._model = SentenceTransformer(
            model_name_or_path=model_source,
            device=self._config.device,
            cache_folder=str(cache_folder) if cache_folder else None,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        model_source = self._config.model_path or self._config.model_name
        LOGGER.info("Generating embeddings using model: %s", model_source)
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
