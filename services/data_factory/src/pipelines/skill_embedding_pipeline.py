from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np

from ..embeddings.embedding_service import EmbeddingService
from ..io.skill_loader import load_skills
from ..models.skill import Skill
from ..services.elasticsearch_service import ElasticsearchService
from ..services.mysql_service import MySQLService
from ..settings import Settings

LOGGER = logging.getLogger("data_factory.pipeline")


class SkillEmbeddingPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        es_service: ElasticsearchService,
        mysql_service: MySQLService,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._es_service = es_service
        self._mysql_service = mysql_service

    def run(self) -> None:
        LOGGER.info("Starting skill embedding pipeline")
        skills = self._load_and_filter_skills()
        LOGGER.info("Loaded %s skills", len(skills))

        payloads = [self._compose_text(skill) for skill in skills]  
        embeddings = self._embedding_service.encode(payloads)
        self._validate_embedding_dim(embeddings)

        self._persist_embeddings(skills, embeddings)
        self._mysql_service.initialize()
        self._mysql_service.upsert_skills([skill.as_index_document() for skill in skills])
        self._es_service.ensure_index()
        documents = self._build_documents(skills, embeddings)
        self._es_service.bulk_index(documents)
        LOGGER.info("Pipeline completed successfully")

    def _load_and_filter_skills(self) -> List[Skill]:
        skills = load_skills(
            skills_path=self._settings.paths.esco_skills,
            relations_path=self._settings.paths.esco_skill_relations,
        )
        min_length = self._settings.pipeline.min_description_length
        filtered = [skill for skill in skills if self._is_valid_skill(skill, min_length)]

        max_records = self._settings.pipeline.max_records
        if max_records:
            filtered = filtered[:max_records]
        return filtered

    def _persist_embeddings(self, skills: List[Skill], embeddings: np.ndarray) -> None:
        output_dir = self._settings.paths.processed_embeddings_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        metadata_path = Path(output_dir) / "skills_metadata.jsonl"
        embeddings_path = Path(output_dir) / "skills_embeddings.npy"

        with metadata_path.open("w", encoding="utf-8") as handle:
            for skill in skills:
                handle.write(json.dumps(skill.as_index_document(), ensure_ascii=False) + "\n")

        np.save(embeddings_path, embeddings)
        LOGGER.info("Persisted embeddings to %s", embeddings_path)

    def _build_documents(self, skills: Iterable[Skill], embeddings: np.ndarray) -> Iterable[dict]:
        for skill, vector in zip(skills, embeddings):
            yield {
                "skill_id": skill.skill_id,
                "vector": vector.tolist(),
            }

    def _compose_text(self, skill: Skill) -> str:
        fragments = [skill.preferred_label]
        if skill.description:
            fragments.append(skill.description)
        if skill.alternative_labels:
            fragments.append("; ".join(skill.alternative_labels))
        return ". ".join(fragment for fragment in fragments if fragment)

    def _validate_embedding_dim(self, embeddings: np.ndarray) -> None:
        expected = self._settings.elasticsearch.vector_dim
        if embeddings.shape[1] != expected:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match configured dimension {expected}"
            )

    @staticmethod
    def _is_valid_skill(skill: Skill, min_description_length: int) -> bool:
        if not skill.description:
            return False
        return len(skill.description.strip()) >= min_description_length
