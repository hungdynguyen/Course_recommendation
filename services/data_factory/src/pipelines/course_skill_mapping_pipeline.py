from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..embeddings.embedding_service import EmbeddingService
from ..io.course_skill_loader import load_course_skills
from ..io.esco_embedding_loader import load_esco_embeddings
from ..models.course_skill import CourseSkill, MappedCourseSkill
from ..services.mysql_course_skill_service import MySQLCourseSkillService
from ..services.reranker_service import RerankerService
from ..settings import Settings

LOGGER = logging.getLogger("data_factory.course_skill_pipeline")


class CourseSkillMappingPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        mysql_service: MySQLCourseSkillService,
        reranker_service: Optional[RerankerService] = None,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._mysql_service = mysql_service
        self._reranker = reranker_service

    def run(self) -> None:
        """Legacy run method - runs all phases together (not memory optimized)"""
        LOGGER.info("Starting course skill mapping pipeline")

        esco_metadata, esco_embeddings = load_esco_embeddings(
            self._settings.paths.processed_embeddings_dir
        )
        LOGGER.info("Loaded %s ESCO skills", len(esco_metadata))

        course_skills = load_course_skills(self._settings.paths.course_catalog_dir)
        if not course_skills:
            LOGGER.warning("No course skills found; terminating pipeline")
            return

        payloads = [item.to_embedding_payload() for item in course_skills]
        course_embeddings = self._embedding_service.encode(payloads)

        mapped_results = self._map_skills(
            course_skills,
            course_embeddings,
            esco_metadata,
            esco_embeddings,
        )

        records = [item.as_record() for item in mapped_results]
        self._write_to_disk(records)
        self._mysql_service.initialize()
        self._mysql_service.insert_records(records)
        LOGGER.info("Finished course skill mapping pipeline")

    def run_embedding_phase(self) -> Tuple[np.ndarray, List[CourseSkill], List[dict], np.ndarray]:
        """Phase 1: Load data and generate embeddings"""
        LOGGER.info("Starting embedding phase")

        esco_metadata, esco_embeddings = load_esco_embeddings(
            self._settings.paths.processed_embeddings_dir
        )
        LOGGER.info("Loaded %s ESCO skills", len(esco_metadata))

        course_skills = load_course_skills(self._settings.paths.course_catalog_dir)
        if not course_skills:
            LOGGER.warning("No course skills found")
            return np.array([]), [], [], np.array([])

        payloads = [item.to_embedding_payload() for item in course_skills]
        course_embeddings = self._embedding_service.encode(payloads)
        LOGGER.info("Embedding phase completed")

        return course_embeddings, course_skills, esco_metadata, esco_embeddings

    def run_mapping_phase(
        self,
        course_skills: List[CourseSkill],
        course_embeddings: np.ndarray,
        esco_metadata: List[dict],
        esco_embeddings: np.ndarray,
    ) -> None:
        """Phase 2: Map skills using embeddings and optional reranking"""
        LOGGER.info("Starting mapping phase")

        if not course_skills:
            LOGGER.warning("No course skills to map")
            return

        mapped_results = self._map_skills(
            course_skills,
            course_embeddings,
            esco_metadata,
            esco_embeddings,
        )

        records = [item.as_record() for item in mapped_results]
        self._write_to_disk(records)
        self._mysql_service.initialize()
        self._mysql_service.insert_records(records)
        LOGGER.info("Finished course skill mapping pipeline")

    def _map_skills(
        self,
        course_skills: List[CourseSkill],
        course_embeddings: np.ndarray,
        esco_metadata: List[dict],
        esco_embeddings: np.ndarray,
    ) -> List[MappedCourseSkill]:
        LOGGER.info("Computing similarity matrix for %d course skills...", len(course_skills))
        normalized_course = _normalize_vectors(course_embeddings)
        normalized_esco = _normalize_vectors(esco_embeddings)

        similarity_matrix = normalized_course @ normalized_esco.T
        LOGGER.info("Similarity matrix computed, shape: %s", similarity_matrix.shape)
        
        results: List[MappedCourseSkill] = []
        top_k = max(1, self._settings.mapping.rerank_top_k)

        # Add progress bar for mapping loop
        LOGGER.info("Starting skill mapping with reranking=%s", self._reranker is not None)
        for idx, skill in enumerate(tqdm(course_skills, desc="Mapping skills", unit="skill")):
            similarities = similarity_matrix[idx]
            if similarities.size == 0:
                results.append(MappedCourseSkill(skill, None, None, None, None))
                continue

            candidate_indices = _top_k_indices(similarities, top_k)
            candidates = []
            for candidate_idx in candidate_indices:
                esco_skill = esco_metadata[candidate_idx]
                candidates.append(
                    {
                        "index": int(candidate_idx),
                        "similarity": float(similarities[candidate_idx]),
                        "text": _compose_esco_text(esco_skill),
                        "metadata": esco_skill,
                    }
                )

            reranked_candidates = None
            if self._reranker and candidates:
                reranked_candidates = self._reranker.rerank(
                    query=skill.to_embedding_payload(),
                    candidates=candidates,
                )

            chosen = (reranked_candidates or candidates)[0]
            best_idx = chosen["index"]
            best_score = float(chosen["similarity"])
            threshold = self._settings.mapping.min_similarity
            if best_score < threshold:
                results.append(MappedCourseSkill(skill, None, None, None, best_score))
                continue

            esco_skill = esco_metadata[best_idx]
            results.append(
                MappedCourseSkill(
                    original=skill,
                    esco_skill_id=esco_skill.get("skill_id"),
                    esco_preferred_label=esco_skill.get("preferred_label"),
                    esco_description=esco_skill.get("description"),
                    similarity_score=best_score,
                )
            )
        return results

    def _write_to_disk(self, records: Iterable[dict]) -> None:
        output_dir = self._settings.paths.course_skill_mappings_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_file = Path(output_dir) / "course_skill_mappings.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("Persisted course skill mappings to %s", output_file)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=int)
    k = min(k, scores.shape[0])
    if k == 1:
        return np.array([int(np.argmax(scores))])
    partition = np.argpartition(-scores, k - 1)[:k]
    return partition[np.argsort(-scores[partition])]


def _compose_esco_text(esco_skill: dict) -> str:
    parts = [esco_skill.get("preferred_label") or "", esco_skill.get("description") or ""]
    alt_labels = esco_skill.get("alternative_labels")
    if isinstance(alt_labels, list) and alt_labels:
        parts.append("; ".join(alt_labels))
    return ". ".join(part for part in parts if part)
