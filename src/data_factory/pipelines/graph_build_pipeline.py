"""Graph Build Pipeline (v2 – không dùng ESCO).

Luồng:
  1. Load course skills từ file JSON (trích xuất bằng LLM).
  2. Embed tất cả skill labels bằng EmbeddingService.
  3. Dedup (2-pass):
     - Pass 1: Fuzzy matching (Jaro-Winkler >= 0.90)
     - Pass 2: Embedding similarity (cosine >= 0.92)
  4. Index canonical skills vào Elasticsearch (serving-time vector search).
  5. Build Neo4j KG: Skill nodes + Course nodes + TEACHES/REQUIRES edges.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from shared.db.es_client import ElasticsearchIndexClient
from shared.db.neo4j_client import Neo4jBatchClient
from shared.embeddings.embedding_service import EmbeddingService
from shared.models.course import CourseNode, CourseSkill, RequiresEdge, TeachesEdge
from shared.models.skill import CanonicalSkill
from data_factory.io.course_skill_loader import load_course_skills
from data_factory.services.skill_dedup_service import SkillDedupService
from data_factory.settings import Settings

logger = logging.getLogger("data_factory.graph_build_pipeline")


class GraphBuildPipeline:

    def __init__(
        self,
        settings: Settings,
        neo4j: Neo4jBatchClient,
        es: ElasticsearchIndexClient,
        embedding: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._neo4j = neo4j
        self._es = es
        self._embedding = embedding
        self._dedup = SkillDedupService(settings.deduplication)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        clear_existing: bool = False,
        output_metrics_report: bool = True,
        metrics_output_path: str = "build_graph_metrics_report.json",
    ) -> None:
        logger.info("=== Graph Build Pipeline bắt đầu ===")
        started_at = time.perf_counter()
        phase_durations: Dict[str, float] = {}
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self._settings.environment,
            "configuration": {
                "course_catalog_dir": str(self._settings.paths.course_catalog_dir),
                "es_index": self._settings.elasticsearch.index,
                "embedding_model": self._settings.embedding.model_name,
                "embedding_device": self._settings.embedding.device,
                "fuzzy_threshold": self._settings.deduplication.fuzzy_threshold,
                "embedding_threshold": self._settings.deduplication.embedding_threshold,
            },
            "phases": {},
        }

        # Phase 1 – Load
        t0 = time.perf_counter()
        course_skills = load_course_skills(self._settings.paths.course_catalog_dir)
        if not course_skills:
            logger.warning("Không tìm thấy course skills – dừng pipeline")
            report["phases"]["load"] = {
                "duration_sec": round(time.perf_counter() - t0, 4),
                "raw_skill_records": 0,
                "message": "No course skills found",
            }
            if output_metrics_report:
                self._write_metrics_report(report, metrics_output_path)
            return
        logger.info("Loaded %d course skill records", len(course_skills))
        phase_durations["load"] = time.perf_counter() - t0

        input_metrics = self._compute_input_metrics(course_skills)
        report["phases"]["load"] = {
            "duration_sec": round(phase_durations["load"], 4),
            **input_metrics,
        }

        # Phase 1.5 – Optional soft-skill filter before embedding/indexing.
        t0 = time.perf_counter()
        if self._settings.soft_skill_filter.enabled:
            filtered_course_skills, removed_soft = self._filter_soft_skills(course_skills)
            logger.info(
                "Soft-skill filter enabled: removed=%d kept=%d",
                removed_soft,
                len(filtered_course_skills),
            )
        else:
            filtered_course_skills = course_skills
            removed_soft = 0

        phase_durations["soft_skill_filter"] = time.perf_counter() - t0
        report["phases"]["soft_skill_filter"] = {
            "duration_sec": round(phase_durations["soft_skill_filter"], 4),
            "enabled": self._settings.soft_skill_filter.enabled,
            "removed_records": removed_soft,
            "remaining_records": len(filtered_course_skills),
        }

        if not filtered_course_skills:
            logger.warning("All course skills removed by soft-skill filter – dừng pipeline")
            report["summary"] = {
                "total_duration_sec": round(time.perf_counter() - started_at, 4),
                "status": "stopped_empty_after_soft_filter",
            }
            if output_metrics_report:
                self._write_metrics_report(report, metrics_output_path)
            return

        # Phase 2 – Embed course skill payloads (name + description + category)
        t0 = time.perf_counter()
        labels     = [cs.skill_name for cs in filtered_course_skills]
        descs      = [cs.description for cs in filtered_course_skills]
        cats       = [cs.category for cs in filtered_course_skills]
        embedding_payloads = [cs.to_embedding_payload() for cs in filtered_course_skills]
        embeddings = self._embedding.encode(embedding_payloads, batch_size=self._settings.embedding.batch_size)
        phase_durations["embedding"] = time.perf_counter() - t0
        report["phases"]["embedding"] = {
            "duration_sec": round(phase_durations["embedding"], 4),
            "skills_encoded": len(labels),
            "vector_dim": int(embeddings.shape[1]) if embeddings.size else 0,
            "batch_size": self._settings.embedding.batch_size,
            "normalize": self._settings.embedding.normalize,
        }

        # Phase 3 – Dedup (2-pass: fuzzy + embedding)
        t0 = time.perf_counter()
        canonical_skills, merge_map = self._dedup.deduplicate(
            labels=labels,
            embeddings=embeddings,
            descriptions=descs,
            categories=cats,
        )
        logger.info("Canonical skills sau dedup: %d", len(canonical_skills))
        phase_durations["dedup"] = time.perf_counter() - t0
        dedup_metrics = self._dedup.get_metrics()
        dedup_details = self._dedup.get_merge_report()
        report["phases"]["dedup"] = {
            "duration_sec": round(phase_durations["dedup"], 4),
            **dedup_metrics,
            "merge_details": dedup_details,
        }

        # Phase 4 – Index vào Elasticsearch
        logger.info("Indexing %d canonical skills vào ES...", len(canonical_skills))
        t0 = time.perf_counter()
        self._es.ensure_index()
        canon_payloads = [
            ". ".join(
                p for p in [cs.canonical_label, cs.description, f"Category: {cs.category}" if cs.category else None] if p
            )
            for cs in canonical_skills
        ]
        canon_vecs     = self._embedding.encode(canon_payloads)
        es_docs = []
        for cs, vec in zip(canonical_skills, canon_vecs):
            doc = cs.as_es_document()
            doc["vector"] = vec.tolist()
            es_docs.append(doc)
        es_result = self._es.bulk_index(iter(es_docs))
        phase_durations["es_index"] = time.perf_counter() - t0
        report["phases"]["es_index"] = {
            "duration_sec": round(phase_durations["es_index"], 4),
            "documents_attempted": len(es_docs),
            "documents_indexed": es_result.get("success_count", 0),
            "error_count": es_result.get("error_count", 0),
            "error_rate": round(
                es_result.get("error_count", 0) / max(len(es_docs), 1),
                6,
            ),
            "error_samples": es_result.get("error_samples", []),
            "vector_dim": self._settings.elasticsearch.vector_dim,
        }

        # Phase 5 – Build Neo4j KG
        logger.info("Xây dựng Neo4j Knowledge Graph...")
        t0 = time.perf_counter()
        if clear_existing:
            self._neo4j.clear_graph()
        self._neo4j.create_indexes()

        self._neo4j.batch_merge_skills(
            (cs.as_neo4j_props() for cs in canonical_skills)
        )

        course_nodes, teaches, requires = self._build_graph_edges(filtered_course_skills, merge_map)
        self._neo4j.batch_merge_courses(n.as_neo4j_props() for n in course_nodes)
        self._neo4j.batch_merge_teaches(e.as_neo4j_props() for e in teaches)
        self._neo4j.batch_merge_requires(e.as_neo4j_props() for e in requires)
        phase_durations["neo4j_graph_build"] = time.perf_counter() - t0

        graph_metrics = self._compute_graph_metrics(
            canonical_skills=canonical_skills,
            course_nodes=course_nodes,
            teaches=teaches,
            requires=requires,
        )
        report["phases"]["neo4j_graph_build"] = {
            "duration_sec": round(phase_durations["neo4j_graph_build"], 4),
            **graph_metrics,
        }

        total_duration = time.perf_counter() - started_at
        report["summary"] = {
            "total_duration_sec": round(total_duration, 4),
            "phase_durations_sec": {
                k: round(v, 4) for k, v in phase_durations.items()
            },
            "status": "completed",
        }

        if output_metrics_report:
            self._write_metrics_report(report, metrics_output_path)

        logger.info(
            "=== Graph Build Pipeline hoàn thành: %d courses, %d TEACHES, %d REQUIRES ===",
            len(course_nodes), len(teaches), len(requires),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_metrics_report(self, report: Dict[str, Any], output_path: str) -> None:
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("Metrics report saved to %s", path)

    def _compute_input_metrics(self, course_skills: List[CourseSkill]) -> Dict[str, Any]:
        unique_courses = {cs.course_id for cs in course_skills}
        unique_labels = {cs.skill_name for cs in course_skills if cs.skill_name}

        skills_per_course: Dict[str, int] = defaultdict(int)
        outcome_count = 0
        entry_count = 0
        for cs in course_skills:
            skills_per_course[cs.course_id] += 1
            if cs.skill_type == "outcome":
                outcome_count += 1
            elif cs.skill_type == "entry":
                entry_count += 1

        counts = list(skills_per_course.values())
        avg_skills = (sum(counts) / len(counts)) if counts else 0.0
        return {
            "raw_skill_records": len(course_skills),
            "unique_courses": len(unique_courses),
            "unique_skill_names": len(unique_labels),
            "skills_per_course": {
                "avg": round(avg_skills, 4),
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
            },
            "skill_type_distribution": {
                "outcome": outcome_count,
                "entry": entry_count,
            },
        }

    def _compute_graph_metrics(
        self,
        canonical_skills: List[CanonicalSkill],
        course_nodes: List[CourseNode],
        teaches: List[TeachesEdge],
        requires: List[RequiresEdge],
    ) -> Dict[str, Any]:
        canonical_ids = {cs.skill_id for cs in canonical_skills}
        course_ids = {c.course_id for c in course_nodes}

        taught_skill_ids = {e.skill_id for e in teaches}
        required_skill_ids = {e.skill_id for e in requires}
        used_skill_ids = taught_skill_ids | required_skill_ids

        orphaned_skill_ids = sorted(canonical_ids - used_skill_ids)

        teaches_by_course: Dict[str, Set[str]] = defaultdict(set)
        requires_by_course: Dict[str, Set[str]] = defaultdict(set)
        for e in teaches:
            teaches_by_course[e.course_id].add(e.skill_id)
        for e in requires:
            requires_by_course[e.course_id].add(e.skill_id)

        disconnected_courses = sorted(
            cid for cid in course_ids
            if not teaches_by_course.get(cid) and not requires_by_course.get(cid)
        )
        multimodal_courses = [
            cid for cid in course_ids
            if teaches_by_course.get(cid) and requires_by_course.get(cid)
        ]

        dependency_adj = self._build_course_dependency_adjacency(
            course_ids=course_ids,
            teaches_by_course=teaches_by_course,
            requires_by_course=requires_by_course,
        )
        undirected = self._to_undirected_adjacency(course_ids, dependency_adj)
        components = self._connected_components(undirected)

        dep_edge_count = sum(len(v) for v in dependency_adj.values())
        largest_component = max((len(c) for c in components), default=0)

        return {
            "skill_nodes": len(canonical_skills),
            "course_nodes": len(course_nodes),
            "teaches_edges": len(teaches),
            "requires_edges": len(requires),
            "orphaned_skills_count": len(orphaned_skill_ids),
            "orphaned_skill_ids_sample": orphaned_skill_ids[:100],
            "disconnected_courses_count": len(disconnected_courses),
            "disconnected_course_ids_sample": disconnected_courses[:100],
            "courses_with_both_teaches_and_requires": len(multimodal_courses),
            "courses_with_both_teaches_and_requires_ratio": round(
                len(multimodal_courses) / max(len(course_ids), 1),
                6,
            ),
            "course_dependency_edges": dep_edge_count,
            "course_dependency_connected_components": len(components),
            "largest_dependency_component_size": largest_component,
        }

    def _build_course_dependency_adjacency(
        self,
        course_ids: Set[str],
        teaches_by_course: Dict[str, Set[str]],
        requires_by_course: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Edge B -> A nếu B dạy một skill mà A yêu cầu."""
        adjacency: Dict[str, Set[str]] = {cid: set() for cid in course_ids}
        ids = list(course_ids)
        for cid in ids:
            req = requires_by_course.get(cid, set())
            if not req:
                continue
            for other in ids:
                if other == cid:
                    continue
                taught = teaches_by_course.get(other, set())
                if taught and (req & taught):
                    adjacency[other].add(cid)
        return adjacency

    def _to_undirected_adjacency(
        self,
        course_ids: Set[str],
        directed: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        undirected: Dict[str, Set[str]] = {cid: set() for cid in course_ids}
        for src, dsts in directed.items():
            for dst in dsts:
                undirected[src].add(dst)
                undirected[dst].add(src)
        return undirected

    def _connected_components(self, undirected: Dict[str, Set[str]]) -> List[List[str]]:
        visited: Set[str] = set()
        components: List[List[str]] = []
        for node in undirected:
            if node in visited:
                continue
            stack = [node]
            visited.add(node)
            comp: List[str] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in undirected[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            components.append(comp)
        return components

    def _build_graph_edges(
        self,
        course_skills: List[CourseSkill],
        merge_map: Dict[str, str],
    ) -> Tuple[List[CourseNode], List[TeachesEdge], List[RequiresEdge]]:
        seen: Dict[str, CourseNode] = {}
        teaches:  List[TeachesEdge]  = []
        requires: List[RequiresEdge] = []

        for cs in course_skills:
            if cs.course_id not in seen:
                seen[cs.course_id] = CourseNode(
                    course_id=cs.course_id,
                    course_title=cs.course_title,
                    category=cs.category,
                    source_file=str(cs.source_file),
                )

            skill_id = merge_map.get(cs.skill_name)
            if not skill_id:
                logger.debug("Skill '%s' không có trong merge_map", cs.skill_name)
                continue

            props = dict(
                course_id=cs.course_id,
                skill_id=skill_id,
                skill_type=cs.skill_type,
                source=str(cs.source_file),
            )
            if cs.skill_type == "outcome":
                teaches.append(TeachesEdge(**props))
            else:
                requires.append(RequiresEdge(**props))

        return list(seen.values()), teaches, requires

    def _filter_soft_skills(self, course_skills: List[CourseSkill]) -> Tuple[List[CourseSkill], int]:
        terms = [t for t in self._settings.soft_skill_filter.exclude_terms if t]
        if not terms:
            return course_skills, 0

        kept: List[CourseSkill] = []
        removed = 0
        for cs in course_skills:
            text = " ".join([
                str(cs.skill_name or ""),
                str(cs.description or ""),
                str(cs.category or ""),
            ]).lower()
            if any(term in text for term in terms):
                removed += 1
                continue
            kept.append(cs)
        return kept, removed
