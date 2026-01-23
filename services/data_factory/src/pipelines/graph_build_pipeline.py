from __future__ import annotations

import logging
from pathlib import Path

from ..io.graph_loader import load_broader_edges, load_course_mappings, load_skill_nodes
from ..io.skill_loader import load_skills
from ..services.neo4j_service import Neo4jService
from ..settings import Settings

LOGGER = logging.getLogger("data_factory.graph_build_pipeline")


class GraphBuildPipeline:
    def __init__(self, settings: Settings, neo4j_service: Neo4jService) -> None:
        self._settings = settings
        self._neo4j = neo4j_service

    def run(self, clear_existing: bool = False) -> None:
        LOGGER.info("Starting graph build pipeline")

        self._neo4j.connect()

        if clear_existing:
            self._neo4j.clear_graph()

        self._neo4j.create_indexes()

        LOGGER.info("Loading ESCO skills and relations")
        skills = load_skills(
            skills_path=self._settings.paths.esco_skills,
            relations_path=self._settings.paths.esco_skill_relations,
        )

        skill_nodes = load_skill_nodes(skills)
        broader_edges = load_broader_edges(skills)

        LOGGER.info("Merging %s skill nodes to graph", len(skill_nodes))
        self._neo4j.batch_merge_skills(
            (node.to_dict() for node in skill_nodes),
            batch_size=self._settings.neo4j.batch_size,
        )

        LOGGER.info("Merging %s BROADER edges to graph", len(broader_edges))
        self._neo4j.batch_merge_broader_edges(
            (edge.to_dict() for edge in broader_edges),
            batch_size=self._settings.neo4j.batch_size,
        )

        mappings_file = (
            self._settings.paths.course_skill_mappings_dir / "course_skill_mappings.jsonl"
        )
        if not mappings_file.exists():
            LOGGER.warning("Course mappings file not found: %s", mappings_file)
            LOGGER.info("Skipping course/TEACHES/REQUIRES loading")
        else:
            LOGGER.info("Loading course mappings from %s", mappings_file)
            courses, teaches_edges, requires_edges = load_course_mappings(mappings_file)

            LOGGER.info("Merging %s course nodes to graph", len(courses))
            self._neo4j.batch_merge_courses(
                (course.to_dict() for course in courses),
                batch_size=self._settings.neo4j.batch_size,
            )

            LOGGER.info("Merging %s TEACHES edges to graph", len(teaches_edges))
            self._neo4j.batch_merge_teaches_edges(
                (edge.to_dict() for edge in teaches_edges),
                batch_size=self._settings.neo4j.batch_size,
            )

            LOGGER.info("Merging %s REQUIRES edges to graph", len(requires_edges))
            self._neo4j.batch_merge_requires_edges(
                (edge.to_dict() for edge in requires_edges),
                batch_size=self._settings.neo4j.batch_size,
            )

        LOGGER.info("Graph build pipeline completed successfully")
        self._neo4j.close()
