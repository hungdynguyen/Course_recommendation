from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase, Session

LOGGER = logging.getLogger("data_factory.neo4j_service")


class Neo4jService:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j") -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver = None

    def connect(self) -> None:
        if self._driver:
            return
        LOGGER.info("Connecting to Neo4j at %s", self._uri)
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            LOGGER.info("Neo4j connection closed")

    def create_indexes(self) -> None:
        LOGGER.info("Creating indexes and constraints")
        with self._driver.session(database=self._database) as session:
            # Drop existing constraints first (they depend on indexes)
            try:
                session.run("DROP CONSTRAINT skill_id_unique IF EXISTS")
            except Exception:
                pass
            try:
                session.run("DROP CONSTRAINT course_id_unique IF EXISTS")
            except Exception:
                pass
            
            # Then drop indexes
            try:
                session.run("DROP INDEX skill_id_idx IF EXISTS")
            except Exception:
                pass
            try:
                session.run("DROP INDEX course_id_idx IF EXISTS")
            except Exception:
                pass
            
            # Create fresh constraints (they create indexes automatically)
            session.run(
                "CREATE CONSTRAINT skill_id_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.skill_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT course_id_unique IF NOT EXISTS FOR (c:Course) REQUIRE c.course_id IS UNIQUE"
            )

    def clear_graph(self) -> None:
        LOGGER.warning("Clearing all nodes and relationships from graph")
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def batch_merge_skills(self, skills: Iterable[Dict[str, Any]], batch_size: int = 5000) -> None:
        LOGGER.info("Merging skill nodes in batches of %s", batch_size)
        batch: List[Dict[str, Any]] = []
        count = 0
        with self._driver.session(database=self._database) as session:
            for skill in skills:
                batch.append(skill)
                if len(batch) >= batch_size:
                    self._merge_skill_batch(session, batch)
                    count += len(batch)
                    batch = []
            if batch:
                self._merge_skill_batch(session, batch)
                count += len(batch)
        LOGGER.info("Merged %s skill nodes", count)

    def batch_merge_broader_edges(
        self, relations: Iterable[Dict[str, str]], batch_size: int = 5000
    ) -> None:
        LOGGER.info("Merging BROADER relationships in batches of %s", batch_size)
        batch: List[Dict[str, str]] = []
        count = 0
        with self._driver.session(database=self._database) as session:
            for rel in relations:
                batch.append(rel)
                if len(batch) >= batch_size:
                    self._merge_broader_batch(session, batch)
                    count += len(batch)
                    batch = []
            if batch:
                self._merge_broader_batch(session, batch)
                count += len(batch)
        LOGGER.info("Merged %s BROADER edges", count)

    def batch_merge_courses(self, courses: Iterable[Dict[str, Any]], batch_size: int = 5000) -> None:
        LOGGER.info("Merging course nodes in batches of %s", batch_size)
        batch: List[Dict[str, Any]] = []
        count = 0
        with self._driver.session(database=self._database) as session:
            for course in courses:
                batch.append(course)
                if len(batch) >= batch_size:
                    self._merge_course_batch(session, batch)
                    count += len(batch)
                    batch = []
            if batch:
                self._merge_course_batch(session, batch)
                count += len(batch)
        LOGGER.info("Merged %s course nodes", count)

    def batch_merge_teaches_edges(
        self, teaches: Iterable[Dict[str, Any]], batch_size: int = 5000
    ) -> None:
        LOGGER.info("Merging TEACHES relationships in batches of %s", batch_size)
        batch: List[Dict[str, Any]] = []
        count = 0
        with self._driver.session(database=self._database) as session:
            for edge in teaches:
                batch.append(edge)
                if len(batch) >= batch_size:
                    self._merge_teaches_batch(session, batch)
                    count += len(batch)
                    batch = []
            if batch:
                self._merge_teaches_batch(session, batch)
                count += len(batch)
        LOGGER.info("Merged %s TEACHES edges", count)

    def batch_merge_requires_edges(
        self, requires: Iterable[Dict[str, Any]], batch_size: int = 5000
    ) -> None:
        LOGGER.info("Merging REQUIRES relationships in batches of %s", batch_size)
        batch: List[Dict[str, Any]] = []
        count = 0
        with self._driver.session(database=self._database) as session:
            for edge in requires:
                batch.append(edge)
                if len(batch) >= batch_size:
                    self._merge_requires_batch(session, batch)
                    count += len(batch)
                    batch = []
            if batch:
                self._merge_requires_batch(session, batch)
                count += len(batch)
        LOGGER.info("Merged %s REQUIRES edges", count)

    @staticmethod
    def _merge_skill_batch(session: Session, batch: List[Dict[str, Any]]) -> None:
        session.run(
            """
            UNWIND $batch AS row
            MERGE (s:Skill {skill_id: row.skill_id})
            SET s.preferred_label = row.preferred_label,
                s.description = row.description,
                s.skill_type = row.skill_type,
                s.alternative_labels = row.alternative_labels
            """,
            batch=batch,
        )

    @staticmethod
    def _merge_broader_batch(session: Session, batch: List[Dict[str, str]]) -> None:
        session.run(
            """
            UNWIND $batch AS row
            MATCH (child:Skill {skill_id: row.child_id})
            MATCH (parent:Skill {skill_id: row.parent_id})
            MERGE (child)-[:BROADER]->(parent)
            """,
            batch=batch,
        )

    @staticmethod
    def _merge_course_batch(session: Session, batch: List[Dict[str, Any]]) -> None:
        session.run(
            """
            UNWIND $batch AS row
            MERGE (c:Course {course_id: row.course_id})
            SET c.course_title = row.course_title,
                c.category = row.category,
                c.source_file = row.source_file
            """,
            batch=batch,
        )

    @staticmethod
    def _merge_teaches_batch(session: Session, batch: List[Dict[str, Any]]) -> None:
        session.run(
            """
            UNWIND $batch AS row
            MATCH (c:Course {course_id: row.course_id})
            MATCH (s:Skill {skill_id: row.skill_id})
            MERGE (c)-[r:TEACHES]->(s)
            SET r.similarity_score = row.similarity_score,
                r.skill_type = row.skill_type,
                r.source = row.source
            """,
            batch=batch,
        )

    @staticmethod
    def _merge_requires_batch(session: Session, batch: List[Dict[str, Any]]) -> None:
        session.run(
            """
            UNWIND $batch AS row
            MATCH (c:Course {course_id: row.course_id})
            MATCH (s:Skill {skill_id: row.skill_id})
            MERGE (c)-[r:REQUIRES]->(s)
            SET r.source = row.source,
                r.skill_type = row.skill_type
            """,
            batch=batch,
        )
