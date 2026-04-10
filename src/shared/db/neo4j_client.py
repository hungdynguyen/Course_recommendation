"""Unified Neo4j client – dùng chung cho data_factory (batch ETL) lẫn service_api (query).

- `Neo4jClient`: thin wrapper dùng trong service_api (read queries).
- `Neo4jBatchClient`: mở rộng thêm batch MERGE operations cho data_factory.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Read-oriented Neo4j client dùng trong service_api."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j") -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        logger.info("Neo4jClient connected to %s", uri)

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def verify_connection(self) -> bool:
        try:
            with self.driver.session(database=self._database) as s:
                s.run("RETURN 1")
            return True
        except Exception:
            return False

    def query(self, cypher: str, params: Optional[dict] = None) -> List[dict]:
        """Execute a Cypher read query and return list of plain dicts."""
        with self.driver.session(database=self._database) as s:
            result = s.run(cypher, params or {})
            return [dict(record) for record in result]


class Neo4jBatchClient(Neo4jClient):
    """Write-oriented Neo4j client dùng trong data_factory (ETL)."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        batch_size: int = 5000,
    ) -> None:
        super().__init__(uri, username, password, database)
        self._batch_size = batch_size

    # ---------- DDL ----------

    def create_indexes(self) -> None:
        stmts = [
            "DROP CONSTRAINT skill_id_unique IF EXISTS",
            "DROP CONSTRAINT course_id_unique IF EXISTS",
            "DROP INDEX skill_id_idx IF EXISTS",
            "DROP INDEX course_id_idx IF EXISTS",
            "CREATE CONSTRAINT skill_id_unique IF NOT EXISTS FOR (s:Skill)  REQUIRE s.skill_id IS UNIQUE",
            "CREATE CONSTRAINT course_id_unique IF NOT EXISTS FOR (c:Course) REQUIRE c.course_id IS UNIQUE",
        ]
        with self.driver.session(database=self._database) as s:
            for stmt in stmts:
                try:
                    s.run(stmt)
                except Exception:
                    pass

    def clear_graph(self) -> None:
        logger.warning("Clearing all nodes and relationships from Neo4j")
        with self.driver.session(database=self._database) as s:
            s.run("MATCH (n) DETACH DELETE n")

    # ---------- Skill nodes ----------

    def batch_merge_skills(self, skills: Iterable[Dict[str, Any]]) -> None:
        self._run_batch(
            skills,
            """
            UNWIND $batch AS row
            MERGE (s:Skill {skill_id: row.skill_id})
            SET s.canonical_label = row.canonical_label,
                s.aliases         = row.aliases,
                s.description     = row.description,
                s.category        = row.category
            """,
        )

    # ---------- Course nodes ----------

    def batch_merge_courses(self, courses: Iterable[Dict[str, Any]]) -> None:
        self._run_batch(
            courses,
            """
            UNWIND $batch AS row
            MERGE (c:Course {course_id: row.course_id})
            SET c.course_title = row.course_title,
                c.category     = row.category,
                c.source_file  = row.source_file
            """,
        )

    # ---------- Edges ----------

    def batch_merge_teaches(self, edges: Iterable[Dict[str, Any]]) -> None:
        self._run_batch(
            edges,
            """
            UNWIND $batch AS row
            MATCH (c:Course {course_id: row.course_id})
            MATCH (s:Skill  {skill_id:  row.skill_id})
            MERGE (c)-[r:TEACHES]->(s)
            SET r.skill_type = row.skill_type,
                r.source     = row.source
            """,
        )

    def batch_merge_requires(self, edges: Iterable[Dict[str, Any]]) -> None:
        self._run_batch(
            edges,
            """
            UNWIND $batch AS row
            MATCH (c:Course {course_id: row.course_id})
            MATCH (s:Skill  {skill_id:  row.skill_id})
            MERGE (c)-[r:REQUIRES]->(s)
            SET r.skill_type = row.skill_type,
                r.source     = row.source
            """,
        )

    # ---------- Private ----------

    def _run_batch(self, items: Iterable[Dict[str, Any]], cypher: str) -> None:
        batch: List[Dict[str, Any]] = []
        total = 0
        with self.driver.session(database=self._database) as s:
            for item in items:
                batch.append(item)
                if len(batch) >= self._batch_size:
                    s.run(cypher, batch=batch)
                    total += len(batch)
                    batch = []
            if batch:
                s.run(cypher, batch=batch)
                total += len(batch)
        logger.info("Batch merged %d records", total)
