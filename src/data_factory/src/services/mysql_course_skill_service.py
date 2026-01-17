from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, Sequence

import pymysql

from ..settings import MySQLConfig

LOGGER = logging.getLogger("data_factory.mysql_course_mappings")


class MySQLCourseSkillService:
    def __init__(self, config: MySQLConfig) -> None:
        self._config = config

    def initialize(self) -> None:
        if not self._config.enabled:
            LOGGER.info("MySQL integration disabled; skipping init for course mappings")
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self._config.mapping_table} (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            course_id VARCHAR(255) NOT NULL,
            course_title VARCHAR(512),
            skill_name VARCHAR(512) NOT NULL,
            skill_type VARCHAR(64) NOT NULL,
            description TEXT,
            category VARCHAR(255),
            proficiency_level INT,
            bloom_taxonomy_level VARCHAR(64),
            source_file VARCHAR(1024),
            esco_skill_id VARCHAR(255),
            esco_preferred_label VARCHAR(512),
            esco_description TEXT,
            similarity_score DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) CHARACTER SET {self._config.charset};
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(ddl)
            conn.commit()
        LOGGER.info("Ensured MySQL mapping table %s exists", self._config.mapping_table)

    def insert_records(self, records: Sequence[dict]) -> None:
        if not self._config.enabled:
            LOGGER.debug("MySQL integration disabled; skipping course mapping insert")
            return
        if not records:
            LOGGER.info("No course skill mappings to insert")
            return

        columns = [
            "course_id",
            "course_title",
            "skill_name",
            "skill_type",
            "description",
            "category",
            "proficiency_level",
            "bloom_taxonomy_level",
            "source_file",
            "esco_skill_id",
            "esco_preferred_label",
            "esco_description",
            "similarity_score",
        ]
        placeholders = ", ".join(["%s" for _ in columns])
        sql = f"INSERT INTO {self._config.mapping_table} ({', '.join(columns)}) VALUES ({placeholders})"

        total = 0
        with self._connect() as conn:
            with conn.cursor() as cursor:
                for chunk in _batched(records, self._config.batch_size):
                    payload = [tuple(record.get(c) for c in columns) for record in chunk]
                    cursor.executemany(sql, payload)
                    total += len(payload)
                conn.commit()
        LOGGER.info("Inserted %s course skill mappings into MySQL", total)

    @contextmanager
    def _connect(self):
        if not self._config.enabled:
            raise RuntimeError("MySQL integration is disabled in configuration")
        conn = pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.username,
            password=self._config.password,
            database=self._config.database,
            charset=self._config.charset,
            connect_timeout=self._config.connect_timeout,
            autocommit=False,
        )
        try:
            yield conn
        finally:
            conn.close()


def _batched(records: Sequence[dict], batch_size: int) -> Iterable[Sequence[dict]]:
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
