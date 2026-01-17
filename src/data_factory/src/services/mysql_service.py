from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, Sequence

import pymysql
import math

from ..settings import MySQLConfig

LOGGER = logging.getLogger("data_factory.mysql")


class MySQLService:
    def __init__(self, config: MySQLConfig) -> None:
        self._config = config

    def initialize(self) -> None:
        if not self._config.enabled:
            LOGGER.info("MySQL integration disabled; skipping initialization")
            return
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {self._config.esco_table} (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            skill_id VARCHAR(255) NOT NULL UNIQUE,
            preferred_label VARCHAR(512),
            description TEXT,
            skill_type VARCHAR(128),
            alternative_labels TEXT,
            broader_skill_ids TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) CHARACTER SET {self._config.charset};
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table)
            conn.commit()
        LOGGER.info("Ensured MySQL table %s exists", self._config.esco_table)

    def upsert_skills(self, records: Sequence[dict]) -> None:
        if not self._config.enabled:
            LOGGER.debug("MySQL integration disabled; skipping upsert")
            return
        if not records:
            LOGGER.info("No skill metadata to upsert into MySQL")
            return

        columns = [
            "skill_id",
            "preferred_label",
            "description",
            "skill_type",
            "alternative_labels",
            "broader_skill_ids",
        ]
        placeholders = ", ".join(["%s" for _ in columns])
        update_clause = ", ".join([f"{col}=VALUES({col})" for col in columns[1:]])
        sql = (
            f"INSERT INTO {self._config.esco_table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {update_clause}"
        )

        batched = _batched(records, self._config.batch_size)
        total = 0
        with self._connect() as conn:
            with conn.cursor() as cursor:
                for chunk in batched:
                    payload = [
                        (
                            r.get("skill_id"),
                            _null_if_nan(r.get("preferred_label")),
                            _null_if_nan(r.get("description")),
                            _null_if_nan(r.get("skill_type")),
                            "|".join([x for x in (r.get("alternative_labels") or []) if _null_if_nan(x)]),
                            "|".join([x for x in (r.get("broader_skill_ids") or []) if _null_if_nan(x)]),
                        )
                        for r in chunk
                    ]
                    cursor.executemany(sql, payload)
                    total += len(payload)
                conn.commit()
        LOGGER.info("Upserted %s skills into MySQL", total)

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


def _null_if_nan(value):
    """Return None when value is NaN; otherwise return the original value."""
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    return value
