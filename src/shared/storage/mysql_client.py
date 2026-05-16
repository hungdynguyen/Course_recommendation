"""MySQL client – SQLAlchemy engine + session factory.

Usage:
    from shared.storage.mysql_client import MySQLClient

    db = MySQLClient(host="mysql", port=3306, database="vietcv", ...)
    with db.session() as session:
        session.execute(text("SELECT 1"))
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base

logger = logging.getLogger(__name__)


Base = declarative_base()


class MySQLClient:
    """Thin wrapper around SQLAlchemy engine + session factory."""

    def __init__(
        self,
        host: str = "mysql",
        port: int = 3306,
        database: str = "vietcv",
        username: str = "vietcv_user",
        password: str = "secure_password",
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ) -> None:
        url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        self._engine: Engine = create_engine(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=echo,
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info("MySQLClient connected to %s:%d/%s", host, port, database)

    @property
    def engine(self) -> Engine:
        return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope for a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def verify_connection(self) -> bool:
        """Check if MySQL is reachable."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def create_tables(self) -> None:
        """Create all ORM-mapped tables if they don't exist."""
        Base.metadata.create_all(self._engine)
        logger.info("MySQL tables created/verified")

    def close(self) -> None:
        """Dispose engine connections."""
        self._engine.dispose()
        logger.info("MySQL connection pool disposed")
