"""Course Engine – FastAPI application (port 8004).

Handles embedding building and Elasticsearch indexing for courses.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8004"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="VietCV Course Engine",
        description="Embedding building + ES indexing service",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "course_engine"}

    @app.post("/build")
    async def build():
        """Build embeddings and index courses into Elasticsearch.

        TODO: Implement in Phase 3.
        """
        return {"status": "not_implemented", "message": "Phase 3 – coming soon"}

    return app


app = create_app()
