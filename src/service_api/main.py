"""FastAPI application entry point."""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service_api.config import settings
from service_api.dependencies import close_all
from service_api.api.v1.api import api_router

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    yield
    logger.info("Shutting down – closing connections")
    close_all()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="GraphRAG-based course recommendation: JD/CV skill gap→KG→courses",
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    def root():
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs":    "/docs",
        }

    return app


app = create_app()
