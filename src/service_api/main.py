"""FastAPI application entry point – VietCV Course Recommendation API."""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service_api.config import settings
from service_api.dependencies import close_all

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
        description="Course recommendation system: skill gap analysis → course suggestions via ES KNN",
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

    # Health endpoints (available immediately)
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/health/live")
    def liveness():
        return {"status": "alive"}

    @app.get("/health/ready")
    def readiness():
        """Readiness probe – checks all dependencies."""
        deps = {}
        try:
            from service_api.dependencies import get_mysql
            deps["mysql"] = "ok" if get_mysql().verify_connection() else "fail"
        except Exception:
            deps["mysql"] = "fail"
        try:
            from service_api.dependencies import get_course_es
            deps["elasticsearch"] = "ok" if get_course_es().verify_connection() else "fail"
        except Exception:
            deps["elasticsearch"] = "fail"
        try:
            from service_api.dependencies import get_s3
            deps["s3"] = "ok" if get_s3().verify_connection() else "fail"
        except Exception:
            deps["s3"] = "fail"

        all_ok = all(v == "ok" for v in deps.values())
        return {
            "status": "ready" if all_ok else "degraded",
            "dependencies": deps,
        }

    @app.get("/")
    def root():
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    # TODO: Phase 2 – register API routers
    # from service_api.api.v1.api import api_router
    # app.include_router(api_router, prefix="/api")

    return app


app = create_app()
