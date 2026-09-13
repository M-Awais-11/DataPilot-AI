"""
FastAPI Main Application Factory.
Configures CORS, lifespan events (database tables creation, directory creation),
rate limiting, API routers, and health checks.
"""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.config.settings import settings
from app.database.connection import engine, Base, check_db_connection
import app.models  # Ensure all models are registered with Base.metadata

# Import API Routers
from app.api.auth import router as auth_router
from app.api.datasets import router as datasets_router
from app.api.experiments import router as experiments_router
from app.api.models import router as models_router
from app.api.predictions import router as predictions_router
from app.api.reports import router as reports_router
from app.api.chat import router as chat_router

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("adsagent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing Autonomous Data Scientist Agent backend...")
    
    # Ensure storage folders exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)
    Path("./mlruns").mkdir(parents=True, exist_ok=True)

    # Automatically create tables in dev/docker environments
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schemas verified/created successfully.")
    except Exception as e:
        logger.warning(f"Database table auto-creation skipped or failed: {e}")

    yield

    logger.info("Shutting down backend...")
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade AI-powered Autonomous Data Scientist Agent API",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.exception_handler(SQLAlchemyError)
async def database_unavailable_handler(request: Request, exc: SQLAlchemyError):
    """Return a clear API error when the configured database cannot be reached."""
    logger.exception("Database request failed for %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": (
                "The database is unavailable. Verify that SQL Server is running and that "
                "its TCP/IP or Named Pipes protocol is enabled, then restart the backend."
            )
        },
    )


# ── Health & Diagnostics ──────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """System health check endpoint."""
    db_healthy = await check_db_connection()
    return {
        "status": "online" if db_healthy else "degraded",
        "service": settings.app_name,
        "version": settings.app_version,
        "database": "connected" if db_healthy else "unreachable",
        "llm_model": settings.openrouter_model,
    }



@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to the Autonomous Data Scientist Agent API",
        "docs": "/docs",
        "health": "/health",
    }


# Keep CORS outside FastAPI's error middleware. This preserves the browser's
# access to structured error responses even when a dependency (such as the
# database) raises an unexpected exception.
app = CORSMiddleware(
    app=app,
    allow_origins=settings.allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
