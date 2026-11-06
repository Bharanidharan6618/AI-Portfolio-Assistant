"""Main FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import health
from app.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="AI Portfolio Assistant API",
    description="Backend API for an AI chatbot that answers questions "
    "about my professional portfolio.",
    version="0.1.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log only the error type, never request contents or secrets.
    logger.error(
        "Unhandled error on %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


app.include_router(health.router)