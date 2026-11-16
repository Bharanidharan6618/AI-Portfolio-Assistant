"""Main FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, health, portfolio
from app.config import get_settings
from app.middleware import BodySizeLimitMiddleware
from app.services.gemini_service import GeminiServiceError
from app.services.knowledge_service import KnowledgeBaseError
from app.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
settings = get_settings()  # fails fast at startup if configuration is invalid

app = FastAPI(
    title="AI Portfolio Assistant API",
    description="Backend API for an AI chatbot that answers questions "
    "about my professional portfolio.",
    version="0.3.0",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

# Middleware order: the LAST one added is the outermost. CORS goes last so that
# even error responses (413, 429, ...) carry CORS headers and browsers can read them.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)


@app.exception_handler(GeminiServiceError)
async def gemini_error_handler(request: Request, exc: GeminiServiceError):
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.user_message},
        headers=headers,
    )


@app.exception_handler(KnowledgeBaseError)
async def knowledge_error_handler(request: Request, exc: KnowledgeBaseError):
    logger.error("Knowledge base problem: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The portfolio information is temporarily unavailable."},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Report which field is wrong, but never echo back what the visitor sent.
    problems = [
        {
            "field": ".".join(str(part) for part in err["loc"] if part != "body"),
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request.", "errors": problems},
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
app.include_router(portfolio.router)
app.include_router(chat.router)