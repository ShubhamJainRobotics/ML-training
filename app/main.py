"""FastAPI application for Jaanch ML inference service."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    HealthResponse,
    MLSignalRequest,
    MLSignalResponse,
    ErrorResponse
)
from app.model_service import model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jaanch_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the calibrated model artifact once at application startup."""
    logger.info("Initializing Jaanch ML Inference Service...")
    loaded = model_service.load_model()
    if not loaded:
        logger.warning(
            "Model artifact could not be loaded at startup. Service will return 503 on inference requests."
        )
    yield
    logger.info("Shutting down Jaanch ML Inference Service.")


app = FastAPI(
    title="Jaanch ML Scam Detection Service",
    version="2.0.0",
    description="Explainable, calibrated TF-IDF + Logistic Regression supplementary triage service.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Validates API key:
    - Required in production or whenever API_KEY is set.
    - Optional in local development if API_KEY is blank.
    """
    configured_key = settings.API_KEY

    # If an API key is configured or in production environment
    if configured_key or settings.ENVIRONMENT == "production":
        provided_key = None
        if x_api_key:
            provided_key = x_api_key
        elif authorization:
            if authorization.startswith("Bearer "):
                provided_key = authorization[7:].strip()
            else:
                provided_key = authorization.strip()

        if not provided_key or provided_key != configured_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid API key.",
            )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and model load status",
)
async def health_check():
    """Returns the service health and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        model_version=settings.MODEL_VERSION,
    )


@app.post(
    "/ml/signal",
    response_model=MLSignalResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - Consent missing"},
        500: {"model": ErrorResponse, "description": "Inference Error"},
        503: {"model": ErrorResponse, "description": "Model Not Ready"},
    },
    tags=["Inference"],
    summary="Generate supplementary ML scam detection signal",
)
async def generate_ml_signal(
    request: MLSignalRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Evaluates job posting text under conservative safety guardrails:
    1. Rejects with 403 if consent is False.
    2. Redacts PII before tokenization.
    3. Returns calibrated probability, severity capped at Medium, and top contributing terms.
    """
    # 1. API key authentication check
    verify_api_key(x_api_key=x_api_key, authorization=authorization)

    # 2. Consent check
    if not request.consent:
        logger.info("Request rejected: User consent not granted.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Explicit user consent is required before analyzing message text.",
        )

    # 3. Model readiness check
    if not model_service.is_loaded:
        logger.error("Inference requested while model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Unavailable: Model artifact is not ready.",
        )

    # 4. Execute inference with safe exception shielding
    try:
        result = model_service.predict(request.text)
        return MLSignalResponse(**result)
    except Exception as e:
        logger.error(f"Inference error encountered: {type(e).__name__}")
        # Safe JSON error response without leaking stack traces or user text
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InferenceError",
                "message": "Unable to compute ML signal safely. Please rely on deterministic rule checks."
            }
        )
