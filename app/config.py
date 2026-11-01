"""Application configuration and environment settings."""

import os
from typing import List
from pathlib import Path


class Settings:
    """Application configuration with safe production defaults."""

    def __init__(self):
        # Base directories
        self.BASE_DIR: Path = Path(__file__).resolve().parent.parent

        # Environment & Debug
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
        self.DEBUG: bool = self.ENVIRONMENT == "development"

        # Model configuration
        model_env = os.getenv("MODEL_PATH", "models/jaanch_scam_model_v2_calibrated.joblib")
        candidate_path = Path(model_env)
        if not candidate_path.is_absolute():
            candidate_path = self.BASE_DIR / candidate_path
        self.MODEL_PATH: Path = candidate_path

        self.MODEL_VERSION: str = os.getenv("MODEL_VERSION", "tfidf-logreg-v2-calibrated")

        # Thresholds
        self.LOW_THRESHOLD: float = float(os.getenv("LOW_THRESHOLD", "0.50"))
        self.MEDIUM_THRESHOLD: float = float(os.getenv("MEDIUM_THRESHOLD", "0.75"))

        # Hard invariant: Even if configured otherwise, max allowed severity is 'medium'
        raw_max = os.getenv("MAX_SEVERITY", "medium").lower()
        self.MAX_SEVERITY: str = "medium" if raw_max in ["medium", "high"] else raw_max

        # Security & Authentication
        self.API_KEY: str = os.getenv("API_KEY", "").strip()

        # In production, an API key is required
        if self.ENVIRONMENT == "production" and not self.API_KEY:
            # We don't crash at import, but enforce at auth check
            pass

        # CORS origins
        raw_origins = os.getenv(
            "CORS_ORIGINS",
            "https://your-jaanch-domain.com,http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
        )
        self.CORS_ORIGINS: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]

        # Request Timeout
        self.REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5.0"))


settings = Settings()


def get_settings() -> Settings:
    """Return application settings instance."""
    return settings

