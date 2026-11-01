"""Pydantic schemas for the Jaanch ML inference API contract."""

from typing import List, Optional
from pydantic import BaseModel, Field


class MLSignalRequest(BaseModel):
    text: str = Field(
        ...,
        description="Raw job posting or recruitment message text to analyze.",
        min_length=1,
        max_length=50000,
    )
    consent: bool = Field(
        ...,
        description="Explicit user consent required before sending message text for ML analysis.",
    )


class PolicyInvariants(BaseModel):
    can_trigger_high: bool = Field(
        default=False,
        description="Hard invariant: ML can never elevate severity to High.",
    )
    can_mark_verified: bool = Field(
        default=False,
        description="Hard invariant: ML can never mark an offer Verified or genuine.",
    )
    can_override_rule: bool = Field(
        default=False,
        description="Hard invariant: ML can never override or remove deterministic rule findings.",
    )


class MLSignalResponse(BaseModel):
    source: str = Field(
        default="ml",
        description="Source of the signal (always 'ml').",
    )
    signal: Optional[str] = Field(
        default=None,
        description="'possible_scam_pattern' when probability >= low threshold, else null.",
    )
    severity: str = Field(
        ...,
        description="Severity level capped at medium ('none', 'low', or 'medium').",
    )
    confidence: float = Field(
        ...,
        description="Calibrated probability score between 0.0 and 1.0.",
    )
    model_version: str = Field(
        ...,
        description="Identifier of the model artifact used for scoring.",
    )
    top_contributing_terms: List[str] = Field(
        default_factory=list,
        description="Top n-gram terms contributing to the scam probability.",
    )
    policy: PolicyInvariants = Field(
        default_factory=PolicyInvariants,
        description="Architectural safety invariants guaranteed by the service.",
    )


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service health status.")
    model_loaded: bool = Field(..., description="Whether the ML model artifact is loaded in memory.")
    model_version: str = Field(..., description="Active model version identifier.")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error category name.")
    message: str = Field(..., description="Safe, sanitized error description.")

