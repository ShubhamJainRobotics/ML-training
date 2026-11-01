"""Threshold policy enforcement and severity mapping."""

from typing import Dict, Any, Tuple
from app.config import settings

# Hard invariant: Max allowed severity is strictly 'medium'
MAX_ALLOWED_SEVERITY = "medium"


def get_thresholds() -> Tuple[float, float]:
    """Retrieves low and medium thresholds from configuration with validation."""
    low = settings.LOW_THRESHOLD
    med = settings.MEDIUM_THRESHOLD

    # Fallback to sensible defaults if invalid
    if not (0.0 < low < med < 1.0):
        low, med = 0.50, 0.75

    return low, med


def evaluate_severity(probability: float) -> str:
    """
    Maps calibrated scam probability to a conservative triage severity band:
    - p < low_threshold: 'none'
    - low_threshold <= p < medium_threshold: 'low'
    - p >= medium_threshold: 'medium'

    Guarantees severity is clamped to 'medium' and NEVER evaluates to 'high'.
    """
    low_th, med_th = get_thresholds()

    if probability >= med_th:
        raw_severity = "medium"
    elif probability >= low_th:
        raw_severity = "low"
    else:
        raw_severity = "none"

    # Clamp severity to maximum allowed cap
    if raw_severity in ["high", "critical"] or settings.MAX_SEVERITY == "medium":
        if raw_severity not in ["none", "low"]:
            raw_severity = MAX_ALLOWED_SEVERITY

    return raw_severity


def get_policy_metadata() -> Dict[str, Any]:
    """Returns fixed architectural safety invariants."""
    return {
        "can_trigger_high": False,
        "can_mark_verified": False,
        "can_override_rule": False,
    }
