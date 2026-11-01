"""Unit tests for threshold policy evaluation and severity clamping."""

import pytest
from app.policy import evaluate_severity, get_policy_metadata
from app.config import settings


def test_severity_below_low_threshold():
    """Scores below 0.50 must yield 'none'."""
    assert evaluate_severity(0.10) == "none"
    assert evaluate_severity(0.49) == "none"
    assert evaluate_severity(0.0) == "none"


def test_severity_between_low_and_medium():
    """Scores between 0.50 and 0.7499 must yield 'low'."""
    assert evaluate_severity(0.50) == "low"
    assert evaluate_severity(0.65) == "low"
    assert evaluate_severity(0.749) == "low"


def test_severity_at_or_above_medium():
    """Scores >= 0.75 must yield 'medium'."""
    assert evaluate_severity(0.75) == "medium"
    assert evaluate_severity(0.85) == "medium"
    assert evaluate_severity(0.999) == "medium"
    assert evaluate_severity(1.0) == "medium"


def test_severity_can_never_exceed_medium():
    """Severity must never return 'high' or 'critical' under any input."""
    for prob in [0.80, 0.95, 0.99, 1.0, 1.5]:
        severity = evaluate_severity(prob)
        assert severity in ["none", "low", "medium"]
        assert severity != "high"


def test_clamped_even_if_config_set_to_high(monkeypatch):
    """If an environment variable or config attempts to set MAX_SEVERITY to 'high', it must clamp to 'medium'."""
    monkeypatch.setattr(settings, "MAX_SEVERITY", "high")
    # Calling evaluate_severity should still clamp to 'medium'
    severity = evaluate_severity(0.95)
    assert severity == "medium"
    assert severity != "high"


def test_policy_invariants_metadata():
    """Confirms all three architectural safety invariants are strictly False."""
    meta = get_policy_metadata()
    assert meta["can_trigger_high"] is False
    assert meta["can_mark_verified"] is False
    assert meta["can_override_rule"] is False
