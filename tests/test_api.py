"""
Unit and integration tests for FastAPI endpoints in jaanch-ml-service.
Tests:
- GET /health
- POST /ml/signal (consent checks, authentication, response structure, severity cap)
- Safe error handling without stack traces
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app, model_service
from app.config import get_settings


@pytest.fixture
def client():
    """Create test client with default test settings (no API key required)."""
    settings = get_settings()
    settings.API_KEY = ""
    settings.ENVIRONMENT = "test"
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_success(client):
    """Test that /health returns status ok and model_loaded status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "calibrated" in data["model_version"]


def test_ml_signal_requires_consent(client):
    """Test that POST /ml/signal rejects requests without consent (HTTP 403)."""
    payload = {
        "text": "Earn Rs 50,000 per day working from home on Telegram tasks.",
        "consent": False
    }
    response = client.post("/ml/signal", json=payload)
    assert response.status_code == 403
    data = response.json()
    assert "detail" in data
    assert "consent" in data["detail"].lower()


def test_ml_signal_valid_request_contract(client):
    """Test POST /ml/signal returns the exact required schema and invariants."""
    payload = {
        "text": "Urgent hiring! Pay registration fee of Rs 500 via Google Pay / UPI to HR and start immediately.",
        "consent": True
    }
    response = client.post("/ml/signal", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Exact contract assertions
    assert data["source"] == "ml"
    assert data["signal"] in ["possible_scam_pattern", None]
    assert data["severity"] in ["none", "low", "medium"]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["top_contributing_terms"], list)
    assert data["model_version"] == get_settings().MODEL_VERSION

    # Policy invariants verification
    policy = data["policy"]
    assert policy["can_trigger_high"] is False
    assert policy["can_mark_verified"] is False
    assert policy["can_override_rule"] is False


def test_ml_signal_severity_never_exceeds_medium(client):
    """Test that severity is strictly capped at medium even when confidence is near 1.0."""
    with patch.object(model_service.model, "predict_proba", return_value=[[0.0001, 0.9999]]):
        payload = {
            "text": "Deposit money immediately to our Telegram group.",
            "consent": True
        }
        response = client.post("/ml/signal", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["confidence"] == 0.9999
        assert data["signal"] == "possible_scam_pattern"
        assert data["severity"] == "medium"
        assert data["severity"] != "high"
        assert data["policy"]["can_trigger_high"] is False


def test_ml_signal_low_confidence_produces_none_severity(client):
    """Test that text below low threshold produces 'none' signal and severity."""
    with patch.object(model_service.model, "predict_proba", return_value=[[0.80, 0.20]]):
        payload = {
            "text": "Looking for a Senior Python Engineer with 5+ years experience in Django.",
            "consent": True
        }
        response = client.post("/ml/signal", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["confidence"] == 0.20
        assert data["signal"] is None
        assert data["severity"] == "none"


def test_ml_signal_intermediate_confidence_produces_low_severity(client):
    """Test that confidence between 0.50 and 0.75 produces 'low' severity."""
    with patch.object(model_service.model, "predict_proba", return_value=[[0.38, 0.62]]):
        payload = {
            "text": "Work from home data entry position open.",
            "consent": True
        }
        response = client.post("/ml/signal", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["confidence"] == 0.62
        assert data["signal"] == "possible_scam_pattern"
        assert data["severity"] == "low"


def test_api_key_authentication_enforced():
    """Test that when API_KEY is set, unauthorized requests receive 401."""
    settings = get_settings()
    settings.API_KEY = "super-secret-jaanch-key"
    settings.ENVIRONMENT = "production"

    with TestClient(app) as auth_client:
        # 1. Missing header
        res_missing = auth_client.post("/ml/signal", json={"text": "Hello", "consent": True})
        assert res_missing.status_code == 401
        assert "Unauthorized" in res_missing.json()["detail"]

        # 2. Invalid header
        res_invalid = auth_client.post(
            "/ml/signal",
            json={"text": "Hello", "consent": True},
            headers={"X-API-Key": "wrong-key"}
        )
        assert res_invalid.status_code == 401

        # 3. Valid header
        res_valid = auth_client.post(
            "/ml/signal",
            json={"text": "Hello", "consent": True},
            headers={"X-API-Key": "super-secret-jaanch-key"}
        )
        assert res_valid.status_code == 200

    # Reset API key after test
    settings.API_KEY = ""
    settings.ENVIRONMENT = "test"


def test_error_handling_no_stack_trace_leak(client):
    """Test that internal exceptions return safe 500 error without exposing stack traces."""
    with patch.object(model_service, "predict", side_effect=RuntimeError("Internal memory fault in C extension")):
        payload = {
            "text": "Some text to process",
            "consent": True
        }
        response = client.post("/ml/signal", json=payload)
        assert response.status_code == 500
        data = response.json()

        assert "error" in data
        assert data["error"] == "InferenceError"
        assert "stack" not in data
        assert "Traceback" not in str(data)
        assert "C extension" not in str(data)  # Detailed internal leak masked
