# Jaanch ML Service (`jaanch-ml-service`)

> Lightweight, self-contained, and explainable ML inference microservice for **Jaanch**, an online job-posting scam detector.

---

## Overview

The `jaanch-ml-service` provides a supplementary statistical scam signal using an explainable, calibrated TF-IDF + Logistic Regression model. It operates as an independent microservice that can be deployed anywhere (e.g., Render, Railway, Fly.io, Docker) and consumed by the Jaanch web application.

### Key Architectural Invariants

- **Supplementary Signal Only**: Never replaces deterministic rule checks or Gemini LLM review.
- **Severity Hard-Capped at `Medium`**: ML signal can only evaluate to `none`, `low`, or `medium`. It **never** outputs `high`.
- **Never Triggers High Risk**: An overall report can never be escalated to High Risk based solely on ML.
- **Never Marks Verified**: Absence of scam patterns does not imply a job is legitimate.
- **Never Overrides Rules**: Deterministic rule findings always take precedence.
- **Fail-Open Design**: If the ML service is down or times out, the main Jaanch application proceeds seamlessly.
- **Privacy First**: Strips emails, phone numbers, URLs, UPI IDs, Aadhaar, PAN, and OTPs via regex redaction before inference.

---

## Repository Structure

```text
jaanch-ml-service/
├── app/
│   ├── __init__.py           # Package marker
│   ├── main.py               # FastAPI application, routes, and lifespan handler
│   ├── config.py             # Settings, environment variables, and defaults
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── redaction.py          # PII redaction (Email, Phone, URL, UPI, Aadhaar, PAN)
│   ├── model_service.py      # Model loader, predictor, and feature explainer
│   └── policy.py             # Threshold evaluation and severity clamping
├── models/
│   ├── .gitkeep              # Git tracking
│   ├── MODEL_VERSION.txt     # Model checksum (SHA256), version, and metadata
│   ├── jaanch_scam_model_v2_calibrated.joblib        # Baseline calibrated model (~18 MB)
│   ├── jaanch_scam_model_v3_calibrated.joblib        # v3 calibrated model (~25 MB)
│   └── jaanch_scam_model_v3_tuned_calibrated.joblib  # Production v3-Tuned calibrated model (~40 MB)
├── training/
│   ├── 16_train_model_v3_tuned.ipynb                 # Kaggle-ready tuning notebook
│   ├── run_v3_tuning.py                              # Local hyperparameter search & calibration script
│   ├── training_monitor.py                           # Real-time training metrics & logging callback
│   ├── training_dashboard.py                         # Streamlit live monitoring dashboard
│   ├── v3_tuned_best_hyperparameters.json            # Optimal hyperparameter search result
│   └── v3_tuned_threshold_policy.json                # Conservative triage threshold policy
├── tests/
│   ├── __init__.py           # Test package
│   ├── test_redaction.py     # PII redaction unit tests
│   ├── test_policy.py        # Severity threshold & clamping tests
│   └── test_api.py           # FastAPI endpoint and auth integration tests
├── docs/
│   ├── INTEGRATION.md                 # Full integration guide with Next.js / TypeScript code
│   ├── MODEL_V3_TUNED_EVALUATION.md   # Untouched test set benchmark & comparison report
│   └── TUNING_WORKFLOW.md             # End-to-end tuning pipeline documentation
├── .env.example              # Environment variable template
├── .gitignore                # Clean exclusions (no raw datasets or notebooks)
├── Dockerfile                # Production Dockerfile (non-root user, healthcheck)
├── render.yaml               # Render Blueprint deployment definition
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.14)
- Git

### 2. Installation

```bash
# Navigate to the service folder
cd jaanch-ml-service

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the template:
```bash
cp .env.example .env
```

Default variables:
```bash
ENVIRONMENT=development
API_KEY=
MODEL_PATH=models/jaanch_scam_model_v2_calibrated.joblib
MODEL_VERSION=tfidf-logreg-v2-calibrated
LOW_THRESHOLD=0.50
MEDIUM_THRESHOLD=0.75
MAX_SEVERITY=medium
PORT=8000
```

### 4. Run Development Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## Running Automated Tests

Run the full test suite using `pytest`:

```bash
pytest
```

The test suite covers:
- **PII Redaction**: Email, Indian/international phone numbers, URLs, UPI handles, Aadhaar, PAN, and OTP masking.
- **Threshold Policy**: Correct band assignment (`none`, `low`, `medium`) and strict enforcement of the `medium` severity ceiling.
- **API Endpoints**: `/health` status, consent validation (403 when false), API key authentication (401), response contract verification, and error shielding without stack trace leakage.

---

## API Endpoints

### 1. Health Check: `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "tfidf-logreg-v2-calibrated"
}
```

### 2. Inference Signal: `POST /ml/signal`

**Request:**
```bash
curl -X POST http://localhost:8000/ml/signal \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Urgent hiring! Work from home on Telegram tasks. Pay 500 registration fee via UPI.",
    "consent": true
  }'
```

**Response (`200 OK`):**
```json
{
  "source": "ml",
  "signal": "possible_scam_pattern",
  "severity": "medium",
  "confidence": 0.8421,
  "model_version": "tfidf-logreg-v2-calibrated",
  "top_contributing_terms": ["registration", "fee", "telegram", "deposit", "daily"],
  "policy": {
    "can_trigger_high": false,
    "can_mark_verified": false,
    "can_override_rule": false
  }
}
```

---

## Deployment to Render

This repository includes a `render.yaml` blueprint and a production-grade `Dockerfile`.

### Method A: One-Click Blueprint
1. Push this repository to GitHub.
2. In the Render Dashboard, click **New +** -> **Blueprint**.
3. Connect this repository; Render will automatically detect `render.yaml` and configure the web service.

### Method B: Manual Docker Service
1. Create a **New Web Service** on Render and select your GitHub repository.
2. Choose **Docker** as the Environment.
3. Configure the following environment variables:
   - `ENVIRONMENT`: `production`
   - `API_KEY`: Generate a random secure token (e.g. `openssl rand -hex 24`)
   - `MODEL_PATH`: `models/jaanch_scam_model_v2_calibrated.joblib`
   - `MODEL_VERSION`: `tfidf-logreg-v2-calibrated`
   - `CORS_ORIGINS`: Your Jaanch frontend domain (e.g., `https://jaanch.vercel.app`)
4. Set Health Check path to `/health`.
5. Deploy!

---

## Updating the Model Artifact

To update to a new model version (e.g., `v3`):
1. Place the new joblib file in `models/` (e.g., `jaanch_scam_model_v3_calibrated.joblib`).
2. Calculate its SHA256 checksum:
   ```bash
   certutil -hashfile models/jaanch_scam_model_v3_calibrated.joblib SHA256
   ```
3. Update `models/MODEL_VERSION.txt` with the new file name, version string, date, and checksum.
4. Update `MODEL_PATH` and `MODEL_VERSION` in your `.env` or deployment environment variables.
5. Run tests (`pytest`) to verify backward compatibility.

---

## Integration Guide

For full instructions and ready-to-use TypeScript code for connecting this service into the main Jaanch frontend and aggregator, see [docs/INTEGRATION.md](file:///c:/Users/Shubkacomputer/OneDrive/Desktop/JAANCH%20codemeet/jaanch-ml-service/docs/INTEGRATION.md).

