# Jaanch ML Service - Git Release & Deployment Report

**Date:** November 1, 2026  
**Target Repository:** `Repo B` (`jaanch-ml-service`)  
**Remote URL:** `https://github.com/ShubhamJainRobotics/ML-training.git`  
**Branch:** `main`  
**Status:** Successfully Pushed  

---

## 1. Summary of Changes

This release promotes the **Jaanch Scam Detection Model v3 (Calibrated)** to production within the standalone ML inference microservice (`jaanch-ml-service`). It updates feature extraction to support `FeatureUnion` (word + char n-grams) with backward compatibility for legacy single-vectorizer models, adapts test suites, and preserves zero-downtime rollback capability.

---

## 2. Commits Pushed to Remote

| Commit Hash | Author | Message |
|-------------|--------|---------|
| `dce743c` | ML Engineering Agent | `feat(ml): promote calibrated scam detection model v3 and update feature extractor` |
| `9a7110b` | ML Engineering Agent | `test(ml): update test suite assertions for model v3 compatibility` |

**Git Push Output:**
```text
To https://github.com/ShubhamJainRobotics/ML-training.git
   86207c7..9a7110b  main -> main
```

---

## 3. Files Committed

| File | Status | Description |
|------|--------|-------------|
| `models/jaanch_scam_model_v3_calibrated.joblib` | Added (25.69 MB) | Production calibrated LogisticRegression + FeatureUnion model artifact |
| `models/MODEL_VERSION.txt` | Modified | Updated model metadata, version string, training set version, and SHA-256 |
| `app/config.py` | Modified | Updated default `MODEL_PATH` and `MODEL_VERSION` to v3 |
| `app/model_service.py` | Modified | Enhanced feature extraction to support both `features` (`FeatureUnion`) and `tfidf` steps; cleans prefix names for token explanations |
| `.env.example` | Modified | Updated environment configuration defaults for v3 |
| `tests/test_api.py` | Modified | Updated test assertions to dynamically match active `MODEL_VERSION` |

---

## 4. Excluded Files & Safety Verification

- **Root Repository Untouched:** All changes were isolated strictly to `Repo B` (`jaanch-ml-service`). The root workspace (`JaanchAI.git`) remains uncommitted and unaffected.
- **Secrets & Credentials:** No `.env` or production secrets were committed (`.env.example` verified clean).
- **Large Files & LFS:** The model artifact size is **25.69 MB**, well below GitHub's 50 MB warning threshold and 100 MB hard limit. Git LFS was not required.
- **Data Privacy:** No raw training datasets, customer data, or PII were staged.

---

## 5. Model v3 Specifications & Verification

- **Artifact Name:** `jaanch_scam_model_v3_calibrated.joblib`
- **File Size:** 25,690,443 bytes (~25.69 MB)
- **SHA-256 Checksum:** `30D71C425703E4A4089C59A6FBBD200B20C83ABE027F3E365E2F556C6CCD0FD8`
- **Architecture:** Word TF-IDF (1-2 n-grams) + Char_wb TF-IDF (3-5 n-grams) $\to$ `LogisticRegression(C=0.8, class_weight='balanced')` $\to$ `CalibratedClassifierCV(method='sigmoid', cv=5)`
- **Policy Compliance:** ML risk score remains capped at **Medium** severity. High Risk is strictly reserved for deterministic rule engines.

---

## 6. Pre-Commit Test Suite Results

All tests executed via `pytest tests` and passed cleanly:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Shubkacomputer\OneDrive\Desktop\JAANCH codemeet\jaanch-ml-service
plugins: anyio-4.15.1, platformdirs-4.14.2
collected 22 items

tests\test_api.py ........                                               [ 36%]
tests\test_policy.py ......                                              [ 63%]
tests\test_redaction.py ........                                         [100%]

============================= 22 passed in 14.62s =============================
```

---

## 7. Zero-Downtime Rollback Instructions

The previous production model `models/jaanch_scam_model_v2_calibrated.joblib` is retained in the repository. If any regression occurs in production, rollback can be performed immediately without code changes via environment variables:

```bash
# In your Render or production deployment environment variables:
MODEL_PATH=models/jaanch_scam_model_v2_calibrated.joblib
MODEL_VERSION=tfidf-logreg-v2-calibrated
```

Restart the FastAPI service, and it will immediately serve using Model v2.
