# Jaanch ML Service - Git Release & Deployment Report

**Date:** November 1, 2026  
**Target Repository:** `ML-training` (`jaanch-ml-service`)  
**Remote URL:** `https://github.com/ShubhamJainRobotics/ML-training.git`  
**Branch:** `main`  
**Latest Remote Commit:** `4ce482a`  
**Status:** Successfully Pushed & Fully Up-to-Date  

---

## 1. Summary of Changes

This repository now contains the production **Jaanch Scam Detection Model v3-Tuned (Calibrated)**, complete model card and evaluation benchmarks, inference microservice logic with automatic `.env` loading, and all reproducible training scripts, notebooks, and live monitors.

---

## 2. Complete Commit History Pushed to Remote

| Commit Hash | Scope | Message |
|-------------|-------|---------|
| `4ce482a` | `feat(training)` | `add model v3 tuning notebook, training scripts, and live monitor` |
| `263d057` | `docs(ml)` | `add v3-tuned model evaluation report and tuning workflow` |
| `914b202` | `feat(ml)` | `promote calibrated scam detection model v3-tuned and update service configuration` |
| `34443a1` | `docs` | `add release and git push report for model v3 deployment` |
| `9a7110b` | `test(ml)` | `update test suite assertions for model v3 compatibility` |
| `dce743c` | `feat(ml)` | `promote calibrated scam detection model v3 and update feature extractor` |
| `86207c7` | `chore` | `Normalize line endings` |
| `aa86744` | `feat(ml)` | `Add reusable Jaanch ML inference service` |

---

## 3. Pushed Files Breakdown

### A. Production Model Artifacts & Metadata
- `models/jaanch_scam_model_v3_tuned_calibrated.joblib` (40.33 MB)  
  *SHA-256:* `9D244527840F65F35E96D606FFCEA5508C151EC57E1135385AAD925D93DC470E`
- `models/MODEL_VERSION.txt` (Version: `tfidf-logreg-v3-tuned-calibrated`, Policy: `jaanch_conservative_triage_v3_tuned`)
- `models/jaanch_scam_model_v3_calibrated.joblib` (25.69 MB, Retained for immediate zero-downtime rollback)
- `models/jaanch_scam_model_v2_calibrated.joblib` (18.74 MB, Retained for backward compatibility)

### B. Inference Microservice & Configuration
- `app/config.py`: Integrated `python-dotenv` for automatic environment loading; defaulted to v3-tuned.
- `app/model_service.py`: Strips `word__` and `char__` prefixes from token explanations; extracts FeatureUnion coefficients cleanly.
- `render.yaml`: Updated deployment blueprint for v3-tuned.
- `.env.example`: Updated environment template with v3-tuned defaults.
- `.gitignore`: Configured to preserve `training/` scripts and notebooks while strictly ignoring raw data files and secrets.
- `README.md`: Documented updated repository structure including `training/` and `docs/`.

### C. Documentation & Benchmarks
- `docs/MODEL_V3_TUNED_EVALUATION.md`: Full 120-sample untouched test set benchmark report (99.17% accuracy, 100% recall, 1.67% FPR).
- `docs/TUNING_WORKFLOW.md`: Step-by-step documentation of hyperparameter optimization, custom scorer, and calibration.

### D. Training Scripts & Tooling (`training/`)
- `training/16_train_model_v3_tuned.ipynb`: Kaggle-ready tuning notebook.
- `training/run_v3_tuning.py`: Local hyperparameter optimization script.
- `training/training_monitor.py`: Real-time training progress callback and metrics logger.
- `training/training_dashboard.py`: Streamlit live monitoring dashboard.
- `training/v3_tuned_best_hyperparameters.json`: Optimal hyperparameter parameters.
- `training/v3_tuned_threshold_policy.json`: Conservative triage threshold policy.

---

## 4. Safety, Test & Quality Verifications

- **Pre-Commit Tests:** All 22 test cases passed (`tests/test_api.py`, `tests/test_policy.py`, `tests/test_redaction.py`).
- **Secrets Check:** `.env` remains ignored and excluded.
- **Root Repository Safety:** Workspace root (`JaanchAI.git`) remains uncommitted and unaffected.
- **GitHub File Size Limit:** The largest file is `40.33 MB`, safely under the 50 MB GitHub warning and 100 MB hard limit.

---

## 5. Rollback Procedures

If rolling back to Model v3 or Model v2 is ever required:
```bash
# In production environment variables:
MODEL_PATH=models/jaanch_scam_model_v3_calibrated.joblib
MODEL_VERSION=tfidf-logreg-v3-calibrated
```
Restart the service to switch models without downtime or code rebuilds.
