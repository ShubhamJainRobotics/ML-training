# Model Comparison Report: v3-Current vs v3-Tuned
**Evaluation Date**: 2026-11-01 18:48:39  
**Baseline Model**: `jaanch_scam_model_v3_calibrated.joblib`  
**Candidate Model**: `jaanch_scam_model_v3_tuned_calibrated.joblib`  
**Test Set**: Untouched Local Jaanch Test Set (`data/jaanch_local_test_set.csv`, N=120)  
**Threshold Policy**: Low = 0.50, Medium = 0.75, Capped at Medium  
**Recommendation**: `deploy_v3_tuned_with_monitoring`

---

## 1. Executive Summary & Benchmark Metrics

| Evaluation Dimension | v3-Current Baseline | v3-Tuned Candidate | Absolute Delta | Trend / Status |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy (N=120)** | 0.9833 | **0.9917** | +0.0084 | Stable |
| **Precision** | 0.9677 | **0.9836** | +0.0159 | Stable |
| **Scam Recall (60/60)** | 1.0000 (60/60) | **1.0000 (60/60)** | +0.0000 | **PASS (100% Retained)** |
| **F1-Score** | 0.9836 | **0.9917** | +0.0081 | Stable |
| **False Positives (on genuine)** | 2 (FPR: 3.33%) | **1 (FPR: 1.67%)** | -1 | **PASS (Zero Regression)** |
| **False Negatives (on scam)** | 0 | **0** | 0 | **PASS (Zero Missed Scams)** |
| **ROC-AUC** | 0.9989 | **1.0000** | +0.0011 | Preserved |
| **Brier Calibration Score** | 0.0229 | **0.0058** | -0.0171 | Well Calibrated |

---

## 2. Category Breakdown on Untouched Local Test Set

| Category | Type | N | Current Pass Rate | Tuned Pass Rate | Current Mean Prob | Tuned Mean Prob | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fake_hr_scam` | scam | 12 | 1.0000 | 1.0000 | 0.8361 | **0.9631** | +0.1270 |
| `genuine_entry_level_job` | genuine | 20 | 0.9000 | 0.9500 | 0.1640 | **0.0787** | -0.0853 |
| `genuine_internship` | genuine | 20 | 1.0000 | 1.0000 | 0.0745 | **0.0398** | -0.0347 |
| `genuine_remote_job` | genuine | 20 | 1.0000 | 1.0000 | 0.0135 | **0.0061** | -0.0074 |
| `registration_fee_scam` | scam | 12 | 1.0000 | 1.0000 | 0.8841 | **0.9846** | +0.1005 |
| `telegram_task_scam` | scam | 12 | 1.0000 | 1.0000 | 0.8840 | **0.9888** | +0.1047 |
| `upi_deposit_scam` | scam | 12 | 1.0000 | 1.0000 | 0.9130 | **0.9920** | +0.0791 |
| `whatsapp_hr_scam` | scam | 12 | 1.0000 | 1.0000 | 0.9234 | **0.9992** | +0.0758 |

---

## 3. Subgroup Performance: Short Messages & Hinglish

- **Short Messages (<150 characters, N=38)**:
  - v3-Current Accuracy: 1.0000 | F1: 1.0000
  - v3-Tuned Accuracy: **1.0000** | F1: **1.0000**
- **Hinglish Vocabulary Messages (N=4)**:
  - v3-Current Accuracy: 1.0000 | F1: 1.0000
  - v3-Tuned Accuracy: **1.0000** | F1: **1.0000**

---

## 4. Best Hyperparameter Configuration

Extensive 5-fold Stratified cross-validation with custom optimization scorer (0.7192):
```json
{
  "features__word__sublinear_tf": false,
  "features__word__ngram_range": [
    1,
    2
  ],
  "features__word__min_df": 1,
  "features__word__max_features": 80000,
  "features__char__sublinear_tf": true,
  "features__char__ngram_range": [
    3,
    5
  ],
  "features__char__min_df": 2,
  "features__char__max_features": 60000,
  "clf__C": 2.0
}
```

---

## 5. Deployment Recommendation & Rollback Governance

### Verdict: `DEPLOY_V3_TUNED_WITH_MONITORING`

**Justification**:
1. **Zero False Positive Regression**: The tuned model maintains identical low false positive count (1/60 genuine postings) preventing genuine student applicants from being falsely alarmed.
2. **100% Scam Recall Retained**: All 60 fraudulent postings in the untouched test set are successfully flagged.
3. **Calibrated Probabilities**: Sigmoid post-processing guarantees well-behaved probability margins (Brier score 0.0058).
4. **Safety Compliance**: The model strictly functions as a supplementary triage signal capped at `medium` severity.

### Rollback Plan
If anomalies arise under production traffic:
1. Set environment variables:
   ```bash
   MODEL_PATH=models/jaanch_scam_model_v3_calibrated.joblib
   MODEL_VERSION=tfidf-logreg-v3-calibrated
   ```
2. Restart the FastAPI service container.
