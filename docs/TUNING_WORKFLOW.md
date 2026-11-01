# Jaanch Model v3 Hyperparameter Tuning & Evaluation Workflow

## 1. Executive Summary & Optimization Mission

This workflow executes comprehensive hyperparameter optimization for the **Jaanch job-scam detector**, upgrading Model v3 to **Model v3-Tuned** using **strictly the existing training pool** without introducing any new or augmented data.

### Core Governance & Safety Constraints
- **Zero New Data**: No data was added, generated, scraped, augmented, or synthesized. Training utilized solely `data/jaanch_training_pool_v3.csv` ($N=16,264$).
- **Strict Zero-Leakage Guarantee**: The held-out evaluation set (`data/jaanch_local_test_set.csv`, $N=120$) remained completely isolated during preprocessing, cross-validation, hyperparameter selection, and threshold tuning. It was evaluated strictly once at the final comparison step.
- **Supplementary Signal Constraint**: ML model predictions are hard-capped at `medium` severity. The model **never** triggers High Risk, never marks a posting Verified, and cannot override or dismiss rule-engine findings.
- **Custom Optimization Metric**: Optimization utilized a custom objective function prioritizing high scam recall and near-zero false-positive rate on genuine postings rather than naive accuracy.

---

## 2. Step-by-Step Technical Execution

### Step 1: Data Audit & Validation
- **Source**: `data/jaanch_training_pool_v3.csv` (16,264 records).
- **Quality Filtering**: Enforced text length $\ge 50$ characters, stripped whitespace, removed invalid labels and exact duplicates.
- **Zero-Leakage Verification**: Verified zero overlap with the 120 texts in `data/jaanch_local_test_set.csv`.
- **Partitioning**: Stratified 85/15 train-validation split (Train: $13,824$ records with $683$ scams; Validation: $2,440$ records with $121$ scams; random_state=42).
- **Report**: Generated `reports/v3_tuned_data_audit.md`.

### Step 2: Tunable Dual-Branch Architecture
- **Pipeline Structure**:
  - `features`: `FeatureUnion` combining:
    - `word`: Word-level `TfidfVectorizer(stop_words='english')`
    - `char`: Sub-word character boundary `TfidfVectorizer(analyzer='char_wb')`
  - `clf`: `LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)`

### Step 3: Hyperparameter Search & Custom Scorer
- **Custom Jaanch Scorer**:
  $$\text{Score} = 0.45 \times F_1 + 0.45 \times \text{Recall} - 2.0 \times \text{FPR}_{\text{genuine}}$$
  *Penalizes false alarms on student job applications while aggressively rewarding complete scam detection.*
- **Search Strategy**: `RandomizedSearchCV` across 40 iterations with 5-fold Stratified Cross-Validation ($200$ fit iterations) using all available CPU threads (`n_jobs=-1`).
- **Optimal Hyperparameters Found** (CV Custom Score: $0.7192$):
  ```json
  {
    "clf__C": 2.0,
    "features__word__max_features": 80000,
    "features__word__min_df": 1,
    "features__word__ngram_range": [1, 2],
    "features__word__sublinear_tf": false,
    "features__char__max_features": 60000,
    "features__char__min_df": 2,
    "features__char__ngram_range": [3, 5],
    "features__char__sublinear_tf": true
  }
  ```
- **Report**: Saved `reports/v3_tuned_best_hyperparameters.json`.

### Step 4: Probability Calibration
- Re-fitted the optimal pipeline on the full training partition ($N=13,824$).
- Calibrated using `CalibratedClassifierCV(estimator=best_pipeline, method="sigmoid", cv=5)`.
- **Exported Artifacts**:
  - `artifacts/models/jaanch_scam_model_v3_tuned.joblib` (Uncalibrated base pipeline: 6.41 MB)
  - `artifacts/models/jaanch_scam_model_v3_tuned_calibrated.joblib` (Calibrated model: 38.46 MB, SHA256: `9D244527840F65F35E96D606FFCEA5508C151EC57E1135385AAD925D93DC470E`)

### Step 5: Decision Threshold Policy (Validation Split)
Evaluated candidate thresholds from $0.30$ to $0.90$ in increments of $0.05$ on the validation split ($N=2,440$):
- **Threshold 0.50**: Precision: $89.69\%$ | Recall: $71.90\%$ | FPR on genuine: $0.43\%$ (10 false alarms out of 2,319 genuine).
- **Threshold 0.75**: Precision: $92.86\%$ | Recall: $64.46\%$ | FPR on genuine: $0.26\%$ (6 false alarms out of 2,319 genuine).
- **Policy Established**:
  - Low Risk Triage Threshold: `0.50`
  - Medium Risk Warning Threshold: `0.75`
  - Maximum Severity: `medium` (Hard ceiling)
- **Reports**: Saved `reports/v3_tuned_threshold_analysis.csv` and `reports/v3_tuned_threshold_policy.json`.

---

## 3. Final Held-Out Evaluation: Model v3 vs Model v3-Tuned

Evaluated strictly once on the untouched local Jaanch test set (`data/jaanch_local_test_set.csv`, $N=120$):

| Metric / Dimension | v3-Current Baseline | v3-Tuned Candidate | Delta | Trend |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy (N=120)** | 0.9833 | **0.9917** | **+0.0084** | **Improved** |
| **Precision** | 0.9677 | **0.9836** | **+0.0159** | **Improved** |
| **Scam Recall (60/60)** | 1.0000 (60/60) | **1.0000 (60/60)** | +0.0000 | **PASS (100% Retained)** |
| **F1-Score** | 0.9836 | **0.9917** | **+0.0081** | **Improved** |
| **False Positives (on genuine)** | 2 (FPR: 3.33%) | **1 (FPR: 1.67%)** | **-1** | **PASS (Halved)** |
| **False Negatives (on scam)** | 0 | **0** | 0 | **PASS (Zero Misses)** |
| **ROC-AUC** | 0.9989 | **1.0000** | **+0.0011** | **Perfect Discriminability** |
| **Brier Calibration Score** | 0.0229 | **0.0058** | **-0.0171** | **75% Calibration Error Reduction** |

### Category-Level Mean Probability Separation

| Category | Type | N | Current Mean Prob | Tuned Mean Prob | Probability Delta | Impact of Tuning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fake_hr_scam` | scam | 12 | 0.8361 | **0.9631** | **+0.1270** | Strong separation on spoofed recruiter copy |
| `registration_fee_scam` | scam | 12 | 0.8841 | **0.9846** | **+0.1005** | High confidence activation on UPI/fee requests |
| `telegram_task_scam` | scam | 12 | 0.8840 | **0.9888** | **+0.1047** | Decisive detection of like/subscribe tasks |
| `upi_deposit_scam` | scam | 12 | 0.9130 | **0.9920** | **+0.0791** | Overwhelming confidence margin |
| `whatsapp_hr_scam` | scam | 12 | 0.9234 | **0.9992** | **+0.0758** | Pinpoint certainty on WhatsApp redirects |
| `genuine_entry_level_job` | genuine | 20 | 0.1640 | **0.0787** | **-0.0853** | Pass rate rose from 90% to **95%** (FP eliminated) |
| `genuine_internship` | genuine | 20 | 0.0745 | **0.0398** | **-0.0347** | Dropped deep into safe negative zone |
| `genuine_remote_job` | genuine | 20 | 0.0135 | **0.0061** | **-0.0074** | Absolute negative baseline |

### Subgroup Performance
- **Short Messages (<150 characters, N=38)**: Accuracy: **1.0000** | F1: **1.0000**
- **Hinglish Copy (N=4)**: Accuracy: **1.0000** | F1: **1.0000**

---

## 4. Promotion Recommendation & Rollback Governance

### Verdict: `DEPLOY_V3_TUNED_WITH_MONITORING`

#### Verification of Promotion Criteria:
1. **Fewer False Positives**: False positives halved from 2 to 1 on genuine job postings ($1.67\%$ FPR).
2. **Improved Category Separation**: Scam mean confidence increased by $+7.5\%$ to $+12.7\%$ across all Indian scam categories, while genuine posting probabilities dropped by up to $-8.5\%$.
3. **No F1 Regression**: Overall F1-score improved from $0.9836$ to $0.9917$.
4. **Superior Probability Calibration**: Brier calibration score dropped by $75\%$ ($0.0229 \rightarrow 0.0058$).

#### Rollback Plan
If production telemetry flags any anomaly:
1. Revert environment variables in `jaanch-ml-service`:
   ```bash
   MODEL_PATH=models/jaanch_scam_model_v3_calibrated.joblib
   MODEL_VERSION=tfidf-logreg-v3-calibrated
   ```
2. Restart the FastAPI service container. Both models share the identical input/output schema and `FeatureUnion` extractor.

---

## 5. Kaggle Execution Guide (Notebook 16)

The standalone notebook is available at `kaggle/16_train_model_v3_tuned.ipynb`.

### Execution Steps on Kaggle:
1. Go to [kaggle.com/code](https://www.kaggle.com/code) -> Click **New Notebook**.
2. **File** -> **Import Notebook** -> Upload `kaggle/16_train_model_v3_tuned.ipynb`.
3. In the right sidebar, under **Input**, click **Add Data**:
   - Upload `jaanch_training_pool_v3.csv` and `jaanch_local_test_set.csv`.
   - Upload existing baseline `jaanch_scam_model_v3_calibrated.joblib`.
4. Click **Run All Cells**. The notebook auto-discovers inputs in `/kaggle/input/` and writes outputs to `/kaggle/working/`.
5. Download artifacts from `/kaggle/working/`:
   - `jaanch_scam_model_v3_tuned_calibrated.joblib`
   - `v3_tuned_best_hyperparameters.json`
   - `v3_tuned_threshold_policy.json`
   - `v3_tuned_threshold_analysis.csv`
   - `reports/v3_vs_v3_tuned_curves.png`

---

## 6. Generated Files & Deliverables Index

- `kaggle/16_train_model_v3_tuned.ipynb` (Kaggle-ready notebook)
- `reports/v3_tuned_data_audit.md` (Dataset audit & zero-leakage verification)
- `reports/v3_tuned_best_hyperparameters.json` (Optimal hyperparameters & CV scores)
- `reports/v3_tuned_threshold_policy.json` (Calibrated triage threshold policy)
- `reports/v3_tuned_threshold_analysis.csv` (Validation threshold trade-off table)
- `reports/model_v3_vs_v3_tuned.md` (Formal comparative evaluation report)
- `reports/model_v3_vs_v3_tuned.json` (Structured JSON benchmark metrics)
- `reports/v3_vs_v3_tuned_curves.png` (ROC and PR comparison curves)
- `reports/v3_vs_v3_tuned_category_prob.png` (Category probability separation plot)
- `artifacts/models/jaanch_scam_model_v3_tuned.joblib` (Uncalibrated pipeline)
- `artifacts/models/jaanch_scam_model_v3_tuned_calibrated.joblib` (Calibrated candidate artifact)
- `README_V3_TUNED_WORKFLOW.md` (This guide)

