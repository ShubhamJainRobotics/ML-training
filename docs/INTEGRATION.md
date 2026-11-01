# Jaanch ML Service Integration Guide

This guide explains how to integrate the self-contained Jaanch ML inference service into the main Jaanch web application (or any client application).

---

## Architecture & Role

The Jaanch ML service provides a **supplementary statistical signal** using an explainable, calibrated TF-IDF + Logistic Regression model (`jaanch_scam_model_v2_calibrated.joblib`).

> [!IMPORTANT]
> **Strict Safety Invariants**
> 1. **Capped Severity**: ML severity is strictly capped at `medium`. It can **NEVER** return `high`.
> 2. **Never Triggers High Risk**: The ML signal alone must never elevate an overall report to `High Risk`.
> 3. **Never Marks Verified**: A low scam score indicates lack of scam signals, **NOT** authenticity. It must never mark a posting as `Verified` or `Safe`.
> 4. **Never Overrides Rules**: Deterministic rule findings (e.g., direct payment requests, fraudulent domains) always take precedence. The ML model cannot override, dismiss, or suppress any rule or Gemini finding.
> 5. **Fail-Open Strategy**: If this service is down, times out, or errors, the main application continues uninterrupted.

---

## 1. Environment Configuration

Add the following environment variables to your main Jaanch web application (`.env.local` or hosting provider dashboard):

```bash
# Jaanch ML Inference Service Endpoint
ML_API_URL=https://jaanch-ml-service.onrender.com

# Shared API authentication key (must match API_KEY in jaanch-ml-service)
ML_API_KEY=your_secure_shared_api_key_here

# Request timeout in milliseconds (fail-open if exceeded)
ML_TIMEOUT_MS=3000
```

---

## 2. API Contract

### Request: `POST /ml/signal`

- **Headers**:
  - `Content-Type: application/json`
  - `X-API-Key: <ML_API_KEY>` (required when configured on server)
- **Body**:
  ```json
  {
    "text": "Full job posting text or description to evaluate...",
    "consent": true
  }
  ```
  *(Note: If `consent` is `false` or missing, the API responds with `403 Forbidden` and `CONSENT_REQUIRED`).*

### Response: `200 OK`

```json
{
  "source": "ml",
  "signal": "possible_scam_pattern",
  "severity": "medium",
  "confidence": 0.842,
  "top_contributing_terms": ["registration", "fee", "telegram", "deposit", "daily"],
  "model_version": "tfidf-logreg-v2-calibrated",
  "policy": {
    "can_trigger_high": false,
    "can_mark_verified": false,
    "can_override_rule": false
  }
}
```

---

## 3. TypeScript Integration Example (Next.js / Node.js)

Below is a production-ready client utility for the main Jaanch application (`lib/mlClient.ts`):

```typescript
// lib/mlClient.ts

export interface MLSignalResult {
  source: 'ml';
  signal: 'possible_scam_pattern' | null;
  severity: 'none' | 'low' | 'medium';
  confidence: number;
  top_contributing_terms: string[];
  model_version: string;
  policy: {
    can_trigger_high: boolean;
    can_mark_verified: boolean;
    can_override_rule: boolean;
  };
}

export async function fetchMLSignal(
  text: string,
  userConsent: boolean
): Promise<MLSignalResult | null> {
  // If user has not consented, skip ML signal completely
  if (!userConsent) {
    return null;
  }

  const mlUrl = process.env.ML_API_URL;
  if (!mlUrl) {
    console.warn('[Jaanch ML] ML_API_URL not set; skipping supplementary ML signal.');
    return null;
  }

  const apiKey = process.env.ML_API_KEY || '';
  const timeoutMs = parseInt(process.env.ML_TIMEOUT_MS || '3000', 10);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${mlUrl}/ml/signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
      body: JSON.stringify({
        text,
        consent: true,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(`[Jaanch ML] Service responded with HTTP ${response.status}`);
      return null;
    }

    const data: MLSignalResult = await response.json();

    // Defensive enforcement: never allow severity to exceed medium
    if (data.severity as string === 'high') {
      data.severity = 'medium';
    }

    return data;
  } catch (error: any) {
    // Fail-open: log error warning and return null so verification continues
    if (error.name === 'AbortError') {
      console.warn(`[Jaanch ML] Inference timed out after ${timeoutMs}ms (fail-open)`);
    } else {
      console.warn(`[Jaanch ML] Connection failed: ${error.message} (fail-open)`);
    }
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}
```

---

## 4. Aggregating Results with Rule Engine & Gemini

When consolidating findings into the final Jaanch safety score:

```typescript
// Example aggregator logic in your scan API route
export function aggregateFindings(
  ruleFindings: Finding[],
  geminiReview: GeminiFinding | null,
  mlSignal: MLSignalResult | null
): FinalReport {
  const allFindings = [...ruleFindings];

  if (geminiReview) {
    allFindings.push(geminiReview);
  }

  // Incorporate ML finding only if it detected a signal
  if (mlSignal && mlSignal.severity !== 'none') {
    allFindings.push({
      id: 'ml_statistical_anomaly',
      source: 'ml',
      title: 'Statistical Pattern Match',
      description: `Language patterns in this posting match known job scam profiles with ${(mlSignal.confidence * 100).toFixed(1)}% confidence. Key trigger terms: ${mlSignal.top_contributing_terms.slice(0, 3).join(', ')}.`,
      severity: mlSignal.severity, // 'low' or 'medium' only
    });
  }

  // Derive final risk level:
  // Rule findings or Gemini can set 'High Risk'
  // ML signal ALONE can at most produce 'Moderate Risk'
  const hasHighRiskRule = ruleFindings.some((f) => f.severity === 'high');
  const hasMediumRiskRule = ruleFindings.some((f) => f.severity === 'medium');

  let riskLevel: 'Safe' | 'Low Risk' | 'Moderate Risk' | 'High Risk';

  if (hasHighRiskRule) {
    riskLevel = 'High Risk';
  } else if (hasMediumRiskRule || (mlSignal && mlSignal.severity === 'medium')) {
    riskLevel = 'Moderate Risk';
  } else if (mlSignal && mlSignal.severity === 'low') {
    riskLevel = 'Low Risk';
  } else {
    riskLevel = 'Safe';
  }

  return {
    riskLevel,
    findings: allFindings,
    mlSignal,
  };
}
```

---

## 5. Security & Privacy Checklist

- [x] **Server-to-Server Only**: Always call `ML_API_URL` from your backend server or API routes. Never expose `ML_API_KEY` in frontend client bundles.
- [x] **PII Redaction**: The ML service automatically strips emails, Indian/international phone numbers, URLs, UPI handles, and identity numbers (Aadhaar/PAN/OTP).
- [x] **Zero Persistent Storage**: The ML service does not store job texts, query logs, or user identifiers to any database or disk.
- [x] **Consent Requirement**: Explicit consent is required per invocation. Ensure user consent toggle is checked before calling.
