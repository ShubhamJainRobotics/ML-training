"""In-memory model service for deterministic, explainable scam detection."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

from app.config import settings
from app.redaction import redact_text
from app.policy import evaluate_severity, get_policy_metadata

logger = logging.getLogger("jaanch_model_service")


class ModelService:
    """Singleton service to load model once at startup and execute inference."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.is_loaded: bool = False
        self.tfidf = None
        self.clf = None

    def load_model(self) -> bool:
        """Loads the calibrated joblib pipeline once at startup."""
        model_path = settings.MODEL_PATH
        if not model_path.exists():
            logger.error(f"Model artifact not found at {model_path}")
            self.is_loaded = False
            return False

        try:
            logger.info(f"Loading calibrated model from {model_path}...")
            self.model = joblib.load(model_path)
            self._extract_feature_components()
            self.is_loaded = True
            logger.info(f"Model successfully loaded. Version: {settings.MODEL_VERSION}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model artifact: {e}")
            self.is_loaded = False
            return False

    def _extract_feature_components(self):
        """Extracts vectorizer/feature-union and base classifier coefficients for token explanations."""
        if self.model is None:
            return

        base_pipeline = None
        # Handle CalibratedClassifierCV
        if hasattr(self.model, "estimator"):
            base_pipeline = self.model.estimator
        elif hasattr(self.model, "calibrated_classifiers_") and len(self.model.calibrated_classifiers_) > 0:
            base_pipeline = self.model.calibrated_classifiers_[0].estimator
        else:
            base_pipeline = self.model

        if hasattr(base_pipeline, "named_steps"):
            self.tfidf = base_pipeline.named_steps.get("features") or base_pipeline.named_steps.get("tfidf")
            self.clf = base_pipeline.named_steps.get("clf")

    def explain_terms(self, redacted_text: str, top_k: int = 5) -> List[str]:
        """Identifies top positive scam-contributing n-grams for transparency."""
        if self.tfidf is None or self.clf is None:
            return []

        try:
            feature_names = np.array(self.tfidf.get_feature_names_out())
            coefs = self.clf.coef_[0]
            vectorized = self.tfidf.transform([redacted_text])

            non_zero_indices = vectorized.indices
            if len(non_zero_indices) == 0:
                return []

            contributions = []
            for idx in non_zero_indices:
                tf_val = vectorized[0, idx]
                weight = coefs[idx]
                score = tf_val * weight
                if score > 0:
                    contributions.append((str(feature_names[idx]), float(score)))

            contributions.sort(key=lambda x: x[1], reverse=True)

            clean_terms: List[str] = []
            seen = set()
            for raw_term, _ in contributions:
                cleaned = raw_term
                for prefix in ("word_tfidf__", "char_tfidf__", "tfidf__", "word__", "char__"):
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):]
                cleaned = cleaned.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    clean_terms.append(cleaned)
                if len(clean_terms) >= top_k:
                    break

            return clean_terms
        except Exception as e:
            logger.warning(f"Could not extract feature explanations: {e}")
            return []

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Executes complete inference workflow:
        1. PII Redaction
        2. Calibrated Probability Prediction
        3. Threshold Policy & Clamped Severity
        4. Feature Explanations
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not initialized or failed to load.")

        redacted = redact_text(raw_text)

        try:
            probs = self.model.predict_proba([redacted])[0]
            scam_prob = float(round(probs[1], 4))
        except Exception as e:
            logger.error("Inference execution failed during predict_proba.")
            raise RuntimeError("Inference execution failed safely.") from e

        severity = evaluate_severity(scam_prob)
        policy_meta = get_policy_metadata()

        if severity == "none":
            signal = None
            terms: List[str] = []
        else:
            signal = "possible_scam_pattern"
            terms = self.explain_terms(redacted, top_k=5)

        return {
            "source": "ml",
            "signal": signal,
            "severity": severity,
            "confidence": scam_prob,
            "model_version": settings.MODEL_VERSION,
            "top_contributing_terms": terms,
            "policy": policy_meta
        }


# Global instance
model_service = ModelService()

