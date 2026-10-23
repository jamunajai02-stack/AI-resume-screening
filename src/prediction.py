"""Prediction and SHAP explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import shap

from src.config import (
    FEATURE_METADATA_PATH,
    FEATURE_NAMES,
    FIT_MODEL_PATH,
    LABEL_MODERATE,
    LABEL_POOR,
    LABEL_STRONG,
    MODERATE_FIT_THRESHOLD,
    SHAP_BACKGROUND_PATH,
    STRONG_FIT_THRESHOLD,
    TFIDF_VECTORIZER_PATH,
)
from src.data_preprocessing import score_to_label
from src.feature_engineering import build_feature_vector


class ModelNotFoundError(FileNotFoundError):
    """Raised when trained model artifacts are missing."""


def load_artifacts() -> tuple[Any, Any, dict[str, Any], Optional[np.ndarray]]:
    """Load model, vectorizer, metadata, and optional SHAP background."""
    if not FIT_MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model not found at {FIT_MODEL_PATH}. Run: python -m src.train_model"
        )
    if not TFIDF_VECTORIZER_PATH.exists():
        raise ModelNotFoundError(
            f"Vectorizer not found at {TFIDF_VECTORIZER_PATH}. Run: python -m src.train_model"
        )
    if not FEATURE_METADATA_PATH.exists():
        raise ModelNotFoundError(
            f"Metadata not found at {FEATURE_METADATA_PATH}. Run: python -m src.train_model"
        )

    model = joblib.load(FIT_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    metadata = joblib.load(FEATURE_METADATA_PATH)

    background = None
    if SHAP_BACKGROUND_PATH.exists():
        background = joblib.load(SHAP_BACKGROUND_PATH)

    return model, vectorizer, metadata, background


def predict_fit(
    resume_text: str,
    job_text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Predict job fit for a resume-job pair.

    Returns dict with fit_score, fit_label, features, skills, and metadata.
    """
    if model is None or vectorizer is None or metadata is None:
        model, vectorizer, metadata, _ = load_artifacts()

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")
    if not job_text or not job_text.strip():
        raise ValueError("Job description is empty.")

    feature_array, feat_meta = build_feature_vector(resume_text, job_text, vectorizer)
    X = feature_array.reshape(1, -1)

    raw_score = float(model.predict(X)[0])
    # Clip to 0-100 range
    fit_score = float(np.clip(raw_score, 0, 100))
    fit_label = score_to_label(fit_score)

    return {
        "fit_score": round(fit_score, 1),
        "fit_label": fit_label,
        "features": {name: feat_meta[name] for name in FEATURE_NAMES},
        "matched_skills": feat_meta["matched_skills"],
        "missing_skills": feat_meta["missing_skills"],
        "resume_skills": feat_meta["resume_skills"],
        "job_skills": feat_meta["job_skills"],
        "feature_vector": feature_array,
        "model_name": metadata.get("best_regressor_name", "Unknown"),
    }


def _get_explainer(
    model: Any,
    background: Optional[np.ndarray] = None,
) -> shap.Explainer:
    """Create appropriate SHAP explainer for the model."""
    model_type = type(model).__name__

    tree_models = (
        "RandomForestRegressor", "RandomForestClassifier",
        "GradientBoostingRegressor", "GradientBoostingClassifier",
        "XGBRegressor", "XGBClassifier",
        "ExtraTreesRegressor", "ExtraTreesClassifier",
    )

    if any(t in model_type for t in tree_models):
        return shap.TreeExplainer(model)

    # Linear or other models - use KernelExplainer with background
    if background is not None and len(background) > 0:
        return shap.KernelExplainer(model.predict, background)

    # Fallback: small zero background
    dummy_bg = np.zeros((5, len(FEATURE_NAMES)))
    return shap.KernelExplainer(model.predict, dummy_bg)


def explain_prediction(
    resume_text: str,
    job_text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
    background: Optional[np.ndarray] = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Generate SHAP explanation for a prediction.

    Returns SHAP values, feature impacts, and prediction result.
    """
    if model is None or vectorizer is None or metadata is None:
        model, vectorizer, metadata, bg_loaded = load_artifacts()
        if background is None:
            background = bg_loaded

    prediction = predict_fit(resume_text, job_text, model, vectorizer, metadata)
    X = prediction["feature_vector"].reshape(1, -1)

    try:
        explainer = _get_explainer(model, background)
        shap_values = explainer.shap_values(X)

        # Handle multi-output (classification) vs single output (regression)
        if isinstance(shap_values, list):
            # Multi-class: use predicted class index
            shap_vals = shap_values[0][0] if isinstance(shap_values[0], np.ndarray) else shap_values[0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                shap_vals = shap_values[0, 0, :]
            elif shap_values.ndim == 2:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
        else:
            shap_vals = np.array(shap_values).flatten()

        shap_vals = np.array(shap_vals).flatten()
        if len(shap_vals) != len(FEATURE_NAMES):
            shap_vals = shap_vals[: len(FEATURE_NAMES)]

    except Exception as exc:
        # Graceful fallback if SHAP fails
        shap_vals = np.zeros(len(FEATURE_NAMES))
        return {
            **prediction,
            "shap_values": shap_vals.tolist(),
            "feature_impacts": [],
            "shap_error": str(exc),
        }

    # Build feature impact table
    impacts = []
    for i, name in enumerate(FEATURE_NAMES):
        val = float(shap_vals[i]) if i < len(shap_vals) else 0.0
        direction = "Positive" if val >= 0 else "Negative"
        impacts.append({
            "feature": name.replace("_", " ").title(),
            "impact": round(abs(val), 4),
            "direction": direction,
            "shap_value": round(val, 4),
            "feature_value": prediction["features"].get(name, 0),
        })

    impacts.sort(key=lambda x: x["impact"], reverse=True)

    return {
        **prediction,
        "shap_values": shap_vals.tolist(),
        "feature_impacts": impacts[:top_n],
        "expected_value": float(explainer.expected_value)
        if hasattr(explainer, "expected_value") and not isinstance(explainer.expected_value, np.ndarray)
        else None,
    }


def get_global_feature_importance(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Get global feature importance from saved metadata."""
    names = metadata.get("feature_names", FEATURE_NAMES)
    importances = metadata.get("feature_importances", [])

    if not importances:
        return []

    result = []
    for name, imp in zip(names, importances):
        result.append({"feature": name.replace("_", " ").title(), "importance": round(float(imp), 4)})

    result.sort(key=lambda x: x["importance"], reverse=True)
    return result
