"""Tests for prediction pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data_preprocessing import score_to_label
from src.config import LABEL_STRONG, LABEL_MODERATE, LABEL_POOR


def test_score_to_label_strong():
    assert score_to_label(82.0) == LABEL_STRONG


def test_score_to_label_moderate():
    assert score_to_label(60.0) == LABEL_MODERATE


def test_score_to_label_poor():
    assert score_to_label(30.0) == LABEL_POOR


def test_score_to_label_boundaries():
    assert score_to_label(75.0) == LABEL_STRONG
    assert score_to_label(50.0) == LABEL_MODERATE
    assert score_to_label(49.9) == LABEL_POOR


@patch("src.prediction.load_artifacts")
def test_predict_fit_mock(mock_load):
    """Test predict_fit with mocked model artifacts."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([78.5])

    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = MagicMock()

    mock_metadata = {"best_regressor_name": "Test Model"}

    mock_load.return_value = (mock_model, mock_vectorizer, mock_metadata, None)

    from src.prediction import predict_fit

    with patch("src.prediction.build_feature_vector") as mock_build:
        mock_build.return_value = (
            np.array([0.5, 5, 10, 3, 2, 60.0, 200, 150, 3.0]),
            {
                "tfidf_similarity": 0.5,
                "required_skill_count": 5,
                "candidate_skill_count": 10,
                "matched_skill_count": 3,
                "missing_skill_count": 2,
                "skill_match_percentage": 60.0,
                "resume_word_count": 200,
                "job_word_count": 150,
                "experience_years": 3.0,
                "matched_skills": ["Python"],
                "missing_skills": ["AWS"],
                "resume_skills": ["Python", "SQL"],
                "job_skills": ["Python", "AWS"],
            },
        )
        result = predict_fit("resume text", "job text")

    assert "fit_score" in result
    assert "fit_label" in result
    assert "matched_skills" in result
    assert result["fit_score"] == 78.5


def test_explain_structure():
    """Test build_explanation returns expected keys."""
    from src.explain import build_explanation

    prediction = {
        "fit_score": 82.0,
        "fit_label": LABEL_STRONG,
        "features": {
            "skill_match_percentage": 70.0,
            "tfidf_similarity": 0.4,
            "experience_years": 5.0,
            "matched_skill_count": 5,
            "missing_skill_count": 2,
        },
        "matched_skills": ["Python", "SQL"],
        "missing_skills": ["AWS"],
    }
    explanation_raw = {
        "feature_impacts": [
            {"feature": "Skill Match %", "impact": 0.5, "direction": "Positive"},
        ],
        "shap_values": [0.1, 0.2, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1],
    }
    result = build_explanation(prediction, explanation_raw)
    assert "summary" in result
    assert "positive_factors" in result
    assert "skill_gaps" in result
    assert "label_color" in result
