"""Tests for feature engineering and skill extraction."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering import (
    build_feature_vector,
    extract_experience_years,
    fit_tfidf_vectorizer,
)
from src.skill_extraction import compute_skill_features, extract_skills


def test_extract_skills_python():
    text = "Experienced Python developer with SQL and Machine Learning expertise."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Machine Learning" in skills


def test_no_false_r_match():
    """Ensure 'R' is not matched in every word containing 'r'."""
    text = "program manager with experience in marketing and operations."
    skills = extract_skills(text)
    assert "R" not in skills


def test_skill_match_features():
    resume_skills = ["Python", "SQL", "Machine Learning"]
    job_skills = ["Python", "SQL", "AWS", "Docker"]
    result = compute_skill_features(resume_skills, job_skills)
    assert result["matched_skill_count"] == 2
    assert result["missing_skill_count"] == 2
    assert result["skill_match_percentage"] == 50.0
    assert "Python" in result["matched_skills"]
    assert "AWS" in result["missing_skills"]


def test_experience_extraction():
    text = "Software engineer with 5+ years of experience in Python development."
    years = extract_experience_years(text)
    assert years >= 5


def test_experience_fallback():
    text = "Recent graduate with internship experience."
    years = extract_experience_years(text)
    assert years == 0.0


def test_tfidf_vectorizer_shape():
    texts = [
        "Python developer with machine learning skills",
        "Java developer with spring boot experience",
        "Data scientist with pandas and numpy",
    ]
    vectorizer = fit_tfidf_vectorizer(texts, max_features=100, min_df=1)
    assert vectorizer.max_features == 100


def test_build_feature_vector():
    resume = "Python developer with 3 years experience in SQL and Machine Learning."
    job = "Looking for Python and AWS expert with Docker experience."
    texts = [resume, job]
    vectorizer = fit_tfidf_vectorizer(texts, max_features=100, min_df=1)
    feat_array, meta = build_feature_vector(resume, job, vectorizer)
    assert feat_array.shape == (9,)
    assert "matched_skills" in meta
    assert "Python" in meta["matched_skills"]
    assert meta["experience_years"] >= 3
