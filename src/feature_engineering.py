"""Feature engineering for resume-job fit prediction."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    FEATURE_NAMES,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from src.data_preprocessing import preprocess_text, word_count
from src.skill_extraction import compute_skill_features, extract_skills


def extract_experience_years(text: str) -> float:
    """
    Extract years of experience from resume text.

    Looks for patterns like '5 years', '5+ years', '10 years of experience'.
    Returns the maximum found, or 0.0 if none detected.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?",
        r"(?:over|more than|at least)\s+(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\+?\s*(?:years?|yrs?)\s+in",
    ]

    years_found: list[float] = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                years_found.append(float(match))
            except ValueError:
                continue

    if years_found:
        return float(max(years_found))
    return 0.0


def compute_tfidf_similarity(
    resume_text: str,
    job_text: str,
    vectorizer: TfidfVectorizer,
) -> float:
    """Compute cosine similarity between resume and job TF-IDF vectors."""
    resume_clean = preprocess_text(resume_text)
    job_clean = preprocess_text(job_text)

    if not resume_clean or not job_clean:
        return 0.0

    combined = [resume_clean, job_clean]
    try:
        tfidf_matrix = vectorizer.transform(combined)
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def build_feature_vector(
    resume_text: str,
    job_text: str,
    vectorizer: TfidfVectorizer,
    skills_path: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build feature vector for a single resume-job pair.

    Returns (feature_array, metadata_dict with skills and intermediate values).
    """
    resume_skills = extract_skills(resume_text, skills_path)
    job_skills = extract_skills(job_text, skills_path)
    skill_info = compute_skill_features(resume_skills, job_skills)

    tfidf_sim = compute_tfidf_similarity(resume_text, job_text, vectorizer)
    exp_years = extract_experience_years(resume_text)

    features = {
        "tfidf_similarity": round(tfidf_sim, 4),
        "required_skill_count": skill_info["required_skill_count"],
        "candidate_skill_count": skill_info["candidate_skill_count"],
        "matched_skill_count": skill_info["matched_skill_count"],
        "missing_skill_count": skill_info["missing_skill_count"],
        "skill_match_percentage": skill_info["skill_match_percentage"],
        "resume_word_count": word_count(resume_text),
        "job_word_count": word_count(job_text),
        "experience_years": exp_years,
    }

    feature_array = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

    metadata = {
        **features,
        "matched_skills": skill_info["matched_skills"],
        "missing_skills": skill_info["missing_skills"],
        "resume_skills": skill_info["resume_skills"],
        "job_skills": skill_info["job_skills"],
    }

    return feature_array, metadata


def build_feature_matrix(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    skills_path: Optional[str] = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Build feature matrix for entire dataframe.

    Returns (X, list of metadata dicts per row).
    """
    X_list: list[np.ndarray] = []
    metadata_list: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        resume_text = str(row.get("resume_text", ""))
        job_text = str(row.get("job_text", ""))
        feat, meta = build_feature_vector(resume_text, job_text, vectorizer, skills_path)
        X_list.append(feat)
        metadata_list.append(meta)

    return np.vstack(X_list), metadata_list


def fit_tfidf_vectorizer(
    texts: list[str],
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
) -> TfidfVectorizer:
    """
    Fit TF-IDF vectorizer on combined resume and job texts.

    Should be fit on TRAINING data only to avoid leakage.
    """
    cleaned = [preprocess_text(t) for t in texts if t]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
    )
    vectorizer.fit(cleaned)
    return vectorizer


def get_combined_texts_from_df(df: pd.DataFrame) -> list[str]:
    """Get all resume and job texts from dataframe for vectorizer fitting."""
    texts: list[str] = []
    for _, row in df.iterrows():
        if row.get("resume_text"):
            texts.append(str(row["resume_text"]))
        if row.get("job_text"):
            texts.append(str(row["job_text"]))
    return texts
