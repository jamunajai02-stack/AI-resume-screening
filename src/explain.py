"""Human-readable explanation generation from SHAP and skill analysis."""

from __future__ import annotations

from typing import Any

from src.config import LABEL_MODERATE, LABEL_POOR, LABEL_STRONG


def generate_positive_factors(prediction: dict[str, Any], explanation: dict[str, Any]) -> list[str]:
    """Generate list of positive contributing factors."""
    factors: list[str] = []

    features = prediction.get("features", {})
    matched = prediction.get("matched_skills", [])

    # Skill match
    skill_pct = features.get("skill_match_percentage", 0)
    if skill_pct >= 50:
        factors.append(f"High skill match ({skill_pct:.0f}%)")
    elif skill_pct >= 25:
        factors.append(f"Moderate skill match ({skill_pct:.0f}%)")

    # TF-IDF similarity
    tfidf = features.get("tfidf_similarity", 0)
    if tfidf >= 0.3:
        factors.append("Strong resume-job text similarity")
    elif tfidf >= 0.15:
        factors.append("Moderate resume-job text similarity")

    # Experience
    exp = features.get("experience_years", 0)
    if exp >= 5:
        factors.append(f"{exp:.0f}+ years of experience detected")
    elif exp >= 2:
        factors.append(f"{exp:.0f} years of experience detected")

    # Matched skills (top 5)
    for skill in matched[:5]:
        factors.append(f"{skill} matched")

    # SHAP positive features
    for impact in explanation.get("feature_impacts", []):
        if impact.get("direction") == "Positive" and impact.get("impact", 0) > 0.01:
            feat = impact["feature"]
            if feat not in [f.split(" matched")[0] for f in factors]:
                factors.append(f"Positive model signal: {feat}")

    return factors[:10]


def generate_skill_gaps(prediction: dict[str, Any]) -> list[str]:
    """Generate list of missing skills / skill gaps."""
    missing = prediction.get("missing_skills", [])
    return missing[:15]


def generate_summary_text(prediction: dict[str, Any], explanation: dict[str, Any]) -> str:
    """Generate a human-readable summary paragraph."""
    score = prediction.get("fit_score", 0)
    label = prediction.get("fit_label", "Unknown")
    features = prediction.get("features", {})

    skill_pct = features.get("skill_match_percentage", 0)
    matched_count = features.get("matched_skill_count", 0)
    missing_count = features.get("missing_skill_count", 0)
    tfidf = features.get("tfidf_similarity", 0)

    summary = (
        f"Predicted job fit: {score:.1f}% ({label}). "
        f"The candidate matches {matched_count} required skills ({skill_pct:.0f}% skill match) "
        f"with {missing_count} skill gaps identified. "
        f"Resume-job text similarity score is {tfidf:.2f}. "
        f"This prediction is based on the trained machine learning model "
        f"using NLP features, skill extraction, and engineered numerical features."
    )
    return summary


def build_explanation(
    prediction: dict[str, Any],
    explanation: dict[str, Any],
) -> dict[str, Any]:
    """
    Build complete human-readable explanation from prediction and SHAP results.

    Returns structured explanation dict for dashboard display.
    """
    positive = generate_positive_factors(prediction, explanation)
    gaps = generate_skill_gaps(prediction)
    summary = generate_summary_text(prediction, explanation)

    # Fit label color coding
    label = prediction.get("fit_label", "")
    if label == LABEL_STRONG:
        label_color = "green"
    elif label == LABEL_MODERATE:
        label_color = "orange"
    else:
        label_color = "red"

    return {
        "summary": summary,
        "positive_factors": positive,
        "skill_gaps": gaps,
        "fit_score": prediction.get("fit_score"),
        "fit_label": label,
        "label_color": label_color,
        "feature_impacts": explanation.get("feature_impacts", []),
        "shap_values": explanation.get("shap_values", []),
    }
