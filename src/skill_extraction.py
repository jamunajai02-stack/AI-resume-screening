"""Skill extraction from resume and job description text."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

from src.config import SKILLS_PATH


# Skills that need exact/special boundary matching to avoid false positives
SPECIAL_BOUNDARY_SKILLS = {"r", "c", "go", "sql", "aws", "gcp", "api", "ui", "ux", "hr", "bi"}


@lru_cache(maxsize=1)
def load_skills_dictionary(path: Optional[str] = None) -> dict[str, list[str]]:
    """Load skills dictionary from JSON config file."""
    path = path or str(SKILLS_PATH)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_skills(path: Optional[str] = None) -> list[str]:
    """Return flat list of all skills, sorted longest-first for matching."""
    skills_dict = load_skills_dictionary(path)
    all_skills: list[str] = []
    for category_skills in skills_dict.values():
        all_skills.extend(category_skills)

    # Deduplicate preserving longest forms
    seen_lower: set[str] = set()
    unique: list[str] = []
    for skill in sorted(all_skills, key=len, reverse=True):
        key = skill.lower()
        if key not in seen_lower:
            seen_lower.add(key)
            unique.append(skill)
    return unique


def _build_pattern(skill: str) -> re.Pattern:
    """Build regex pattern for a skill with appropriate word boundaries."""
    escaped = re.escape(skill.lower())

    # Special handling for skills with special characters
    if skill.lower() in SPECIAL_BOUNDARY_SKILLS:
        # Require word boundaries or start/end of string
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    elif "+" in skill or "#" in skill or "." in skill:
        # C++, C#, Node.js etc - allow adjacent non-alpha
        pattern = rf"(?<![a-z0-9+#.]){escaped}(?![a-z0-9+#.])"
    else:
        # Standard multi-word or single word
        if " " in skill:
            pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        else:
            pattern = rf"\b{escaped}\b"

    return re.compile(pattern, re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_skill_patterns(path: Optional[str] = None) -> list[tuple[str, re.Pattern]]:
    """Get compiled patterns for all skills."""
    skills = get_all_skills(path)
    return [(skill, _build_pattern(skill)) for skill in skills]


def extract_skills(text: str, path: Optional[str] = None) -> list[str]:
    """
    Extract skills from text using dictionary matching.

    Returns normalized skill names (canonical form from dictionary).
    Avoids false substring matches (e.g., 'R' in 'program').
    """
    if not text or not isinstance(text, str):
        return []

    text_lower = text.lower()
    matched: list[str] = []
    matched_lower: set[str] = set()

    for skill, pattern in _get_skill_patterns(path):
        if pattern.search(text_lower):
            key = skill.lower()
            if key not in matched_lower:
                matched_lower.add(key)
                matched.append(skill)

    return sorted(matched, key=str.lower)


def compute_skill_features(
    resume_skills: list[str],
    job_skills: list[str],
) -> dict[str, float | int | list[str]]:
    """
    Compute skill match features from extracted skill lists.

    Returns dict with counts, percentage, matched and missing skills.
    """
    resume_set = {s.lower() for s in resume_skills}
    job_set = {s.lower() for s in job_skills}

    # Preserve canonical names for display
    resume_canonical = {s.lower(): s for s in resume_skills}
    job_canonical = {s.lower(): s for s in job_skills}

    matched_lower = resume_set & job_set
    missing_lower = job_set - resume_set

    matched = sorted([job_canonical[s] for s in matched_lower], key=str.lower)
    missing = sorted([job_canonical[s] for s in missing_lower], key=str.lower)

    required_count = len(job_skills)
    candidate_count = len(resume_skills)
    matched_count = len(matched)
    missing_count = len(missing)

    if required_count > 0:
        match_pct = (matched_count / required_count) * 100.0
    else:
        match_pct = 0.0

    return {
        "required_skill_count": required_count,
        "candidate_skill_count": candidate_count,
        "matched_skill_count": matched_count,
        "missing_skill_count": missing_count,
        "skill_match_percentage": round(match_pct, 2),
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
    }
