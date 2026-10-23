"""Job recommendation engine — search links only (no scraping)."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote_plus

import pandas as pd

from src.config import DATA_DIR, DATASET_PATH
from src.skill_extraction import extract_skills

COMPANY_CAREERS_PATH = DATA_DIR / "company_careers.csv"

LOCATIONS = [
    "India",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Mumbai",
    "Delhi NCR",
    "Remote",
    "Gurgaon",
    "Noida",
    "Kolkata",
]

# Role catalog: title, category, key skills for matching
ROLE_CATALOG: list[dict[str, Any]] = [
    {
        "title": "Data Scientist",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["Python", "Machine Learning", "SQL", "Pandas", "NumPy", "Scikit-learn", "Deep Learning", "TensorFlow", "PyTorch", "NLP"],
    },
    {
        "title": "Machine Learning Engineer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Docker", "AWS", "Deep Learning", "Scikit-learn"],
    },
    {
        "title": "Data Analyst",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Pandas", "Data Analysis", "Data Visualization"],
    },
    {
        "title": "Software Engineer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["Python", "Java", "JavaScript", "SQL", "Git", "Docker", "React", "Node.js"],
    },
    {
        "title": "Full Stack Developer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["JavaScript", "React", "Node.js", "HTML", "CSS", "SQL", "MongoDB", "Git"],
    },
    {
        "title": "DevOps Engineer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["Docker", "Kubernetes", "AWS", "Jenkins", "CI/CD", "Linux", "Git", "Terraform"],
    },
    {
        "title": "Cloud Engineer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Linux", "Python"],
    },
    {
        "title": "Backend Developer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["Python", "Java", "SQL", "Django", "Flask", "Node.js", "REST API", "MongoDB"],
    },
    {
        "title": "Frontend Developer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["JavaScript", "React", "HTML", "CSS", "TypeScript", "Angular", "Vue.js"],
    },
    {
        "title": "Business Analyst",
        "category": "BUSINESS-DEVELOPMENT",
        "skills": ["Excel", "SQL", "Power BI", "Communication", "Data Analysis", "Stakeholder Management"],
    },
    {
        "title": "HR Executive",
        "category": "HR",
        "skills": ["Recruitment", "Employee Relations", "HRIS", "Payroll", "Onboarding", "Communication"],
    },
    {
        "title": "Talent Acquisition Specialist",
        "category": "HR",
        "skills": ["Talent Acquisition", "Recruitment", "Interviewing", "Communication", "LinkedIn"],
    },
    {
        "title": "Financial Analyst",
        "category": "FINANCE",
        "skills": ["Financial Analysis", "Excel", "Accounting", "Budgeting", "Financial Modeling", "SQL"],
    },
    {
        "title": "Accountant",
        "category": "ACCOUNTANT",
        "skills": ["Accounting", "Excel", "Tally", "GST", "Financial Reporting", "Auditing"],
    },
    {
        "title": "Marketing Executive",
        "category": "DIGITAL-MEDIA",
        "skills": ["Digital Marketing", "SEO", "Social Media Marketing", "Google Analytics", "Content Marketing"],
    },
    {
        "title": "Sales Executive",
        "category": "SALES",
        "skills": ["Sales", "CRM", "Communication", "Lead Generation", "Client Relations"],
    },
    {
        "title": "Project Manager",
        "category": "CONSULTANT",
        "skills": ["Project Management", "Agile", "Scrum", "Communication", "MS Project", "Leadership"],
    },
    {
        "title": "QA / Test Engineer",
        "category": "INFORMATION-TECHNOLOGY",
        "skills": ["Quality Assurance", "Selenium", "Java", "Python", "SQL", "Agile"],
    },
    {
        "title": "UI/UX Designer",
        "category": "DESIGNER",
        "skills": ["UI Design", "UX Design", "Figma", "Adobe Photoshop", "Adobe Illustrator"],
    },
    {
        "title": "Teacher / Trainer",
        "category": "TEACHER",
        "skills": ["Teaching", "Training", "Communication", "Curriculum Development", "E-Learning"],
    },
    {
        "title": "Healthcare Administrator",
        "category": "HEALTHCARE",
        "skills": ["Healthcare Administration", "Patient Care", "Communication", "HIPAA"],
    },
    {
        "title": "Civil / Site Engineer",
        "category": "CONSTRUCTION",
        "skills": ["AutoCAD", "Project Management", "Civil Engineering", "Quality Control"],
    },
    {
        "title": "Banking Officer",
        "category": "BANKING",
        "skills": ["Customer Service", "Communication", "Banking", "Excel", "Sales"],
    },
]


def build_search_urls(role: str, skills: list[str], location: str) -> dict[str, str]:
    """Build legitimate job-portal search URLs (no scraping)."""
    skill_query = " ".join(skills[:4]) if skills else ""
    query = f"{role} {skill_query}".strip()
    q = quote_plus(query)
    loc = quote_plus(location)
    loc_lower = location.lower().replace(" ", "-")

    urls = {
        "LinkedIn": f"https://www.linkedin.com/jobs/search/?keywords={q}&location={loc}",
        "Naukri": f"https://www.naukri.com/{quote_plus(role.lower().replace(' ', '-'))}-jobs-in-{loc_lower}",
        "Indeed": f"https://in.indeed.com/jobs?q={q}&l={loc}",
        "Internshala": f"https://internshala.com/jobs/{quote_plus(role.lower().replace(' ', '-'))}-jobs/",
        "Foundit": f"https://www.foundit.in/search/{quote_plus(role)}?locations={loc}",
        "Glassdoor": f"https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={q}&locT=N&locId=115",
        "Cutshort": f"https://cutshort.io/profile/recommended-jobs?q={q}",
        "Wellfound": f"https://wellfound.com/jobs?q={q}",
        "Freshersworld": f"https://www.freshersworld.com/jobs/jobsearch/{quote_plus(role)}-jobs",
        "Shine": f"https://www.shine.com/job-search/{quote_plus(role.lower().replace(' ', '-'))}-jobs",
    }
    return urls


def _score_role(
    resume_skills: list[str],
    role: dict[str, Any],
    experience_years: float,
    preferred_category: Optional[str] = None,
    jd_skills: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Score a role against candidate skills."""
    resume_set = {s.lower() for s in resume_skills}
    role_skills = role["skills"]
    role_set = {s.lower(): s for s in role_skills}

    matched = [role_set[s] for s in role_set if s in resume_set]
    missing = [role_set[s] for s in role_set if s not in resume_set]

    if not role_skills:
        skill_pct = 0.0
    else:
        skill_pct = (len(matched) / len(role_skills)) * 100.0

    # Category boost
    cat_boost = 0.0
    if preferred_category and preferred_category.upper() == role["category"].upper():
        cat_boost = 12.0

    # Experience soft boost (cap)
    exp_boost = min(experience_years, 10) * 1.2

    # Overlap with JD skills (if analyzing against a JD)
    jd_boost = 0.0
    if jd_skills:
        jd_set = {s.lower() for s in jd_skills}
        overlap = len(resume_set & jd_set & set(role_set.keys()))
        jd_boost = min(overlap * 3.0, 15.0)

    match_pct = min(100.0, skill_pct * 0.7 + cat_boost + exp_boost + jd_boost)

    return {
        "title": role["title"],
        "category": role["category"],
        "match_pct": round(match_pct, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "required_skills": role_skills,
    }


def recommend_jobs(
    resume_text: str,
    job_text: str = "",
    prediction: Optional[dict[str, Any]] = None,
    location: str = "India",
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """
    Recommend 5–10 suitable roles with platform search links.

    Uses resume skills, optional JD skills, experience, and category heuristics.
    """
    if prediction:
        resume_skills = prediction.get("resume_skills") or extract_skills(resume_text)
        jd_skills = prediction.get("job_skills") or (extract_skills(job_text) if job_text else [])
        experience_years = float(prediction.get("features", {}).get("experience_years", 0) or 0)
    else:
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_text) if job_text else []
        experience_years = 0.0

    preferred_category = _infer_category(resume_skills, jd_skills, job_text)

    scored = [
        _score_role(resume_skills, role, experience_years, preferred_category, jd_skills)
        for role in ROLE_CATALOG
    ]
    scored.sort(key=lambda x: x["match_pct"], reverse=True)
    top = scored[: max(5, min(top_n, 10))]

    recommendations: list[dict[str, Any]] = []
    platforms = ["LinkedIn", "Naukri", "Indeed", "Internshala", "Foundit",
                 "Glassdoor", "Cutshort", "Wellfound", "Freshersworld", "Shine"]

    for i, item in enumerate(top):
        # Rotate primary platform for variety in demo
        primary = platforms[i % len(platforms)]
        urls = build_search_urls(item["title"], item["matched_skills"] or item["required_skills"][:3], location)
        recommendations.append({
            **item,
            "platform": primary,
            "search_url": urls[primary],
            "all_platform_urls": urls,
            "location": location,
        })

    return recommendations


def _infer_category(
    resume_skills: list[str],
    jd_skills: list[str],
    job_text: str,
) -> Optional[str]:
    """Infer preferred job category from skills / JD text."""
    text = (job_text or "").lower()
    keywords = {
        "INFORMATION-TECHNOLOGY": ["software", "developer", "python", "data scientist", "engineer", "devops", "cloud"],
        "HR": ["human resource", "hr ", "recruitment", "talent acquisition"],
        "FINANCE": ["finance", "financial analyst", "accounting"],
        "ACCOUNTANT": ["accountant", "bookkeeping", "gst"],
        "SALES": ["sales", "business development", "account executive"],
        "BANKING": ["bank", "banking"],
        "HEALTHCARE": ["hospital", "healthcare", "nurse", "clinical"],
        "TEACHER": ["teacher", "trainer", "education", "faculty"],
        "DESIGNER": ["designer", "ui/ux", "figma", "graphic"],
        "CONSULTANT": ["consultant", "project manager", "pmp"],
        "DIGITAL-MEDIA": ["marketing", "seo", "digital media", "content"],
        "CONSTRUCTION": ["civil", "construction", "site engineer"],
    }
    for cat, words in keywords.items():
        if any(w in text for w in words):
            return cat

    # Fall back to skill-based category from ROLE_CATALOG majority
    skill_set = {s.lower() for s in resume_skills + jd_skills}
    best_cat = None
    best_score = 0
    for role in ROLE_CATALOG:
        overlap = len(skill_set & {s.lower() for s in role["skills"]})
        if overlap > best_score:
            best_score = overlap
            best_cat = role["category"]
    return best_cat


def load_company_careers() -> pd.DataFrame:
    """Load company career-page CSV."""
    if not COMPANY_CAREERS_PATH.exists():
        return pd.DataFrame(columns=["company", "category", "career_url", "notes"])
    return pd.read_csv(COMPANY_CAREERS_PATH)


def recommend_companies(
    recommendations: list[dict[str, Any]],
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """
    Recommend companies with official career page links.

    Does NOT claim current openings — only links to careers pages.
    """
    df = load_company_careers()
    if df.empty:
        return []

    preferred_cats = {r["category"] for r in recommendations[:5]}
    matched = df[df["category"].isin(preferred_cats)]
    if matched.empty:
        matched = df

    # Prefer category matches first
    rows = matched.head(top_n)
    if len(rows) < top_n:
        extra = df[~df["company"].isin(rows["company"])].head(top_n - len(rows))
        rows = pd.concat([rows, extra], ignore_index=True)

    result = []
    for _, row in rows.iterrows():
        result.append({
            "company": row["company"],
            "category": row["category"],
            "career_url": row["career_url"],
            "notes": row.get("notes", "Official careers page"),
        })
    return result


def get_dataset_category_hints(n: int = 5) -> list[str]:
    """Optional: sample categories from training dataset if available."""
    if not DATASET_PATH.exists():
        return []
    try:
        df = pd.read_csv(DATASET_PATH, usecols=["category"])
        return df["category"].value_counts().head(n).index.tolist()
    except Exception:
        return []
