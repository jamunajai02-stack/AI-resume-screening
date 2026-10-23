"""Project configuration and path constants."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Files
DATASET_PATH = DATA_DIR / "job_resume_fit.csv"
SKILLS_PATH = CONFIG_DIR / "skills.json"
FIT_MODEL_PATH = MODELS_DIR / "fit_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
FEATURE_METADATA_PATH = MODELS_DIR / "feature_metadata.pkl"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.pkl"

# Hugging Face dataset URL for optional download
DATASET_URL = (
    "https://huggingface.co/datasets/batuhanmtl/job_resume_fit/resolve/main/job_resume_fit.csv"
)

# Reproducibility
RANDOM_STATE = 42

# Target thresholds (applied after score is on 0-100 scale)
STRONG_FIT_THRESHOLD = 75.0
MODERATE_FIT_THRESHOLD = 50.0

# Fit labels
LABEL_STRONG = "Strong Fit"
LABEL_MODERATE = "Moderate Fit"
LABEL_POOR = "Poor Fit"
FIT_LABELS = [LABEL_POOR, LABEL_MODERATE, LABEL_STRONG]

# Columns that must NOT be used as model features (data leakage)
LEAKAGE_COLUMNS = [
    "ai_match_score",
    "skill_string_match_score",
    "fuzzy_match_score",
    "ai_matched_skills",
]

# TF-IDF settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# Train/test split
TEST_SIZE = 0.2

# Feature names (order matters for model input)
FEATURE_NAMES = [
    "tfidf_similarity",
    "required_skill_count",
    "candidate_skill_count",
    "matched_skill_count",
    "missing_skill_count",
    "skill_match_percentage",
    "resume_word_count",
    "job_word_count",
    "experience_years",
]

# Academic disclaimer
DISCLAIMER = (
    "This system is an academic decision-support prototype. Predictions are based "
    "on the available dataset and extracted resume/job features and should not be "
    "used as the sole basis for employment decisions."
)

# Target construction note
TARGET_NOTE = (
    "Fit labels (Strong/Moderate/Poor Fit) are derived from the dataset's "
    "ai_match_score using predefined thresholds (>=75, >=50, <50). "
    "They are NOT human HR labels."
)
