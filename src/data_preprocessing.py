"""Text preprocessing, dataset inspection, and target construction."""

from __future__ import annotations

import ast
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import (
    DATASET_PATH,
    LABEL_MODERATE,
    LABEL_POOR,
    LABEL_STRONG,
    LEAKAGE_COLUMNS,
    MODERATE_FIT_THRESHOLD,
    STRONG_FIT_THRESHOLD,
)


def ensure_nltk_data() -> None:
    """Download required NLTK data if not present."""
    import nltk

    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# Technical terms to preserve during preprocessing
PRESERVE_TERMS = {
    "python", "c++", "c#", "sql", "aws", "azure", "power bi", "tensorflow",
    "pytorch", "scikit-learn", "machine learning", "deep learning", "nlp",
    "docker", "kubernetes", ".net", "node.js", "react", "angular", "vue.js",
    "xgboost", "pandas", "numpy", "matplotlib", "tableau", "spark", "hadoop",
    "git", "github", "jenkins", "flask", "django", "html", "css", "javascript",
    "typescript", "mongodb", "postgresql", "mysql", "oracle", "redis",
    "google cloud", "gcp", "devops", "ci/cd", "rest api", "api",
}


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocess text while preserving technical terms.

    - Lowercase
    - Normalize whitespace
    - Light punctuation handling
    - Optional stopword removal (preserves technical terms)
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode and whitespace
    text = text.replace("\xa0", " ").replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase while protecting multi-char technical terms
    lower_text = text.lower()

    if remove_stopwords:
        ensure_nltk_data()
        from nltk.corpus import stopwords

        stops = set(stopwords.words("english"))
        # Remove stopwords but keep preserved terms
        words = lower_text.split()
        filtered = []
        for word in words:
            clean_word = re.sub(r"[^\w+#.+/-]", "", word)
            if clean_word in PRESERVE_TERMS or word in PRESERVE_TERMS:
                filtered.append(word)
            elif clean_word not in stops:
                filtered.append(word)
        lower_text = " ".join(filtered)
    else:
        lower_text = lower_text

    return lower_text


def tokenize_text(text: str) -> list[str]:
    """Basic tokenization after preprocessing."""
    ensure_nltk_data()
    from nltk.tokenize import word_tokenize

    processed = preprocess_text(text)
    return word_tokenize(processed)


def word_count(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(preprocess_text(text).split())


def parse_skill_list(value: Any) -> list[str]:
    """Parse skill list from string representation in dataset."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(s).strip() for s in value if s]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if s]
        except (ValueError, SyntaxError):
            pass
        # Fallback: comma-separated
        return [s.strip() for s in value.split(",") if s.strip()]
    return []


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Perform comprehensive dataset inspection.

    Returns a dictionary with inspection results for logging/display.
    """
    report: dict[str, Any] = {}

    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)

    # Shape
    report["shape"] = df.shape
    print(f"\n1. Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Column names
    report["columns"] = list(df.columns)
    print(f"\n2. Columns: {list(df.columns)}")

    # Data types
    report["dtypes"] = df.dtypes.astype(str).to_dict()
    print(f"\n3. Data types:\n{df.dtypes}")

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing.to_dict()
    print(f"\n4. Missing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}")

    # Duplicates
    dup_rows = df.duplicated().sum()
    report["duplicate_rows"] = int(dup_rows)
    print(f"\n5. Duplicate rows: {dup_rows}")

    # Sample rows
    print(f"\n6. Sample rows (first 3):")
    print(df.head(3).to_string(max_colwidth=50))

    # Score distributions
    score_cols = ["ai_match_score", "skill_string_match_score", "fuzzy_match_score"]
    score_stats = {}
    for col in score_cols:
        if col in df.columns:
            stats = df[col].describe()
            score_stats[col] = stats.to_dict()
            print(f"\n7. {col} distribution:\n{stats}")
    report["score_stats"] = score_stats

    # Unique categories
    if "category" in df.columns:
        n_categories = df["category"].nunique()
        categories = sorted(df["category"].unique().tolist())
        report["n_categories"] = n_categories
        report["categories"] = categories
        print(f"\n8. Unique job categories: {n_categories}")
        print(f"   Categories: {categories}")

    # Duplicate resume/job pairs
    if "resume_text" in df.columns and "job_text" in df.columns:
        pair_dupes = df.duplicated(subset=["resume_text", "job_text"]).sum()
        report["duplicate_pairs"] = int(pair_dupes)
        print(f"\n9. Duplicate resume/job pairs: {pair_dupes}")

    # ID duplicates
    if "ID" in df.columns:
        id_dupes = df["ID"].duplicated().sum()
        report["duplicate_ids"] = int(id_dupes)
        print(f"\n10. Duplicate IDs: {id_dupes}")

    print("\n" + "=" * 60)
    return report


def normalize_score_to_100(series: pd.Series) -> pd.Series:
    """Normalize score to 0-100 scale if needed."""
    min_val = series.min()
    max_val = series.max()

    if max_val <= 100 and min_val >= 0:
        # Already on 0-100 scale
        return series.clip(0, 100)

    # Min-max normalize to 0-100
    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)

    normalized = (series - min_val) / (max_val - min_val) * 100
    return normalized.clip(0, 100)


def score_to_label(score: float) -> str:
    """Convert continuous fit score to label using thresholds."""
    if score >= STRONG_FIT_THRESHOLD:
        return LABEL_STRONG
    if score >= MODERATE_FIT_THRESHOLD:
        return LABEL_MODERATE
    return LABEL_POOR


def create_targets(df: pd.DataFrame, score_column: str = "ai_match_score") -> pd.DataFrame:
    """
    Create fit_score and fit_label targets from matching score column.

    Uses ai_match_score (0-100) as primary target source.
    """
    df = df.copy()

    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in dataset.")

    print(f"\nCreating targets from '{score_column}'...")
    print(f"  Original range: [{df[score_column].min():.2f}, {df[score_column].max():.2f}]")

    df["fit_score"] = normalize_score_to_100(df[score_column])
    df["fit_label"] = df["fit_score"].apply(score_to_label)

    print(f"  Normalized fit_score range: [{df['fit_score'].min():.2f}, {df['fit_score'].max():.2f}]")
    print(f"  Label distribution:\n{df['fit_label'].value_counts()}")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: drop nulls in key columns, remove duplicate pairs."""
    df = df.copy()
    initial_len = len(df)

    # Drop rows with missing resume or job text
    required_cols = ["resume_text", "job_text"]
    for col in required_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
            df = df[df[col].astype(str).str.strip() != ""]

    # Drop exact duplicate resume/job pairs
    if "resume_text" in df.columns and "job_text" in df.columns:
        df = df.drop_duplicates(subset=["resume_text", "job_text"], keep="first")

    # Drop rows with missing target score
    if "ai_match_score" in df.columns:
        df = df.dropna(subset=["ai_match_score"])

    print(f"\nCleaning: {initial_len} -> {len(df)} rows "
          f"({initial_len - len(df)} removed)")

    return df.reset_index(drop=True)


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from CSV path."""
    path = path or str(DATASET_PATH)
    df = pd.read_csv(path)
    return df


def prepare_dataset(path: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full dataset preparation pipeline.

    Returns cleaned dataframe with targets and inspection report.
    """
    df = load_dataset(path)
    report = inspect_dataset(df)
    df = clean_dataset(df)
    df = create_targets(df)
    report["final_shape"] = df.shape
    report["label_distribution"] = df["fit_label"].value_counts().to_dict()
    return df, report
