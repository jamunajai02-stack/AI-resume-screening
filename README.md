# AI-Based Resume Screening and Job Fit Prediction Using Explainable Machine Learning

An end-to-end academic Data Science project that screens resumes against job descriptions, predicts job-fit scores using machine learning, and explains predictions with SHAP.

---

## Problem Statement

Manual resume screening is time-consuming and inconsistent. HR teams must compare hundreds of resumes against job requirements, often relying on subjective judgment. This project builds an AI-powered decision-support system that:

- Extracts text from PDF/DOCX resumes
- Compares candidate profiles against job descriptions using NLP
- Predicts a continuous job-fit score and categorical fit label
- Explains predictions using SHAP (Explainable AI)
- Provides an interactive Streamlit dashboard

---

## Objectives

1. **Data Preprocessing** — Clean and inspect resume/job text data
2. **NLP Feature Engineering** — TF-IDF cosine similarity between resume and job description
3. **Skill Extraction** — Dictionary-based skill matching with false-positive prevention
4. **Machine Learning** — Train and compare Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
5. **Model Evaluation** — Classification and regression metrics with visualization
6. **Explainable AI** — SHAP-based global and individual prediction explanations
7. **Interactive Dashboard** — Streamlit app for resume screening and analysis

---

## Dataset

**Source:** [Hugging Face — batuhanmtl/job_resume_fit](https://huggingface.co/datasets/batuhanmtl/job_resume_fit)

| Column | Description |
|--------|-------------|
| `ID` | Unique candidate identifier |
| `resume_text` | Full resume text |
| `job_text` | Job posting text |
| `category` | Job category (23 categories) |
| `job_required_skills` | Required skills for the job |
| `resume_skill_list` | Skills extracted from resume |
| `ai_matched_skills` | AI-matched skills |
| `ai_match_score` | AI-based matching score (0–100) |
| `skill_string_match_score` | String-based matching score |
| `fuzzy_match_score` | Fuzzy token matching score |

**Place the dataset at:** `data/job_resume_fit.csv`

### Target Construction

- **`fit_score`** = `ai_match_score` (already on 0–100 scale)
- **`fit_label`** derived using thresholds:
  - Score ≥ 75 → **Strong Fit**
  - Score ≥ 50 and < 75 → **Moderate Fit**
  - Score < 50 → **Poor Fit**

> **Important:** These labels are derived from the dataset's matching score using predefined thresholds. They are **NOT** human HR labels.

### Data Leakage Prevention

The following columns are **never used as model features:**
- `ai_match_score`
- `skill_string_match_score`
- `fuzzy_match_score`
- `ai_matched_skills`

---

## Methodology

```
Resume PDF/DOCX → Text Extraction → Preprocessing
                                          |
Job Description ──────────────────────────+
                                          |
                                          v
                              Skill Extraction
                                          |
                                          v
                              Feature Engineering
                    (TF-IDF similarity, skill match,
                     text stats, experience)
                                          |
                                          v
                              ML Model (Regression)
                                          |
                              Fit Score + Fit Label
                                          |
                                          v
                              SHAP Explanation
                                          |
                                          v
                              Streamlit Dashboard
```

1. **Data Collection** — Load CSV, inspect shape, types, missing values, duplicates
2. **Preprocessing** — Lowercase, whitespace normalization, preserve technical terms
3. **NLP** — TF-IDF vectorization (max_features=5000, ngram_range=(1,2))
4. **Skill Extraction** — Configurable dictionary with word-boundary matching
5. **Feature Engineering** — 9 numerical features per resume-job pair
6. **ML Training** — 80/20 stratified split, compare 4+ models, select best by F1/RMSE
7. **Evaluation** — Accuracy, F1, confusion matrix, MAE, RMSE, R²
8. **SHAP** — TreeExplainer/KernelExplainer for feature impact
9. **Dashboard** — Streamlit with screening, insights, and dataset analysis pages

---

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Place Dataset

Download from Hugging Face and place at:
```
data/job_resume_fit.csv
```

### 2. Train Model

```bash
python -m src.train_model
```

This automatically:
- Loads and inspects the dataset
- Creates targets and features
- Trains classifiers and regressors
- Evaluates and selects best models
- Saves model artifacts to `models/`
- Generates evaluation plots in `outputs/figures/`

### 3. Launch Dashboard

```bash
streamlit run app.py
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── config/
│   └── skills.json                 # Configurable skills dictionary
├── data/
│   └── job_resume_fit.csv          # Dataset (user-provided)
├── models/                         # Trained artifacts (generated)
│   ├── fit_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── feature_metadata.pkl
│   └── shap_background.pkl
├── src/
│   ├── config.py                   # Paths, constants, thresholds
│   ├── data_preprocessing.py       # Text preprocessing, dataset inspection
│   ├── feature_engineering.py      # TF-IDF, feature vectors
│   ├── skill_extraction.py         # Dictionary-based skill matching
│   ├── resume_parser.py            # PDF/DOCX text extraction
│   ├── train_model.py              # Training pipeline
│   ├── evaluate_model.py           # Metrics and plots
│   ├── prediction.py               # predict_fit(), explain_prediction()
│   ├── explain.py                  # Human-readable explanations
│   ├── report.py                   # HTML report generation
│   └── dashboard.py                # Streamlit pages
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebook
├── outputs/
│   ├── figures/                    # Evaluation plots
│   └── reports/                    # Generated reports
└── tests/
    ├── test_parser.py
    ├── test_features.py
    └── test_prediction.py
```

---

## Features Engineered

| Feature | Description |
|---------|-------------|
| `tfidf_similarity` | Cosine similarity of TF-IDF vectors |
| `required_skill_count` | Skills found in job description |
| `candidate_skill_count` | Skills found in resume |
| `matched_skill_count` | Intersection of resume and job skills |
| `missing_skill_count` | Job skills not in resume |
| `skill_match_percentage` | matched / required × 100 |
| `resume_word_count` | Word count of resume |
| `job_word_count` | Word count of job description |
| `experience_years` | Extracted years of experience |

---

## Screenshots

<!-- Add screenshots after running the dashboard -->

| Page | Screenshot |
|------|-----------|
| Home | `outputs/figures/screenshot_home.png` |
| Resume Screening | `outputs/figures/screenshot_screening.png` |
| Model Insights | `outputs/figures/screenshot_insights.png` |
| Dataset Analysis | `outputs/figures/screenshot_dataset.png` |

---

## Results

| Metric | Value |
|--------|-------|
| Best Regressor | Random Forest |
| MAE | 8.97 |
| RMSE | 11.40 |
| R² | 0.7788 |
| Best Classifier | Random Forest |
| Accuracy | 0.7904 |
| F1 (macro) | 0.7482 |

**Dataset:** 2,385 rows (2,382 after deduplication), 23 job categories  
**Label distribution:** Poor Fit (60%), Moderate Fit (25%), Strong Fit (15%)

---

## Responsible AI

- No protected attributes (gender, race, religion, etc.) are used
- Predictions are labeled as "Predicted job fit based on the trained model"
- System includes academic disclaimer on all result pages
- Skill gaps are noted as extraction-based, not definitive proof of absence

---

## License

Academic project — for educational purposes only.
