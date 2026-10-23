"""Streamlit dashboard pages for Resume-Job Fit application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DATASET_PATH,
    DISCLAIMER,
    FEATURE_METADATA_PATH,
    FIGURES_DIR,
    LABEL_MODERATE,
    LABEL_POOR,
    LABEL_STRONG,
    TARGET_NOTE,
)
from src.data_preprocessing import create_targets, inspect_dataset, load_dataset, parse_skill_list, score_to_label
from src.explain import build_explanation
from src.prediction import ModelNotFoundError, explain_prediction, get_global_feature_importance, load_artifacts, predict_fit
from src.report import generate_html_report
from src.resume_parser import (
    EmptyDocumentError,
    ResumeParserError,
    UnsupportedFormatError,
    extract_text_from_bytes,
)


def render_disclaimer() -> None:
    """Render academic disclaimer."""
    st.info(DISCLAIMER)


def page_home() -> None:
    """Home page with project overview."""
    st.title("AI-Based Resume Screening & Job Fit Prediction")
    st.subheader("Explainable Machine Learning for Resume-Job Matching")

    st.markdown("""
    ### Project Objective
    This academic project demonstrates an end-to-end AI pipeline that:
    - Extracts and preprocesses resume text from PDF/DOCX files
    - Compares candidate profiles against job descriptions
    - Predicts job-fit scores using machine learning
    - Explains predictions using SHAP (Explainable AI)
    - Provides an interactive dashboard for analysis

    ### How It Works
    ```
    Resume + Job Description
            |
            v
    NLP Preprocessing
            |
            v
    TF-IDF Similarity + Skill Extraction
            |
            v
    Feature Engineering
            |
            v
    Machine Learning Model
            |
            v
    Fit Score + Fit Category
            |
            v
    SHAP Explanation
            |
            v
    Explainable Recommendation
    ```

    ### Technology Stack
    - **Python 3.10+** | **scikit-learn** | **XGBoost** | **SHAP**
    - **NLTK** | **TF-IDF** | **PyMuPDF** | **python-docx**
    - **Streamlit** | **Plotly** | **pandas** | **numpy**

    ### Fit Categories
    | Score Range | Category |
    |-------------|----------|
    | 75 - 100 | Strong Fit |
    | 50 - 74 | Moderate Fit |
    | 0 - 49 | Poor Fit |

    """)
    st.warning(TARGET_NOTE)
    render_disclaimer()


def _fit_gauge(score: float) -> go.Figure:
    """Create Plotly gauge chart for fit score."""
    if score >= 75:
        bar_color = "#27ae60"
    elif score >= 50:
        bar_color = "#f39c12"
    else:
        bar_color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 49], "color": "#fadbd8"},
                {"range": [50, 74], "color": "#fdebd0"},
                {"range": [75, 100], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": "Job Fit Score", "font": {"size": 18}},
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=30, r=30))
    return fig


def page_resume_screening() -> None:
    """Resume screening and analysis page."""
    st.title("Resume Screening")
    st.markdown("Upload a resume and provide a job description to analyze candidate fit.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resume Upload")
        resume_file = st.file_uploader(
            "Upload Resume (PDF or DOCX)",
            type=["pdf", "docx", "doc"],
            key="resume_upload",
        )

    with col2:
        st.subheader("Job Description")
        jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"], key="jd_upload")
        job_text = st.text_area(
            "Or paste Job Description here",
            height=200,
            placeholder="Paste the full job description text...",
            key="jd_text",
        )

    if jd_file is not None:
        try:
            job_text = jd_file.read().decode("utf-8")
            st.success("Job description loaded from file.")
        except Exception:
            st.error("Could not read job description file.")

    analyze = st.button("Analyze Resume", type="primary", use_container_width=True)

    if not analyze:
        return

    # Validation
    if resume_file is None:
        st.error("Please upload a resume file (PDF or DOCX).")
        return
    if not job_text or not job_text.strip():
        st.error("Please provide a job description (paste or upload .txt file).")
        return

    # Extract resume text
    try:
        resume_text = extract_text_from_bytes(resume_file.read(), resume_file.name)
        st.success("Resume extracted successfully.")
    except EmptyDocumentError as exc:
        st.error(f"Empty document: {exc}")
        return
    except UnsupportedFormatError as exc:
        st.error(str(exc))
        return
    except ResumeParserError as exc:
        st.error(f"Resume parsing error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error reading resume: {exc}")
        return

    # Run prediction
    try:
        with st.spinner("Analyzing candidate fit..."):
            prediction = predict_fit(resume_text, job_text.strip())
            explanation_raw = explain_prediction(resume_text, job_text.strip())
            explanation = build_explanation(prediction, explanation_raw)
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first: `python -m src.train_model`")
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        return

    render_disclaimer()

    # Results
    st.markdown("---")
    st.subheader("Candidate Fit Analysis")

    score = prediction["fit_score"]
    label = prediction["fit_label"]
    features = prediction["features"]

    col_gauge, col_info = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(_fit_gauge(score), use_container_width=True)
    with col_info:
        label_color = explanation["label_color"]
        st.markdown(f"### Classification: :{label_color}[{label}]")
        st.markdown(f"**Predicted job fit based on the trained model.**")
        st.markdown(f"Model: `{prediction.get('model_name', 'N/A')}`")

        m1, m2, m3 = st.columns(3)
        m1.metric("TF-IDF Similarity", f"{features.get('tfidf_similarity', 0):.3f}")
        m2.metric("Skill Match %", f"{features.get('skill_match_percentage', 0):.1f}%")
        m3.metric("Experience", f"{features.get('experience_years', 0):.0f} yrs")
        m4, m5, m6 = st.columns(3)
        m4.metric("Matched Skills", features.get("matched_skill_count", 0))
        m5.metric("Missing Skills", features.get("missing_skill_count", 0))
        m6.metric("Resume Words", features.get("resume_word_count", 0))

    # Matched skills
    st.subheader("Matched Skills")
    matched = prediction.get("matched_skills", [])
    if matched:
        cols = st.columns(min(len(matched), 6))
        for i, skill in enumerate(matched):
            cols[i % len(cols)].success(skill)
    else:
        st.info("No matched skills detected by the configured skill dictionary.")

    # Missing skills
    st.subheader("Skill Gaps")
    st.caption("Missing according to configured skill extraction; not proof candidate lacks the skill.")
    missing = prediction.get("missing_skills", [])
    if missing:
        cols = st.columns(min(len(missing), 6))
        for i, skill in enumerate(missing):
            cols[i % len(cols)].error(skill)
    else:
        st.info("No skill gaps detected.")

    # Human explanation
    st.subheader("Why did the model predict this fit?")
    st.markdown(explanation["summary"])

    col_pos, col_gap = st.columns(2)
    with col_pos:
        st.markdown("**Positive factors:**")
        for factor in explanation["positive_factors"]:
            st.markdown(f"- {factor}")
    with col_gap:
        st.markdown("**Skill gaps:**")
        for gap in explanation["skill_gaps"][:8]:
            st.markdown(f"- {gap}")

    # SHAP table
    impacts = explanation.get("feature_impacts", [])
    if impacts:
        st.subheader("Feature Impact (SHAP)")
        impact_df = pd.DataFrame(impacts)[["feature", "impact", "direction"]]
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

        # SHAP bar chart
        fig = px.bar(
            impact_df,
            x="impact",
            y="feature",
            color="direction",
            color_discrete_map={"Positive": "#27ae60", "Negative": "#e74c3c"},
            orientation="h",
            title="SHAP Feature Impact",
        )
        fig.update_layout(height=350, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Download report
    st.markdown("---")
    html_report = generate_html_report(prediction, explanation)
    st.download_button(
        label="Download Candidate Report (HTML)",
        data=html_report,
        file_name="candidate_analysis_report.html",
        mime="text/html",
        use_container_width=True,
    )


def page_model_insights() -> None:
    """Model insights and performance page."""
    st.title("Model Insights")

    try:
        _, _, metadata, _ = load_artifacts()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first: `python -m src.train_model`")
        return

    st.subheader("Selected Models")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Production Regressor", metadata.get("best_regressor_name", "N/A"))
    with col2:
        st.metric("Best Classifier", metadata.get("best_classifier_name", "N/A"))

    # Regression metrics
    st.subheader("Regression Performance (Production Model)")
    reg_metrics = metadata.get("regression_metrics", {})
    if reg_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{reg_metrics.get('mae', 0):.4f}")
        c2.metric("RMSE", f"{reg_metrics.get('rmse', 0):.4f}")
        c3.metric("R²", f"{reg_metrics.get('r2', 0):.4f}")

    # Classification metrics
    st.subheader("Classification Performance")
    clf_metrics = metadata.get("classification_metrics", {})
    if clf_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{clf_metrics.get('accuracy', 0):.4f}")
        c2.metric("Precision (macro)", f"{clf_metrics.get('precision_macro', 0):.4f}")
        c3.metric("Recall (macro)", f"{clf_metrics.get('recall_macro', 0):.4f}")
        c4.metric("F1 (macro)", f"{clf_metrics.get('f1_macro', 0):.4f}")

    # Feature importance
    st.subheader("Global Feature Importance")
    importances = get_global_feature_importance(metadata)
    if importances:
        imp_df = pd.DataFrame(importances)
        fig = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Most Influential Features",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    st.subheader("Model Comparison")
    tab1, tab2 = st.tabs(["Regressors", "Classifiers"])
    with tab1:
        reg_comp = metadata.get("regressor_comparison", {})
        if reg_comp:
            reg_df = pd.DataFrame(reg_comp).T.reset_index()
            reg_df.columns = ["Model", "MAE", "RMSE", "R2"]
            st.dataframe(reg_df, use_container_width=True, hide_index=True)
    with tab2:
        clf_comp = metadata.get("classifier_comparison", {})
        if clf_comp:
            clf_df = pd.DataFrame(clf_comp).T.reset_index()
            st.dataframe(clf_df, use_container_width=True, hide_index=True)

    # Confusion matrix image
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.subheader("Confusion Matrix")
        st.image(str(cm_path), use_container_width=True)

    # Training info
    st.subheader("Training Configuration")
    st.markdown(f"""
    - **Random State:** {metadata.get('random_state', 42)}
    - **Test Size:** {metadata.get('test_size', 0.2)}
    - **Training Samples:** {metadata.get('n_train', 'N/A')}
    - **Test Samples:** {metadata.get('n_test', 'N/A')}
    """)
    st.warning(TARGET_NOTE)
    render_disclaimer()


def page_dataset_analysis() -> None:
    """Dataset exploration and analysis page."""
    st.title("Dataset Analysis")

    if not DATASET_PATH.exists():
        st.error(f"Dataset not found at `{DATASET_PATH}`")
        st.info(
            "Download from Hugging Face:\n\n"
            "https://huggingface.co/datasets/batuhanmtl/job_resume_fit\n\n"
            f"Place the CSV at: `{DATASET_PATH}`"
        )
        return

    try:
        df = load_dataset()
        df = create_targets(df)
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        return

    # Summary metrics
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Job Categories", df["category"].nunique() if "category" in df.columns else "N/A")
    c3.metric("Avg Fit Score", f"{df['fit_score'].mean():.1f}" if "fit_score" in df.columns else "N/A")
    c4.metric("Unique Resumes", df["resume_text"].nunique() if "resume_text" in df.columns else "N/A")

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        if "fit_label" in df.columns:
            label_counts = df["fit_label"].value_counts().reindex([LABEL_POOR, LABEL_MODERATE, LABEL_STRONG]).fillna(0)
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Fit Label Distribution",
                color=label_counts.index,
                color_discrete_map={
                    LABEL_STRONG: "#27ae60",
                    LABEL_MODERATE: "#f39c12",
                    LABEL_POOR: "#e74c3c",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "category" in df.columns:
            cat_counts = df["category"].value_counts().head(15)
            fig = px.bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                title="Top Job Categories",
                labels={"x": "Count", "y": "Category"},
            )
            fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # Score histogram
    if "fit_score" in df.columns:
        fig = px.histogram(
            df,
            x="fit_score",
            nbins=30,
            title="Fit Score Distribution",
            labels={"fit_score": "Fit Score", "count": "Frequency"},
        )
        fig.add_vline(x=50, line_dash="dash", line_color="#f39c12", annotation_text="Moderate (50)")
        fig.add_vline(x=75, line_dash="dash", line_color="#27ae60", annotation_text="Strong (75)")
        st.plotly_chart(fig, use_container_width=True)

    # Top skills from dataset
    st.subheader("Top Skills in Dataset")
    if "resume_skill_list" in df.columns:
        all_skills: list[str] = []
        for val in df["resume_skill_list"].dropna().head(500):
            all_skills.extend(parse_skill_list(val))
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts().head(20)
            fig = px.bar(
                x=skill_counts.values,
                y=skill_counts.index,
                orientation="h",
                title="Top 20 Resume Skills (from dataset)",
                labels={"x": "Count", "y": "Skill"},
            )
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance from saved model
    if FEATURE_METADATA_PATH.exists():
        metadata = joblib.load(FEATURE_METADATA_PATH)
        importances = get_global_feature_importance(metadata)
        if importances:
            st.subheader("Model Feature Importance")
            imp_df = pd.DataFrame(importances)
            fig = px.bar(
                imp_df.head(10),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Feature Importances",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.warning(TARGET_NOTE)


def run_dashboard() -> None:
    """Main dashboard entry with sidebar navigation."""
    st.set_page_config(
        page_title="Resume-Job Fit Prediction",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Resume Screening", "Model Insights", "Dataset Analysis"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI Resume Screening**")
    st.sidebar.markdown("Explainable ML Project")

    pages = {
        "Home": page_home,
        "Resume Screening": page_resume_screening,
        "Model Insights": page_model_insights,
        "Dataset Analysis": page_dataset_analysis,
    }

    pages[page]()
