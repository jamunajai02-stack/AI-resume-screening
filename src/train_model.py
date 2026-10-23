"""Model training pipeline - one command workflow."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    DATASET_PATH,
    DATASET_URL,
    FEATURE_METADATA_PATH,
    FEATURE_NAMES,
    FIT_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SHAP_BACKGROUND_PATH,
    TARGET_NOTE,
    TEST_SIZE,
    TFIDF_VECTORIZER_PATH,
)
from src.data_preprocessing import prepare_dataset
from src.evaluate_model import (
    evaluate_classification,
    evaluate_regression,
    generate_evaluation_plots,
)
from src.feature_engineering import (
    build_feature_matrix,
    fit_tfidf_vectorizer,
    get_combined_texts_from_df,
)


def download_dataset(dest: Path = DATASET_PATH) -> bool:
    """Attempt to download dataset from Hugging Face."""
    print(f"Dataset not found at {dest}")
    print(f"Attempting download from Hugging Face...")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, str(dest))
        print(f"Downloaded dataset to {dest}")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        print(f"\nPlease place job_resume_fit.csv at:\n  {dest.absolute()}")
        return False


def get_classifiers() -> dict[str, Any]:
    """Return dictionary of classification models."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric="mlogloss", verbosity=0,
        )
    except ImportError:
        print("XGBoost not available, skipping XGBClassifier.")
    return models


def get_regressors() -> dict[str, Any]:
    """Return dictionary of regression models."""
    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, random_state=RANDOM_STATE, verbosity=0
        )
    except ImportError:
        print("XGBoost not available, skipping XGBRegressor.")
    return models


def select_best_classifier(
    results: dict[str, dict[str, Any]],
) -> tuple[str, Any, dict[str, Any]]:
    """Select best classifier by macro F1, then ROC-AUC."""
    best_name = max(
        results.keys(),
        key=lambda k: (
            results[k]["metrics"].get("f1_macro", 0),
            results[k]["metrics"].get("roc_auc_ovr") or 0,
        ),
    )
    return best_name, results[best_name]["model"], results[best_name]["metrics"]


def select_best_regressor(
    results: dict[str, dict[str, Any]],
) -> tuple[str, Any, dict[str, float]]:
    """Select best regressor by RMSE (lower), then R2 (higher)."""
    best_name = min(
        results.keys(),
        key=lambda k: (
            results[k]["metrics"]["rmse"],
            -results[k]["metrics"]["r2"],
        ),
    )
    return best_name, results[best_name]["model"], results[best_name]["metrics"]


def get_feature_importances(model: Any, feature_names: list[str]) -> np.ndarray:
    """Extract feature importances from model."""
    # Unwrap pipeline
    est = model
    if hasattr(model, "named_steps"):
        for step in ("clf", "reg"):
            if step in model.named_steps:
                est = model.named_steps[step]
                break

    if hasattr(est, "feature_importances_"):
        return np.array(est.feature_importances_)
    if hasattr(est, "coef_"):
        coef = est.coef_
        if coef.ndim > 1:
            return np.abs(coef).mean(axis=0)
        return np.abs(coef)
    return np.ones(len(feature_names)) / len(feature_names)


def train() -> dict[str, Any]:
    """Full training pipeline."""
    print("=" * 60)
    print("RESUME-JOB FIT MODEL TRAINING")
    print("=" * 60)

    # Load or download dataset
    if not DATASET_PATH.exists():
        if not download_dataset():
            sys.exit(1)

    # Prepare dataset
    df, inspection_report = prepare_dataset()

    # Train/test split indices first (for TF-IDF fitting on train only)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["fit_label"],
    )
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    print(f"\nTrain size: {len(df_train)}, Test size: {len(df_test)}")

    # Fit TF-IDF on training texts only
    print("\nFitting TF-IDF vectorizer on training data...")
    train_texts = get_combined_texts_from_df(df_train)
    vectorizer = fit_tfidf_vectorizer(train_texts)

    # Build feature matrices
    print("Building feature matrices...")
    X_train, _ = build_feature_matrix(df_train, vectorizer)
    X_test, _ = build_feature_matrix(df_test, vectorizer)
    y_train_score = df_train["fit_score"].values
    y_test_score = df_test["fit_score"].values
    y_train_label = df_train["fit_label"].values
    y_test_label = df_test["fit_label"].values

    # Label encoder for classifiers
    le = LabelEncoder()
    le.fit(["Poor Fit", "Moderate Fit", "Strong Fit"])
    y_train_enc = le.transform(y_train_label)
    y_test_enc = le.transform(y_test_label)

    # --- Classification ---
    print("\n" + "-" * 40)
    print("TRAINING CLASSIFIERS")
    print("-" * 40)
    clf_results: dict[str, dict[str, Any]] = {}
    clf_comparison: dict[str, dict[str, float]] = {}

    for name, model in get_classifiers().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", None), "predict_proba"):
            y_proba = model.predict_proba(X_test)

        y_pred_labels = le.inverse_transform(y_pred)
        metrics = evaluate_classification(y_test_label, y_pred_labels, y_proba)
        clf_results[name] = {"model": model, "metrics": metrics}
        clf_comparison[name] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        }
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

    best_clf_name, best_clf, best_clf_metrics = select_best_classifier(clf_results)
    print(f"\nBest classifier: {best_clf_name} (F1={best_clf_metrics['f1_macro']:.4f})")

    # --- Regression ---
    print("\n" + "-" * 40)
    print("TRAINING REGRESSORS")
    print("-" * 40)
    reg_results: dict[str, dict[str, Any]] = {}
    reg_comparison: dict[str, dict[str, float]] = {}

    for name, model in get_regressors().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train_score)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 100)
        metrics = evaluate_regression(y_test_score, y_pred)
        reg_results[name] = {"model": model, "metrics": metrics}
        reg_comparison[name] = metrics
        print(f"  MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

    best_reg_name, best_reg, best_reg_metrics = select_best_regressor(reg_results)
    print(f"\nBest regressor: {best_reg_name} (RMSE={best_reg_metrics['rmse']:.4f}, R2={best_reg_metrics['r2']:.4f})")

    # Feature importances from best regressor
    importances = get_feature_importances(best_reg, FEATURE_NAMES)

    # Generate plots
    print("\nGenerating evaluation plots...")
    saved_plots = generate_evaluation_plots(
        df=df,
        clf_metrics=best_clf_metrics,
        reg_metrics=best_reg_metrics,
        clf_comparison=clf_comparison,
        reg_comparison=reg_comparison,
        feature_names=FEATURE_NAMES,
        feature_importances=importances,
        best_clf_name=best_clf_name,
    )
    print(f"Saved {len(saved_plots)} plots to outputs/figures/")

    # SHAP background sample (from training data)
    n_bg = min(100, len(X_train))
    rng = np.random.RandomState(RANDOM_STATE)
    bg_indices = rng.choice(len(X_train), n_bg, replace=False)
    shap_background = X_train[bg_indices]

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_reg, FIT_MODEL_PATH)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(shap_background, SHAP_BACKGROUND_PATH)

    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances.tolist(),
        "best_regressor_name": best_reg_name,
        "best_classifier_name": best_clf_name,
        "regression_metrics": best_reg_metrics,
        "classification_metrics": {
            k: v for k, v in best_clf_metrics.items()
            if k != "classification_report"
        },
        "regressor_comparison": reg_comparison,
        "classifier_comparison": clf_comparison,
        "target_note": TARGET_NOTE,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "label_encoder_classes": le.classes_.tolist(),
        "inspection_summary": {
            "shape": inspection_report.get("shape"),
            "n_categories": inspection_report.get("n_categories"),
            "label_distribution": inspection_report.get("label_distribution"),
        },
    }
    joblib.dump(metadata, FEATURE_METADATA_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Production model (regressor): {best_reg_name}")
    print(f"  MAE:  {best_reg_metrics['mae']:.4f}")
    print(f"  RMSE: {best_reg_metrics['rmse']:.4f}")
    print(f"  R2:   {best_reg_metrics['r2']:.4f}")
    print(f"\nBest classifier: {best_clf_name}")
    print(f"  Accuracy: {best_clf_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {best_clf_metrics['f1_macro']:.4f}")
    print(f"\nSaved artifacts:")
    print(f"  {FIT_MODEL_PATH}")
    print(f"  {TFIDF_VECTORIZER_PATH}")
    print(f"  {FEATURE_METADATA_PATH}")
    print(f"  {SHAP_BACKGROUND_PATH}")

    return metadata


def main() -> None:
    """Entry point for python -m src.train_model"""
    train()


if __name__ == "__main__":
    main()
