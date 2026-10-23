"""Model evaluation metrics and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR, FIT_LABELS, LABEL_MODERATE, LABEL_POOR, LABEL_STRONG


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compute classification metrics."""
    labels = labels or FIT_LABELS

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["roc_auc_ovr"] = None

    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    labels: pd.Series,
    title: str = "Fit Label Distribution",
    save_path: Optional[Path] = None,
) -> None:
    """Plot class distribution bar chart."""
    plt.figure(figsize=(8, 5))
    counts = labels.value_counts().reindex([LABEL_POOR, LABEL_MODERATE, LABEL_STRONG]).fillna(0)
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    counts.plot(kind="bar", color=colors)
    plt.title(title)
    plt.xlabel("Fit Label")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None,
    top_n: int = 15,
) -> None:
    """Plot feature importance bar chart."""
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=names, hue=names, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "f1_macro",
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
) -> None:
    """Plot model comparison bar chart."""
    models = list(results.keys())
    scores = [results[m].get(metric, 0) for m in models]

    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("husl", len(models))
    bars = plt.bar(models, scores, color=colors)
    plt.title(f"{title} ({metric})")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(
    scores: pd.Series,
    title: str = "Fit Score Distribution",
    save_path: Optional[Path] = None,
) -> None:
    """Plot fit score histogram."""
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    plt.axvline(x=50, color="#f39c12", linestyle="--", label="Moderate threshold (50)")
    plt.axvline(x=75, color="#27ae60", linestyle="--", label="Strong threshold (75)")
    plt.title(title)
    plt.xlabel("Fit Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_evaluation_plots(
    df: pd.DataFrame,
    clf_metrics: dict[str, Any],
    reg_metrics: dict[str, float],
    clf_comparison: dict[str, dict[str, float]],
    reg_comparison: dict[str, dict[str, float]],
    feature_names: list[str],
    feature_importances: np.ndarray,
    best_clf_name: str,
    output_dir: Optional[Path] = None,
) -> list[str]:
    """Generate all evaluation plots and return list of saved paths."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Confusion matrix
    if "confusion_matrix" in clf_metrics:
        cm = np.array(clf_metrics["confusion_matrix"])
        path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, FIT_LABELS, f"Confusion Matrix ({best_clf_name})", path)
        saved.append(str(path))

    # Class distribution
    if "fit_label" in df.columns:
        path = output_dir / "class_distribution.png"
        plot_class_distribution(df["fit_label"], save_path=path)
        saved.append(str(path))

    # Feature importance
    path = output_dir / "feature_importance.png"
    plot_feature_importance(feature_names, feature_importances, save_path=path)
    saved.append(str(path))

    # Model comparison - classification
    if clf_comparison:
        path = output_dir / "model_comparison_classification.png"
        plot_model_comparison(clf_comparison, metric="f1_macro",
                              title="Classification Model Comparison", save_path=path)
        saved.append(str(path))

    # Model comparison - regression
    if reg_comparison:
        path = output_dir / "model_comparison_regression.png"
        plot_model_comparison(reg_comparison, metric="rmse",
                              title="Regression Model Comparison (lower is better)",
                              save_path=path)
        saved.append(str(path))

    # Score distribution
    if "fit_score" in df.columns:
        path = output_dir / "score_distribution.png"
        plot_score_distribution(df["fit_score"], save_path=path)
        saved.append(str(path))

    return saved
