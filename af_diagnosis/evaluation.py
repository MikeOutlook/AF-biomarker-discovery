"""Model evaluation and metrics."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Try to import matplotlib (optional dependency)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def evaluate(y_true, y_pred, y_proba=None, labels=None):
    """Compute comprehensive evaluation metrics."""
    if labels is None:
        labels = ["Sinus Rhythm", "AF"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def evaluate_model(model, X_test, y_test, labels=None):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass

    return evaluate(y_test.values, y_pred, y_proba, labels)


def print_evaluation(metrics):
    """Print evaluation results nicely formatted."""
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    print("\nOverall Metrics:")
    print("  Accuracy:  %.2f%%" % (metrics["accuracy"] * 100))
    print("  Precision: %.2f%%" % (metrics["precision"] * 100))
    print("  Recall:    %.2f%%" % (metrics["recall"] * 100))
    print("  F1 Score:  %.2f%%" % (metrics["f1"] * 100))

    if "auc" in metrics and metrics["auc"] is not None:
        print("  AUC:       %.2f%%" % (metrics["auc"] * 100))

    print("\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print("  Predicted    SR    AF")
    print("  Actual SR  [%3d  %3d]" % (cm[0, 0], cm[0, 1]))
    print("  Actual AF [%3d  %3d]" % (cm[1, 0], cm[1, 1]))


def plot_confusion_matrix(cm, labels=None, save_path=None):
    """Plot confusion matrix heatmap."""
    if not HAS_PLOT:
        print("Warning: matplotlib not available, skipping plot")
        return

    if labels is None:
        labels = ["Sinus Rhythm", "AF"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved confusion matrix to %s" % save_path)

    plt.close()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Plot ROC curve."""
    if not HAS_PLOT:
        print("Warning: matplotlib not available, skipping plot")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label="ROC (AUC = %.2f)" % auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved ROC curve to %s" % save_path)

    plt.close()


def plot_feature_importance(feature_scores, top_n=20, save_path=None):
    """Plot feature importance bar chart."""
    if not HAS_PLOT:
        print("Warning: matplotlib not available, skipping plot")
        return

    top_features = feature_scores.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["score"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("ANOVA Score")
    plt.ylabel("Gene")
    plt.title("Top %d Discriminative Genes" % top_n)
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved feature importance to %s" % save_path)

    plt.close()


def plot_all(metrics, y_proba=None, feature_scores=None, output_dir="results/"):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)

    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, save_path=output_dir + "/confusion_matrix.png")

    if feature_scores is not None:
        plot_feature_importance(feature_scores, save_path=output_dir + "/feature_importance.png")