"""
evaluate.py
-----------
Evaluates trained models and generates:
  - Detailed classification reports (per-class precision/recall/F1)
  - Confusion matrix heatmaps
  - Multi-class ROC curves (One-vs-Rest)
All plots are saved to the plots/ directory.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
CLASS_LABELS = {
    0: "Healthy",
    1: "Type 1 HD",
    2: "Type 2 HD",
    3: "Type 3 HD",
    4: "Type 4 HD",
}

PALETTE = ["#4CAF50", "#FF5722", "#2196F3", "#FF9800", "#9C27B0"]


def make_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def print_classification_report(name: str, y_test, y_pred, classes):
    """Prints a per-class classification report for one model."""
    print(f"\n{'='*60}")
    print(f"📊 Classification Report: {name}")
    print("=" * 60)
    target_names = [CLASS_LABELS.get(c, str(c)) for c in sorted(classes)]
    print(classification_report(y_test, y_pred, target_names=target_names))


def plot_confusion_matrix(name: str, y_test, y_pred, classes):
    """Saves a confusion matrix heatmap for one model."""
    make_plots_dir()

    cm = confusion_matrix(y_test, y_pred, labels=sorted(classes))
    labels = [CLASS_LABELS.get(c, str(c)) for c in sorted(classes)]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = os.path.join(PLOTS_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"   💾 Confusion matrix saved: {fname}")


def plot_roc_curves(name: str, model, X_test, y_test, classes):
    """Saves multi-class ROC curves (One-vs-Rest) for one model."""
    make_plots_dir()
    classes_sorted = sorted(classes)
    n_classes = len(classes_sorted)

    # Binarize labels for OvR ROC
    y_bin = label_binarize(y_test, classes=classes_sorted)

    # Get probability scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        # normalize to (0,1) range for plotting
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-8)
    else:
        print(f"   ⚠️  {name} doesn't support probability output. Skipping ROC.")
        return

    # If only 2 columns returned for multi-class, skip
    if y_score.shape[1] < n_classes:
        print(f"   ⚠️  {name} returned fewer probability columns than classes. Skipping ROC.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes_sorted):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        label = CLASS_LABELS.get(cls, f"Class {cls}")
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)],
                label=f"{label} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(PLOTS_DIR, f"roc_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"   💾 ROC curves saved     : {fname}")


def evaluate_all(results: dict, X_test, y_test):
    """
    Runs full evaluation for every model in results dict.

    Parameters
    ----------
    results : dict  (from train.train_all_models)
    X_test  : np.ndarray
    y_test  : np.ndarray
    """
    classes = np.unique(y_test)

    for name, info in results.items():
        model = info["model"]
        y_pred = info["predictions"]

        print_classification_report(name, y_test, y_pred, classes)
        plot_confusion_matrix(name, y_test, y_pred, classes)
        plot_roc_curves(name, model, X_test, y_test, classes)

    print("\n✅ All evaluation plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    from data_loader import load_heart_disease_data
    from preprocessing import preprocess
    from train import train_all_models

    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(X, y)
    num_classes = len(np.unique(y_train))
    results, best = train_all_models(X_train, X_test, y_train, y_test, num_classes)
    evaluate_all(results, X_test, y_test)
