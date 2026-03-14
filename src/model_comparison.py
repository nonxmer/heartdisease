"""
model_comparison.py
-------------------
Generates a side-by-side visual comparison of all trained models.
Plots:
  1. Accuracy bar chart (with value labels)
  2. Training time bar chart
  3. Combined accuracy + F1 grouped bar chart
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
COLORS = ["#4361EE", "#3A0CA3", "#F72585", "#4CC9F0", "#7209B7"]


def plot_accuracy_comparison(results: dict):
    """Bar chart of accuracy per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accuracies, color=COLORS[:len(names)], width=0.55, edgecolor="white", linewidth=0.8)

    # Value labels on top
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticklabels(names, fontsize=11)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Baseline (50%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = os.path.join(PLOTS_DIR, "comparison_accuracy.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"💾 Accuracy comparison saved: {fname}")


def plot_training_time_comparison(results: dict):
    """Bar chart of training time per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = list(results.keys())
    times = [results[n]["train_time"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, times, color=COLORS[:len(names)], height=0.5, edgecolor="white")

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s",
            va="center", fontsize=10, fontweight="bold"
        )

    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Model Training Time Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.set_yticklabels(names, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = os.path.join(PLOTS_DIR, "comparison_training_time.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"💾 Training time comparison saved: {fname}")


def plot_accuracy_vs_f1(results: dict, y_test: np.ndarray):
    """Grouped bar chart: accuracy vs macro-F1 per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]
    f1_scores = [
        f1_score(y_test, results[n]["predictions"], average="macro", zero_division=0) * 100
        for n in names
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)",
                   color="#4361EE", edgecolor="white")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="Macro F1 (%)",
                   color="#F72585", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Accuracy vs Macro F1 — All Models", fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = os.path.join(PLOTS_DIR, "comparison_accuracy_vs_f1.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"💾 Accuracy vs F1 comparison saved: {fname}")


def compare_all(results: dict, y_test: np.ndarray):
    """Runs all comparison plots."""
    print("\n📊 Generating model comparison plots...")
    plot_accuracy_comparison(results)
    plot_training_time_comparison(results)
    plot_accuracy_vs_f1(results, y_test)
    print("✅ Comparison plots complete!")


if __name__ == "__main__":
    from data_loader import load_heart_disease_data
    from preprocessing import preprocess
    from train import train_all_models

    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(X, y)
    num_classes = len(np.unique(y_train))
    results, best = train_all_models(X_train, X_test, y_train, y_test, num_classes)
    compare_all(results, y_test)
