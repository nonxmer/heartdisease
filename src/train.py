"""
train.py
--------
Trains 5 classifiers on the preprocessed Heart Disease dataset:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost (Gradient Boosting)

Saves each model to models/ directory.
Prints accuracy for each model and identifies the best one.
"""

import os
import sys
import time
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Add src/ to path when running as script
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_heart_disease_data
from preprocessing import preprocess


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_models(num_classes: int) -> dict:
    """Returns a dictionary of model name → sklearn estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr", random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="euclidean"
        ),
    }


def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    save: bool = True,
) -> dict:
    """
    Trains all models and returns a results dictionary.

    Returns
    -------
    results : dict
        { model_name: {"model": estimator, "accuracy": float, "train_time": float} }
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_models(num_classes)
    results = {}

    print("=" * 60)
    print("🚀 Training Models on Heart Disease Dataset")
    print("=" * 60)

    best_acc = 0
    best_name = ""

    for name, model in models.items():
        print(f"\n🔄 Training: {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results[name] = {
            "model":      model,
            "accuracy":   acc,
            "train_time": elapsed,
            "predictions": preds,
        }

        print(f"   ✅ Accuracy : {acc * 100:.2f}%  |  Time: {elapsed:.2f}s")

        if save:
            model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
            joblib.dump(model, model_path)

        if acc > best_acc:
            best_acc = acc
            best_name = name

    print("\n" + "=" * 60)
    print(f"🏆 Best Model: {best_name}  ({best_acc * 100:.2f}% accuracy)")
    print("=" * 60)

    if save:
        best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(results[best_name]["model"], best_path)
        print(f"💾 Best model saved to: {best_path}")

    return results, best_name


if __name__ == "__main__":
    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(X, y)
    num_classes = len(np.unique(y_train))
    results, best = train_all_models(X_train, X_test, y_train, y_test, num_classes)
