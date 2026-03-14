"""
run.py
------
Main entry point for the Heart Disease ML pipeline.
Runs the complete pipeline end-to-end:
  1. Load data
  2. Preprocess
  3. Train all models
  4. Evaluate (classification report, confusion matrix, ROC)
  5. Compare models (accuracy, F1, training time charts)

Usage:
    python run.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_heart_disease_data
from preprocessing import preprocess
from train import train_all_models
from evaluate import evaluate_all
from model_comparison import compare_all


def main():
    print("\n" + "=" * 65)
    print("  ❤️   Heart Disease Prediction & Classification Pipeline")
    print("=" * 65)

    # Step 1: Load data
    X, y = load_heart_disease_data()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, scaler, features = preprocess(X, y)
    num_classes = len(np.unique(y_train))

    # Step 3: Train
    results, best_name = train_all_models(X_train, X_test, y_train, y_test, num_classes, save=True)

    # Step 4: Evaluate
    evaluate_all(results, X_test, y_test)

    # Step 5: Compare
    compare_all(results, y_test)

    print("\n" + "=" * 65)
    print(f"  🏁  Pipeline Complete!")
    print(f"  🏆  Best Model: {best_name}")
    print(f"  📂  Saved models → models/")
    print(f"  📊  Saved plots  → plots/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
