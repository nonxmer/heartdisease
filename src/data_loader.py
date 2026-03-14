"""
data_loader.py
--------------
Loads the Cleveland Heart Disease dataset from the UCI ML Repository
using the `ucimlrepo` package. Returns raw features (X) and target (y)
where y is the `num` column (0=no disease, 1-4=different heart disease types).
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_heart_disease_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetches the Cleveland Heart Disease dataset from UCI ML Repository.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 13 clinical attributes.
    y : pd.Series
        Target labels (0 = healthy, 1-4 = heart disease severity/type).
    """
    print("📥 Fetching Cleveland Heart Disease Dataset from UCI...")
    heart_disease = fetch_ucirepo(id=45)  # ID 45 = Heart Disease dataset

    X: pd.DataFrame = heart_disease.data.features
    y: pd.Series = heart_disease.data.targets.squeeze()

    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Target classes: {sorted(y.unique())}")
    print(f"\n📋 Feature columns:\n   {list(X.columns)}")
    print(f"\n📊 Class distribution:\n{y.value_counts().sort_index()}\n")

    return X, y


def get_feature_descriptions() -> dict:
    """Returns a human-readable description for each feature."""
    return {
        "age":      "Age in years",
        "sex":      "Sex (1=male, 0=female)",
        "cp":       "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol":     "Serum cholesterol (mg/dl)",
        "fbs":      "Fasting blood sugar > 120 mg/dl (1=True)",
        "restecg":  "Resting ECG results (0-2)",
        "thalach":  "Max heart rate achieved",
        "exang":    "Exercise induced angina (1=Yes)",
        "oldpeak":  "ST depression induced by exercise",
        "slope":    "Slope of peak exercise ST segment (0-2)",
        "ca":       "Number of major vessels colored by fluoroscopy (0-3)",
        "thal":     "Thalassemia type (1=normal, 2=fixed defect, 3=reversible defect)",
    }


if __name__ == "__main__":
    X, y = load_heart_disease_data()
    print("\nFirst 5 rows of features:")
    print(X.head())
