"""
preprocessing.py
----------------
Handles all data preprocessing for the Heart Disease dataset:
  - Missing value imputation
  - Feature encoding
  - Numerical scaling
  - Train/test splitting (stratified)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Columns that are categorical (will be mode-imputed)
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]

# Columns that are numerical (will be mean-imputed + scaled)
NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    binarize_target: bool = False,
) -> tuple:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    X             : Raw feature DataFrame
    y             : Raw target Series (0-4)
    test_size     : Fraction of data to use for testing
    random_state  : Random seed for reproducibility
    binarize_target : If True, map y > 0 → 1 (binary problem)

    Returns
    -------
    X_train, X_test, y_train, y_test : Split and scaled arrays
    scaler                           : Fitted StandardScaler (for later inference)
    feature_names                    : List of feature names after preprocessing
    """
    print("🔧 Starting preprocessing pipeline...")

    # --- 1. Copy to avoid mutating the original ---
    X = X.copy()
    y = y.copy()

    # --- 2. Optionally binarize target ---
    if binarize_target:
        y = (y > 0).astype(int)
        print(f"   Binary target: healthy={( y==0).sum()}, disease={(y==1).sum()}")
    else:
        print(f"   Multi-class target distribution:\n{y.value_counts().sort_index().to_string()}")

    # --- 3. Impute missing values ---
    # Numerical: mean imputation
    num_imputer = SimpleImputer(strategy="mean")
    X[NUMERICAL_COLS] = num_imputer.fit_transform(X[NUMERICAL_COLS])

    # Categorical: mode imputation
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[CATEGORICAL_COLS] = cat_imputer.fit_transform(X[CATEGORICAL_COLS])

    print(f"   Missing values after imputation: {X.isnull().sum().sum()}")

    # --- 4. Ensure correct dtypes ---
    X[CATEGORICAL_COLS] = X[CATEGORICAL_COLS].astype(float)
    X[NUMERICAL_COLS] = X[NUMERICAL_COLS].astype(float)

    feature_names = NUMERICAL_COLS + CATEGORICAL_COLS

    # --- 5. Reorder columns consistently ---
    X = X[feature_names]

    # --- 6. Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # --- 7. Scale numerical features ---
    scaler = StandardScaler()
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    print("✅ Preprocessing complete!\n")
    return X_train.values, X_test.values, y_train.values, y_test.values, scaler, feature_names


if __name__ == "__main__":
    from data_loader import load_heart_disease_data
    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(X, y)
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Features   :", features)
