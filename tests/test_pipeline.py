"""
test_pipeline.py
----------------
Unit tests for the Heart Disease ML pipeline.
Run with: python -m pytest tests/test_pipeline.py -v
"""

import sys
import os
import numpy as np
import pytest

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


# --------------------------------------------------------------------------- #
#  Data Loading Tests                                                         #
# --------------------------------------------------------------------------- #
class TestDataLoader:
    """Tests for data_loader.py"""

    def test_load_returns_nonempty(self):
        """Dataset should have more than 100 rows."""
        from data_loader import load_heart_disease_data
        X, y = load_heart_disease_data()
        assert len(X) > 100, "Dataset too small — something went wrong with loading."

    def test_feature_count(self):
        """Should have exactly 13 features."""
        from data_loader import load_heart_disease_data
        X, y = load_heart_disease_data()
        assert X.shape[1] == 13, f"Expected 13 features, got {X.shape[1]}"

    def test_target_classes(self):
        """Target should have values in {0, 1, 2, 3, 4}."""
        from data_loader import load_heart_disease_data
        X, y = load_heart_disease_data()
        unique = set(y.unique())
        assert unique.issubset({0, 1, 2, 3, 4}), f"Unexpected target values: {unique}"

    def test_target_has_multiple_classes(self):
        """Target should have at least 2 distinct class values."""
        from data_loader import load_heart_disease_data
        X, y = load_heart_disease_data()
        assert y.nunique() >= 2, "Target does not have multiple classes."


# --------------------------------------------------------------------------- #
#  Preprocessing Tests                                                        #
# --------------------------------------------------------------------------- #
class TestPreprocessing:
    """Tests for preprocessing.py"""

    @pytest.fixture(scope="class")
    def data(self):
        from data_loader import load_heart_disease_data
        from preprocessing import preprocess
        X, y = load_heart_disease_data()
        return preprocess(X, y)

    def test_no_nan_in_train(self, data):
        X_train = data[0]
        assert not np.any(np.isnan(X_train)), "NaN values found in X_train after preprocessing."

    def test_no_nan_in_test(self, data):
        X_test = data[1]
        assert not np.any(np.isnan(X_test)), "NaN values found in X_test after preprocessing."

    def test_shapes_match(self, data):
        X_train, X_test, y_train, y_test, _, _ = data
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train row counts differ."
        assert X_test.shape[0] == y_test.shape[0], "X_test and y_test row counts differ."

    def test_feature_count(self, data):
        X_train = data[0]
        assert X_train.shape[1] == 13, f"Expected 13 features, got {X_train.shape[1]}"

    def test_test_size(self, data):
        """Test split should be approximately 20% of the data."""
        X_train, X_test = data[0], data[1]
        total = X_train.shape[0] + X_test.shape[0]
        test_fraction = X_test.shape[0] / total
        assert 0.15 <= test_fraction <= 0.25, f"Unexpected test split fraction: {test_fraction:.2f}"


# --------------------------------------------------------------------------- #
#  Training Tests                                                             #
# --------------------------------------------------------------------------- #
class TestTraining:
    """Tests for train.py"""

    @pytest.fixture(scope="class")
    def trained(self):
        from data_loader import load_heart_disease_data
        from preprocessing import preprocess
        from train import train_all_models
        X, y = load_heart_disease_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)
        num_classes = len(np.unique(y_train))
        results, best = train_all_models(X_train, X_test, y_train, y_test, num_classes, save=False)
        return results, X_test, y_test

    def test_all_models_trained(self, trained):
        results, _, _ = trained
        expected = {"Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost"}
        assert set(results.keys()) == expected, f"Missing models: {expected - set(results.keys())}"

    def test_accuracies_above_chance(self, trained):
        """Each model should beat naive chance (20% for 5 classes)."""
        results, _, _ = trained
        for name, info in results.items():
            assert info["accuracy"] > 0.20, f"{name} accuracy too low: {info['accuracy']:.2f}"

    def test_predictions_shape(self, trained):
        results, _, y_test = trained
        for name, info in results.items():
            assert len(info["predictions"]) == len(y_test), \
                f"{name}: predictions length mismatch."

    def test_best_model_accuracy(self, trained):
        """Best model should achieve at least 50% accuracy (generous threshold)."""
        results, _, _ = trained
        best_acc = max(info["accuracy"] for info in results.values())
        assert best_acc >= 0.50, f"Best model accuracy too low: {best_acc:.2f}"
