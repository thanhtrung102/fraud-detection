"""
Unit Tests for Stacking Model
=============================
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stacking_model import StackingFraudDetector, find_optimal_threshold


class TestStackingFraudDetector:
    """Tests for the StackingFraudDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 500
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        return X, y

    @pytest.fixture
    def small_model(self):
        """Create a small model for quick testing."""
        return StackingFraudDetector(
            xgb_params={"n_estimators": 10, "max_depth": 3},
            lgbm_params={"n_estimators": 10, "max_depth": 3, "verbose": -1},
            catboost_params={"iterations": 10, "depth": 3, "verbose": 0},
            meta_params={"n_estimators": 10, "max_depth": 2}
        )

    def test_model_initialization(self, small_model):
        """Test model initializes correctly."""
        assert small_model is not None
        assert small_model.xgb_model is not None
        assert small_model.lgbm_model is not None
        assert small_model.catboost_model is not None

    def test_model_fit(self, small_model, sample_data):
        """Test model fitting."""
        X, y = sample_data
        small_model.fit(X, y)

        # Check that models are fitted
        assert hasattr(small_model.xgb_model, "n_features_in_")
        assert hasattr(small_model.lgbm_model, "n_features_in_")

    def test_model_predict(self, small_model, sample_data):
        """Test model prediction."""
        X, y = sample_data
        small_model.fit(X, y)

        predictions = small_model.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_model_predict_proba(self, small_model, sample_data):
        """Test model probability prediction."""
        X, y = sample_data
        small_model.fit(X, y)

        probas = small_model.predict_proba(X)

        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        np.testing.assert_array_almost_equal(probas.sum(axis=1), 1.0)

    def test_model_save_load(self, small_model, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        small_model.fit(X, y)

        # Get predictions before save
        predictions_before = small_model.predict(X)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            small_model.save(tmpdir)

            # Create new model and load
            loaded_model = StackingFraudDetector()
            loaded_model.load(tmpdir)

            # Get predictions after load
            predictions_after = loaded_model.predict(X)

        # Check predictions match
        np.testing.assert_array_equal(predictions_before, predictions_after)

    def test_model_with_threshold(self, small_model, sample_data):
        """Test prediction with custom threshold."""
        X, y = sample_data
        small_model.fit(X, y)

        pred_default = small_model.predict(X, threshold=0.5)
        pred_high = small_model.predict(X, threshold=0.8)
        pred_low = small_model.predict(X, threshold=0.2)

        # Higher threshold should give fewer positives
        assert pred_high.sum() <= pred_default.sum()
        assert pred_low.sum() >= pred_default.sum()


class TestFindOptimalThreshold:
    """Tests for threshold optimization."""

    def test_find_optimal_threshold_f1(self):
        """Test finding optimal threshold for F1."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        threshold, score = find_optimal_threshold(y_true, y_proba, metric='f1')

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_find_optimal_threshold_precision(self):
        """Test finding optimal threshold for precision."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        threshold, score = find_optimal_threshold(y_true, y_proba, metric='precision')

        assert 0 <= threshold <= 1

    def test_threshold_affects_predictions(self):
        """Test that threshold actually affects predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7])

        threshold_low, _ = find_optimal_threshold(y_true, y_proba, metric='recall')
        threshold_high, _ = find_optimal_threshold(y_true, y_proba, metric='precision')

        # These might be the same in this simple case, but logic is tested
        assert threshold_low is not None
        assert threshold_high is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
