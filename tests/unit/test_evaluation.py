"""
Unit Tests for Evaluation Module
================================
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import compare_with_paper, compute_metrics


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.9])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics for all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_metrics_range(self):
        """Test that all metrics are in valid range."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.random(100)

        metrics = compute_metrics(y_true, y_pred, y_proba)

        for key in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0 <= metrics[key] <= 1, f"{key} out of range"

        if "auc_roc" in metrics:
            assert 0 <= metrics["auc_roc"] <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        metrics = compute_metrics(y_true, y_pred)

        cm = metrics["confusion_matrix"]
        assert "true_negatives" in cm
        assert "false_positives" in cm
        assert "false_negatives" in cm
        assert "true_positives" in cm

        # Check totals
        total = (
            cm["true_negatives"]
            + cm["false_positives"]
            + cm["false_negatives"]
            + cm["true_positives"]
        )
        assert total == len(y_true)

    def test_no_proba(self):
        """Test metrics without probability scores."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        metrics = compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "auc_roc" not in metrics


class TestCompareWithPaper:
    """Tests for comparison with paper results."""

    def test_compare_with_paper(self):
        """Test comparison function."""
        metrics = {
            "accuracy": 0.98,
            "precision": 0.95,
            "recall": 0.90,
            "f1_score": 0.92,
            "auc_roc": 0.97,
        }

        comparison = compare_with_paper(metrics)

        for key in ["accuracy", "auc_roc"]:
            assert key in comparison
            assert "achieved" in comparison[key]
            assert "paper" in comparison[key]
            assert "difference" in comparison[key]

    def test_within_tolerance(self):
        """Test tolerance checking."""
        # Close to paper results
        metrics = {"accuracy": 0.98, "auc_roc": 0.98}

        comparison = compare_with_paper(metrics)

        # 2% tolerance
        assert comparison["accuracy"]["within_tolerance"] is True
        assert comparison["auc_roc"]["within_tolerance"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
