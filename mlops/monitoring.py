"""
Model Monitoring with Evidently
===============================

Data drift detection and model performance monitoring.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report


class FraudMonitor:
    """Monitor for fraud detection model performance and data drift."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_column: str = "isFraud",
        prediction_column: str = "prediction",
        numerical_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None
    ):
        """
        Initialize the fraud monitor.

        Args:
            reference_data: Reference dataset (training data)
            target_column: Name of target column
            prediction_column: Name of prediction column
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.prediction_column = prediction_column

        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Remove target and prediction columns
            numerical_features = [
                f for f in numerical_features
                if f not in [target_column, prediction_column]
            ]

        if categorical_features is None:
            categorical_features = reference_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # Set up column mapping
        self.column_mapping = ColumnMapping(
            target=target_column,
            prediction=prediction_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )

    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> tuple[Report, dict[str, Any]]:
        """
        Generate a data drift report.

        Args:
            current_data: Current/production data
            output_path: Optional path to save HTML report

        Returns:
            Tuple of (Report, drift_metrics_dict)
        """
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetMissingValuesMetric()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        if output_path:
            report.save_html(output_path)
            print(f"Data drift report saved to {output_path}")

        # Extract key metrics
        result = report.as_dict()
        metrics = self._extract_drift_metrics(result)

        return report, metrics

    def generate_model_performance_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> tuple[Report, dict[str, Any]]:
        """
        Generate a model performance report.

        Args:
            current_data: Current data with predictions and actuals
            output_path: Optional path to save HTML report

        Returns:
            Tuple of (Report, performance_metrics_dict)
        """
        report = Report(metrics=[
            ClassificationPreset(),
            TargetDriftPreset()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        if output_path:
            report.save_html(output_path)
            print(f"Model performance report saved to {output_path}")

        result = report.as_dict()
        metrics = self._extract_performance_metrics(result)

        return report, metrics

    def generate_data_quality_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> tuple[Report, dict[str, Any]]:
        """
        Generate a data quality report.

        Args:
            current_data: Current data to analyze
            output_path: Optional path to save HTML report

        Returns:
            Tuple of (Report, quality_metrics_dict)
        """
        report = Report(metrics=[
            DataQualityPreset()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        if output_path:
            report.save_html(output_path)
            print(f"Data quality report saved to {output_path}")

        return report, report.as_dict()

    def check_drift_threshold(
        self,
        current_data: pd.DataFrame,
        drift_threshold: float = 0.5,
        feature_drift_threshold: float = 0.3
    ) -> dict[str, Any]:
        """
        Check if data drift exceeds thresholds.

        Args:
            current_data: Current data to check
            drift_threshold: Overall dataset drift threshold
            feature_drift_threshold: Per-feature drift threshold

        Returns:
            Dictionary with drift status and details
        """
        _, metrics = self.generate_data_drift_report(current_data)

        dataset_drift = metrics.get("dataset_drift", 0)
        drifted_features = metrics.get("drifted_features", [])
        drift_share = metrics.get("drift_share", 0)

        alert = False
        alerts = []

        if drift_share > drift_threshold:
            alert = True
            alerts.append(f"Dataset drift ({drift_share:.2%}) exceeds threshold ({drift_threshold:.2%})")

        if len(drifted_features) > len(self.numerical_features) * feature_drift_threshold:
            alert = True
            alerts.append(f"Too many features drifted: {len(drifted_features)}")

        return {
            "alert": alert,
            "alerts": alerts,
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_drift_metrics(self, result: dict) -> dict[str, Any]:
        """Extract key drift metrics from report result."""
        metrics = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "drifted_features": [],
            "total_features": 0
        }

        try:
            for metric in result.get("metrics", []):
                metric_id = metric.get("metric", "")

                if "DatasetDriftMetric" in metric_id:
                    result_data = metric.get("result", {})
                    metrics["dataset_drift"] = result_data.get("dataset_drift", False)
                    metrics["drift_share"] = result_data.get("drift_share", 0)
                    metrics["total_features"] = result_data.get("number_of_columns", 0)

                    # Get drifted columns
                    drift_by_columns = result_data.get("drift_by_columns", {})
                    metrics["drifted_features"] = [
                        col for col, data in drift_by_columns.items()
                        if data.get("drift_detected", False)
                    ]
        except Exception as e:
            print(f"Error extracting drift metrics: {e}")

        return metrics

    def _extract_performance_metrics(self, result: dict) -> dict[str, Any]:
        """Extract key performance metrics from report result."""
        metrics = {}

        try:
            for metric in result.get("metrics", []):
                metric_id = metric.get("metric", "")
                result_data = metric.get("result", {})

                if "ClassificationQualityMetric" in metric_id:
                    current = result_data.get("current", {})
                    metrics.update({
                        "accuracy": current.get("accuracy"),
                        "precision": current.get("precision"),
                        "recall": current.get("recall"),
                        "f1": current.get("f1")
                    })
        except Exception as e:
            print(f"Error extracting performance metrics: {e}")

        return metrics


def create_monitoring_report(
    reference_path: str,
    current_path: str,
    output_dir: str = "monitoring/evidently_reports",
    target_column: str = "isFraud",
    prediction_column: str = "prediction"
) -> dict[str, Any]:
    """
    Create all monitoring reports.

    Args:
        reference_path: Path to reference data CSV
        current_path: Path to current data CSV
        output_dir: Output directory for reports
        target_column: Target column name
        prediction_column: Prediction column name

    Returns:
        Dictionary with all metrics
    """
    # Load data
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize monitor
    monitor = FraudMonitor(
        reference_data=reference_data,
        target_column=target_column,
        prediction_column=prediction_column
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate reports
    _, drift_metrics = monitor.generate_data_drift_report(
        current_data,
        output_path=str(output_dir / f"data_drift_{timestamp}.html")
    )

    _, perf_metrics = monitor.generate_model_performance_report(
        current_data,
        output_path=str(output_dir / f"model_performance_{timestamp}.html")
    )

    # Check for alerts
    drift_check = monitor.check_drift_threshold(current_data)

    # Combine all metrics
    all_metrics = {
        "timestamp": timestamp,
        "drift": drift_metrics,
        "performance": perf_metrics,
        "alerts": drift_check
    }

    # Save metrics JSON
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"Monitoring metrics saved to {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    # Example usage
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    reference = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(5, 2, n_samples),
        "isFraud": np.random.binomial(1, 0.035, n_samples),
        "prediction": np.random.binomial(1, 0.04, n_samples)
    })

    # Simulate drift in current data
    current = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1.2, n_samples),  # Drift
        "feature2": np.random.normal(5, 2, n_samples),
        "isFraud": np.random.binomial(1, 0.05, n_samples),
        "prediction": np.random.binomial(1, 0.06, n_samples)
    })

    monitor = FraudMonitor(reference)
    drift_check = monitor.check_drift_threshold(current)

    print(f"Alert: {drift_check['alert']}")
    print(f"Drift share: {drift_check['drift_share']:.2%}")
    print(f"Drifted features: {drift_check['drifted_features']}")
