"""
Data Validation Module
======================

Comprehensive data validation for fraud detection pipelines.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data for fraud detection pipelines."""

    def __init__(self, config_path: str = "config/validation.yaml"):
        """
        Initialize the data validator.

        Args:
            config_path: Path to validation configuration file
        """
        self.config_path = Path(config_path)

        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            self.config = self._get_default_config()

        self.validation_config = self.config.get("validation", {})
        self.required_columns = self.validation_config.get("required_columns", [])
        self.data_types = self.validation_config.get("data_types", {})
        self.value_ranges = self.validation_config.get("value_ranges", {})
        self.thresholds = self.validation_config.get("thresholds", {})

    def _get_default_config(self) -> dict:
        """Get default validation configuration."""
        return {
            "validation": {
                "required_columns": [
                    "TransactionDT",
                    "TransactionAmt",
                    "card1",
                ],
                "data_types": {
                    "TransactionAmt": "float64",
                    "card1": "int64",
                },
                "value_ranges": {
                    "TransactionAmt": {"min": 0.01, "max": 999999.99},
                    "card1": {"min": 1000, "max": 20000},
                },
                "thresholds": {
                    "max_missing_pct": 30,
                    "max_duplicate_pct": 1,
                },
            }
        }

    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate DataFrame schema (columns and types).

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")

        # Check and attempt to convert data types
        for col, expected_type in self.data_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    try:
                        if expected_type == "datetime64[ns]":
                            df[col] = pd.to_datetime(df[col])
                        else:
                            df[col] = df[col].astype(expected_type)
                        logger.info(f"Converted {col} from {actual_type} to {expected_type}")
                    except Exception as e:
                        errors.append(f"Cannot convert {col} from {actual_type} to {expected_type}: {e}")

        return len(errors) == 0, errors

    def validate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Perform comprehensive data quality checks.

        Args:
            df: Input DataFrame

        Returns:
            Quality report dictionary
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "column_stats": {},
            "quality_issues": [],
        }

        max_missing_pct = self.thresholds.get("max_missing_pct", 30)
        max_duplicate_pct = self.thresholds.get("max_duplicate_pct", 1)

        # Check for duplicates
        duplicates = df.duplicated().sum()
        duplicate_pct = (duplicates / len(df)) * 100 if len(df) > 0 else 0
        if duplicate_pct > max_duplicate_pct:
            report["quality_issues"].append(
                f"High duplicate rate: {duplicates} rows ({duplicate_pct:.2f}%)"
            )

        report["duplicates"] = {"count": duplicates, "percentage": duplicate_pct}

        # Column-wise analysis
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_values": int(df[col].nunique()),
                "unique_percentage": float((df[col].nunique() / len(df)) * 100),
            }

            # Check missing values threshold
            if col_stats["null_percentage"] > max_missing_pct:
                report["quality_issues"].append(
                    f"High missing rate in {col}: {col_stats['null_percentage']:.2f}%"
                )

            # Numeric column statistics
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                col_stats.update(
                    {
                        "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                        "std": float(df[col].std()) if not df[col].isnull().all() else None,
                        "min": float(df[col].min()) if not df[col].isnull().all() else None,
                        "max": float(df[col].max()) if not df[col].isnull().all() else None,
                        "median": float(df[col].median()) if not df[col].isnull().all() else None,
                        "outliers": self._detect_outliers(df[col]),
                    }
                )

            report["column_stats"][col] = col_stats

        return report

    def validate_value_ranges(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate that values are within expected ranges.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for col, ranges in self.value_ranges.items():
            if col not in df.columns:
                continue

            col_data = df[col].dropna()

            if "min" in ranges:
                below_min = (col_data < ranges["min"]).sum()
                if below_min > 0:
                    errors.append(f"{col}: {below_min} values below minimum ({ranges['min']})")

            if "max" in ranges:
                above_max = (col_data > ranges["max"]).sum()
                if above_max > 0:
                    errors.append(f"{col}: {above_max} values above maximum ({ranges['max']})")

        return len(errors) == 0, errors

    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> dict[str, Any]:
        """
        Detect outliers in a numeric series.

        Args:
            series: Pandas Series
            method: Detection method ('iqr' or 'zscore')

        Returns:
            Outlier statistics
        """
        series = series.dropna()

        if len(series) == 0:
            return {"count": 0, "percentage": 0, "method": method}

        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = (z_scores > 3).sum()
        else:
            outliers = 0

        return {
            "count": int(outliers),
            "percentage": float((outliers / len(series)) * 100),
            "method": method,
        }

    def validate_for_training(self, df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """
        Run all validations required before training.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Starting training data validation")

        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
            },
        }

        # Schema validation
        schema_valid, schema_errors = self.validate_schema(df)
        report["schema_validation"] = {"is_valid": schema_valid, "errors": schema_errors}

        # Data quality
        report["data_quality"] = self.validate_data_quality(df)

        # Value ranges
        ranges_valid, range_errors = self.validate_value_ranges(df)
        report["value_ranges"] = {"is_valid": ranges_valid, "errors": range_errors}

        # Overall validation result
        is_valid = schema_valid and ranges_valid and len(report["data_quality"]["quality_issues"]) == 0

        report["overall_valid"] = is_valid
        report["total_issues"] = (
            len(schema_errors)
            + len(range_errors)
            + len(report["data_quality"]["quality_issues"])
        )

        if is_valid:
            logger.info("Validation passed")
        else:
            logger.warning(f"Validation failed with {report['total_issues']} issues")

        return is_valid, report

    def validate_for_inference(
        self,
        df: pd.DataFrame,
        training_stats: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate data for inference, including distribution shift detection.

        Args:
            df: Input DataFrame
            training_stats: Optional training data statistics for drift detection

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Starting inference data validation")

        # Run basic validations
        is_valid, report = self.validate_for_training(df)

        # Check for distribution shift if training stats provided
        if training_stats:
            drift_results = self._detect_distribution_shift(df, training_stats)
            report["distribution_shift"] = drift_results
            if drift_results.get("drift_detected", False):
                logger.warning("Distribution shift detected in inference data")
                report["warnings"] = report.get("warnings", []) + ["Distribution shift detected"]

        return is_valid, report

    def _detect_distribution_shift(
        self,
        df: pd.DataFrame,
        training_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Detect distribution shift between training and inference data.

        Args:
            df: Inference DataFrame
            training_stats: Training data statistics

        Returns:
            Shift detection results
        """
        results = {
            "drift_detected": False,
            "shifted_features": [],
            "shift_details": {},
        }

        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in training_stats:
                continue

            train_mean = training_stats[col].get("mean")
            train_std = training_stats[col].get("std")

            if train_mean is None or train_std is None or train_std == 0:
                continue

            inference_mean = df[col].mean()

            # Check for significant shift (more than 3 standard deviations)
            if abs(inference_mean - train_mean) > 3 * train_std:
                results["drift_detected"] = True
                results["shifted_features"].append(col)
                results["shift_details"][col] = {
                    "training_mean": train_mean,
                    "inference_mean": float(inference_mean),
                    "shift_magnitude": abs(inference_mean - train_mean) / train_std,
                }

        return results

    def save_report(self, report: dict[str, Any], output_path: str) -> None:
        """
        Save validation report to JSON file.

        Args:
            report: Validation report dictionary
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to {output_path}")


def validate_training_data(
    df: pd.DataFrame,
    config_path: str = "config/validation.yaml",
) -> tuple[bool, dict[str, Any]]:
    """
    Convenience function to validate training data.

    Args:
        df: Input DataFrame
        config_path: Path to validation config

    Returns:
        Tuple of (is_valid, validation_report)
    """
    validator = DataValidator(config_path)
    return validator.validate_for_training(df)


if __name__ == "__main__":
    # Test validation
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    test_data = pd.DataFrame(
        {
            "TransactionDT": np.random.randint(0, 1000000, 1000),
            "TransactionAmt": np.random.exponential(100, 1000),
            "card1": np.random.randint(1000, 20000, 1000),
            "C1": np.random.exponential(2, 1000),
            "C14": np.random.exponential(2, 1000),
        }
    )

    # Add some issues
    test_data.loc[0:10, "TransactionAmt"] = -100  # Invalid negative amounts
    test_data.loc[11:20, "card1"] = None  # Missing values

    validator = DataValidator()
    is_valid, report = validator.validate_for_training(test_data)

    print(f"Validation passed: {is_valid}")
    print(f"Total issues: {report['total_issues']}")
    print(f"Schema errors: {report['schema_validation']['errors']}")
    print(f"Range errors: {report['value_ranges']['errors']}")
