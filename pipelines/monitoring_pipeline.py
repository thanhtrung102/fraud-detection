"""
Monitoring Pipeline
===================

Prefect flow for model monitoring and drift detection.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.monitoring import FraudMonitor


@task(name="load_reference_data")
def load_reference_data(reference_path: str) -> pd.DataFrame:
    """Load reference (training) data for comparison."""
    logger = get_run_logger()
    logger.info(f"Loading reference data from {reference_path}")

    df = pd.read_csv(reference_path)
    logger.info(f"Loaded {len(df)} reference records")

    return df


@task(name="load_production_data")
def load_production_data(production_path: str) -> pd.DataFrame:
    """Load production data for monitoring."""
    logger = get_run_logger()
    logger.info(f"Loading production data from {production_path}")

    df = pd.read_csv(production_path)
    logger.info(f"Loaded {len(df)} production records")

    return df


@task(name="check_data_drift")
def check_data_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    drift_threshold: float = 0.5
) -> dict[str, Any]:
    """Check for data drift between reference and production data."""
    logger = get_run_logger()
    logger.info("Checking for data drift...")

    monitor = FraudMonitor(reference_data)
    drift_result = monitor.check_drift_threshold(
        production_data,
        drift_threshold=drift_threshold
    )

    if drift_result["alert"]:
        logger.warning(f"DATA DRIFT DETECTED: {drift_result['alerts']}")
    else:
        logger.info("No significant data drift detected")

    return drift_result


@task(name="generate_drift_report")
def generate_drift_report(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    output_dir: str
) -> str:
    """Generate detailed drift report."""
    logger = get_run_logger()

    monitor = FraudMonitor(reference_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_dir) / f"drift_report_{timestamp}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report, metrics = monitor.generate_data_drift_report(
        production_data,
        output_path=str(output_path)
    )

    logger.info(f"Drift report saved to {output_path}")
    return str(output_path)


@task(name="generate_performance_report")
def generate_performance_report(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    output_dir: str
) -> str:
    """Generate model performance report."""
    logger = get_run_logger()

    monitor = FraudMonitor(reference_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_dir) / f"performance_report_{timestamp}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report, metrics = monitor.generate_model_performance_report(
        production_data,
        output_path=str(output_path)
    )

    logger.info(f"Performance report saved to {output_path}")
    return str(output_path)


@task(name="send_alert")
def send_alert(drift_result: dict[str, Any], alert_config: dict[str, Any]) -> None:
    """Send alert if drift is detected."""
    logger = get_run_logger()

    if not drift_result["alert"]:
        logger.info("No alert needed")
        return

    # Log alert details
    logger.warning("=" * 50)
    logger.warning("DRIFT ALERT")
    logger.warning("=" * 50)
    for alert_msg in drift_result["alerts"]:
        logger.warning(f"  - {alert_msg}")
    logger.warning(f"  Drift share: {drift_result['drift_share']:.2%}")
    logger.warning(f"  Drifted features: {drift_result['drifted_features']}")
    logger.warning("=" * 50)

    # Here you would integrate with your alerting system:
    # - Slack webhook
    # - Email via SMTP
    # - PagerDuty
    # - Custom webhook

    if alert_config.get("slack_webhook"):
        logger.info("Would send Slack alert (not implemented)")

    if alert_config.get("email"):
        logger.info("Would send email alert (not implemented)")


@task(name="save_monitoring_metrics")
def save_monitoring_metrics(
    drift_result: dict[str, Any],
    output_dir: str
) -> str:
    """Save monitoring metrics to JSON."""
    logger = get_run_logger()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"monitoring_metrics_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(drift_result, f, indent=2, default=str)

    logger.info(f"Monitoring metrics saved to {output_path}")
    return str(output_path)


@flow(name="fraud-detection-monitoring", log_prints=True)
def monitoring_flow(
    reference_path: str,
    production_path: str,
    output_dir: str = "monitoring/evidently_reports",
    drift_threshold: float = 0.5,
    alert_config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Model monitoring pipeline flow.

    Args:
        reference_path: Path to reference (training) data
        production_path: Path to production data
        output_dir: Directory for output reports
        drift_threshold: Threshold for drift detection
        alert_config: Configuration for alerts

    Returns:
        Dictionary with monitoring results
    """
    logger = get_run_logger()
    logger.info("Starting fraud detection monitoring pipeline")

    alert_config = alert_config or {}

    # Load data
    reference_data = load_reference_data(reference_path)
    production_data = load_production_data(production_path)

    # Check drift
    drift_result = check_data_drift(
        reference_data,
        production_data,
        drift_threshold
    )

    # Generate reports
    drift_report_path = generate_drift_report(
        reference_data,
        production_data,
        output_dir
    )

    # Save metrics
    metrics_path = save_monitoring_metrics(drift_result, output_dir)

    # Send alerts if needed
    send_alert(drift_result, alert_config)

    logger.info("Monitoring pipeline complete")

    return {
        "drift_detected": drift_result["alert"],
        "drift_share": drift_result["drift_share"],
        "drifted_features": drift_result["drifted_features"],
        "drift_report": drift_report_path,
        "metrics_path": metrics_path
    }


# Scheduled deployment (for Prefect Cloud/Server)
def create_monitoring_deployment():
    """Create a scheduled deployment for monitoring."""
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=monitoring_flow,
        name="daily-monitoring",
        schedule=CronSchedule(cron="0 6 * * *"),  # Daily at 6 AM
        parameters={
            "reference_path": "data/reference_data.csv",
            "production_path": "data/production_data.csv",
            "drift_threshold": 0.5
        }
    )

    return deployment


if __name__ == "__main__":
    # Example usage with sample data
    import numpy as np

    # Create sample reference data
    np.random.seed(42)
    n_samples = 1000

    reference = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(5, 2, n_samples),
        "isFraud": np.random.binomial(1, 0.035, n_samples),
        "prediction": np.random.binomial(1, 0.04, n_samples)
    })

    # Create sample production data with some drift
    production = pd.DataFrame({
        "feature1": np.random.normal(0.3, 1.1, n_samples),  # Slight drift
        "feature2": np.random.normal(5, 2, n_samples),
        "isFraud": np.random.binomial(1, 0.04, n_samples),
        "prediction": np.random.binomial(1, 0.05, n_samples)
    })

    # Save sample data
    Path("data").mkdir(exist_ok=True)
    reference.to_csv("data/sample_reference.csv", index=False)
    production.to_csv("data/sample_production.csv", index=False)

    # Run monitoring
    result = monitoring_flow(
        reference_path="data/sample_reference.csv",
        production_path="data/sample_production.csv",
        drift_threshold=0.3
    )

    print("\nMonitoring Results:")
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.2%}")
    print(f"  Report: {result['drift_report']}")
