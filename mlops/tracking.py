"""
MLflow Experiment Tracking
==========================

Experiment tracking and logging utilities using MLflow.
"""

import os
from typing import Any, Optional

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient


def setup_mlflow(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "fraud-detection"
) -> str:
    """
    Set up MLflow tracking.

    Args:
        tracking_uri: MLflow tracking URI (SQLite, PostgreSQL, or remote)
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "fraud-detection", "version": "1.0"}
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id


def log_params(params: dict[str, Any], prefix: str = "") -> None:
    """
    Log parameters to MLflow.

    Args:
        params: Dictionary of parameters
        prefix: Optional prefix for parameter names
    """
    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_params(value, param_name)
        else:
            mlflow.log_param(param_name, value)


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metrics
        step: Optional step number
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            mlflow.log_metric(key, float(value), step=step)


def log_artifacts(artifact_dir: str, artifact_path: Optional[str] = None) -> None:
    """
    Log artifacts directory to MLflow.

    Args:
        artifact_dir: Local directory containing artifacts
        artifact_path: Optional destination path in MLflow
    """
    if os.path.exists(artifact_dir):
        mlflow.log_artifacts(artifact_dir, artifact_path)
        print(f"Logged artifacts from {artifact_dir}")


def log_model_artifact(model_path: str, artifact_name: str = "model") -> None:
    """
    Log a single model file as artifact.

    Args:
        model_path: Path to model file
        artifact_name: Name for the artifact
    """
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_name)


def log_figure(fig, filename: str) -> None:
    """
    Log a matplotlib figure to MLflow.

    Args:
        fig: Matplotlib figure
        filename: Filename for the figure
    """
    mlflow.log_figure(fig, filename)


def start_run(
    run_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    nested: bool = False
) -> mlflow.ActiveRun:
    """
    Start an MLflow run.

    Args:
        run_name: Optional name for the run
        tags: Optional tags for the run
        nested: Whether this is a nested run

    Returns:
        Active MLflow run
    """
    return mlflow.start_run(run_name=run_name, tags=tags, nested=nested)


def end_run(status: str = "FINISHED") -> None:
    """
    End the current MLflow run.

    Args:
        status: Run status (FINISHED, FAILED, KILLED)
    """
    mlflow.end_run(status=status)


def log_training_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    model_path: str,
    artifacts_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None
) -> str:
    """
    Log a complete training run to MLflow.

    Args:
        params: Training parameters
        metrics: Evaluation metrics
        model_path: Path to saved model
        artifacts_dir: Optional directory with additional artifacts
        run_name: Optional name for the run
        tags: Optional tags

    Returns:
        Run ID
    """
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Log parameters
        log_params(params)

        # Log metrics
        log_metrics(metrics)

        # Log model
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                mlflow.log_artifacts(model_path, "model")
            else:
                mlflow.log_artifact(model_path, "model")

        # Log additional artifacts
        if artifacts_dir and os.path.exists(artifacts_dir):
            mlflow.log_artifacts(artifacts_dir, "artifacts")

        print(f"Logged run: {run.info.run_id}")
        return run.info.run_id


def get_best_run(
    experiment_name: str,
    metric: str = "auc_roc",
    ascending: bool = False
) -> Optional[dict[str, Any]]:
    """
    Get the best run from an experiment.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        ascending: Whether lower is better

    Returns:
        Best run info or None
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1
    )

    if not runs:
        return None

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "artifacts_uri": best_run.info.artifact_uri
    }


def compare_runs(
    experiment_name: str,
    metrics: Optional[list[str]] = None,
    max_runs: int = 10
) -> list[dict[str, Any]]:
    """
    Compare multiple runs from an experiment.

    Args:
        experiment_name: Name of the experiment
        metrics: Metrics to compare
        max_runs: Maximum number of runs to return

    Returns:
        List of run comparisons
    """
    if metrics is None:
        metrics = ["accuracy", "auc_roc", "f1_score"]

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_runs
    )

    comparisons = []
    for run in runs:
        comparison = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
        }
        for metric in metrics:
            comparison[metric] = run.data.metrics.get(metric)
        comparisons.append(comparison)

    return comparisons


if __name__ == "__main__":
    # Test MLflow setup
    setup_mlflow()

    with mlflow.start_run(run_name="test-run"):
        mlflow.log_param("test_param", "value")
        mlflow.log_metric("test_metric", 0.95)

    print("MLflow test successful!")
