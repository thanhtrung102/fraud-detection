"""
Training Pipeline
=================

Prefect flow for model training with MLflow tracking.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from prefect import flow, task
from prefect.logging import get_run_logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.registry import ModelRegistry
from mlops.tracking import log_artifacts, log_metrics, log_params, setup_mlflow, start_run
from src.data_preprocessing import load_config, preprocess_pipeline
from src.evaluation import compute_metrics, generate_all_plots
from src.feature_selection import apply_feature_selection, shap_feature_selection
from src.optuna_tuning import tune_all_models
from src.stacking_model import StackingFraudDetector, find_optimal_threshold


@task(name="load_data", retries=2, retry_delay_seconds=30)
def load_and_preprocess_data(config: dict[str, Any]) -> tuple:
    """Load and preprocess the fraud detection data."""
    logger = get_run_logger()
    logger.info("Loading and preprocessing data...")

    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

    # Convert to numpy
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    logger.info(f"Data loaded: {X_train_np.shape[0]} training samples, {X_test_np.shape[0]} test samples")

    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names


@task(name="feature_selection")
def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    n_top_features: int = 30
) -> tuple:
    """Perform SHAP-based feature selection."""
    logger = get_run_logger()
    logger.info(f"Selecting top {n_top_features} features using SHAP...")

    feature_indices, selected_features, importance_df = shap_feature_selection(
        X_train, y_train, feature_names,
        n_top_features=n_top_features,
        sample_size=min(50000, len(X_train))
    )

    X_train_selected = apply_feature_selection(X_train, feature_indices)
    X_test_selected = apply_feature_selection(X_test, feature_indices)

    logger.info(f"Selected features: {selected_features[:5]}...")

    return X_train_selected, X_test_selected, selected_features


@task(name="hyperparameter_tuning")
def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 20,
    cv_folds: int = 5
) -> dict[str, Any]:
    """Tune model hyperparameters using Optuna."""
    logger = get_run_logger()
    logger.info(f"Tuning hyperparameters with {n_trials} trials...")

    best_params = tune_all_models(
        X_train, y_train,
        n_trials=n_trials,
        cv_folds=cv_folds,
        random_state=42
    )

    logger.info("Hyperparameter tuning complete")
    return best_params


@task(name="train_model")
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    config: dict[str, Any]
) -> StackingFraudDetector:
    """Train the stacking ensemble model."""
    logger = get_run_logger()
    logger.info("Training stacking ensemble...")

    model = StackingFraudDetector(
        xgb_params=params.get('xgboost'),
        lgbm_params=params.get('lightgbm'),
        catboost_params=params.get('catboost'),
        meta_params=config.get('meta_learner')
    )
    model.fit(X_train, y_train)

    logger.info("Model training complete")
    return model


@task(name="evaluate_model")
def evaluate_model(
    model: StackingFraudDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "results"
) -> tuple[dict[str, float], np.ndarray, float]:
    """Evaluate the trained model."""
    logger = get_run_logger()
    logger.info("Evaluating model...")

    y_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, y_proba, metric='f1')
    y_pred = (y_proba >= optimal_threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Generate plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generate_all_plots(y_test, y_pred, y_proba, output_dir)

    logger.info(f"Evaluation complete: AUC-ROC={metrics['auc_roc']:.4f}, Accuracy={metrics['accuracy']:.4f}")

    return metrics, y_pred, optimal_threshold


@task(name="save_model")
def save_model(
    model: StackingFraudDetector,
    output_dir: str = "models"
) -> str:
    """Save the trained model."""
    logger = get_run_logger()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    logger.info(f"Model saved to {output_dir}")
    return output_dir


@task(name="register_model")
def register_model_to_mlflow(
    run_id: str,
    metrics: dict[str, float],
    stage: str = "Staging"
) -> Optional[str]:
    """Register the model to MLflow registry."""
    logger = get_run_logger()

    registry = ModelRegistry()
    version = registry.register_model(
        run_id=run_id,
        model_path="model",
        description=f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}"
    )

    registry.transition_model_stage(version, stage)
    logger.info(f"Model registered: version {version}, stage {stage}")

    return version


@flow(name="fraud-detection-training", log_prints=True)
def training_flow(
    config_path: Optional[str] = None,
    use_optuna: bool = True,
    use_feature_selection: bool = True,
    n_trials: int = 20,
    n_top_features: int = 30,
    register_model: bool = True,
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
) -> dict[str, Any]:
    """
    Complete training pipeline flow.

    Args:
        config_path: Path to configuration file
        use_optuna: Whether to use Optuna tuning
        use_feature_selection: Whether to use SHAP feature selection
        n_trials: Number of Optuna trials
        n_top_features: Number of features to select
        register_model: Whether to register model to MLflow
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info("Starting fraud detection training pipeline")

    # Setup MLflow
    setup_mlflow(tracking_uri=mlflow_tracking_uri, experiment_name="fraud-detection")

    # Load config
    config = load_config(config_path)

    run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Log configuration
        log_params(config)
        log_params({"use_optuna": use_optuna, "use_feature_selection": use_feature_selection})

        # Load data
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(config)

        # Feature selection
        if use_feature_selection:
            X_train, X_test, feature_names = select_features(
                X_train, X_test, y_train, feature_names, n_top_features
            )
            log_params({"n_selected_features": len(feature_names)})

        # Hyperparameter tuning
        if use_optuna:
            best_params = tune_hyperparameters(X_train, y_train, n_trials)
            log_params(best_params, prefix="tuned")
        else:
            best_params = {
                'xgboost': config.get('base_models', {}).get('xgboost'),
                'lightgbm': config.get('base_models', {}).get('lightgbm'),
                'catboost': config.get('base_models', {}).get('catboost')
            }

        # Train model
        model = train_model(X_train, y_train, best_params, config)

        # Evaluate
        metrics, y_pred, threshold = evaluate_model(model, X_test, y_test)

        # Log metrics
        log_metrics(metrics)
        log_metrics({"optimal_threshold": threshold})

        # Save model
        model_dir = save_model(model)

        # Log artifacts
        log_artifacts("results", "plots")
        log_artifacts(model_dir, "model")

        # Register model
        model_version = None
        if register_model:
            model_version = register_model_to_mlflow(run_id, metrics)

    logger.info("Training pipeline complete")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "threshold": threshold,
        "model_version": model_version,
        "feature_names": feature_names
    }


if __name__ == "__main__":
    # Run the training flow
    result = training_flow(
        use_optuna=False,  # Skip for quick test
        use_feature_selection=True,
        register_model=False
    )

    print(f"\nTraining Results:")
    print(f"  Run ID: {result['run_id']}")
    print(f"  AUC-ROC: {result['metrics']['auc_roc']:.4f}")
    print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
