"""
Inference Pipeline
==================

Prefect flow for batch inference with model monitoring.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import load_config
from src.stacking_model import StackingFraudDetector


@task(name="load_model", retries=2)
def load_model(model_dir: str = "models") -> StackingFraudDetector:
    """Load the trained model."""
    logger = get_run_logger()
    logger.info(f"Loading model from {model_dir}")

    model = StackingFraudDetector()
    model.load(model_dir)

    logger.info("Model loaded successfully")
    return model


@task(name="load_inference_data")
def load_inference_data(data_path: str) -> pd.DataFrame:
    """Load data for inference."""
    logger = get_run_logger()
    logger.info(f"Loading inference data from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")

    return df


@task(name="preprocess_inference_data")
def preprocess_inference_data(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Preprocess data for inference."""
    logger = get_run_logger()

    # Select only required features
    missing_features = [f for f in feature_names if f not in df.columns]

    if missing_features:
        logger.warning(f"Missing features: {missing_features[:5]}...")
        # Fill missing features with 0
        for f in missing_features:
            df[f] = 0

    X = df[feature_names].values
    logger.info(f"Preprocessed data shape: {X.shape}")

    return X


@task(name="run_inference")
def run_inference(
    model: StackingFraudDetector, X: np.ndarray, threshold: float = 0.44
) -> dict[str, np.ndarray]:
    """Run inference on data."""
    logger = get_run_logger()
    logger.info(f"Running inference on {X.shape[0]} samples")

    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    n_fraud = y_pred.sum()
    fraud_rate = n_fraud / len(y_pred)

    logger.info(f"Predictions: {n_fraud} fraud ({fraud_rate:.2%})")

    return {"probabilities": y_proba, "predictions": y_pred}


@task(name="save_predictions")
def save_predictions(df: pd.DataFrame, predictions: dict[str, np.ndarray], output_path: str) -> str:
    """Save predictions to file."""
    logger = get_run_logger()

    df = df.copy()
    df["fraud_probability"] = predictions["probabilities"]
    df["fraud_prediction"] = predictions["predictions"]

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    return output_path


@task(name="generate_inference_report")
def generate_inference_report(
    predictions: dict[str, np.ndarray], output_dir: str = "results"
) -> dict[str, Any]:
    """Generate inference summary report."""
    logger = get_run_logger()

    y_pred = predictions["predictions"]
    y_proba = predictions["probabilities"]

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(y_pred),
        "predicted_fraud": int(y_pred.sum()),
        "predicted_legitimate": int(len(y_pred) - y_pred.sum()),
        "fraud_rate": float(y_pred.mean()),
        "avg_fraud_probability": float(y_proba.mean()),
        "max_fraud_probability": float(y_proba.max()),
        "high_risk_count": int((y_proba > 0.8).sum()),
        "medium_risk_count": int(((y_proba > 0.5) & (y_proba <= 0.8)).sum()),
        "low_risk_count": int((y_proba <= 0.5).sum()),
    }

    # Save report
    import json

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(output_dir) / f"inference_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Inference report saved to {report_path}")
    return report


@flow(name="fraud-detection-inference", log_prints=True)
def inference_flow(
    data_path: str,
    model_dir: str = "models",
    output_path: Optional[str] = None,
    feature_names: Optional[list[str]] = None,
    threshold: float = 0.44,
) -> dict[str, Any]:
    """
    Batch inference pipeline flow.

    Args:
        data_path: Path to input data CSV
        model_dir: Directory containing saved model
        output_path: Path for output predictions
        feature_names: List of feature names (auto-detect if None)
        threshold: Classification threshold

    Returns:
        Dictionary with inference results
    """
    logger = get_run_logger()
    logger.info("Starting fraud detection inference pipeline")

    # Set default output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/predictions_{timestamp}.csv"

    # Load model
    model = load_model(model_dir)

    # Load data
    df = load_inference_data(data_path)

    # Load feature names from model directory
    if feature_names is None:
        import json
        from pathlib import Path

        feature_file = Path(model_dir) / "feature_names.json"
        if feature_file.exists():
            with open(feature_file) as f:
                feature_names = json.load(f)
            logger.info(f"Loaded {len(feature_names)} feature names from {feature_file}")
        else:
            # Fallback: use all numeric columns (may cause shape mismatch)
            logger.warning(f"Feature names file not found at {feature_file}, using all numeric columns")
            feature_names = df.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocess
    X = preprocess_inference_data(df, feature_names)

    # Run inference
    predictions = run_inference(model, X, threshold)

    # Save predictions
    save_predictions(df, predictions, output_path)

    # Generate report
    report = generate_inference_report(predictions)

    logger.info("Inference pipeline complete")

    return {"output_path": output_path, "report": report, "total_processed": len(df)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Inference Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train_transaction.csv",
        help="Path to input data CSV (default: data/train_transaction.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained model (default: models)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/predictions.csv",
        help="Path for output predictions (default: results/predictions.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.44,
        help="Classification threshold (default: 0.44)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of rows to sample for inference (default: 1000, use 0 for all)",
    )
    args = parser.parse_args()

    # Sample data if needed to avoid memory issues
    if args.sample_size > 0:
        import pandas as pd

        print(f"Sampling {args.sample_size} rows from {args.data_path}...")
        df = pd.read_csv(args.data_path, nrows=args.sample_size)
        sample_path = "data/inference_sample.csv"
        df.to_csv(sample_path, index=False)
        data_path = sample_path
    else:
        data_path = args.data_path

    result = inference_flow(
        data_path=data_path,
        model_dir=args.model_dir,
        output_path=args.output_path,
        threshold=args.threshold,
    )

    print("\nInference Results:")
    print(f"  Output: {result['output_path']}")
    print(f"  Total processed: {result['total_processed']}")
    print(f"  Fraud rate: {result['report']['fraud_rate']:.2%}")
