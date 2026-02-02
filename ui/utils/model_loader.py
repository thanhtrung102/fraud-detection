"""
Fraud Detection Model Loader
============================

Utilities for loading fraud detection models from MLflow or local storage.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import joblib
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class FraudModelLoader:
    """Load fraud detection models from MLflow or local storage."""

    def __init__(self):
        """Initialize the model loader."""
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_dir = os.getenv("MODEL_DIR", "models")

        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)

        # Set up S3 credentials if available
        s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        if s3_endpoint:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

        self.client = MlflowClient(tracking_uri=self.mlflow_uri)

        # Loaded models and metadata
        self.model = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.44
        self.loaded = False
        self.run_id: Optional[str] = None
        self.model_info: dict[str, Any] = {}

    def get_latest_run(self, experiment_name: str = "fraud-detection") -> Optional[str]:
        """
        Get the latest successful training run from MLflow.

        Args:
            experiment_name: Name of the MLflow experiment

        Returns:
            Run ID or None if not found
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return None

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1,
            )

            if len(runs) > 0:
                return runs[0].info.run_id

            return None

        except Exception as e:
            logger.error(f"Failed to get latest run: {e}")
            return None

    def get_production_model_run(self) -> Optional[str]:
        """
        Get the run ID of the production model from MLflow registry.

        Returns:
            Run ID or None if not found
        """
        try:
            model_name = "fraud-detection-model"
            versions = self.client.search_model_versions(f"name='{model_name}'")

            for version in versions:
                if version.current_stage == "Production":
                    return version.run_id

            return None

        except Exception as e:
            logger.warning(f"Failed to get production model: {e}")
            return None

    def load_from_mlflow(self, run_id: str) -> bool:
        """
        Load model and artifacts from an MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            True if loading successful
        """
        try:
            logger.info(f"Loading model from MLflow run: {run_id}")

            # Download artifacts
            local_dir = tempfile.mkdtemp(prefix=f"fraud_model_{run_id[:8]}_")
            artifacts_path = self.client.download_artifacts(run_id, "", dst_path=local_dir)

            # Load the stacking model
            model_path = Path(artifacts_path) / "model"
            if model_path.exists():
                self._load_model_from_dir(str(model_path))
            else:
                # Try loading from models subdirectory
                models_path = Path(artifacts_path) / "models"
                if models_path.exists():
                    self._load_model_from_dir(str(models_path))

            # Load feature names
            feature_names_path = Path(artifacts_path) / "model" / "feature_names.json"
            if not feature_names_path.exists():
                feature_names_path = Path(artifacts_path) / "feature_names.json"

            if feature_names_path.exists():
                with open(feature_names_path) as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")

            # Get run metrics
            run = self.client.get_run(run_id)
            self.model_info = {
                "run_id": run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "start_time": run.info.start_time,
            }

            # Get threshold from params or metrics
            if "optimal_threshold" in run.data.metrics:
                self.threshold = run.data.metrics["optimal_threshold"]
            elif "threshold" in run.data.params:
                self.threshold = float(run.data.params["threshold"])

            self.run_id = run_id
            self.loaded = True
            logger.info(f"Model loaded successfully from run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            return False

    def load_from_local(self, model_dir: Optional[str] = None) -> bool:
        """
        Load model from local directory.

        Args:
            model_dir: Path to model directory

        Returns:
            True if loading successful
        """
        try:
            model_dir = model_dir or self.model_dir
            logger.info(f"Loading model from local directory: {model_dir}")

            self._load_model_from_dir(model_dir)

            # Load feature names
            feature_names_path = Path(model_dir) / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path) as f:
                    self.feature_names = json.load(f)
            else:
                # Use default feature names
                self.feature_names = self._get_default_features()

            self.model_info = {"source": "local", "path": model_dir}
            self.loaded = True
            logger.info("Model loaded successfully from local directory")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from local: {e}")
            return False

    def _load_model_from_dir(self, model_dir: str) -> None:
        """Load the stacking fraud detector from a directory."""
        import sys

        # Add project root to path for imports
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

        from src.stacking_model import StackingFraudDetector

        self.model = StackingFraudDetector()
        self.model.load(model_dir)
        logger.info("Stacking model loaded")

    def _get_default_features(self) -> list[str]:
        """Get default feature names (top 30 from SHAP analysis)."""
        return [
            "C14",
            "C12",
            "card6",
            "C1",
            "V308",
            "V258",
            "V317",
            "V282",
            "C11",
            "V280",
            "card2",
            "TransactionDT",
            "TransactionAmt",
            "P_emaildomain",
            "V95",
            "D15",
            "V283",
            "D1",
            "addr1",
            "card1",
            "V285",
            "D4",
            "C13",
            "D10",
            "C2",
            "V310",
            "card5",
            "D11",
            "dist1",
            "C6",
        ]

    def load_model(self) -> bool:
        """
        Load model with automatic source detection.

        Tries in order:
        1. Production model from MLflow registry
        2. Latest run from MLflow
        3. Local model directory

        Returns:
            True if model loaded successfully
        """
        # Try production model first
        run_id = self.get_production_model_run()
        if run_id and self.load_from_mlflow(run_id):
            logger.info("Loaded production model from MLflow registry")
            return True

        # Try latest run
        run_id = self.get_latest_run()
        if run_id and self.load_from_mlflow(run_id):
            logger.info("Loaded latest model from MLflow")
            return True

        # Fall back to local
        if self.load_from_local():
            logger.info("Loaded model from local directory")
            return True

        logger.error("Failed to load model from any source")
        return False

    def get_model_summary(self) -> dict[str, Any]:
        """Get summary of loaded model."""
        if not self.loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "run_id": self.run_id,
            "threshold": self.threshold,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names[:10],  # First 10
            "metrics": self.model_info.get("metrics", {}),
        }
