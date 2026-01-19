"""
MLflow Model Registry
=====================

Model versioning and staging using MLflow Model Registry.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from typing import Optional, Dict, Any, List
import os


class ModelRegistry:
    """MLflow Model Registry wrapper for fraud detection models."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """
        Initialize the model registry.

        Args:
            tracking_uri: MLflow tracking URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = "fraud-detection-model"

    def register_model(
        self,
        run_id: str,
        model_path: str = "model",
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model from an MLflow run.

        Args:
            run_id: MLflow run ID
            model_path: Path to model artifacts within the run
            model_name: Optional custom model name
            tags: Optional tags for the model version
            description: Optional description

        Returns:
            Model version string
        """
        model_name = model_name or self.model_name
        model_uri = f"runs:/{run_id}/{model_path}"

        # Register the model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, version, key, value)

        # Add description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )

        print(f"Registered model '{model_name}' version {version}")
        return version

    def transition_model_stage(
        self,
        version: str,
        stage: str,
        model_name: Optional[str] = None,
        archive_existing: bool = True
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            model_name: Optional model name
            archive_existing: Whether to archive existing models in target stage
        """
        model_name = model_name or self.model_name

        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        print(f"Transitioned model '{model_name}' v{version} to {stage}")

    def get_latest_version(
        self,
        stage: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest model version.

        Args:
            stage: Optional stage filter (Staging, Production)
            model_name: Optional model name

        Returns:
            Model version info or None
        """
        model_name = model_name or self.model_name

        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = self.client.get_latest_versions(model_name)

            if not versions:
                return None

            latest = versions[0]
            return {
                "version": latest.version,
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "source": latest.source,
                "status": latest.status,
                "description": latest.description
            }
        except MlflowException:
            return None

    def get_production_model_uri(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Get the URI for the production model.

        Args:
            model_name: Optional model name

        Returns:
            Model URI or None
        """
        model_name = model_name or self.model_name
        latest = self.get_latest_version(stage="Production", model_name=model_name)

        if latest:
            return f"models:/{model_name}/Production"
        return None

    def load_model(
        self,
        stage: str = "Production",
        model_name: Optional[str] = None
    ):
        """
        Load a model from the registry.

        Args:
            stage: Model stage to load from
            model_name: Optional model name

        Returns:
            Loaded model
        """
        model_name = model_name or self.model_name
        model_uri = f"models:/{model_name}/{stage}"

        return mlflow.pyfunc.load_model(model_uri)

    def list_versions(
        self,
        model_name: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a model.

        Args:
            model_name: Optional model name
            max_results: Maximum number of versions to return

        Returns:
            List of model version info
        """
        model_name = model_name or self.model_name

        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")

            result = []
            for v in versions[:max_results]:
                result.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp
                })
            return result
        except MlflowException:
            return []

    def delete_version(
        self,
        version: str,
        model_name: Optional[str] = None
    ) -> None:
        """
        Delete a model version.

        Args:
            version: Version to delete
            model_name: Optional model name
        """
        model_name = model_name or self.model_name
        self.client.delete_model_version(name=model_name, version=version)
        print(f"Deleted model '{model_name}' version {version}")

    def set_model_alias(
        self,
        alias: str,
        version: str,
        model_name: Optional[str] = None
    ) -> None:
        """
        Set an alias for a model version.

        Args:
            alias: Alias name (e.g., 'champion', 'challenger')
            version: Model version
            model_name: Optional model name
        """
        model_name = model_name or self.model_name
        self.client.set_registered_model_alias(model_name, alias, version)
        print(f"Set alias '{alias}' for model '{model_name}' v{version}")


def register_best_model(
    experiment_name: str,
    metric: str = "auc_roc",
    model_name: str = "fraud-detection-model",
    stage: str = "Staging"
) -> Optional[str]:
    """
    Register the best model from an experiment.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        model_name: Name for registered model
        stage: Initial stage for the model

    Returns:
        Model version or None
    """
    from .tracking import get_best_run

    best_run = get_best_run(experiment_name, metric)
    if not best_run:
        print("No runs found in experiment")
        return None

    registry = ModelRegistry()
    version = registry.register_model(
        run_id=best_run["run_id"],
        model_name=model_name,
        description=f"Best model by {metric}: {best_run['metrics'].get(metric, 'N/A')}"
    )

    registry.transition_model_stage(version, stage, model_name)
    return version


if __name__ == "__main__":
    # Test model registry
    registry = ModelRegistry()

    # List versions
    versions = registry.list_versions()
    print(f"Found {len(versions)} model versions")

    # Get latest production model
    prod_model = registry.get_latest_version(stage="Production")
    if prod_model:
        print(f"Production model: v{prod_model['version']}")
    else:
        print("No production model found")
