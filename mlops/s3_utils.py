"""
S3/MinIO Utilities for MLflow Artifact Storage
===============================================

Utilities for managing MLflow artifacts in S3-compatible storage (MinIO).
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_minio_endpoint() -> str:
    """
    Get MinIO endpoint URL with environment-aware discovery.

    Returns:
        MinIO endpoint URL
    """
    # Check explicit environment variable first
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if endpoint:
        return endpoint

    # Detect if running in container
    in_container = os.path.exists("/.dockerenv") or os.environ.get("AIRFLOW__CORE__EXECUTOR")

    if in_container:
        return "http://minio:9000"
    else:
        return "http://localhost:9000"


class S3ArtifactManager:
    """Manage artifacts in S3-compatible storage (MinIO)."""

    def __init__(
        self,
        bucket_name: str = "mlflow-artifacts",
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize S3 artifact manager.

        Args:
            bucket_name: S3 bucket name for artifacts
            endpoint_url: S3/MinIO endpoint URL
            access_key: AWS/MinIO access key
            secret_key: AWS/MinIO secret key
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url or get_minio_endpoint()
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

        logger.info(f"S3ArtifactManager initialized with endpoint: {self.endpoint_url}")

    def ensure_bucket_exists(self) -> bool:
        """
        Ensure the artifact bucket exists, create if not.

        Returns:
            True if bucket exists or was created
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created bucket {self.bucket_name}")
                    return True
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    return False
            else:
                logger.error(f"Error checking bucket: {e}")
                return False

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> bool:
        """
        Upload a single file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata dictionary

        Returns:
            True if upload successful
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3_client.upload_file(
                local_path, self.bucket_name, s3_key, ExtraArgs=extra_args if extra_args else None
            )
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3.

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            True if download successful
        """
        try:
            # Ensure parent directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False

    def upload_artifact(
        self,
        local_path: str,
        run_id: str,
        artifact_path: Optional[str] = None,
    ) -> str:
        """
        Upload artifact to S3 with MLflow-compatible key structure.

        Args:
            local_path: Local file or directory path
            run_id: MLflow run ID
            artifact_path: Optional subdirectory in artifacts

        Returns:
            S3 key prefix for uploaded artifacts
        """
        # MLflow S3 key structure: {run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/
        base_key = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts"

        if artifact_path:
            base_key = f"{base_key}/{artifact_path}"

        local_path = Path(local_path)

        if local_path.is_file():
            s3_key = f"{base_key}/{local_path.name}"
            self.upload_file(str(local_path), s3_key)
            return s3_key
        elif local_path.is_dir():
            uploaded_keys = []
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{base_key}/{relative_path}"
                    self.upload_file(str(file_path), s3_key)
                    uploaded_keys.append(s3_key)
            logger.info(f"Uploaded {len(uploaded_keys)} files from {local_path}")
            return base_key
        else:
            raise ValueError(f"Path does not exist: {local_path}")

    def list_artifacts(self, run_id: str, artifact_path: Optional[str] = None) -> list[str]:
        """
        List artifacts for a specific MLflow run.

        Args:
            run_id: MLflow run ID
            artifact_path: Optional subdirectory filter

        Returns:
            List of S3 keys
        """
        prefix = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts"
        if artifact_path:
            prefix = f"{prefix}/{artifact_path}"

        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            logger.error(f"Failed to list artifacts: {e}")
            return []

    def download_artifacts(
        self,
        run_id: str,
        artifact_path: Optional[str] = None,
        dst_path: Optional[str] = None,
    ) -> str:
        """
        Download all artifacts for a run.

        Args:
            run_id: MLflow run ID
            artifact_path: Optional subdirectory to download
            dst_path: Destination directory (default: temp directory)

        Returns:
            Local path to downloaded artifacts
        """
        if dst_path is None:
            dst_path = tempfile.mkdtemp(prefix=f"mlflow_artifacts_{run_id[:8]}_")

        artifacts = self.list_artifacts(run_id, artifact_path)

        base_prefix = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts"
        if artifact_path:
            base_prefix = f"{base_prefix}/{artifact_path}"

        for s3_key in artifacts:
            relative_path = s3_key[len(base_prefix) :].lstrip("/")
            local_path = Path(dst_path) / relative_path
            self.download_file(s3_key, str(local_path))

        logger.info(f"Downloaded {len(artifacts)} artifacts to {dst_path}")
        return dst_path

    def sync_mlflow_artifacts_to_s3(self, run_id: str, local_artifacts_dir: str) -> int:
        """
        Sync local MLflow artifacts to S3.

        Args:
            run_id: MLflow run ID
            local_artifacts_dir: Local directory containing artifacts

        Returns:
            Number of files synced
        """
        count = 0
        local_path = Path(local_artifacts_dir)

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{relative_path}"
                if self.upload_file(str(file_path), s3_key):
                    count += 1

        logger.info(f"Synced {count} files to S3 for run {run_id}")
        return count

    def delete_artifacts(self, run_id: str) -> int:
        """
        Delete all artifacts for a run.

        Args:
            run_id: MLflow run ID

        Returns:
            Number of deleted objects
        """
        artifacts = self.list_artifacts(run_id)

        if not artifacts:
            return 0

        delete_objects = [{"Key": key} for key in artifacts]

        try:
            response = self.s3_client.delete_objects(
                Bucket=self.bucket_name, Delete={"Objects": delete_objects}
            )
            deleted = len(response.get("Deleted", []))
            logger.info(f"Deleted {deleted} artifacts for run {run_id}")
            return deleted
        except ClientError as e:
            logger.error(f"Failed to delete artifacts: {e}")
            return 0


def verify_s3_artifacts(
    run_id: str,
    expected_artifacts: list[str],
    bucket_name: str = "mlflow-artifacts",
) -> dict[str, Any]:
    """
    Verify that expected artifacts exist in S3.

    Args:
        run_id: MLflow run ID
        expected_artifacts: List of expected artifact paths
        bucket_name: S3 bucket name

    Returns:
        Verification results dictionary
    """
    manager = S3ArtifactManager(bucket_name=bucket_name)
    artifacts = manager.list_artifacts(run_id)

    results = {
        "success": True,
        "run_id": run_id,
        "total_artifacts": len(artifacts),
        "expected": expected_artifacts,
        "found": [],
        "missing": [],
    }

    for expected in expected_artifacts:
        # Check if any artifact contains the expected path
        found = any(expected in artifact for artifact in artifacts)
        if found:
            results["found"].append(expected)
        else:
            results["missing"].append(expected)
            results["success"] = False

    return results


def log_s3_verification_results(results: dict[str, Any]) -> None:
    """Log S3 verification results."""
    if results["success"]:
        logger.info(
            f"S3 verification passed for run {results['run_id']}: "
            f"{results['total_artifacts']} artifacts found"
        )
    else:
        logger.warning(
            f"S3 verification failed for run {results['run_id']}: "
            f"Missing artifacts: {results['missing']}"
        )


if __name__ == "__main__":
    # Test S3 connection
    logging.basicConfig(level=logging.INFO)

    manager = S3ArtifactManager()
    if manager.ensure_bucket_exists():
        print("S3/MinIO connection successful!")

        # List existing artifacts
        # This would require a valid run_id
        # artifacts = manager.list_artifacts("test_run_id")
        # print(f"Found {len(artifacts)} artifacts")
    else:
        print("Failed to connect to S3/MinIO")
