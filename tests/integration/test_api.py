"""
Integration Tests for FastAPI
=============================
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from deployment.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200

    def test_predict_endpoint_structure(self, client):
        """Test predict endpoint accepts correct structure."""
        # This test may fail if model is not loaded
        # but it tests the API structure
        payload = {
            "transaction": {
                "TransactionAmt": 100.0,
                "card1": 1234,
                "C14": 1.0
            },
            "threshold": 0.5
        }

        response = client.post("/predict", json=payload)

        # Either success or model not loaded error
        assert response.status_code in [200, 503]

    def test_predict_batch_endpoint_structure(self, client):
        """Test batch predict endpoint accepts correct structure."""
        payload = {
            "transactions": [
                {"TransactionAmt": 100.0},
                {"TransactionAmt": 200.0}
            ],
            "threshold": 0.5
        }

        response = client.post("/predict/batch", json=payload)

        assert response.status_code in [200, 503]

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")

        # Either returns info or model not loaded
        assert response.status_code in [200, 503]

    def test_invalid_threshold(self, client):
        """Test that invalid threshold is rejected."""
        payload = {
            "transaction": {"TransactionAmt": 100.0},
            "threshold": 1.5  # Invalid: > 1
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error


class TestAPISchemas:
    """Tests for API schemas."""

    def test_transaction_features_schema(self):
        """Test TransactionFeatures schema."""
        from deployment.api.schemas import TransactionFeatures

        # Valid transaction
        transaction = TransactionFeatures(
            TransactionAmt=100.0,
            card1=1234,
            C14=1.0
        )
        assert transaction.TransactionAmt == 100.0

        # Transaction with extra fields (should be allowed)
        transaction = TransactionFeatures(
            TransactionAmt=100.0,
            custom_field="value"
        )
        assert hasattr(transaction, "custom_field")

    def test_prediction_response_schema(self):
        """Test PredictionResponse schema."""
        from deployment.api.schemas import PredictionResponse

        response = PredictionResponse(
            is_fraud=True,
            fraud_probability=0.85,
            risk_level="high",
            threshold_used=0.5
        )

        assert response.is_fraud == True
        assert response.fraud_probability == 0.85
        assert response.risk_level == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
