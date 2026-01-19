"""
FastAPI Fraud Detection Service
===============================

REST API for real-time fraud detection predictions.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    TransactionFeatures,
)

# Global model instance
model = None
feature_names = None
model_info = {}


def load_model():
    """Load the trained model."""
    global model, feature_names, model_info

    from src.stacking_model import StackingFraudDetector

    model_dir = os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models"))

    try:
        model = StackingFraudDetector()
        model.load(model_dir)

        # Load feature names from config or use defaults
        feature_names = get_default_features()

        model_info = {
            "model_name": "fraud-detection-stacking",
            "version": "1.1.0",
            "features_count": len(feature_names),
            "threshold": 0.44,
            "last_updated": datetime.now(),
        }

        print(f"Model loaded from {model_dir}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def get_default_features() -> list[str]:
    """Get default feature names."""
    # Top 30 features from SHAP analysis
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


def transaction_to_array(transaction: TransactionFeatures) -> np.ndarray:
    """Convert transaction features to numpy array."""
    global feature_names

    values = []
    transaction_dict = transaction.model_dump()

    for feature in feature_names:
        value = transaction_dict.get(feature, 0)
        # Handle None and string values
        if value is None:
            value = 0
        elif isinstance(value, str):
            # Simple encoding for categorical features
            value = hash(value) % 10000
        values.append(float(value))

    return np.array(values).reshape(1, -1)


def get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability >= 0.8:
        return "high"
    elif probability >= 0.5:
        return "medium"
    else:
        return "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting Fraud Detection API...")
    success = load_model()
    if not success:
        print("Warning: Model not loaded. Some endpoints may not work.")
    yield
    # Shutdown
    print("Shutting down Fraud Detection API...")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using stacking ensemble",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(status="healthy", model_loaded=model is not None, version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return ModelInfoResponse(
        model_name=model_info.get("model_name", "unknown"),
        version=model_info.get("version", "unknown"),
        features_count=model_info.get("features_count", 0),
        threshold=model_info.get("threshold", 0.44),
        metrics={"auc_roc": 0.92, "accuracy": 0.98},  # Would load from registry
        last_updated=model_info.get("last_updated"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud for a single transaction.

    Args:
        request: Transaction features and optional threshold

    Returns:
        Fraud prediction with probability and risk level
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        # Convert to array
        X = transaction_to_array(request.transaction)

        # Get prediction
        proba = model.predict_proba(X)[0, 1]
        is_fraud = proba >= request.threshold

        return PredictionResponse(
            is_fraud=bool(is_fraud),
            fraud_probability=float(proba),
            risk_level=get_risk_level(proba),
            threshold_used=request.threshold,
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions.

    Args:
        request: List of transactions and optional threshold

    Returns:
        Batch predictions with summary statistics
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        start_time = time.time()

        # Convert all transactions to array
        X = np.vstack([transaction_to_array(t) for t in request.transactions])

        # Get predictions
        probas = model.predict_proba(X)[:, 1]
        predictions = []

        for proba in probas:
            is_fraud = proba >= request.threshold
            predictions.append(
                PredictionResponse(
                    is_fraud=bool(is_fraud),
                    fraud_probability=float(proba),
                    risk_level=get_risk_level(proba),
                    threshold_used=request.threshold,
                )
            )

        processing_time = (time.time() - start_time) * 1000
        fraud_count = sum(1 for p in predictions if p.is_fraud)

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            fraud_count=fraud_count,
            fraud_rate=fraud_count / len(predictions) if predictions else 0,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/model/reload")
async def reload_model():
    """Reload the model from disk."""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reload model"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
