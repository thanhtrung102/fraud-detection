"""
Pydantic Schemas
================

Request and response models for the fraud detection API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Single transaction features for prediction."""

    TransactionDT: Optional[float] = Field(None, description="Transaction datetime delta")
    TransactionAmt: Optional[float] = Field(None, description="Transaction amount")
    ProductCD: Optional[str] = Field(None, description="Product code")
    card1: Optional[float] = Field(None, description="Card identifier 1")
    card2: Optional[float] = Field(None, description="Card identifier 2")
    card3: Optional[float] = Field(None, description="Card identifier 3")
    card4: Optional[str] = Field(None, description="Card type")
    card5: Optional[float] = Field(None, description="Card identifier 5")
    card6: Optional[str] = Field(None, description="Card category")
    addr1: Optional[float] = Field(None, description="Address 1")
    addr2: Optional[float] = Field(None, description="Address 2")
    dist1: Optional[float] = Field(None, description="Distance 1")
    dist2: Optional[float] = Field(None, description="Distance 2")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")

    # C columns (count features)
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None

    # D columns (time delta features)
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D15: Optional[float] = None

    # V columns (Vesta features) - selected important ones
    V258: Optional[float] = None
    V280: Optional[float] = None
    V282: Optional[float] = None
    V283: Optional[float] = None
    V285: Optional[float] = None
    V308: Optional[float] = None
    V310: Optional[float] = None
    V317: Optional[float] = None

    # Allow additional features
    class Config:
        extra = "allow"


class PredictionRequest(BaseModel):
    """Request model for single prediction."""

    transaction: TransactionFeatures
    threshold: Optional[float] = Field(0.44, ge=0, le=1, description="Classification threshold")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    transactions: list[TransactionFeatures]
    threshold: Optional[float] = Field(0.44, ge=0, le=1, description="Classification threshold")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    threshold_used: float = Field(..., description="Threshold used for classification")
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[PredictionResponse]
    total_count: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = True
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    version: str
    features_count: int
    threshold: float
    metrics: dict[str, float]
    last_updated: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
