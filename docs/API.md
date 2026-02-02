# API Reference

Complete API documentation for the Fraud Detection REST API built with FastAPI.

---

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Model Information](#model-information)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction](#batch-prediction)
  - [Model Reload](#model-reload)
- [Request/Response Schemas](#requestresponse-schemas)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

---

## Overview

The Fraud Detection API provides real-time credit card fraud detection using a stacking ensemble model (XGBoost, LightGBM, CatBoost with Logistic Regression meta-learner).

### Key Features

- Single and batch transaction predictions
- Configurable classification threshold
- Risk level classification (low/medium/high)
- Model hot-reload capability
- OpenAPI/Swagger documentation

### Quick Start

```bash
# Start API server
uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
make docker-serve

# Access Swagger UI
open http://localhost:8000/docs
```

---

## Base URL

| Environment | URL |
|-------------|-----|
| Local Development | `http://localhost:8000` |
| Docker | `http://localhost:8000` |
| Cloud Run | `https://fraud-detection-api-xxxxx.run.app` |

---

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:

- API Key authentication
- OAuth2/JWT tokens
- Cloud IAM (for GCP Cloud Run)

---

## Endpoints

### Health Check

Check API and model status.

#### `GET /`

Root endpoint returning basic health status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2026-02-02T10:30:00.000Z"
}
```

#### `GET /health`

Detailed health check with model status.

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` or `degraded` |
| `model_loaded` | boolean | Whether model is loaded |
| `version` | string | API version |
| `timestamp` | datetime | Current server time |

**Example:**

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2026-02-02T10:30:00.000Z"
}
```

---

### Model Information

#### `GET /model/info`

Get information about the loaded model.

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Model identifier |
| `version` | string | Model version |
| `features_count` | integer | Number of input features |
| `threshold` | float | Default classification threshold |
| `metrics` | object | Model performance metrics |
| `last_updated` | datetime | Model load timestamp |

**Example:**

```bash
curl http://localhost:8000/model/info
```

```json
{
  "model_name": "fraud-detection-stacking",
  "version": "1.1.0",
  "features_count": 30,
  "threshold": 0.44,
  "metrics": {
    "auc_roc": 0.92,
    "accuracy": 0.98
  },
  "last_updated": "2026-02-02T10:00:00.000Z"
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 503 | Model not loaded |

---

### Single Prediction

#### `POST /predict`

Predict fraud probability for a single transaction.

**Request Body:**

```json
{
  "transaction": {
    "TransactionAmt": 150.0,
    "card1": 12345,
    "C14": 1.0,
    "C1": 2.0
  },
  "threshold": 0.44
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `transaction` | object | Yes | Transaction features |
| `threshold` | float | No | Classification threshold (0-1), default: 0.44 |

**Transaction Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `TransactionDT` | float | Transaction datetime delta |
| `TransactionAmt` | float | Transaction amount |
| `card1` - `card6` | float/string | Card identifiers |
| `addr1`, `addr2` | float | Address codes |
| `dist1`, `dist2` | float | Distance features |
| `P_emaildomain` | string | Purchaser email domain |
| `R_emaildomain` | string | Recipient email domain |
| `C1` - `C14` | float | Count features |
| `D1` - `D15` | float | Time delta features |
| `V258`, `V280`, etc. | float | Vesta engineered features |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `is_fraud` | boolean | Fraud prediction |
| `fraud_probability` | float | Probability score (0-1) |
| `risk_level` | string | `low`, `medium`, or `high` |
| `threshold_used` | float | Threshold used for classification |
| `timestamp` | datetime | Prediction timestamp |

**Risk Level Thresholds:**

| Risk Level | Probability Range |
|------------|-------------------|
| Low | < 0.5 |
| Medium | 0.5 - 0.8 |
| High | >= 0.8 |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "TransactionAmt": 150.0,
      "card1": 12345,
      "C14": 1.0,
      "C1": 2.0,
      "V258": 0.5
    },
    "threshold": 0.5
  }'
```

```json
{
  "is_fraud": false,
  "fraud_probability": 0.23,
  "risk_level": "low",
  "threshold_used": 0.5,
  "timestamp": "2026-02-02T10:30:00.000Z"
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 422 | Validation error (invalid input) |
| 500 | Internal server error |
| 503 | Model not loaded |

---

### Batch Prediction

#### `POST /predict/batch`

Predict fraud for multiple transactions in a single request.

**Request Body:**

```json
{
  "transactions": [
    {"TransactionAmt": 100.0, "card1": 1234, "C14": 1.0},
    {"TransactionAmt": 500.0, "card1": 5678, "C14": 2.0},
    {"TransactionAmt": 50.0, "card1": 9012, "C14": 0.5}
  ],
  "threshold": 0.44
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `transactions` | array | Yes | List of transaction objects |
| `threshold` | float | No | Classification threshold (0-1) |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array | List of prediction results |
| `total_count` | integer | Total transactions processed |
| `fraud_count` | integer | Number flagged as fraud |
| `fraud_rate` | float | Percentage of fraud (0-1) |
| `processing_time_ms` | float | Processing time in milliseconds |

**Example:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionAmt": 100.0, "card1": 1234},
      {"TransactionAmt": 5000.0, "card1": 5678}
    ],
    "threshold": 0.44
  }'
```

```json
{
  "predictions": [
    {
      "is_fraud": false,
      "fraud_probability": 0.12,
      "risk_level": "low",
      "threshold_used": 0.44,
      "timestamp": "2026-02-02T10:30:00.000Z"
    },
    {
      "is_fraud": true,
      "fraud_probability": 0.87,
      "risk_level": "high",
      "threshold_used": 0.44,
      "timestamp": "2026-02-02T10:30:00.000Z"
    }
  ],
  "total_count": 2,
  "fraud_count": 1,
  "fraud_rate": 0.5,
  "processing_time_ms": 45.2
}
```

**Performance Notes:**

- Batch processing is more efficient than multiple single requests
- Recommended batch size: 100-1000 transactions
- Maximum batch size: 10,000 transactions

---

### Model Reload

#### `POST /model/reload`

Hot-reload the model from disk without restarting the server.

**Response:**

```json
{
  "status": "success",
  "message": "Model reloaded successfully"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/model/reload
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 500 | Failed to reload model |

---

## Request/Response Schemas

### TransactionFeatures

```python
class TransactionFeatures(BaseModel):
    TransactionDT: Optional[float] = None
    TransactionAmt: Optional[float] = None
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    C1: Optional[float] = None
    # ... C2-C14
    D1: Optional[float] = None
    # ... D2-D15
    V258: Optional[float] = None
    # ... V280, V282, V283, V285, V308, V310, V317

    class Config:
        extra = "allow"  # Allows additional features
```

### PredictionRequest

```python
class PredictionRequest(BaseModel):
    transaction: TransactionFeatures
    threshold: Optional[float] = Field(0.44, ge=0, le=1)
```

### PredictionResponse

```python
class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float  # 0.0 - 1.0
    risk_level: str  # "low", "medium", "high"
    threshold_used: float
    timestamp: datetime
```

### BatchPredictionRequest

```python
class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionFeatures]
    threshold: Optional[float] = Field(0.44, ge=0, le=1)
```

### BatchPredictionResponse

```python
class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_count: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing the issue"
}
```

### HTTP Status Codes

| Status | Description |
|--------|-------------|
| 200 | Success |
| 422 | Validation Error - Invalid request body |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

### Validation Errors

```json
{
  "detail": [
    {
      "loc": ["body", "transaction", "TransactionAmt"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production, consider:

| Endpoint | Recommended Limit |
|----------|-------------------|
| `/health` | 100 req/min |
| `/predict` | 1000 req/min |
| `/predict/batch` | 100 req/min |
| `/model/reload` | 1 req/min |

---

## Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "transaction": {
            "TransactionAmt": 150.0,
            "card1": 12345,
            "C14": 1.0,
            "C1": 2.0
        },
        "threshold": 0.5
    }
)
result = response.json()
print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']:.2%}")

# Batch prediction
transactions = [
    {"TransactionAmt": 100.0, "card1": 1234},
    {"TransactionAmt": 500.0, "card1": 5678},
    {"TransactionAmt": 50.0, "card1": 9012}
]

response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={"transactions": transactions, "threshold": 0.44}
)
batch_result = response.json()
print(f"Fraud rate: {batch_result['fraud_rate']:.2%}")
print(f"Processing time: {batch_result['processing_time_ms']:.1f}ms")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"TransactionAmt": 150.0, "card1": 12345}}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionAmt": 100.0, "card1": 1234},
      {"TransactionAmt": 500.0, "card1": 5678}
    ]
  }'

# Model info
curl http://localhost:8000/model/info

# Reload model
curl -X POST http://localhost:8000/model/reload
```

### JavaScript/Fetch

```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    transaction: {
      TransactionAmt: 150.0,
      card1: 12345,
      C14: 1.0
    },
    threshold: 0.5
  })
});

const result = await response.json();
console.log(`Fraud: ${result.is_fraud}, Risk: ${result.risk_level}`);
```

---

## OpenAPI Specification

The full OpenAPI specification is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Changelog

### v1.0.0

- Initial release
- Single and batch prediction endpoints
- Model info and reload endpoints
- CORS middleware enabled

### v1.1.0

- Added risk level classification
- Improved error handling
- Added processing time metrics for batch predictions
