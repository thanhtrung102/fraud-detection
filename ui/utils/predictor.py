"""
Fraud Predictor
===============

Prediction utilities for fraud detection.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ui.utils.model_loader import FraudModelLoader

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Fraud prediction handler."""

    def __init__(self, model_loader: FraudModelLoader):
        """
        Initialize predictor with a model loader.

        Args:
            model_loader: Loaded FraudModelLoader instance
        """
        self.model_loader = model_loader

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            df: Input DataFrame with transaction data

        Returns:
            DataFrame with features aligned to model expectations
        """
        feature_names = self.model_loader.feature_names

        # Create output DataFrame with all required features
        result = pd.DataFrame(index=df.index)

        for feature in feature_names:
            if feature in df.columns:
                result[feature] = df[feature]
            else:
                # Fill missing features with 0
                result[feature] = 0

        # Handle categorical features
        for col in result.columns:
            if result[col].dtype == "object":
                # Simple hash encoding for categorical features
                result[col] = result[col].apply(lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0)

        # Fill NaN values
        result = result.fillna(0)

        return result

    def get_risk_level(self, probability: float) -> str:
        """
        Determine risk level from fraud probability.

        Args:
            probability: Fraud probability (0-1)

        Returns:
            Risk level string
        """
        if probability >= 0.8:
            return "high"
        elif probability >= 0.5:
            return "medium"
        else:
            return "low"

    def predict_single(
        self,
        transaction: dict[str, Any],
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Predict fraud for a single transaction.

        Args:
            transaction: Transaction features as dictionary
            threshold: Optional custom threshold

        Returns:
            Prediction results
        """
        if not self.model_loader.loaded:
            return {"success": False, "error": "Model not loaded"}

        threshold = threshold or self.model_loader.threshold

        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction])
            X = self.prepare_features(df)

            # Get prediction
            proba = self.model_loader.model.predict_proba(X.values)[0, 1]
            is_fraud = proba >= threshold

            return {
                "success": True,
                "is_fraud": bool(is_fraud),
                "fraud_probability": float(proba),
                "risk_level": self.get_risk_level(proba),
                "threshold_used": threshold,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e)}

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Predict fraud for multiple transactions.

        Args:
            df: DataFrame with transaction data
            threshold: Optional custom threshold

        Returns:
            Batch prediction results
        """
        if not self.model_loader.loaded:
            return {"success": False, "error": "Model not loaded"}

        threshold = threshold or self.model_loader.threshold

        try:
            X = self.prepare_features(df)

            # Get predictions
            probas = self.model_loader.model.predict_proba(X.values)[:, 1]
            predictions = probas >= threshold

            # Build results DataFrame
            results_df = df.copy()
            results_df["fraud_probability"] = probas
            results_df["is_fraud"] = predictions
            results_df["risk_level"] = [self.get_risk_level(p) for p in probas]

            # Calculate summary statistics
            fraud_count = int(predictions.sum())
            total_count = len(predictions)

            return {
                "success": True,
                "predictions": results_df,
                "summary": {
                    "total_count": total_count,
                    "fraud_count": fraud_count,
                    "legitimate_count": total_count - fraud_count,
                    "fraud_rate": fraud_count / total_count if total_count > 0 else 0,
                    "avg_probability": float(probas.mean()),
                    "max_probability": float(probas.max()),
                    "min_probability": float(probas.min()),
                    "high_risk_count": int((probas >= 0.8).sum()),
                    "medium_risk_count": int(((probas >= 0.5) & (probas < 0.8)).sum()),
                    "low_risk_count": int((probas < 0.5).sum()),
                },
                "threshold_used": threshold,
            }

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_sample_data(
        self,
        n_samples: int = 100,
        fraud_rate: float = 0.035,
    ) -> pd.DataFrame:
        """
        Generate sample transaction data for testing.

        Args:
            n_samples: Number of samples to generate
            fraud_rate: Approximate fraud rate in generated data

        Returns:
            DataFrame with sample transactions
        """
        np.random.seed(42)

        feature_names = self.model_loader.feature_names

        # Generate base features
        data = {}

        for feature in feature_names:
            if feature.startswith("C"):
                # C-features are typically counts
                data[feature] = np.random.exponential(2, n_samples)
            elif feature.startswith("V"):
                # V-features are various numerical
                data[feature] = np.random.normal(0, 1, n_samples)
            elif feature.startswith("D"):
                # D-features are typically time deltas
                data[feature] = np.random.exponential(100, n_samples)
            elif feature == "TransactionAmt":
                data[feature] = np.random.exponential(100, n_samples) + 10
            elif feature == "TransactionDT":
                data[feature] = np.random.randint(0, 15811131, n_samples)
            elif feature in ["card1", "card2", "card5"]:
                data[feature] = np.random.randint(1000, 20000, n_samples)
            elif feature == "card6":
                data[feature] = np.random.choice([0, 1], n_samples)
            elif feature in ["addr1", "dist1"]:
                data[feature] = np.random.randint(0, 500, n_samples)
            else:
                data[feature] = np.random.normal(0, 1, n_samples)

        df = pd.DataFrame(data)

        # Add transaction IDs
        df.insert(0, "transaction_id", [f"TXN_{i:06d}" for i in range(n_samples)])

        return df
