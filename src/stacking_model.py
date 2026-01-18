"""
Stacking Ensemble Model Module
==============================

Implements the stacking ensemble with XGBoost, LightGBM, and CatBoost
base learners and XGBoost meta-learner.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import joblib
from pathlib import Path


class StackingFraudDetector:
    """
    Stacking ensemble for fraud detection.

    Base learners: XGBoost, LightGBM, CatBoost
    Meta-learner: XGBoost
    """

    def __init__(self, xgb_params: dict = None, lgbm_params: dict = None,
                 catboost_params: dict = None, meta_params: dict = None):
        """
        Initialize stacking ensemble.

        Args:
            xgb_params: XGBoost hyperparameters
            lgbm_params: LightGBM hyperparameters
            catboost_params: CatBoost hyperparameters
            meta_params: Meta-learner hyperparameters
        """
        # Default XGBoost params
        self.xgb_params = xgb_params or {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.1,
            'random_state': 42
        }

        # Default LightGBM params
        self.lgbm_params = lgbm_params or {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }

        # Default CatBoost params
        self.catboost_params = catboost_params or {
            'iterations': 300,
            'depth': 7,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': 0
        }

        # Default meta-learner params
        self.meta_params = meta_params or {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42
        }

        # Initialize models
        self.xgb_model = XGBClassifier(**self.xgb_params)
        self.lgbm_model = LGBMClassifier(**self.lgbm_params)
        self.catboost_model = CatBoostClassifier(**self.catboost_params)
        self.meta_learner = XGBClassifier(**self.meta_params)

        self.is_fitted = False

    def _get_base_predictions(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Get predictions from all base models.

        Args:
            X: Feature matrix
            training: Whether this is for training (use predict_proba directly)

        Returns:
            Stacked predictions array
        """
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        lgbm_pred = self.lgbm_model.predict_proba(X)[:, 1]
        catboost_pred = self.catboost_model.predict_proba(X)[:, 1]

        return np.column_stack([xgb_pred, lgbm_pred, catboost_pred])

    def fit(self, X: np.ndarray, y: np.ndarray, use_cv_stacking: bool = False):
        """
        Fit the stacking ensemble.

        Args:
            X: Training features
            y: Training labels
            use_cv_stacking: Whether to use cross-validation for meta-features
        """
        print("Training XGBoost base model...")
        self.xgb_model.fit(X, y)

        print("Training LightGBM base model...")
        self.lgbm_model.fit(X, y)

        print("Training CatBoost base model...")
        self.catboost_model.fit(X, y)

        if use_cv_stacking:
            # Use CV predictions to avoid overfitting
            print("Generating CV meta-features...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            xgb_cv = cross_val_predict(
                XGBClassifier(**self.xgb_params), X, y, cv=cv, method='predict_proba'
            )[:, 1]
            lgbm_cv = cross_val_predict(
                LGBMClassifier(**self.lgbm_params), X, y, cv=cv, method='predict_proba'
            )[:, 1]
            catboost_cv = cross_val_predict(
                CatBoostClassifier(**self.catboost_params), X, y, cv=cv, method='predict_proba'
            )[:, 1]

            meta_train = np.column_stack([xgb_cv, lgbm_cv, catboost_cv])
        else:
            # Use direct predictions (as per paper)
            meta_train = self._get_base_predictions(X, training=True)

        print("Training meta-learner...")
        self.meta_learner.fit(meta_train, y)

        self.is_fitted = True
        print("Stacking ensemble training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        meta_features = self._get_base_predictions(X, training=False)
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def save(self, path: str):
        """Save the ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.xgb_model, path / 'xgb_model.joblib')
        joblib.dump(self.lgbm_model, path / 'lgbm_model.joblib')
        joblib.dump(self.catboost_model, path / 'catboost_model.joblib')
        joblib.dump(self.meta_learner, path / 'meta_learner.joblib')
        print(f"Models saved to {path}")

    def load(self, path: str):
        """Load the ensemble from disk."""
        path = Path(path)

        self.xgb_model = joblib.load(path / 'xgb_model.joblib')
        self.lgbm_model = joblib.load(path / 'lgbm_model.joblib')
        self.catboost_model = joblib.load(path / 'catboost_model.joblib')
        self.meta_learner = joblib.load(path / 'meta_learner.joblib')
        self.is_fitted = True
        print(f"Models loaded from {path}")


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                          metric: str = 'f1') -> tuple:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Optimization metric ('f1', 'precision', 'recall')

    Returns:
        (optimal_threshold, best_score)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }

    metric_func = metric_funcs[metric]

    best_threshold = 0.5
    best_score = 0

    for thresh in np.arange(0.30, 0.70, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        score = metric_func(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"Optimal threshold: {best_threshold:.2f} ({metric}: {best_score:.4f})")
    return best_threshold, best_score


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20,
                               n_classes=2, random_state=42)

    model = StackingFraudDetector()
    model.fit(X, y)

    proba = model.predict_proba(X)[:, 1]
    threshold, score = find_optimal_threshold(y, proba)

    print(f"\nTest complete. Threshold: {threshold}, F1: {score:.4f}")
