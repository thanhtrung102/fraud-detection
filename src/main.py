"""
Main Pipeline
=============

End-to-end fraud detection pipeline.
"""

import time
import numpy as np
from pathlib import Path

from data_preprocessing import preprocess_pipeline, load_config
from stacking_model import StackingFraudDetector, find_optimal_threshold
from evaluation import (
    compute_metrics, print_results, generate_all_plots, save_results
)
from explainability import generate_all_explanations


def main():
    """Run the complete fraud detection pipeline."""
    print("=" * 60)
    print("  Credit Card Fraud Detection Pipeline")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Phase 1: Data Preprocessing
    print("\n[1/5] Loading and preprocessing data...")
    start_time = time.time()

    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

    preprocess_time = time.time() - start_time
    print(f"      Preprocessing completed in {preprocess_time:.2f}s")

    # Phase 2: Model Training
    print("\n[2/5] Training stacking ensemble...")
    start_time = time.time()

    model = StackingFraudDetector(
        xgb_params=config.get('base_models', {}).get('xgboost'),
        lgbm_params=config.get('base_models', {}).get('lightgbm'),
        catboost_params=config.get('base_models', {}).get('catboost'),
        meta_params=config.get('meta_learner')
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"      Training completed in {train_time:.2f}s")

    # Phase 3: Prediction & Threshold Optimization
    print("\n[3/5] Generating predictions...")

    y_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, y_proba, metric='f1')
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Phase 4: Evaluation
    print("\n[4/5] Evaluating model performance...")

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_results(metrics, title="Fraud Detection Results")

    # Save results
    output_dir = Path(config.get('output', {}).get('results_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_all_plots(y_test, y_pred, y_proba, str(output_dir))
    save_results(metrics, {}, str(output_dir))

    # Phase 5: Explainability
    print("\n[5/5] Generating explainability reports...")

    # Use the XGBoost base model for explanations
    top_features = generate_all_explanations(
        model.xgb_model,
        X_train if isinstance(X_train, np.ndarray) else X_train.values,
        X_test if isinstance(X_test, np.ndarray) else X_test.values,
        y_test if isinstance(y_test, np.ndarray) else y_test.values,
        feature_names,
        str(output_dir)
    )

    # Save model
    if config.get('output', {}).get('save_models', True):
        models_dir = config.get('output', {}).get('models_dir', 'models')
        model.save(models_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Threshold: {optimal_threshold:.2f}")
    print(f"  Results:   {output_dir}/")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
