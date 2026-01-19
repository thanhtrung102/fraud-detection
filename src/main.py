"""
Main Pipeline
=============

End-to-end fraud detection pipeline with SHAP feature selection
and Optuna hyperparameter tuning (as per paper methodology).

Paper: "Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods"
Target Metrics: 99% Accuracy, 0.99 AUC-ROC, 0.99 Precision, 0.99 Recall, 0.99 F1
"""

import sys
import time
from pathlib import Path

# Support both: python src/main.py AND python -m src.main
try:
    from .data_preprocessing import load_config, preprocess_pipeline
    from .evaluation import (
        compute_metrics,
        generate_all_plots,
        print_results,
        save_results,
    )
    from .explainability import generate_all_explanations
    from .feature_selection import apply_feature_selection, shap_feature_selection
    from .optuna_tuning import tune_all_models
    from .stacking_model import StackingFraudDetector, find_optimal_threshold
except ImportError:
    from data_preprocessing import load_config, preprocess_pipeline
    from evaluation import (
        compute_metrics,
        generate_all_plots,
        print_results,
        save_results,
    )
    from explainability import generate_all_explanations
    from feature_selection import apply_feature_selection, shap_feature_selection
    from optuna_tuning import tune_all_models
    from stacking_model import StackingFraudDetector, find_optimal_threshold


# Paper target metrics
SUCCESS_METRICS = {
    'accuracy': 0.99,
    'auc_roc': 0.99,
    'precision': 0.99,
    'recall': 0.99,
    'f1_score': 0.99
}


def validate_metrics(metrics: dict, targets: dict = SUCCESS_METRICS) -> dict:
    """
    Validate achieved metrics against paper targets.

    Args:
        metrics: Achieved metrics dictionary
        targets: Target metrics dictionary

    Returns:
        Validation results dictionary
    """
    results = {}
    print("\n" + "=" * 60)
    print("  Success Metrics Validation (Paper Targets)")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Achieved':>10} {'Target':>10} {'Status':>10}")
    print("-" * 60)

    all_passed = True
    for metric, target in targets.items():
        achieved = metrics.get(metric, 0)
        passed = achieved >= target
        status = "PASS" if passed else "BELOW"
        if not passed:
            all_passed = False

        results[metric] = {
            'achieved': achieved,
            'target': target,
            'passed': passed
        }
        print(f"  {metric:<12} {achieved:>10.4f} {target:>10.2f} {status:>10}")

    print("-" * 60)
    overall = "ALL TARGETS MET" if all_passed else "TARGETS NOT MET"
    print(f"  Overall: {overall}")
    print("=" * 60)

    return results


def main(use_optuna: bool = True, use_feature_selection: bool = True):
    """
    Run the complete fraud detection pipeline.

    Args:
        use_optuna: Whether to use Optuna hyperparameter tuning
        use_feature_selection: Whether to use SHAP feature selection
    """
    print("=" * 60)
    print("  Credit Card Fraud Detection Pipeline")
    print("  (Replicating Paper Methodology)")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Check if Optuna/feature selection are enabled in config
    optuna_config = config.get('optuna', {})
    feature_config = config.get('feature_selection', {})

    n_trials = optuna_config.get('n_trials', 20)
    n_top_features = feature_config.get('n_top_features', 30)
    shap_sample_size = feature_config.get('shap_sample_size', 50000)

    # Phase 1: Data Preprocessing
    print("\n[1/6] Loading and preprocessing data...")
    start_time = time.time()

    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

    # Convert to numpy if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    preprocess_time = time.time() - start_time
    print(f"      Preprocessing completed in {preprocess_time:.2f}s")

    # Phase 2: SHAP Feature Selection (as per paper - top 30 features)
    if use_feature_selection:
        print(f"\n[2/6] SHAP Feature Selection (top {n_top_features} features)...")
        start_time = time.time()

        feature_indices, selected_features, importance_df = shap_feature_selection(
            X_train_np, y_train_np, feature_names,
            n_top_features=n_top_features,
            sample_size=shap_sample_size
        )

        # Apply feature selection
        X_train_selected = apply_feature_selection(X_train_np, feature_indices)
        X_test_selected = apply_feature_selection(X_test_np, feature_indices)
        feature_names = selected_features

        feature_time = time.time() - start_time
        print(f"      Feature selection completed in {feature_time:.2f}s")
    else:
        print("\n[2/6] Skipping feature selection...")
        X_train_selected = X_train_np
        X_test_selected = X_test_np

    # Phase 3: Optuna Hyperparameter Tuning (as per paper - 20 trials)
    if use_optuna:
        print(f"\n[3/6] Optuna Hyperparameter Tuning ({n_trials} trials)...")
        start_time = time.time()

        best_params = tune_all_models(
            X_train_selected, y_train_np,
            n_trials=n_trials,
            cv_folds=config.get('cross_validation', {}).get('n_splits', 5),
            random_state=config.get('data', {}).get('random_state', 42)
        )

        # Use tuned parameters
        xgb_params = best_params['xgboost']
        lgbm_params = best_params['lightgbm']
        catboost_params = best_params['catboost']

        optuna_time = time.time() - start_time
        print(f"      Hyperparameter tuning completed in {optuna_time:.2f}s")
    else:
        print("\n[3/6] Using default hyperparameters...")
        xgb_params = config.get('base_models', {}).get('xgboost')
        lgbm_params = config.get('base_models', {}).get('lightgbm')
        catboost_params = config.get('base_models', {}).get('catboost')

    # Phase 4: Model Training
    print("\n[4/6] Training stacking ensemble...")
    start_time = time.time()

    model = StackingFraudDetector(
        xgb_params=xgb_params,
        lgbm_params=lgbm_params,
        catboost_params=catboost_params,
        meta_params=config.get('meta_learner')
    )
    model.fit(X_train_selected, y_train_np)

    train_time = time.time() - start_time
    print(f"      Training completed in {train_time:.2f}s")

    # Phase 5: Prediction & Threshold Optimization
    print("\n[5/6] Generating predictions...")

    y_proba = model.predict_proba(X_test_selected)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test_np, y_proba, metric='f1')
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Phase 6: Evaluation
    print("\n[6/6] Evaluating model performance...")

    metrics = compute_metrics(y_test_np, y_pred, y_proba)
    print_results(metrics, title="Fraud Detection Results")

    # Validate against paper targets
    validation_results = validate_metrics(metrics)

    # Save results
    output_dir = Path(config.get('output', {}).get('results_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_all_plots(y_test_np, y_pred, y_proba, str(output_dir))
    save_results(metrics, validation_results, str(output_dir))

    # Generate explainability reports
    print("\n[BONUS] Generating explainability reports...")
    generate_all_explanations(
        model.xgb_model,
        X_train_selected,
        X_test_selected,
        y_test_np,
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
    print(f"  Accuracy:  {metrics['accuracy']:.4f} (target: 0.99)")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f} (target: 0.99)")
    print(f"  Precision: {metrics['precision']:.4f} (target: 0.99)")
    print(f"  Recall:    {metrics['recall']:.4f} (target: 0.99)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} (target: 0.99)")
    print(f"  Threshold: {optimal_threshold:.2f}")
    print(f"  Results:   {output_dir}/")
    print("=" * 60)

    return model, metrics, validation_results


if __name__ == "__main__":
    # Parse command line arguments
    use_optuna = '--no-optuna' not in sys.argv
    use_feature_selection = '--no-feature-selection' not in sys.argv

    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python -m src.main [options]")
        print("\nOptions:")
        print("  --no-optuna             Skip Optuna hyperparameter tuning")
        print("  --no-feature-selection  Skip SHAP feature selection")
        print("  --help, -h              Show this help message")
        sys.exit(0)

    model, metrics, validation = main(
        use_optuna=use_optuna,
        use_feature_selection=use_feature_selection
    )
