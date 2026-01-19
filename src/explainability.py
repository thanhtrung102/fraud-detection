"""
Explainability Module
=====================

SHAP, LIME, and Partial Dependence Plot analysis for model interpretability.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: np.ndarray, feature_names: list = None):
    """
    Compute SHAP values for model predictions.

    Args:
        model: Trained model (tree-based)
        X: Feature matrix
        feature_names: List of feature names

    Returns:
        SHAP explainer and values
    """
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return explainer, shap_values


def get_top_features_shap(shap_values: np.ndarray, feature_names: list,
                          n_top: int = 30) -> list:
    """
    Get top N features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        n_top: Number of top features to return

    Returns:
        List of top feature names
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    top_features = feature_importance.head(n_top)['feature'].tolist()

    print(f"\nTop {n_top} features by SHAP importance:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

    return top_features


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray,
                     feature_names: list = None, save_path: str = None):
    """
    Create SHAP summary plot.

    Args:
        shap_values: SHAP values
        X: Feature matrix
        feature_names: Feature names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))

    if feature_names is not None:
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X

    shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")

    plt.close()


def plot_shap_bar(shap_values: np.ndarray, feature_names: list,
                 save_path: str = None):
    """
    Create SHAP bar plot (feature importance).

    Args:
        shap_values: SHAP values
        feature_names: Feature names
        save_path: Path to save plot
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True).tail(20)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 20 Feature Importances (SHAP)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP bar plot saved to {save_path}")

    plt.close()


def explain_single_prediction_lime(model, X_train: np.ndarray,
                                   X_instance: np.ndarray,
                                   feature_names: list,
                                   save_path: str = None):
    """
    Generate LIME explanation for a single prediction.

    Args:
        model: Trained model
        X_train: Training data for LIME explainer
        X_instance: Single instance to explain
        feature_names: Feature names
        save_path: Path to save HTML explanation
    """
    import lime
    import lime.lime_tabular

    print("Generating LIME explanation...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraud'],
        mode='classification'
    )

    exp = explainer.explain_instance(
        X_instance,
        model.predict_proba,
        num_features=10
    )

    if save_path:
        exp.save_to_file(save_path)
        print(f"LIME explanation saved to {save_path}")

    return exp


def plot_partial_dependence(model, X: np.ndarray, feature_names: list,
                           features_to_plot: list = None,
                           save_path: str = None):
    """
    Create Partial Dependence Plots.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Feature names
        features_to_plot: Specific features to plot
        save_path: Path to save plot
    """
    from sklearn.inspection import PartialDependenceDisplay

    if features_to_plot is None:
        features_to_plot = feature_names[:4]

    # Get feature indices
    feature_indices = [feature_names.index(f) for f in features_to_plot]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    X_df = pd.DataFrame(X, columns=feature_names)

    PartialDependenceDisplay.from_estimator(
        model, X_df, features=feature_indices,
        ax=axes, grid_resolution=50
    )

    plt.suptitle('Partial Dependence Plots', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PDP saved to {save_path}")

    plt.close()


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                   feature_names: list, n_repeats: int = 10):
    """
    Compute permutation feature importance.

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        feature_names: Feature names
        n_repeats: Number of permutation repeats

    Returns:
        DataFrame with importance scores
    """
    from sklearn.inspection import permutation_importance

    print("Computing permutation importance...")
    result = permutation_importance(model, X, y, n_repeats=n_repeats,
                                   random_state=42, scoring='roc_auc')

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    print("\nTop 10 features by permutation importance:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance_mean']:.4f} "
              f"(+/- {row['importance_std']:.4f})")

    return importance_df


def generate_all_explanations(model, X_train: np.ndarray, X_test: np.ndarray,
                             y_test: np.ndarray, feature_names: list,
                             output_dir: str = 'results'):
    """
    Generate all explainability artifacts.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SHAP analysis
    explainer, shap_values = compute_shap_values(model, X_test, feature_names)
    top_features = get_top_features_shap(shap_values, feature_names, n_top=30)

    plot_shap_summary(shap_values, X_test, feature_names,
                     str(output_dir / 'shap_summary.png'))
    plot_shap_bar(shap_values, feature_names,
                 str(output_dir / 'shap_bar.png'))

    # Find a fraud case for LIME
    fraud_indices = np.where(y_test == 1)[0]
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        explain_single_prediction_lime(
            model, X_train, X_test[fraud_idx], feature_names,
            str(output_dir / 'lime_explanation.html')
        )

    # Partial dependence (use top 4 features)
    if len(top_features) >= 4:
        plot_partial_dependence(model, X_test, feature_names,
                               top_features[:4],
                               str(output_dir / 'partial_dependence.png'))

    # Permutation importance
    perm_importance = compute_permutation_importance(
        model, X_test, y_test, feature_names
    )
    perm_importance.to_csv(output_dir / 'permutation_importance.csv', index=False)

    print(f"\nAll explanations saved to {output_dir}")
    return top_features


if __name__ == "__main__":
    print("Explainability module loaded successfully.")
    print("Use generate_all_explanations() for complete XAI analysis.")
