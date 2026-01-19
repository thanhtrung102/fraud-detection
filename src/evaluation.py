"""
Evaluation Module
=================

Metrics computation, visualization, and result reporting.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray = None) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }

    return metrics


def print_results(metrics: dict, title: str = "Classification Results"):
    """Print formatted results."""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print("-" * 60)
    print("  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TN: {cm['true_negatives']:>8}  |  FP: {cm['false_positives']:>8}")
    print(f"    FN: {cm['false_negatives']:>8}  |  TP: {cm['true_positives']:>8}")
    print("=" * 60)


def compare_with_paper(metrics: dict) -> dict:
    """
    Compare achieved metrics with paper's reported results.

    Args:
        metrics: Computed metrics

    Returns:
        Comparison dictionary
    """
    paper_metrics = {
        'accuracy': 0.99,
        'precision': 0.99,
        'recall': 0.99,
        'f1_score': 0.99,
        'auc_roc': 0.99
    }

    comparison = {}
    for key in paper_metrics:
        if key in metrics:
            diff = metrics[key] - paper_metrics[key]
            comparison[key] = {
                'achieved': metrics[key],
                'paper': paper_metrics[key],
                'difference': diff,
                'within_tolerance': abs(diff) <= 0.02  # 2% tolerance
            }

    return comparison


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  save_path: str = None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                save_path: str = None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def save_results(metrics: dict, comparison: dict, output_dir: str = 'results'):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'metrics': metrics,
        'comparison_with_paper': comparison
    }

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results = convert(results)

    output_path = output_dir / 'metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def generate_all_plots(y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: np.ndarray, output_dir: str = 'results'):
    """Generate and save all evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred,
                         str(output_dir / 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_proba,
                  str(output_dir / 'roc_curve.png'))
    plot_precision_recall_curve(y_true, y_proba,
                               str(output_dir / 'precision_recall_curve.png'))

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_proba = np.random.random(1000)
    y_pred = (y_proba > 0.5).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    print_results(metrics)

    comparison = compare_with_paper(metrics)
    print("\nComparison with paper:")
    for key, val in comparison.items():
        status = "✓" if val['within_tolerance'] else "✗"
        print(f"  {key}: {val['achieved']:.4f} vs {val['paper']:.4f} [{status}]")
