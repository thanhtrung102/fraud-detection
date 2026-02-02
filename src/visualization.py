"""
Model Visualization Module
==========================

Comprehensive visualization suite for fraud detection models.
"""

import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class ModelVisualizer:
    """Generate comprehensive visualizations for fraud detection models."""

    def __init__(self, figsize: tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure DPI for saving
        """
        self.figsize = figsize
        self.dpi = dpi

    def create_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {save_path}")

        return fig

    def create_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create ROC curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name for title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.2, color="darkorange")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve - {model_name}", fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved ROC curve to {save_path}")

        return fig

    def create_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        threshold: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create precision-recall curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name for title
            threshold: Optional threshold to mark
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, color="green", lw=2, label="Precision-Recall curve")
        ax.fill_between(recall, precision, alpha=0.2, color="green")

        if threshold is not None:
            # Find precision and recall at threshold
            idx = np.argmin(np.abs(thresholds - threshold))
            ax.scatter(
                recall[idx],
                precision[idx],
                color="red",
                s=100,
                zorder=5,
                label=f"Threshold = {threshold:.2f}",
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved PR curve to {save_path}")

        return fig

    def create_metrics_comparison(
        self,
        metrics_dict: dict[str, dict[str, float]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create 4-panel metrics comparison chart.

        Args:
            metrics_dict: Dictionary of {model_name: {metric: value}}
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics_to_plot = [
            ("accuracy", "Accuracy", False),
            ("auc_roc", "AUC-ROC", False),
            ("precision", "Precision", False),
            ("recall", "Recall", False),
        ]

        models = list(metrics_dict.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        for idx, (metric, title, lower_better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [metrics_dict[model].get(metric, 0) for model in models]

            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor="black")

            # Highlight best
            if values:
                best_idx = values.index(min(values) if lower_better else max(values))
                bars[best_idx].set_edgecolor("green")
                bars[best_idx].set_linewidth(3)

            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved metrics comparison to {save_path}")

        return fig

    def create_roc_comparison(
        self,
        results_dict: dict[str, dict[str, Any]],
        y_true: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create ROC curves for multiple models on single plot.

        Args:
            results_dict: Dictionary of {model_name: {'y_proba': array}}
            y_true: True labels
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))

        for idx, (model_name, results) in enumerate(results_dict.items()):
            y_proba = results.get("y_proba")
            if y_proba is None:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved ROC comparison to {save_path}")

        return fig

    def create_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create feature importance bar chart.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            model_name: Name for title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Sort and get top N
        df = importance_df.sort_values("importance", ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        bars = ax.barh(df["feature"], df["importance"], color=colors, edgecolor="black", alpha=0.8)

        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importance - {model_name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved feature importance to {save_path}")

        return fig

    def create_probability_distribution(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create probability distribution for fraud vs legitimate.

        Args:
            y_proba: Predicted probabilities
            y_true: True labels
            threshold: Classification threshold
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separate by class
        legitimate_proba = y_proba[y_true == 0]
        fraud_proba = y_proba[y_true == 1]

        # Create histograms
        bins = np.linspace(0, 1, 50)
        ax.hist(
            legitimate_proba,
            bins=bins,
            alpha=0.6,
            label=f"Legitimate (n={len(legitimate_proba)})",
            color="green",
            density=True,
        )
        ax.hist(
            fraud_proba,
            bins=bins,
            alpha=0.6,
            label=f"Fraud (n={len(fraud_proba)})",
            color="red",
            density=True,
        )

        # Add threshold line
        ax.axvline(x=threshold, color="black", linestyle="--", lw=2, label=f"Threshold = {threshold}")

        ax.set_xlabel("Fraud Probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Probability Distribution by Class", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved probability distribution to {save_path}")

        return fig

    def create_comprehensive_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        metrics: dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None,
        threshold: float = 0.5,
        model_name: str = "Stacking Ensemble",
        save_dir: str = "results/visualizations",
    ) -> dict[str, str]:
        """
        Generate all visualizations and save to directory.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            metrics: Model metrics
            feature_importance: Optional feature importance DataFrame
            threshold: Classification threshold
            model_name: Model name
            save_dir: Directory to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        saved_files = {}

        # Confusion matrix
        path = os.path.join(save_dir, "confusion_matrix.png")
        self.create_confusion_matrix(y_true, y_pred, model_name, path)
        saved_files["confusion_matrix"] = path

        # ROC curve
        path = os.path.join(save_dir, "roc_curve.png")
        self.create_roc_curve(y_true, y_proba, model_name, path)
        saved_files["roc_curve"] = path

        # Precision-Recall curve
        path = os.path.join(save_dir, "precision_recall_curve.png")
        self.create_precision_recall_curve(y_true, y_proba, model_name, threshold, path)
        saved_files["precision_recall_curve"] = path

        # Probability distribution
        path = os.path.join(save_dir, "probability_distribution.png")
        self.create_probability_distribution(y_proba, y_true, threshold, path)
        saved_files["probability_distribution"] = path

        # Feature importance (if available)
        if feature_importance is not None:
            path = os.path.join(save_dir, "feature_importance.png")
            self.create_feature_importance(feature_importance, 20, model_name, path)
            saved_files["feature_importance"] = path

        # Generate HTML report
        html_path = os.path.join(save_dir, "model_report.html")
        self._generate_html_report(saved_files, metrics, model_name, html_path)
        saved_files["html_report"] = html_path

        logger.info(f"Generated {len(saved_files)} visualizations in {save_dir}")
        return saved_files

    def _generate_html_report(
        self,
        image_paths: dict[str, str],
        metrics: dict[str, float],
        model_name: str,
        output_path: str,
    ) -> None:
        """Generate HTML report with embedded images."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection Model Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .timestamp {{ color: #888; font-size: 14px; text-align: center; }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
    </style>
</head>
<body>
    <h1>Fraud Detection Model Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    <p class="timestamp">Model: {model_name}</p>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            {metrics_html}
        </div>
    </div>

    <div class="section">
        <h2>Visualizations</h2>
        <div class="image-grid">
            {images_html}
        </div>
    </div>
</body>
</html>
"""

        # Generate metrics HTML
        metrics_html = ""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                </div>
                """

        # Generate images HTML with base64 encoding
        images_html = ""
        for name, path in image_paths.items():
            if path.endswith(".png") and os.path.exists(path):
                with open(path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                images_html += f"""
                <div>
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{name}">
                </div>
                """

        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name=model_name,
            metrics_html=metrics_html,
            images_html=images_html,
        )

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {output_path}")


def generate_training_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics: dict[str, float],
    feature_importance: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
    output_dir: str = "results",
) -> dict[str, str]:
    """
    Convenience function to generate all training visualizations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        metrics: Model metrics
        feature_importance: Optional feature importance
        threshold: Classification threshold
        output_dir: Output directory

    Returns:
        Dictionary of saved file paths
    """
    visualizer = ModelVisualizer()
    return visualizer.create_comprehensive_report(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        feature_importance=feature_importance,
        threshold=threshold,
        save_dir=output_dir,
    )
