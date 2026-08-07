"""
Evaluation module for BiLSTM-Transformer model (TensorFlow/Keras).
Computes metrics, generates reports, and plots confusion matrix & training curves.
"""

import json
import os
import sys

# Allow running this file directly (e.g., from VS Code)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import LABEL_NAMES, RESULTS_DIR, ensure_dirs


class Evaluator:
    """
    Evaluates a trained Keras model on a test tf.data.Dataset.
    Produces:
    - Accuracy, Macro F1, Weighted F1
    - Per-class classification report
    - Confusion matrix heatmap
    - Training curves plot
    """

    def __init__(self, model: keras.Model, test_dataset: tf.data.Dataset):
        self.model = model
        self.test_dataset = test_dataset
        ensure_dirs()

    def predict(self) -> tuple[list[int], list[int]]:
        """
        Run inference on the test set.

        Returns:
            (all_preds, all_labels) as lists of ints
        """
        all_preds = []
        all_labels = []

        for (input_ids, attention_mask), labels in self.test_dataset:
            logits = self.model.predict_on_batch([input_ids, attention_mask])
            preds = np.argmax(logits, axis=1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

        return all_preds, all_labels

    def evaluate(self) -> dict:
        """
        Full evaluation: compute all metrics and save reports + plots.

        Returns:
            Dict with accuracy, macro_f1, weighted_f1.
        """
        print("\n" + "=" * 60)
        print(" Evaluating on Test Set")
        print("=" * 60)

        preds, labels = self.predict()

        # ---- Metrics ----
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")
        report = classification_report(
            labels, preds, target_names=LABEL_NAMES, digits=4
        )
        cm = confusion_matrix(labels, preds)

        print(f"\n  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"\n  Classification Report:\n{report}")

        # ---- Save text report ----
        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("BiLSTM-Transformer Sentiment Classification - Test Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy:    {acc:.4f}\n")
            f.write(f"Macro F1:    {macro_f1:.4f}\n")
            f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
        print(f"  [OK] Report saved to {report_path}")

        # ---- Save JSON metrics ----
        metrics = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix": cm.tolist(),
        }
        metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # ---- Plot confusion matrix ----
        self._plot_confusion_matrix(cm)

        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Save confusion matrix as a heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix - BiLSTM-Transformer", fontsize=14)
        plt.tight_layout()

        path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [OK] Confusion matrix saved to {path}")

    @staticmethod
    def plot_training_curves(history: dict):
        """
        Plot training & validation loss/accuracy curves.
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Loss ---
        axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
        axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Accuracy ---
        axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
        axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("BiLSTM-Transformer Training Curves", fontsize=14, y=1.02)
        plt.tight_layout()

        path = os.path.join(RESULTS_DIR, "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Training curves saved to {path}")
