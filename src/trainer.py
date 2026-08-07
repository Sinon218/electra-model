"""
Training module for BiLSTM-Transformer model using TensorFlow/Keras.
Includes callbacks for early stopping, LR scheduling, and checkpointing.
"""

import json
import os
import sys

# Allow running this file directly (e.g., from VS Code)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    LR_FACTOR,
    LR_PATIENCE,
    NUM_EPOCHS,
    RESULTS_DIR,
    ensure_dirs,
)


def compile_model(model: keras.Model) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    """
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks() -> list:
    """
    Create training callbacks:
    - ModelCheckpoint: save best model by val_accuracy
    - EarlyStopping: stop if no improvement for EARLY_STOP_PATIENCE epochs
    - ReduceLROnPlateau: reduce LR if val_accuracy plateaus
    - CSVLogger: log metrics to CSV file
    """
    ensure_dirs()

    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        # Log training history to CSV
        keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, "training_log.csv"),
            separator=",",
            append=False,
        ),
    ]

    return callbacks


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    class_weights: dict | None = None,
) -> dict:
    """
    Train the model.

    Args:
        model:         Compiled Keras model.
        train_dataset: Training tf.data.Dataset.
        val_dataset:   Validation tf.data.Dataset.
        class_weights: Optional dict {class_index: weight}.

    Returns:
        Training history dict.
    """
    ensure_dirs()

    print(f"\n{'=' * 60}")
    print(f" Training BiLSTM-Transformer (TensorFlow)")
    print(f" Parameters: {model.count_params():,}")
    print(f" Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f" Early stop patience: {EARLY_STOP_PATIENCE}")
    print(f"{'=' * 60}\n")

    callbacks = get_callbacks()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save training history as JSON
    history_dict = {
        "train_loss": [float(v) for v in history.history["loss"]],
        "train_acc": [float(v) for v in history.history["accuracy"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "val_acc": [float(v) for v in history.history["val_accuracy"]],
        "lr": [float(v) for v in history.history.get("learning_rate", history.history.get("lr", [LEARNING_RATE]))],
    }

    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\n[OK] Training history saved to {history_path}")

    best_val_acc = max(history_dict["val_acc"])
    print(f"\n{'=' * 60}")
    print(f" Training complete!")
    print(f" Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'=' * 60}\n")

    return history_dict
