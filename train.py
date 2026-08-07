"""
train.py - Main entry point for training the BiLSTM-Transformer model (TensorFlow).

Usage:
    python train.py
"""

import os
import sys

# Suppress TensorFlow C++ log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_CSV,
    TRAIN_RATIO,
    VAL_CSV,
    VAL_RATIO,
    TEST_CSV,
    ensure_dirs,
)
from src.dataset import create_dataset, get_class_weights
from src.evaluator import Evaluator
from src.model import build_bilstm_transformer
from src.tokenizer import WordTokenizer
from src.trainer import compile_model, train_model


def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)





def main():
    set_seed()
    ensure_dirs()

    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    device_info = f"GPU ({gpus[0].name})" if gpus else "CPU"

    print("=" * 60)
    print(" BiLSTM-Transformer Sentiment Classification (TensorFlow)")
    print(f" Device: {device_info}")
    print(f" TensorFlow version: {tf.__version__}")
    print(f" Split:  {TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
    print("=" * 60)

    # ---- 1. Load Training Data & Build Tokenizer ----
    print(f"\n[Step 1/6] Loading pre-split data from {DATA_DIR}...")
    train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["text", "label"])
    val_df = pd.read_csv(VAL_CSV).dropna(subset=["text", "label"])
    test_df = pd.read_csv(TEST_CSV).dropna(subset=["text", "label"])
    print(f"  [OK] Train: {len(train_df)} samples ({TRAIN_CSV})")
    print(f"  [OK] Val:   {len(val_df)} samples ({VAL_CSV})")
    print(f"  [OK] Test:  {len(test_df)} samples ({TEST_CSV})")

    print("\n[Step 2/6] Building tokenizer from training data...")
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(train_df["text"].tolist())
    tokenizer.save()

    # ---- 2. Create Datasets ----
    print("\n[Step 3/6] Creating tf.data datasets...")
    train_dataset = create_dataset(TRAIN_CSV, tokenizer, shuffle=True)
    val_dataset = create_dataset(VAL_CSV, tokenizer, shuffle=False)
    test_dataset = create_dataset(TEST_CSV, tokenizer, shuffle=False)

    # ---- 4. Build & Compile Model ----
    print("\n[Step 4/7] Building model...")
    model = build_bilstm_transformer(vocab_size=tokenizer.vocab_size)
    model = compile_model(model)
    model.summary()

    # ---- 5. Train ----
    print("\n[Step 5/7] Training...")
    class_weights = get_class_weights(TRAIN_CSV)
    history = train_model(model, train_dataset, val_dataset, class_weights=class_weights)

    # ---- 6. Evaluate on Test Set ----
    print("\n[Step 6/7] Evaluating on test set...")
    evaluator = Evaluator(model, test_dataset)
    metrics = evaluator.evaluate()

    # ---- 7. Plot training curves ----
    print("\n[Step 7/7] Plotting training curves...")
    Evaluator.plot_training_curves(history)

    print("\n" + "=" * 60)
    print(" All done! Check the following outputs:")
    print(f"   - Model checkpoint:      checkpoints/best_model.keras")
    print(f"   - Tokenizer vocab:       checkpoints/tokenizer_vocab.json")
    print(f"   - Classification report: results/classification_report.txt")
    print(f"   - Confusion matrix:      results/confusion_matrix.png")
    print(f"   - Training curves:       results/training_curves.png")
    print(f"   - Test metrics (JSON):   results/test_metrics.json")
    print(f"   - Training log (CSV):    results/training_log.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
