"""
evaluate.py - Evaluate a trained BiLSTM-Transformer model on the test set (TensorFlow).

Usage:
    python evaluate.py
    python evaluate.py --checkpoint path/to/model.keras
"""

import os
import sys

# Suppress TensorFlow C++ log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    TEST_CSV,
    ensure_dirs,
)
from src.dataset import create_dataset
from src.evaluator import Evaluator
from src.tokenizer import WordTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM-Transformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        help="Path to model checkpoint (.keras file)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.json"),
        help="Path to tokenizer vocabulary (.json file)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=TEST_CSV,
        help="Path to test CSV file",
    )
    args = parser.parse_args()

    ensure_dirs()

    print("=" * 60)
    print(" BiLSTM-Transformer - Test Evaluation (TensorFlow)")
    print(f" TensorFlow version: {tf.__version__}")
    print("=" * 60)

    # ---- 1. Load tokenizer ----
    print("\n[Step 1/3] Loading tokenizer...")
    tokenizer = WordTokenizer()
    tokenizer.load(args.vocab)

    # ---- 2. Load model ----
    print("\n[Step 2/3] Loading model...")
    model = keras.models.load_model(args.checkpoint)
    print(f"  Loaded from: {args.checkpoint}")
    print(f"  Parameters: {model.count_params():,}")

    # ---- 3. Evaluate ----
    print("\n[Step 3/3] Running evaluation...")
    test_dataset = create_dataset(args.test_csv, tokenizer, shuffle=False)

    evaluator = Evaluator(model, test_dataset)
    metrics = evaluator.evaluate()

    print("\n" + "=" * 60)
    print(" Evaluation complete!")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"   Weighted F1: {metrics['weighted_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
