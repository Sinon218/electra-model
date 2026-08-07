"""
TensorFlow Dataset utilities for sentiment classification.
Creates tf.data.Dataset pipelines from CSV files.
"""

import os
import sys

# Allow running this file directly (e.g., from VS Code)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.tokenizer import WordTokenizer
from src.config import MAX_SEQ_LEN, BATCH_SIZE


def create_dataset(
    csv_path: str,
    tokenizer: WordTokenizer,
    max_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from a sentiment CSV file.

    Args:
        csv_path:   Path to CSV with 'text' and 'label' columns.
        tokenizer:  A fitted WordTokenizer instance.
        max_len:    Maximum sequence length.
        batch_size: Batch size.
        shuffle:    Whether to shuffle the data.

    Returns:
        tf.data.Dataset yielding ((input_ids, attention_mask), labels) batches.
    """
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    texts = df["text"].tolist()
    labels = df["label"].astype(int).values

    # Encode all texts
    all_input_ids = []
    all_attention_masks = []
    for text in texts:
        ids, mask = tokenizer.encode(text, max_len=max_len)
        all_input_ids.append(ids)
        all_attention_masks.append(mask)

    input_ids = np.array(all_input_ids, dtype=np.int32)
    attention_masks = np.array(all_attention_masks, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    print(f"[Dataset] Loaded {len(labels)} samples from {csv_path}")

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        ((input_ids, attention_masks), labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels), seed=42)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def get_class_weights(csv_path: str) -> dict:
    """
    Compute class weights for imbalanced data.
    Returns a dict {class_index: weight} for use with model.fit().
    """
    df = pd.read_csv(csv_path)
    label_counts = df["label"].value_counts().sort_index()
    total = len(df)
    n_classes = len(label_counts)

    # Inverse frequency: weight = total / (num_classes * count_per_class)
    weights = {}
    for label, count in label_counts.items():
        weights[int(label)] = total / (n_classes * count)

    print(f"[Dataset] Class weights: {weights}")
    return weights
