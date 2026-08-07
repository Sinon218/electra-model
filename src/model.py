"""
BiLSTM-Transformer hybrid model for sentiment classification using TensorFlow/Keras 3.

Architecture:
    Input IDs -> Embedding -> BiLSTM -> Transformer Encoder -> Pooling -> FC -> 3 classes
"""

import os
import sys

# Allow running this file directly (e.g., from VS Code)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow C++ log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops

from src.config import (
    EMBED_DIM,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    NUM_HEADS,
    FF_DIM,
    TRANS_LAYERS,
    TRANS_DROPOUT,
    CLASSIFIER_DROPOUT,
    NUM_CLASSES,
    MAX_SEQ_LEN,
)


class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer Encoder block with Multi-Head Attention + FFN.
    Uses pre-norm (LayerNormalization before attention/FFN) for stable training.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, attention_mask=None, training=False):
        # Pre-norm + Multi-Head Self-Attention
        x_norm = self.layernorm1(inputs)

        if attention_mask is not None:
            # Expand mask: [batch, seq_len] -> [batch, 1, seq_len]
            keras_mask = ops.expand_dims(attention_mask, axis=1)
            keras_mask = ops.cast(keras_mask, "bool")
        else:
            keras_mask = None

        attn_output = self.att(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attention_mask=keras_mask,
            training=training,
        )
        attn_output = self.dropout1(attn_output, training=training)
        x = inputs + attn_output

        # Pre-norm + Feed-Forward Network
        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output

        return x


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding."""

    def __init__(self, max_len=MAX_SEQ_LEN, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        # Compute positional encoding matrix
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)  # [1, max_len, d_model]
        super().build(input_shape)

    def call(self, x):
        seq_len = ops.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


def build_bilstm_transformer(vocab_size: int) -> keras.Model:
    """
    Build a BiLSTM-Transformer hybrid model for text classification.

    Flow:
        1. Embedding layer: token indices -> dense vectors
        2. BiLSTM: captures local sequential patterns (forward + backward)
        3. Positional Encoding: injects position info for Transformer
        4. Transformer Encoder: captures global long-range dependencies
        5. Masked Global Average Pooling
        6. Classification Head: FC layers -> class logits

    Args:
        vocab_size: Size of the vocabulary.

    Returns:
        Compiled Keras Model.
    """
    # ---- Inputs ----
    input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype="int32", name="input_ids")
    attention_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype="int32", name="attention_mask")

    # ---- 1. Embedding ----
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        mask_zero=False,
        name="embedding",
    )(input_ids)

    # ---- 2. BiLSTM ----
    for i in range(LSTM_LAYERS):
        x = layers.Bidirectional(
            layers.LSTM(
                LSTM_HIDDEN,
                return_sequences=True,
                dropout=LSTM_DROPOUT if i < LSTM_LAYERS - 1 else 0.0,
                recurrent_dropout=0.0,
            ),
            name=f"bilstm_{i+1}",
        )(x)
    # BiLSTM output: [batch, seq_len, 2 * LSTM_HIDDEN]

    # ---- 3. Positional Encoding ----
    x = PositionalEncoding(max_len=MAX_SEQ_LEN, name="pos_encoding")(x)
    x = layers.Dropout(TRANS_DROPOUT, name="pos_dropout")(x)

    # ---- 4. Transformer Encoder ----
    mask_float = ops.cast(attention_mask, "float32")
    for i in range(TRANS_LAYERS):
        x = TransformerEncoderBlock(
            d_model=2 * LSTM_HIDDEN,  # BiLSTM output dim
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout_rate=TRANS_DROPOUT,
            name=f"transformer_block_{i+1}",
        )(x, attention_mask=mask_float)

    # ---- 5. Masked Global Average Pooling ----
    # Expand mask: [batch, seq_len] -> [batch, seq_len, 1]
    mask_expanded = ops.expand_dims(mask_float, axis=-1)
    # Masked sum / count
    x = ops.sum(x * mask_expanded, axis=1) / (ops.sum(mask_expanded, axis=1) + 1e-9)
    # Result: [batch, 2 * LSTM_HIDDEN]

    # ---- 6. Classification Head ----
    x = layers.LayerNormalization(name="final_norm")(x)
    x = layers.Dropout(CLASSIFIER_DROPOUT, name="classifier_dropout_1")(x)
    x = layers.Dense(256, activation="gelu", name="fc_hidden")(x)
    x = layers.Dropout(CLASSIFIER_DROPOUT, name="classifier_dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=outputs, name="BiLSTM_Transformer")
    return model
