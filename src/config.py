"""
Configuration and hyperparameters for BiLSTM-Transformer model.
All settings are centralized here for easy tuning.
"""

import os


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Tokenizer
# ============================================================
MAX_VOCAB_SIZE = 20_000      # Top-k most frequent words
MAX_SEQ_LEN = 64             # Max sequence length (p95 word len = 30)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# ============================================================
# Model Architecture (Optimized for Fast & Clear Explanation)
# ============================================================
EMBED_DIM = 64               # Word embedding dimension
LSTM_HIDDEN = 128            # BiLSTM hidden size per direction (Output = 256)
LSTM_LAYERS = 1              # 1 BiLSTM Layer (Sequential Context)
LSTM_DROPOUT = 0.2           # Dropout rate

NUM_HEADS = 4                # 4 Attention Heads (256 / 4 = 64 per head)
FF_DIM = 256                 # Transformer feed-forward dimension
TRANS_LAYERS = 1             # 1 Transformer Encoder Block (Self-Attention)
TRANS_DROPOUT = 0.2          # Transformer dropout

CLASSIFIER_DROPOUT = 0.2     # Dropout before final FC layer
NUM_CLASSES = 3              # 0: Negative, 1: Neutral, 2: Positive

# ============================================================
# Training (Fast execution)
# ============================================================
BATCH_SIZE = 128             # Larger batch size for faster CPU throughput
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5               # 5 epochs completes in ~2 minutes
EARLY_STOP_PATIENCE = 2      # Stop early if no improvement
LR_PATIENCE = 1              # Reduce LR after 1 epoch without progress
LR_FACTOR = 0.5

RANDOM_SEED = 42

# ============================================================
# Label names (for reporting)
# ============================================================
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
