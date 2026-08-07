"""
Word-level tokenizer built from training data.
Supports building vocabulary, encoding/decoding, and persistence.
"""

import json
import os
import re
import sys
from collections import Counter

# Allow running this file directly (e.g., from VS Code)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MAX_SEQ_LEN,
    MAX_VOCAB_SIZE,
    PAD_IDX,
    PAD_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
    CHECKPOINT_DIR,
)


class WordTokenizer:
    """
    Simple word-level tokenizer.
    - Builds vocab from training texts (top MAX_VOCAB_SIZE words).
    - Encodes text -> list of token indices.
    - Pads/truncates to MAX_SEQ_LEN.
    - Can save/load vocab for inference reuse.
    """

    def __init__(self):
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_freq = Counter()
        self.vocab_size = 2  # PAD + UNK

    # ----------------------------------------------------------
    # Text preprocessing
    # ----------------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Light cleaning: lowercase, remove URLs, @mentions, extra whitespace."""
        text = text.lower()
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # Remove @user mentions
        text = re.sub(r"@\w+", "", text)
        # Keep emojis, hashtags (#word -> word), punctuation
        text = re.sub(r"#(\w+)", r"\1", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into word tokens."""
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    # ----------------------------------------------------------
    # Build vocabulary
    # ----------------------------------------------------------
    def build_vocab(self, texts: list[str]):
        """
        Build vocabulary from a list of training texts.
        Keeps the top MAX_VOCAB_SIZE most frequent words.
        """
        self.word_freq = Counter()
        for text in texts:
            cleaned = self._clean_text(str(text))
            tokens = self._tokenize(cleaned)
            self.word_freq.update(tokens)

        # Take top-k most common words
        most_common = self.word_freq.most_common(MAX_VOCAB_SIZE - 2)  # -2 for PAD, UNK

        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

        for idx, (word, _freq) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"[Tokenizer] Vocabulary built: {self.vocab_size} tokens "
              f"(from {len(self.word_freq)} unique words)")

    # ----------------------------------------------------------
    # Encode / Decode
    # ----------------------------------------------------------
    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> tuple[list[int], list[int]]:
        """
        Encode a single text string into token indices + attention mask.

        Returns:
            input_ids:      list[int] of length max_len (padded/truncated)
            attention_mask:  list[int] of length max_len (1 = real token, 0 = padding)
        """
        cleaned = self._clean_text(str(text))
        tokens = self._tokenize(cleaned)

        # Convert to indices
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]

        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len]

        # Build attention mask (1 for real tokens, 0 for padding)
        attn_mask = [1] * len(ids) + [0] * (max_len - len(ids))

        # Pad
        ids = ids + [PAD_IDX] * (max_len - len(ids))

        return ids, attn_mask

    def decode(self, ids: list[int]) -> str:
        """Decode token indices back to text (for debugging)."""
        words = [self.idx2word.get(i, UNK_TOKEN) for i in ids if i != PAD_IDX]
        return " ".join(words)

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------
    def save(self, path: str | None = None):
        """Save vocabulary to JSON file."""
        if path is None:
            path = os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "word2idx": self.word2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Vocabulary saved to {path}")

    def load(self, path: str | None = None):
        """Load vocabulary from JSON file."""
        if path is None:
            path = os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.vocab_size = data["vocab_size"]
        print(f"[Tokenizer] Vocabulary loaded: {self.vocab_size} tokens from {path}")
