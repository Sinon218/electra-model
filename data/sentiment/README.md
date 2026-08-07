# 3-Class Social Media Sentiment Dataset

## Overview

This directory contains the standardized, preprocessed shared dataset for **3-Class Social Media Sentiment Classification**.

## Task

> **Sentiment Classification on Social Media Data (Tweets)**

## Classes

Strict label mapping:

| Label Code | Class Name | Description |
| :---: | :--- | :--- |
| `0` | **Negative** | Expresses negative sentiment or emotion |
| `1` | **Neutral** | Factual, objective, or neutral sentiment |
| `2` | **Positive** | Expresses positive sentiment or emotion |

## Files

```text
data/sentiment/
├── README.md                 # Dataset documentation and instructions
├── dataset_statistics.json   # Machine-readable dataset metadata and counts
├── label_distribution.csv    # Class counts and percentages
├── train.csv                 # Training split (~80%, 55,892 samples)
├── validation.csv            # Validation split (~10%, 6,986 samples)
└── test.csv                  # Test split (~10%, 6,987 samples)
```

Each split CSV file contains the required schema:

```text
text,label
"I absolutely love this!",2
"This is terrible.",0
"It is okay.",1
```

## Dataset Size

- **Total Cleaned Samples**: **69,865**
- **Train Samples**: **55,892** (80.00%)
- **Validation Samples**: **6,986** (10.00%)
- **Test Samples**: **6,987** (10.00%)

### Class Distribution

| Label | Name | Count | Percentage |
| :---: | :--- | :---: | :---: |
| `0` | Negative | 21,370 | 30.59% |
| `1` | Neutral | 27,461 | 39.31% |
| `2` | Positive | 21,034 | 30.11% |

## Source

The dataset is synthesized from:
1. **TweetEval Sentiment**: Full benchmark dataset (Train + Validation + Test).
2. **Twitter_Data.csv**: 10,000 Negative samples added to balance the negative class ratio.

## Preprocessing Pipeline

The dataset was processed through a non-destructive pipeline:
```text
Data validation
    ↓
Text cleaning (preserved emojis, emotion punctuation, hashtags, casing)
    ↓
Duplicate removal
    ↓
Label validation (strictly {0, 1, 2})
    ↓
Stratified splitting (80% Train / 10% Val / 10% Test with RANDOM_SEED = 42)
    ↓
Leakage checking (0 overlap verified)
```

> **Note**: Text tokenization and sequence truncation were **NOT** performed in this preprocessing phase and must be handled in the downstream Tokenizer task.

## How to Use

1. Clone or pull the repository.
2. Navigate to the project root directory.
3. Validate dataset integrity:
   ```bash
   python scripts/validate_dataset.py
   ```
4. Load dataset splits in Python for Transformer tokenization and model training:
   - **Train set**: `data/sentiment/train.csv`
   - **Validation set**: `data/sentiment/validation.csv`
   - **Test set**: `data/sentiment/test.csv`
