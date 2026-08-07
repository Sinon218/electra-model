# BiLSTM-Transformer Sentiment Classification (3-Class)

Mô hình học máy kết hợp **BiLSTM (Bidirectional LSTM)** và **Transformer Encoder** phục vụ phân loại cảm xúc (Sentiment Analysis) trên dữ liệu văn bản mạng xã hội (Tweets) cho 3 nhãn:
- **`0`**: Negative (Âm tính)
- **`1`**: Neutral (Trung tính)
- **`2`**: Positive (Dương tính)

---

## 1. Sơ Đồ Kiến Trúc Mô Hình (Architecture Diagram)

```text
[Input Text (Token IDs)]
          ↓
[Embedding Layer (64-dim)]
          ↓
[BiLSTM Layer (Hidden: 128x2 = 256-dim)]   ---> Bắt ngữ cảnh chuỗi tuần tự (Sequential Context)
          ↓
[Positional Encoding (Sinusoidal)]         ---> Bổ sung thông tin vị trí từ
          ↓
[Transformer Encoder (4 Attention Heads)]  ---> Bắt mối quan hệ toàn cục (Self-Attention)
          ↓
[Masked Global Average Pooling]            ---> Gom vector đặc trưng của câu
          ↓
[Dense Layers (256 → 3) + Softmax]         ---> Phân loại 3 lớp (Neg / Neu / Pos)
```

### Tại sao lại chọn BiLSTM + Transformer Encoder?
- **BiLSTM**: Giúp mô hình hiểu được ngữ cảnh hai chiều (trái ⇄ phải) của các từ phụ cận trong câu ngắn/vừa.
- **Transformer Encoder**: Dùng cơ chế **Multi-Head Self-Attention** để tập trung vào các từ mang sắc thái cảm xúc quan trọng (như *love, terrible, great, hate, emoji...*) bất kể vị trí của chúng trong câu.
- **Sự kết hợp**: Trích xuất tối đa cả đặc trưng thứ tự câu (từ BiLSTM) và đặc trưng mối quan hệ xa (từ Transformer).

---

## 2. Dữ Liệu Huấn Luyện (Dataset)

- **Nguồn dữ liệu**: TweetEval Sentiment benchmark dataset
- **Phân chia dữ liệu chuẩn (Splits)**:
  - **Train set (65%)**: 38,914 samples
  - **Validation set (15%)**: 8,980 samples
  - **Test set (20%)**: 11,975 samples

---

## 3. Kết Quả Đánh Giá Trên Tập Test (Test Results)

| Chỉ số (Metric) | Điểm số (Score) |
|:---|:---:|
| **Accuracy (Độ chính xác)** | **64.06%** |
| **Weighted F1-Score** | **63.81%** |
| **Macro F1-Score** | **62.67%** |

### Classification Report Chi Tiết:

```text
              precision    recall  f1-score   support

    Negative     0.5409    0.6454    0.5885      2276
     Neutral     0.6135    0.7505    0.6751      5492
    Positive     0.8189    0.4944    0.6166      4207

    accuracy                         0.6406     11975
   macro avg     0.6578    0.6301    0.6267     11975
weighted avg     0.6718    0.6406    0.6381     11975
```

---

## 4. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
transformer-main/
├── data/
│   └── splits/                 # Dữ liệu train/val/test đã phân chia (65/15/20)
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── src/
│   ├── config.py               # Trung tâm cấu hình tham số (Hyperparameters)
│   ├── tokenizer.py            # Tokenizer cấp từ (Word-level Tokenizer)
│   ├── dataset.py              # Xử lý tf.data.Dataset pipeline
│   ├── model.py                # Kiến trúc BiLSTM-Transformer
│   ├── trainer.py              # Luồng huấn luyện (model.fit + Callbacks)
│   └── evaluator.py            # Đánh giá tập test & vẽ đồ thị
├── results/                    # Kết quả đánh giá & biểu đồ
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_curves.png
├── train.py                    # Script chính: Train -> Evaluate -> Export
├── evaluate.py                 # Script kiểm thử độc lập
├── requirements.txt            # Thư viện phụ thuộc
└── README.md                   # Báo cáo mô hình
```

---

## 5. Hướng Dẫn Chạy (How to Run)

### Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### Huấn luyện mô hình & xuất báo cáo:
```bash
python train.py
```

### Đánh giá mô hình đã lưu trên tập test:
```bash
python evaluate.py
```
