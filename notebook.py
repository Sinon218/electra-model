import os, re, random, gc, math, json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, ops
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "splits")
MAX_VOCAB = 20_000
MAX_SEQLEN = 64
NUM_CLASSES = 3
BATCH = 128
EMBED_DIM = 64
LSTM_HIDDEN = 128
NUM_HEADS = 4
FF_DIM = 256
D_MODEL = 2 * LSTM_HIDDEN

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def encode(texts, vocab):
    ids = np.zeros((len(texts), MAX_SEQLEN), dtype=np.int32)
    mask = np.zeros((len(texts), MAX_SEQLEN), dtype=np.int32)
    for i, t in enumerate(texts):
        toks = re.findall(r"\w+|[^\w\s]", clean(str(t)), re.UNICODE)[:MAX_SEQLEN]
        for j, w in enumerate(toks):
            ids[i, j] = vocab.get(w, 1)
        mask[i, :len(toks)] = 1
    return ids, mask

def augment(text, p=0.15):
    words = text.split()
    if len(words) < 3:
        return text
    words = [w for w in words if random.random() > p]
    if not words:
        words = text.split()[:2]
    for i in range(len(words) - 1):
        if random.random() < p:
            words[i], words[i+1] = words[i+1], words[i]
    return ' '.join(words)

vocab_path = os.path.join(os.path.dirname(DATA), '..', 'checkpoints', 'tokenizer_vocab.json')
with open(vocab_path, 'rb') as f:
    vocab = json.loads(f.read().decode('utf-8'))['word2idx']
V = len(vocab)
print(f"Vocab: {V} tokens")

AUG = 3
tdf = pd.read_csv(os.path.join(DATA, 'train.csv')).dropna(subset=['text', 'label'])
texts, labels = [], []
for _, row in tdf.iterrows():
    t, l = str(row['text']), int(row['label'])
    texts.append(t); labels.append(l)
    for _ in range(AUG - 1):
        texts.append(augment(t)); labels.append(l)
print(f"Train: {len(labels)} samples ({AUG}x augmented)")
Xtr, Mtr = encode(texts, vocab)
Ytr = np.array(labels, np.int32)
del tdf, texts, labels; gc.collect()

def load_split(p):
    df = pd.read_csv(p).dropna(subset=['text', 'label'])
    ids, m = encode(df['text'], vocab)
    return ids, m, df['label'].astype(int).values

Xv, Mv, Yv = load_split(os.path.join(DATA, 'validation.csv'))
Xte, Mte, Yte = load_split(os.path.join(DATA, 'test.csv'))
print(f"Val:{len(Yv)} Test:{len(Yte)}")

train_ds = tf.data.Dataset.from_tensor_slices(((Xtr, Mtr), Ytr)).shuffle(len(Ytr), seed=SEED).batch(BATCH).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(((Xv, Mv), Yv)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices(((Xte, Mte), Yte)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
cw = {i: len(Ytr)/(NUM_CLASSES*c) for i, c in enumerate(np.bincount(Ytr))}

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.d1 = layers.Dropout(dropout_rate)
        self.d2 = layers.Dropout(dropout_rate)

    def call(self, inputs, attention_mask=None, training=False):
        xn = self.ln1(inputs)
        if attention_mask is not None:
            km = ops.expand_dims(attention_mask, axis=1)
            km = ops.cast(km, "bool")
        else:
            km = None
        a = self.att(query=xn, key=xn, value=xn, attention_mask=km, training=training)
        x = inputs + self.d1(a, training=training)
        xn = self.ln2(x)
        return x + self.d2(self.ffn(xn), training=training)

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=MAX_SEQLEN, **kw):
        super().__init__(**kw)
        self.max_len = max_len

    def build(self, input_shape):
        d = input_shape[-1]
        pos = np.arange(self.max_len)[:, None]
        dim = np.arange(d)[None, :]
        angles = pos / np.power(10000, (2 * (dim // 2)) / d)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[np.newaxis], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        return x + self.pe[:, :ops.shape(x)[1], :]

def build_model():
    inp_id = layers.Input((MAX_SEQLEN,), dtype="int32", name="input_ids")
    inp_mask = layers.Input((MAX_SEQLEN,), dtype="int32", name="attention_mask")

    x = layers.Embedding(V, EMBED_DIM)(inp_id)
    x = layers.Bidirectional(layers.LSTM(LSTM_HIDDEN, return_sequences=True))(x)
    x = PositionalEncoding()(x)
    x = TransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout_rate=0.2)(x)

    mask_expanded = ops.expand_dims(ops.cast(inp_mask, "float32"), axis=-1)
    x = x * mask_expanded
    seq_len = ops.cast(ops.sum(mask_expanded, axis=1), "float32")
    x = ops.sum(x, axis=1) / ops.maximum(seq_len, 1.0)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model([inp_id, inp_mask], out)

model = build_model()
model.summary()

EPOCHS = 30
INIT_LR = 3e-4

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INIT_LR, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping('val_accuracy', mode='max', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=2),
    ],
    class_weight=cw, verbose=1
)

yp = np.argmax(model.predict(test_ds, verbose=0), 1)
acc = accuracy_score(Yte, yp)
print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(Yte, yp, target_names=['Neg','Neu','Pos'], digits=4))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(history.history['loss'], label='Train'); axes[0].plot(history.history['val_loss'], label='Val'); axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train'); axes[1].plot(history.history['val_accuracy'], label='Val'); axes[1].set_title('Accuracy'); axes[1].legend()
sns.heatmap(confusion_matrix(Yte, yp), annot=True, fmt='d', cmap='Blues', xticklabels=['N','Ne','P'], yticklabels=['N','Ne','P'], ax=axes[2])
axes[2].set_title('CM'); plt.tight_layout(); plt.show()
