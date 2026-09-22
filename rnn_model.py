# ============================================================
# FINANCIAL NEWS SENTIMENT ANALYSIS
# SIMPLE RNN MODEL
# ============================================================

import os
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SimpleRNN,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_FILE = "sent_train.csv"
VALID_FILE = "sent_valid.xlsx"

MAX_WORDS = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 128

EPOCHS = 10
BATCH_SIZE = 32


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("FINANCIAL NEWS SENTIMENT ANALYSIS - SIMPLE RNN")
print("=" * 70)

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_FILE)

print("Loading validation data...")

valid_df = pd.read_excel(VALID_FILE)

print("\nTraining shape:", train_df.shape)
print("Validation shape:", valid_df.shape)

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(valid_df.columns.tolist())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values in training data:")
print(train_df.isnull().sum())

print("\nMissing values in validation data:")
print(valid_df.isnull().sum())


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

train_df = train_df.dropna(
    subset=["text", "label"]
).copy()

valid_df = valid_df.dropna(
    subset=["text", "label"]
).copy()


# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean financial news text while preserving
    important financial terms and ticker symbols.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Keep letters, numbers, $, %, and #
    text = re.sub(
        r"[^a-zA-Z0-9$%#\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning text...")

train_df["clean_text"] = train_df["text"].apply(
    clean_text
)

valid_df["clean_text"] = valid_df["text"].apply(
    clean_text
)


# ============================================================
# 6. BALANCE TRAINING DATA
# ============================================================

print("\nOriginal training distribution:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

# Find the size of the largest class
max_class_size = (
    train_df["label"]
    .value_counts()
    .max()
)

balanced_parts = []

# Balance each class
for label in sorted(
    train_df["label"].unique()
):

    class_data = train_df[
        train_df["label"] == label
    ]

    # Oversample minority classes
    class_data = class_data.sample(
        n=max_class_size,
        replace=True,
        random_state=42
    )

    balanced_parts.append(
        class_data
    )


# Combine all classes
balanced_train_df = pd.concat(
    balanced_parts,
    ignore_index=True
)


# Shuffle balanced dataset
balanced_train_df = (
    balanced_train_df
    .sample(
        frac=1,
        random_state=42
    )
    .reset_index(drop=True)
)


print("\nBalanced training distribution:")

print(
    balanced_train_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. PREPARE TRAINING AND VALIDATION DATA
# ============================================================

X_train_text = (
    balanced_train_df["clean_text"]
    .values
)

y_train = (
    balanced_train_df["label"]
    .astype(int)
    .values
)

# Validation data is NOT balanced
X_valid_text = (
    valid_df["clean_text"]
    .values
)

y_valid = (
    valid_df["label"]
    .astype(int)
    .values
)


# ============================================================
# 8. CHECK LABEL DISTRIBUTION
# ============================================================

print("\nTraining label distribution:")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nValidation label distribution:")

print(
    pd.Series(y_valid)
    .value_counts()
    .sort_index()
)

print("\nLabel mapping:")

print("0 = Bearish")
print("1 = Bullish")
print("2 = Neutral")


# ============================================================
# 9. TOKENIZATION
# ============================================================

print("\nCreating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)


# IMPORTANT:
# Fit tokenizer ONLY on training data
# to avoid data leakage.

tokenizer.fit_on_texts(
    X_train_text
)


# ============================================================
# 10. CONVERT TEXT TO SEQUENCES
# ============================================================

X_train_sequences = (
    tokenizer.texts_to_sequences(
        X_train_text
    )
)

X_valid_sequences = (
    tokenizer.texts_to_sequences(
        X_valid_text
    )
)


# ============================================================
# 11. PADDING
# ============================================================

X_train = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_valid = pad_sequences(
    X_valid_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


print("\nTraining tensor shape:")

print(X_train.shape)

print("\nValidation tensor shape:")

print(X_valid.shape)


# ============================================================
# 12. VOCABULARY SIZE
# ============================================================

vocab_size = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

print("\nVocabulary size:")

print(vocab_size)


# ============================================================
# 13. BUILD SIMPLE RNN MODEL
# ============================================================

model = Sequential([

    Input(
        shape=(MAX_LENGTH,)
    ),

    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    SimpleRNN(
        128,
        return_sequences=False
    ),

    Dropout(
        0.3
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.2
    ),

    Dense(
        3,
        activation="softmax"
    )
])


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 15. DISPLAY MODEL
# ============================================================

print("\nModel Architecture:")

model.summary()


# ============================================================
# 16. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)

print("STARTING SIMPLE RNN TRAINING")

print("=" * 70)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_valid,
        y_valid
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 18. GENERATE VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

probabilities = model.predict(
    X_valid,
    batch_size=BATCH_SIZE,
    verbose=1
)


# Convert probabilities to class predictions
y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 19. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_valid,
    y_pred
)

precision = precision_score(
    y_valid,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_valid,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_valid,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 20. DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)

print("SIMPLE RNN MODEL PERFORMANCE")

print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"Macro F1 : {f1:.4f}"
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_valid,
        y_pred,
        target_names=[
            "Bearish",
            "Bullish",
            "Neutral"
        ],
        zero_division=0
    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_valid,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 23. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 24. SAVE SIMPLE RNN MODEL
# ============================================================

model.save(
    "models/simple_rnn_model.keras"
)


# ============================================================
# 25. SAVE TOKENIZER
# ============================================================

joblib.dump(
    tokenizer,
    "models/rnn_tokenizer.pkl"
)


# ============================================================
# 26. SAVE MODEL METRICS
# ============================================================

rnn_results = pd.DataFrame({

    "Model": [
        "Simple RNN"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1 Score": [
        f1
    ]

})


rnn_results.to_csv(
    "models/rnn_results.csv",
    index=False
)


# ============================================================
# 27. SAVE VALIDATION PREDICTIONS
# ============================================================

evaluation_df = pd.DataFrame({

    "Actual": y_valid,

    "Predicted": y_pred

})


evaluation_df.to_csv(
    "models/simple_rnn_predictions.csv",
    index=False
)


# ============================================================
# 28. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "models/simple_rnn_training_history.csv",
    index=False
)


# ============================================================
# 29. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)

print("SIMPLE RNN TRAINING COMPLETED")

print("=" * 70)

print("\nSaved files:")

print(
    "1. models/simple_rnn_model.keras"
)

print(
    "2. models/rnn_tokenizer.pkl"
)

print(
    "3. models/rnn_results.csv"
)

print(
    "4. models/simple_rnn_predictions.csv"
)

print(
    "5. models/simple_rnn_training_history.csv"
)

print("\nDone!")