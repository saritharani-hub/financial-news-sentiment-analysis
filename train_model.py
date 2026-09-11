# ============================================================
# FINANCIAL NEWS SENTIMENT PREDICTION
# COMPLETE MODEL TRAINING PROGRAM
# ============================================================

import os
import re
import joblib
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING DATASET")
print("=" * 70)

dataset = load_dataset(
    "zeroshot/twitter-financial-news-sentiment"
)

train_df = dataset["train"].to_pandas()
valid_df = dataset["validation"].to_pandas()

print("\nTraining Shape:", train_df.shape)
print("Validation Shape:", valid_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())


# ============================================================
# 2. LABEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - LABEL INFORMATION")
print("=" * 70)

label_names = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

print("\nLabel Mapping:")
for number, name in label_names.items():
    print(number, "=", name)

print("\nTraining Label Distribution:")
print(train_df["label"].value_counts().sort_index())


# ============================================================
# 3. TEXT CLEANING
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - TEXT PREPROCESSING")
print("=" * 70)


def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove stock ticker symbols such as $AAPL
    text = re.sub(
        r"\$[a-zA-Z]+",
        "",
        text
    )

    # Keep only alphabets and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
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


train_df["clean_text"] = train_df["text"].apply(clean_text)
valid_df["clean_text"] = valid_df["text"].apply(clean_text)


# Remove empty texts
train_df = train_df[
    train_df["clean_text"].str.strip() != ""
].copy()

valid_df = valid_df[
    valid_df["clean_text"].str.strip() != ""
].copy()


print("\nTraining rows after cleaning:", len(train_df))
print("Validation rows after cleaning:", len(valid_df))


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - TF-IDF VECTORIZATION")
print("=" * 70)


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)


X_train = tfidf.fit_transform(
    train_df["clean_text"]
)

X_valid = tfidf.transform(
    valid_df["clean_text"]
)

y_train = train_df["label"]
y_valid = valid_df["label"]


print("\nTraining feature shape:", X_train.shape)
print("Validation feature shape:", X_valid.shape)
print(
    "Number of features:",
    len(tfidf.get_feature_names_out())
)


# ============================================================
# 5. CREATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - CREATING MACHINE LEARNING MODELS")
print("=" * 70)


models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    )
}


# ============================================================
# 6. TRAIN MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - TRAINING MODELS")
print("=" * 70)


trained_models = {}


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    print("Training completed.")


print("\nAll models trained successfully.")


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - MODEL EVALUATION")
print("=" * 70)


results = []


for name, model in trained_models.items():

    print("\n")
    print("=" * 70)
    print("MODEL:", name)
    print("=" * 70)

    # Predictions
    y_pred = model.predict(X_valid)

    # Metrics
    accuracy = accuracy_score(
        y_valid,
        y_pred
    )

    precision = precision_score(
        y_valid,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_valid,
            y_pred,
            target_names=[
                "Negative",
                "Positive",
                "Neutral"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_valid,
            y_pred
        )
    )

    # Store results
    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    })


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - MODEL COMPARISON")
print("=" * 70)


results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n")
print(results_df)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\nBest Model:", best_model_name)


# ============================================================
# 10. CREATE MODELS FOLDER
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/linear_svm_model.pkl"
)

print(
    "\nSaved:",
    "models/linear_svm_model.pkl"
)


# ============================================================
# 12. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print(
    "Saved:",
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# 13. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print(
    "Saved:",
    "models/model_comparison.csv"
)


# ============================================================
# 14. SAVE LINEAR SVM PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - SAVING LINEAR SVM PREDICTIONS")
print("=" * 70)


# We specifically use Linear SVM because it is our selected model
linear_svm_model = trained_models[
    "Linear SVM"
]


linear_svm_predictions = linear_svm_model.predict(
    X_valid
)


evaluation_df = pd.DataFrame({

    "Actual": y_valid.values,

    "Predicted": linear_svm_predictions

})


evaluation_df.to_csv(
    "models/linear_svm_predictions.csv",
    index=False
)


print(
    "\nSaved:",
    "models/linear_svm_predictions.csv"
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nBest Model:")
print(best_model_name)

print("\nSaved Files:")

print("1. models/linear_svm_model.pkl")
print("2. models/tfidf_vectorizer.pkl")
print("3. models/model_comparison.csv")
print("4. models/linear_svm_predictions.csv")

print("\nAll files created successfully!")