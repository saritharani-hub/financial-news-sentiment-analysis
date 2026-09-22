# ============================================================
# FINANCIAL NEWS SENTIMENT ANALYSIS
# MODEL COMPARISON
# Linear SVM vs Simple RNN vs LSTM
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. FILE PATHS
# ============================================================

SVM_RESULTS = "models/model_comparison.csv"
RNN_RESULTS = "models/rnn_results.csv"
LSTM_RESULTS = "models/lstm_results.csv"

OUTPUT_FILE = "models/deep_learning_model_comparison.csv"


# ============================================================
# 2. CHECK FILES
# ============================================================

print("=" * 70)
print("FINANCIAL NEWS SENTIMENT ANALYSIS")
print("MODEL COMPARISON")
print("=" * 70)

required_files = [
    SVM_RESULTS,
    RNN_RESULTS,
    LSTM_RESULTS
]

for file in required_files:

    if os.path.exists(file):

        print(f"Found: {file}")

    else:

        print(f"ERROR: File not found: {file}")


# ============================================================
# 3. LOAD LINEAR SVM RESULTS
# ============================================================

print("\nLoading Linear SVM results...")

svm_df = pd.read_csv(
    SVM_RESULTS
)

print("\nLinear SVM results:")

print(svm_df)


# ============================================================
# 4. FIND LINEAR SVM MODEL
# ============================================================

# Your existing model_comparison.csv contains
# Logistic Regression, Naive Bayes and Linear SVM.
#
# We select only Linear SVM for this comparison.

svm_row = svm_df[
    svm_df["Model"].str.contains(
        "Linear SVM",
        case=False,
        na=False
    )
].copy()


# ============================================================
# 5. LOAD SIMPLE RNN RESULTS
# ============================================================

print("\nLoading Simple RNN results...")

rnn_df = pd.read_csv(
    RNN_RESULTS
)

print("\nSimple RNN results:")

print(rnn_df)


# ============================================================
# 6. LOAD LSTM RESULTS
# ============================================================

print("\nLoading LSTM results...")

lstm_df = pd.read_csv(
    LSTM_RESULTS
)

print("\nLSTM results:")

print(lstm_df)


# ============================================================
# 7. STANDARDIZE COLUMN NAMES
# ============================================================

# The classical model uses:
# F1
#
# The deep learning models use:
# F1 Score
#
# Rename them so all models have the same structure.

if "F1" in svm_row.columns:

    svm_row = svm_row.rename(
        columns={
            "F1": "F1 Score"
        }
    )


# ============================================================
# 8. SELECT REQUIRED COLUMNS
# ============================================================

svm_comparison = svm_row[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()


rnn_comparison = rnn_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()


lstm_comparison = lstm_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()


# ============================================================
# 9. COMBINE ALL MODELS
# ============================================================

comparison_df = pd.concat(
    [
        svm_comparison,
        rnn_comparison,
        lstm_comparison
    ],
    ignore_index=True
)


# ============================================================
# 10. CONVERT METRICS TO NUMERIC
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for column in metric_columns:

    comparison_df[column] = pd.to_numeric(
        comparison_df[column],
        errors="coerce"
    )


# ============================================================
# 11. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)

print("FINAL MODEL COMPARISON")

print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE COMPARISON CSV
# ============================================================

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nComparison saved to: {OUTPUT_FILE}"
)


# ============================================================
# 13. ACCURACY COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    comparison_df["Model"],
    comparison_df["Accuracy"]
)

plt.title(
    "Model Accuracy Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "models/model_accuracy_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. F1 SCORE COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    comparison_df["Model"],
    comparison_df["F1 Score"]
)

plt.title(
    "Model Macro F1 Score Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Macro F1 Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "models/model_f1_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. ALL METRICS COMPARISON
# ============================================================

plot_df = comparison_df.set_index(
    "Model"
)[metric_columns]


plot_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Performance Comparison of Machine Learning and RNN Models"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    "models/all_model_metrics_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)

print("MODEL COMPARISON COMPLETED")

print("=" * 70)

print("\nCreated files:")

print(
    "1. models/deep_learning_model_comparison.csv"
)

print(
    "2. models/model_accuracy_comparison.png"
)

print(
    "3. models/model_f1_comparison.png"
)

print(
    "4. models/all_model_metrics_comparison.png"
)

print("\nDone!")