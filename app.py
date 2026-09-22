# ============================================================
# FINANCIAL NEWS SENTIMENT ANALYSIS
# STREAMLIT DASHBOARD
# Linear SVM + Simple RNN + LSTM
# ============================================================

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

SVM_MODEL_FILE = "models/linear_svm_model.pkl"
TFIDF_FILE = "models/tfidf_vectorizer.pkl"
SVM_RESULTS_FILE = "models/model_comparison.csv"
SVM_PREDICTIONS_FILE = "models/linear_svm_predictions.csv"

RNN_MODEL_FILE = "models/simple_rnn_model.keras"
RNN_TOKENIZER_FILE = "models/rnn_tokenizer.pkl"
RNN_RESULTS_FILE = "models/rnn_results.csv"
RNN_PREDICTIONS_FILE = "models/simple_rnn_predictions.csv"

LSTM_MODEL_FILE = "models/lstm_model.keras"
LSTM_TOKENIZER_FILE = "models/lstm_tokenizer.pkl"
LSTM_RESULTS_FILE = "models/lstm_results.csv"
LSTM_PREDICTIONS_FILE = "models/lstm_predictions.csv"

COMPARISON_FILE = "models/deep_learning_model_comparison.csv"

VALID_FILE = "sent_valid.xlsx"


# ============================================================
# LOAD CLASSICAL MODEL
# ============================================================

model = joblib.load(
    SVM_MODEL_FILE
)


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

tfidf = joblib.load(
    TFIDF_FILE
)


# ============================================================
# LOAD CLASSICAL RESULTS
# ============================================================

results_df = pd.read_csv(
    SVM_RESULTS_FILE
)


# ============================================================
# LOAD CLASSICAL PREDICTIONS
# ============================================================

evaluation_df = pd.read_csv(
    SVM_PREDICTIONS_FILE
)


# ============================================================
# LOAD SIMPLE RNN
# ============================================================

@st.cache_resource
def load_rnn_model():

    return load_model(
        RNN_MODEL_FILE
    )


@st.cache_resource
def load_rnn_tokenizer():

    return joblib.load(
        RNN_TOKENIZER_FILE
    )


rnn_model = load_rnn_model()

rnn_tokenizer = load_rnn_tokenizer()


# ============================================================
# LOAD LSTM
# ============================================================

@st.cache_resource
def load_lstm_model():

    return load_model(
        LSTM_MODEL_FILE
    )


@st.cache_resource
def load_lstm_tokenizer():

    return joblib.load(
        LSTM_TOKENIZER_FILE
    )


lstm_model = load_lstm_model()

lstm_tokenizer = load_lstm_tokenizer()


# ============================================================
# LOAD RNN/LSTM RESULTS
# ============================================================

rnn_results_df = pd.read_csv(
    RNN_RESULTS_FILE
)

lstm_results_df = pd.read_csv(
    LSTM_RESULTS_FILE
)


# ============================================================
# LOAD COMBINED MODEL COMPARISON
# ============================================================

comparison_df = pd.read_csv(
    COMPARISON_FILE
)


# ============================================================
# LOAD RNN/LSTM PREDICTIONS
# ============================================================

rnn_predictions_df = pd.read_csv(
    RNN_PREDICTIONS_FILE
)

lstm_predictions_df = pd.read_csv(
    LSTM_PREDICTIONS_FILE
)


# ============================================================
# LABEL MAPPING
# ============================================================

label_names = {

    0: "Bearish",

    1: "Bullish",

    2: "Neutral"
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

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

    # Keep financial symbols
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


# ============================================================
# LSTM PREDICTION FUNCTION
# ============================================================

def predict_lstm(text):

    cleaned_text = clean_text(
        text
    )

    sequence = (
        lstm_tokenizer
        .texts_to_sequences(
            [cleaned_text]
        )
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=100,
        padding="post",
        truncating="post"
    )

    probabilities = lstm_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    sentiment = label_names[
        predicted_class
    ]

    return sentiment, probabilities


# ============================================================
# LINEAR SVM PREDICTION FUNCTION
# ============================================================

def predict_svm(text):

    text_tfidf = tfidf.transform(
        [text]
    )

    prediction = model.predict(
        text_tfidf
    )[0]

    return label_names[
        int(prediction)
    ]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Navigation"
)

page = st.sidebar.radio(

    "Select a page:",

    [
        "📰 Sentiment Prediction",
        "📊 Model Comparison",
        "📈 Model Performance",
        "🔍 Sample Predictions"
    ]
)


# ============================================================
# PAGE 1 - SENTIMENT PREDICTION
# ============================================================

if page == "📰 Sentiment Prediction":

    st.title(
        "📰 Financial News Sentiment Prediction"
    )

    st.write(
        "Enter a financial news statement or "
        "financial tweet to predict its sentiment."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    prediction_model = st.selectbox(

        "Select Prediction Model",

        [
            "LSTM",
            "Linear SVM"
        ]
    )

    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------

    news_text = st.text_area(

        "Enter Financial News",

        placeholder=(
            "Example: "
            "The company reported strong revenue growth "
            "and excellent profits."
        ),

        height=150
    )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Sentiment"
    ):

        if news_text.strip() == "":

            st.warning(
                "Please enter some financial news."
            )

        else:

            # =================================================
            # LSTM PREDICTION
            # =================================================

            if prediction_model == "LSTM":

                sentiment, probabilities = (
                    predict_lstm(
                        news_text
                    )
                )

                st.subheader(
                    "Prediction Result"
                )

                # Display sentiment
                if sentiment == "Bullish":

                    st.success(
                        "📈 Sentiment: Bullish"
                    )

                elif sentiment == "Bearish":

                    st.error(
                        "📉 Sentiment: Bearish"
                    )

                else:

                    st.info(
                        "➖ Sentiment: Neutral"
                    )

                # ------------------------------------------------
                # PROBABILITIES
                # ------------------------------------------------

                st.subheader(
                    "Sentiment Probabilities"
                )

                probability_df = pd.DataFrame({

                    "Sentiment": [
                        "Bearish",
                        "Bullish",
                        "Neutral"
                    ],

                    "Probability": [
                        probabilities[0],
                        probabilities[1],
                        probabilities[2]
                    ]

                })

                probability_df[
                    "Probability (%)"
                ] = (
                    probability_df[
                        "Probability"
                    ] * 100
                )

                st.dataframe(

                    probability_df[
                        [
                            "Sentiment",
                            "Probability (%)"
                        ]
                    ].style.format(
                        {
                            "Probability (%)":
                            "{:.2f}%"
                        }
                    ),

                    use_container_width=True,

                    hide_index=True
                )

                # ------------------------------------------------
                # PROBABILITY BAR CHART
                # ------------------------------------------------

                st.subheader(
                    "Probability Distribution"
                )

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                ax.bar(

                    probability_df[
                        "Sentiment"
                    ],

                    probability_df[
                        "Probability"
                    ]

                )

                ax.set_ylim(
                    0,
                    1
                )

                ax.set_ylabel(
                    "Probability"
                )

                ax.set_xlabel(
                    "Sentiment"
                )

                ax.set_title(
                    "LSTM Sentiment Probability"
                )

                st.pyplot(
                    fig
                )

                plt.close(fig)

            # =================================================
            # LINEAR SVM PREDICTION
            # =================================================

            else:

                sentiment = predict_svm(
                    news_text
                )

                st.subheader(
                    "Prediction Result"
                )

                if sentiment == "Bullish":

                    st.success(
                        "📈 Sentiment: Bullish"
                    )

                elif sentiment == "Bearish":

                    st.error(
                        "📉 Sentiment: Bearish"
                    )

                else:

                    st.info(
                        "➖ Sentiment: Neutral"
                    )

                st.info(
                    "Linear SVM does not provide probability "
                    "outputs in the current saved model. "
                    "Select LSTM to view class probabilities."
                )


# ============================================================
# PAGE 2 - MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.title(
        "📊 Model Comparison"
    )

    st.write(
        "Comparison of Linear SVM, Simple RNN "
        "and LSTM models."
    )

    st.divider()

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    st.subheader(
        "Model Performance Comparison"
    )

    display_comparison = comparison_df.copy()

    display_comparison[
        "Accuracy"
    ] = (
        display_comparison[
            "Accuracy"
        ] * 100
    )

    display_comparison[
        "Precision"
    ] = (
        display_comparison[
            "Precision"
        ] * 100
    )

    display_comparison[
        "Recall"
    ] = (
        display_comparison[
            "Recall"
        ] * 100
    )

    display_comparison[
        "F1 Score"
    ] = (
        display_comparison[
            "F1 Score"
        ] * 100
    )

    st.dataframe(

        display_comparison.style.format(
            {
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1 Score": "{:.2f}%"
            }
        ),

        use_container_width=True,

        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    st.subheader(
        "Accuracy Comparison"
    )

    accuracy_chart = comparison_df.set_index(
        "Model"
    )["Accuracy"]

    st.bar_chart(
        accuracy_chart
    )

    # --------------------------------------------------------
    # F1 SCORE CHART
    # --------------------------------------------------------

    st.subheader(
        "Macro F1 Score Comparison"
    )

    f1_chart = comparison_df.set_index(
        "Model"
    )["F1 Score"]

    st.bar_chart(
        f1_chart
    )

    # --------------------------------------------------------
    # ALL METRICS
    # --------------------------------------------------------

    st.subheader(
        "All Model Metrics"
    )

    metrics_df = comparison_df.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]

    st.bar_chart(
        metrics_df
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Models Used"
    )

    st.write(
        "• Linear SVM — TF-IDF based classical ML model"
    )

    st.write(
        "• Simple RNN — Embedding + Simple RNN"
    )

    st.write(
        "• LSTM — Embedding + LSTM"
    )


# ============================================================
# PAGE 3 - MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Model Performance"
    )

    st.write(
        "Detailed performance analysis of "
        "the three sentiment classification models."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL SELECTOR
    # --------------------------------------------------------

    selected_model = st.selectbox(

        "Select Model",

        [
            "Linear SVM",
            "Simple RNN",
            "LSTM"
        ]
    )

    # ========================================================
    # LINEAR SVM
    # ========================================================

    if selected_model == "Linear SVM":

        st.subheader(
            "Linear SVM Performance"
        )

        svm_row = results_df[
            results_df["Model"] == "Linear SVM"
        ].iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{svm_row['Accuracy']:.2%}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{svm_row['Precision']:.2%}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{svm_row['Recall']:.2%}"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{svm_row['F1 Score']:.2%}"
            )

        st.divider()

        # Confusion matrix
        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(

            evaluation_df["Actual"],

            evaluation_df["Predicted"]

        )

        labels = [
            "Bearish",
            "Bullish",
            "Neutral"
        ]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=labels,

            yticklabels=labels,

            ax=ax
        )

        ax.set_xlabel(
            "Predicted Sentiment"
        )

        ax.set_ylabel(
            "Actual Sentiment"
        )

        ax.set_title(
            "Linear SVM Confusion Matrix"
        )

        st.pyplot(
            fig
        )

        plt.close(fig)

    # ========================================================
    # SIMPLE RNN
    # ========================================================

    elif selected_model == "Simple RNN":

        st.subheader(
            "Simple RNN Performance"
        )

        rnn_row = rnn_results_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{rnn_row['Accuracy']:.2%}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{rnn_row['Precision']:.2%}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{rnn_row['Recall']:.2%}"
            )

        with col4:

            st.metric(
                "Macro F1",
                f"{rnn_row['F1 Score']:.2%}"
            )

        st.divider()

        # Confusion matrix
        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(

            rnn_predictions_df["Actual"],

            rnn_predictions_df["Predicted"]

        )

        labels = [
            "Bearish",
            "Bullish",
            "Neutral"
        ]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=labels,

            yticklabels=labels,

            ax=ax
        )

        ax.set_xlabel(
            "Predicted Sentiment"
        )

        ax.set_ylabel(
            "Actual Sentiment"
        )

        ax.set_title(
            "Simple RNN Confusion Matrix"
        )

        st.pyplot(
            fig
        )

        plt.close(fig)

    # ========================================================
    # LSTM
    # ========================================================

    else:

        st.subheader(
            "LSTM Performance"
        )

        lstm_row = lstm_results_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{lstm_row['Accuracy']:.2%}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{lstm_row['Precision']:.2%}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{lstm_row['Recall']:.2%}"
            )

        with col4:

            st.metric(
                "Macro F1",
                f"{lstm_row['F1 Score']:.2%}"
            )

        st.divider()

        # Confusion matrix
        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(

            lstm_predictions_df["Actual"],

            lstm_predictions_df["Predicted"]

        )

        labels = [
            "Bearish",
            "Bullish",
            "Neutral"
        ]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=labels,

            yticklabels=labels,

            ax=ax
        )

        ax.set_xlabel(
            "Predicted Sentiment"
        )

        ax.set_ylabel(
            "Actual Sentiment"
        )

        ax.set_title(
            "LSTM Confusion Matrix"
        )

        st.pyplot(
            fig
        )

        plt.close(fig)


# ============================================================
# PAGE 4 - SAMPLE PREDICTIONS
# ============================================================

elif page == "🔍 Sample Predictions":

    st.title(
        "🔍 Sample Financial News Predictions"
    )

    st.write(
        "Test the trained LSTM model with "
        "sample financial news."
    )

    st.divider()

    sample_news = [

        "The company reported strong revenue growth "
        "and excellent profits.",

        "The company announced major losses "
        "and weak financial results.",

        "The company reported its quarterly "
        "financial results today.",

        "The company achieved record sales "
        "and increased its earnings.",

        "The company faced a significant decline "
        "in revenue and profits."
    ]

    for news in sample_news:

        sentiment, probabilities = (
            predict_lstm(
                news
            )
        )

        st.write(
            f"**News:** {news}"
        )

        if sentiment == "Bullish":

            st.success(
                f"📈 Sentiment: {sentiment}"
            )

        elif sentiment == "Bearish":

            st.error(
                f"📉 Sentiment: {sentiment}"
            )

        else:

            st.info(
                f"➖ Sentiment: {sentiment}"
            )

        probability_df = pd.DataFrame({

            "Sentiment": [
                "Bearish",
                "Bullish",
                "Neutral"
            ],

            "Probability": [
                probabilities[0],
                probabilities[1],
                probabilities[2]
            ]

        })

        probability_df[
            "Probability (%)"
        ] = (
            probability_df[
                "Probability"
            ] * 100
        )

        st.dataframe(

            probability_df[
                [
                    "Sentiment",
                    "Probability (%)"
                ]
            ].style.format(
                {
                    "Probability (%)":
                    "{:.2f}%"
                }
            ),

            use_container_width=True,

            hide_index=True
        )

        st.divider()


# ============================================================
# PAGE 5 - CLASS DISTRIBUTION
# ============================================================

if page == "📊 Model Comparison":

    st.divider()

    st.subheader(
        "Validation Class Distribution"
    )

    if os.path.exists(
        VALID_FILE
    ):

        valid_df = pd.read_excel(
            VALID_FILE
        )

        valid_df["Sentiment"] = (
            valid_df["label"]
            .map(label_names)
        )

        class_counts = (
            valid_df["Sentiment"]
            .value_counts()
            .reindex(
                [
                    "Bearish",
                    "Bullish",
                    "Neutral"
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            class_counts
        )

        st.write(
            "The chart above shows the distribution "
            "of Bearish, Bullish, and Neutral samples "
            "in the validation dataset."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(

    "Financial News Sentiment Prediction\n\n"

    "Models:\n"
    "• Linear SVM\n"
    "• Simple RNN\n"
    "• LSTM\n\n"

    "Classes:\n"
    "• Bearish\n"
    "• Bullish\n"
    "• Neutral"
)