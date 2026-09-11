# ============================================================
# FINANCIAL NEWS SENTIMENT ANALYSIS
# STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load(
    "models/linear_svm_model.pkl"
)


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD MODEL COMPARISON RESULTS
# ============================================================

results_df = pd.read_csv(
    "models/model_comparison.csv"
)


# ============================================================
# LOAD LINEAR SVM PREDICTIONS
# ============================================================

evaluation_df = pd.read_csv(
    "models/linear_svm_predictions.csv"
)


# ============================================================
# LABEL MAPPING
# ============================================================

label_names = {

    0: "Negative",

    1: "Positive",

    2: "Neutral"
}


# ============================================================
# SENTIMENT PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    # Convert input text into TF-IDF
    text_tfidf = tfidf.transform(
        [text]
    )

    # Predict sentiment
    prediction = model.predict(
        text_tfidf
    )[0]

    return label_names[prediction]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Navigation")

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
        "Enter a financial news statement below "
        "to predict its sentiment."
    )

    st.divider()

    news_text = st.text_area(

        "Enter Financial News",

        placeholder=(
            "Example: "
            "The company reported strong revenue growth "
            "and excellent profits."
        ),

        height=150
    )


    if st.button(
        "🔮 Predict Sentiment"
    ):

        if news_text.strip() == "":

            st.warning(
                "Please enter some financial news."
            )

        else:

            sentiment = predict_sentiment(
                news_text
            )

            st.subheader(
                "Prediction Result"
            )

            if sentiment == "Positive":

                st.success(
                    "📈 Sentiment: Positive"
                )

            elif sentiment == "Negative":

                st.error(
                    "📉 Sentiment: Negative"
                )

            else:

                st.info(
                    "➖ Sentiment: Neutral"
                )


# ============================================================
# PAGE 2 - MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.title(
        "📊 Model Comparison"
    )

    st.write(
        "Comparison of machine learning models "
        "used for financial sentiment classification."
    )

    st.divider()


    # Display table

    st.subheader(
        "Model Performance"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Accuracy Chart
    # --------------------------------------------------------

    st.subheader(
        "Accuracy Comparison"
    )

    accuracy_chart = results_df.set_index(
        "Model"
    )["Accuracy"]

    st.bar_chart(
        accuracy_chart
    )


    # --------------------------------------------------------
    # F1 Score Chart
    # --------------------------------------------------------

    st.subheader(
        "F1 Score Comparison"
    )

    f1_chart = results_df.set_index(
        "Model"
    )["F1 Score"]

    st.bar_chart(
        f1_chart
    )


    # --------------------------------------------------------
    # Best Model
    # --------------------------------------------------------

    best_model_row = results_df.iloc[0]

    st.success(
        f"🏆 Best Model: {best_model_row['Model']}"
    )

    st.write(
        f"Accuracy: "
        f"{best_model_row['Accuracy']:.4f}"
    )

    st.write(
        f"F1 Score: "
        f"{best_model_row['F1 Score']:.4f}"
    )


# ============================================================
# PAGE 3 - MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Linear SVM Model Performance"
    )

    st.write(
        "Detailed performance analysis of the "
        "selected Linear SVM model."
    )

    st.divider()


    # --------------------------------------------------------
    # Get Linear SVM Metrics
    # --------------------------------------------------------

    svm_row = results_df[
        results_df["Model"] == "Linear SVM"
    ].iloc[0]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )


    cm = confusion_matrix(

        evaluation_df["Actual"],

        evaluation_df["Predicted"]
    )


    labels = [

        "Negative",

        "Positive",

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


    st.divider()


    # --------------------------------------------------------
    # Actual Sentiment Distribution
    # --------------------------------------------------------

    st.subheader(
        "Actual Sentiment Distribution"
    )


    sentiment_counts = (

        evaluation_df["Actual"]

        .map(label_names)

        .value_counts()

        .reindex([
            "Negative",
            "Positive",
            "Neutral"
        ])
    )


    st.bar_chart(
        sentiment_counts
    )


# ============================================================
# PAGE 4 - SAMPLE PREDICTIONS
# ============================================================

elif page == "🔍 Sample Predictions":

    st.title(
        "🔍 Sample Financial News Predictions"
    )

    st.write(
        "Test the saved Linear SVM model "
        "with sample financial news."
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

        sentiment = predict_sentiment(
            news
        )


        st.write(
            f"**News:** {news}"
        )


        if sentiment == "Positive":

            st.success(
                f"Sentiment: {sentiment}"
            )

        elif sentiment == "Negative":

            st.error(
                f"Sentiment: {sentiment}"
            )

        else:

            st.info(
                f"Sentiment: {sentiment}"
            )


        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    "Financial News Sentiment Prediction\n\n"
    "Model: Linear SVM\n"
    "Features: TF-IDF\n"
    "Classes: Negative, Positive, Neutral"
)