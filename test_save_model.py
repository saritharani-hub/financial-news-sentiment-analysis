# ============================================================
# TEST SAVED MODEL
# Financial News Sentiment Prediction
# ============================================================

import joblib

# ------------------------------------------------------------
# 1. Load Saved Model and TF-IDF Vectorizer
# ------------------------------------------------------------

model = joblib.load(
    "models/linear_svm_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("Model and vectorizer loaded successfully!")


# ------------------------------------------------------------
# 2. Sentiment Prediction Function
# ------------------------------------------------------------

def predict_sentiment(text):

    # Convert text into TF-IDF features
    text_tfidf = tfidf.transform([text])

    # Predict sentiment
    prediction = model.predict(text_tfidf)[0]

    # Convert numerical label to sentiment name
    label_names = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }

    return label_names[prediction]


# ------------------------------------------------------------
# 3. Test News Examples
# ------------------------------------------------------------

news_examples = [

    "The company reported strong revenue growth and excellent profits.",

    "The company announced major losses and weak financial results.",

    "The company reported its quarterly financial results today.",

    "The company's stock price increased after strong earnings.",

    "The company is facing a decline in sales and increasing losses.",

    "The company announced its new financial report."
]


# ------------------------------------------------------------
# 4. Make Predictions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINANCIAL NEWS SENTIMENT PREDICTIONS")
print("=" * 70)

for news in news_examples:

    sentiment = predict_sentiment(news)

    print("\nNews:")
    print(news)

    print("Predicted Sentiment:", sentiment)


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)