# Financial News Sentiment Prediction using Deep Learning

## Project Overview

Financial news and social media content can contain important information about market sentiment. This project develops a three-class sentiment classification system for financial text.

The system classifies financial news into:

- **Bearish**
- **Bullish**
- **Neutral**

The project compares classical machine learning and deep learning approaches.

The implemented models are:

1. Logistic Regression
2. Naive Bayes
3. Linear SVM
4. Simple RNN
5. LSTM

The deep-learning part of the project focuses on two RNN-based architectures:

- Embedding + Simple RNN
- Embedding + LSTM

A Streamlit dashboard is also developed for interactive financial sentiment prediction.


---

## Project Objective

The main objectives of this project are:

- Load and understand a financial sentiment dataset.
- Clean and preprocess financial text.
- Handle class imbalance in the training dataset.
- Build classical machine learning sentiment classifiers.
- Build Simple RNN and LSTM deep-learning models.
- Evaluate the models using multiple classification metrics.
- Compare classical ML and deep-learning approaches.
- Develop an interactive Streamlit dashboard for sentiment prediction.


---

## Dataset

The project uses the Financial News Sentiment dataset.

Local project files:

```text
sent_train.csv
sent_valid.xlsx