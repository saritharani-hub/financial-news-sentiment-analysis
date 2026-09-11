# 📈 Financial News Sentiment Prediction

## 📌 Project Overview

Financial news can strongly influence investor decisions and stock market movements. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify financial news into three sentiment categories:

* 🔴 Negative
* 🟢 Positive
* ⚪ Neutral

A **Streamlit dashboard** is developed to allow users to enter financial news and receive a predicted sentiment.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Load and explore a financial news sentiment dataset.
2. Clean and preprocess financial news text.
3. Convert text into numerical features using **TF-IDF**.
4. Train multiple machine learning classification models.
5. Compare model performance.
6. Select the best-performing model.
7. Save the trained model for future predictions.
8. Develop an interactive Streamlit dashboard.
9. Predict sentiment for new financial news statements.

---

## 📊 Dataset

The dataset used in this project is:

**Twitter Financial News Sentiment Dataset**

Source:

`zeroshot/twitter-financial-news-sentiment`

The dataset contains financial news text with three sentiment labels.

### Dataset Labels

| Label | Sentiment |
| ----- | --------- |
| 0     | Negative  |
| 1     | Positive  |
| 2     | Neutral   |

### Dataset Size

| Dataset    | Records |
| ---------- | ------: |
| Training   |   9,543 |
| Validation |   2,388 |

After removing empty cleaned text records:

| Dataset    | Records |
| ---------- | ------: |
| Training   |   9,529 |
| Validation |   2,385 |

---

## 🔄 Project Workflow

```text
Financial News Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Save Model
        ↓
Streamlit Dashboard
        ↓
Financial News Sentiment Prediction
```

---

## 🧹 Text Preprocessing

The following preprocessing steps were performed:

* Converted text to lowercase
* Removed URLs
* Removed stock ticker symbols such as `$AAPL`
* Removed punctuation and special characters
* Removed extra spaces
* Removed empty text records

Example:

```text
Original:
The company reported STRONG profits! Visit www.example.com

Cleaned:
the company reported strong profits
```

---

## 🔢 Feature Engineering

### TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** was used to convert financial news text into numerical features.

The vectorizer was configured with:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

The final feature matrix contained:

```text
5000 features
```

Both unigrams and bigrams were used.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)

---

## 📊 Model Comparison

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

### Results

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| **Linear SVM**      | **81.97%** | **82.13%** | **81.97%** | **82.05%** |
| Logistic Regression |     79.20% |     80.98% |     79.20% |     79.82% |
| Naive Bayes         |     79.08% |     79.57% |     79.08% |     76.62% |

### 🏆 Best Model

**Linear SVM** achieved the best overall performance.

* Accuracy: **81.97%**
* Precision: **82.13%**
* Recall: **81.97%**
* F1 Score: **82.05%**

Therefore, Linear SVM was selected as the final sentiment prediction model.

---

## 💾 Saved Model Files

The following files are stored inside the `models` directory:

```text
models/
│
├── linear_svm_model.pkl
├── tfidf_vectorizer.pkl
├── model_comparison.csv
└── linear_svm_predictions.csv
```

### File Description

| File                         | Purpose                                |
| ---------------------------- | -------------------------------------- |
| `linear_svm_model.pkl`       | Saved Linear SVM model                 |
| `tfidf_vectorizer.pkl`       | Saved TF-IDF vectorizer                |
| `model_comparison.csv`       | Model performance results              |
| `linear_svm_predictions.csv` | Actual and predicted validation labels |

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard with four pages.

### 📰 1. Sentiment Prediction

Users can enter financial news and receive a sentiment prediction.

Example:

```text
The company reported strong revenue growth and excellent profits.
```

Output:

```text
Predicted Sentiment: Positive
```

---

### 📊 2. Model Comparison

This page displays:

* Model comparison table
* Accuracy comparison
* F1 Score comparison
* Best-performing model

---

### 📈 3. Model Performance

This page displays:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Sentiment distribution

---

### 🔍 4. Sample Predictions

This page contains example financial news statements and displays their predicted sentiment.

---

## 📁 Project Structure

```text
financial_sentimental_analysis/
│
├── models/
│   ├── linear_svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── model_comparison.csv
│   └── linear_svm_predictions.csv
│
├── train_model.py
├── test_saved_model.py
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository and install the required libraries.

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install datasets
pip install joblib
pip install matplotlib
pip install seaborn
pip install streamlit
```

Or install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Train the model

```bash
python train_model.py
```

This creates the model and evaluation files inside the `models` folder.

### Step 2 — Test the saved model

```bash
python test_saved_model.py
```

### Step 3 — Start the Streamlit dashboard

```bash
python -m streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🧪 Example Predictions

### Example 1

```text
The company reported strong revenue growth and excellent profits.
```

Expected sentiment:

```text
Positive
```

### Example 2

```text
The company announced major losses and weak financial results.
```

Expected sentiment:

```text
Negative
```

### Example 3

```text
The company reported its quarterly financial results today.
```

The trained model determines the sentiment based on learned financial language patterns.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Hugging Face Datasets
* TF-IDF
* Linear SVM
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Git
* GitHub

---

## 📌 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Natural Language Processing
* Text preprocessing
* TF-IDF vectorization
* N-grams
* Imbalanced classification
* Machine learning model comparison
* Classification metrics
* Confusion matrix
* Model persistence using Joblib
* Streamlit dashboard development
* Git and GitHub project management

---

## 🚀 Future Improvements

Possible future improvements include:

* Using transformer-based models such as BERT
* Adding probability/confidence scores
* Adding real-time financial news collection
* Adding stock market data
* Adding sentiment trends over time
* Deploying the Streamlit application online
* Improving performance on the minority sentiment classes

---

## 👩‍💻 Author

**Saritha**

AI/ML Project — Financial News Sentiment Prediction
