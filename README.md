# 🚨 Disaster Tweet Classification System Using Transformers

## Overview

This project implements an end-to-end pipeline for detecting whether a tweet refers to a real disaster or not. The system combines **state-of-the-art natural language embeddings**, **structured categorical features**, and **gradient boosting techniques** to deliver accurate and explainable predictions. An intuitive **web-based interface** built using **Streamlit** allows users to interact with the model in real-time.

---

## Problem Statement

During disaster events, social media platforms—particularly Twitter—serve as critical communication channels. However, distinguishing between actual disaster-related tweets and unrelated or metaphorical tweets poses a challenge for emergency response systems. The goal of this project is to **automatically classify tweets** into two categories:

* **1 (True)** — Tweet refers to a real disaster event.
* **0 (False)** — Tweet does **not** refer to a real disaster event.

---

## Solution Architecture

```
          +----------------+       +------------------------+
          | Raw CSV Input  | --->  | Preprocessing & Feature |
          | (tweets.csv)   |       | Engineering (Pandas)    |
          +----------------+       +------------------------+
                   |                          |
                   v                          v
     +-------------------------+    +------------------------+
     | Sentence Embedding      |    | One-Hot Encoding       |
     | (SentenceTransformers) |    | (keyword, location)     |
     +-------------------------+    +------------------------+
                   |                          |
                   +-----------+--------------+
                               |
                               v
               +-------------------------------+
               | Feature Concatenation (Numpy) |
               +-------------------------------+
                               |
                               v
                  +--------------------------+
                  | Model Training (XGBoost) |
                  +--------------------------+
                               |
                               v
               +------------------------------+
               | Web UI (Streamlit)           |
               | Real-time Predictions        |
               +------------------------------+
```

---

## Technologies Used

| Tool/Library            | Purpose                        |
| ----------------------- | ------------------------------ |
| `pandas`                | Data cleaning and manipulation |
| `scikit-learn`          | Data splitting and evaluation  |
| `xgboost`               | Model training (classifier)    |
| `sentence-transformers` | Pre-trained NLP embeddings     |
| `streamlit`             | Interactive web application    |
| `joblib`                | Model and object serialization |

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-org/disaster-tweet-classifier.git
cd disaster-tweet-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pandas
numpy
xgboost
scikit-learn
sentence-transformers
streamlit
joblib
```

---

## Training Pipeline

The training pipeline is encapsulated in `train_model.py`.

### Workflow:

1. Load and preprocess dataset from `tweets.csv`
2. Fill missing values (e.g., unknown locations or keywords)
3. Perform one-hot encoding for `keyword` and `location`
4. Embed tweet `text` using `all-MiniLM-L6-v2` transformer
5. Concatenate embeddings with encoded features
6. Split into training and validation sets (stratified)
7. Train an XGBoost classifier
8. Save model and one-hot schema to disk

### Run Training:

```bash
python train_model.py
```

Artifacts:

* `xgb_disaster_model.joblib`: Trained XGBoost model
* `onehot_columns.joblib`: One-hot encoding column mapping

---

## Web Application

The app allows users to classify custom tweet input in real time.

### Launch the App:

```bash
streamlit run app.py
```

### Features:

* Text area to input tweet
* Optional inputs: `keyword`, `location`
* Model prediction and probability confidence

The app loads:

* The serialized XGBoost model
* The one-hot column schema
* The sentence transformer model (for text embeddings)

---

## Input and Output Schema

### Input Format

| Field      | Description                             | Required |
| ---------- | --------------------------------------- | -------- |
| `text`     | Tweet content                           | ✅        |
| `keyword`  | Disaster-related keyword (e.g. "flood") | Optional |
| `location` | Location of tweet                       | Optional |

### Output Format

* **Label**: `Real Disaster` or `Not a Disaster`
* **Confidence Score**: Probability of predicted class (0–100%)

---

## Evaluation Metrics

The model performance is evaluated using:

* **Precision**
* **Recall**
* **F1-Score**
* **Accuracy**

Output (from `classification_report`):

```text
              precision    recall  f1-score   support

           0       0.85      0.89      0.87       870
           1       0.84      0.78      0.81       652

    accuracy                           0.84      1522
   macro avg       0.84      0.84      0.84      1522
weighted avg       0.84      0.84      0.84      1522
```

---

## Potential Improvements

To further improve performance and usability:

### Modeling

* Incorporate **transformer fine-tuning** on tweet-specific datasets.
* Explore **multi-modal** inputs (e.g., images, hashtags, time).
* Introduce **class balancing techniques** (SMOTE, focal loss).

### UI/UX

* Provide **interpretability features** using SHAP or LIME.
* Auto-detect keyword and location via NLP extraction (e.g., named entity recognition).
* Batch prediction interface for CSV uploads.

### Deployment

* Wrap the Streamlit app as a **Docker container** for portability.
* Integrate into real-time data pipelines (e.g., Twitter streaming API).

---

## Acknowledgements

* The [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) model from SentenceTransformers.
* [Kaggle Disaster Tweets Dataset](https://www.kaggle.com/competitions/nlp-getting-started) for the original dataset.
* Contributors and open-source community for XGBoost, Streamlit, and Hugging Face.

---
# License

This repository is proprietary and all rights are reserved. No usage, modification, or distribution is allowed without permission.

