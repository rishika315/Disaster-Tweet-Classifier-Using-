# 🚨 Disaster Tweet Classification System Using Transformers

---

During disasters, social media often becomes a noisy mix of real alerts, public reactions, and irrelevant metaphorical language. The objective of this system is to distinguish between tweets that actually report disaster-related events and those that do not. This aids in:

* Improving **disaster response coordination**
* Reducing **manual filtering effort** for responders
* Enhancing **real-time situational awareness**

---

## 🧪 Solution Architecture

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

## 🛠 Technologies Used

| Library                 | Purpose                               |
| ----------------------- | ------------------------------------- |
| `pandas`, `numpy`       | Data handling and transformation      |
| `xgboost`               | Gradient boosting classifier          |
| `scikit-learn`          | Model evaluation and data splitting   |
| `sentence-transformers` | Pre-trained NLP embeddings (`MiniLM`) |
| `streamlit`             | Interactive front-end for prediction  |
| `joblib`                | Model and schema serialization        |

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-org/disaster-tweet-classifier.git
cd disaster-tweet-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### requirements.txt:

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

## 🧠 Training the Model

The training logic is encapsulated in `train_model.py`.

### Steps:

1. Load raw data (`tweets.csv`)
2. Preprocess `keyword` and `location` using one-hot encoding
3. Encode `text` using SentenceTransformer (`all-MiniLM-L6-v2`)
4. Concatenate all features
5. Train an XGBoost classifier
6. Save the model and encoding schema

### Run:

```bash
python train_model.py
```

### Output:

* `xgb_disaster_model.joblib` – Trained classifier
* `onehot_columns.joblib` – Column schema for inference

---

## 💻 Web Application

An interactive UI built with Streamlit allows users to test the classifier in real time.

### Launch:

```bash
streamlit run app.py
```

### Features:

* Input tweet text
* Optional: `keyword` and `location`
* Displays:

  * Predicted label (`Real Disaster` or `Not a Disaster`)
  * Prediction confidence (%)

---

## 🔍 Input/Output Schema

### Input Fields:

| Field      | Description                         | Required |
| ---------- | ----------------------------------- | -------- |
| `text`     | Content of the tweet                | ✅ Yes    |
| `keyword`  | Disaster-related keyword            | Optional |
| `location` | Location information (if available) | Optional |

### Output:

* `Prediction`: Disaster-related or not
* `Confidence`: Probability of prediction

---

## 📊 Model Evaluation

Evaluation uses a stratified 80/20 split with `classification_report` from `scikit-learn`.

**Sample Output:**

```
              precision    recall  f1-score   support

           0       0.85      0.89      0.87       870
           1       0.84      0.78      0.81       652

    accuracy                           0.84      1522
   macro avg       0.84      0.84      0.84      1522
weighted avg       0.84      0.84      0.84      1522
```

---

## 🔧 Project Structure

```
.
├── app.py                    # Streamlit web application
├── train_model.py            # Model training pipeline
├── tweets.csv                # Dataset (input)
├── xgb_disaster_model.joblib # Trained model
├── onehot_columns.joblib     # One-hot encoding column list
└── README.md                 # Project documentation
```

---

## 📈 Possible Enhancements

* **Modeling**

  * Fine-tune transformer embeddings on tweet-specific corpora
  * Add TF-IDF or POS-tag features
  * Experiment with deep learning (e.g., LSTM, BERT-based classifier)

* **User Experience**

  * Visualize model decisions using SHAP or LIME
  * Auto-fill `keyword` and `location` using NER/NLP pipelines
  * Add batch upload support (CSV)

* **Deployment**

  * Dockerize the app
  * Integrate with real-time APIs (e.g., Twitter streaming)

---

## 🙏 Acknowledgements

* Dataset inspired by [Kaggle Disaster Tweets Challenge](https://www.kaggle.com/competitions/nlp-getting-started)
* Embedding model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* Thanks to the open-source community for tools like XGBoost, Streamlit, and Sentence Transformers.

---
# License

This repository is proprietary and all rights are reserved. No usage, modification, or distribution is allowed without permission.

