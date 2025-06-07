import os
os.environ["USE_TF"] = "0"  # <--- Force disable TensorFlow

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv('tweets.csv')

df['location'].fillna('unknown', inplace=True)
df = pd.get_dummies(df, columns=['keyword', 'location'])

model_embed = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings for tweet texts...")
embeddings = model_embed.encode(df['text'].tolist(), show_progress_bar=True)

categorical_features = df.drop(columns=['id', 'text', 'target']).values
onehot_columns = df.drop(columns=['id', 'text', 'target']).columns.tolist()

# Save to file
joblib.dump(onehot_columns, 'onehot_columns.joblib')
print("One-hot columns saved as onehot_columns.joblib")
X = np.hstack((embeddings, categorical_features))

y = df['target'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
print("Training XGBoost model...")
model.fit(X_train, y_train)
joblib.dump(model, 'xgb_disaster_model.joblib')
print("Model saved as xgb_disaster_model.joblib")

y_pred = model.predict(X_val)
print("Classification Report:\n")
print(classification_report(y_val, y_pred))
