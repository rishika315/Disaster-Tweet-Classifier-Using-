import os
os.environ["USE_TF"] = "0"  # <--- Force disable TensorFlow

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

model = joblib.load('xgb_disaster_model.joblib')
onehot_columns = joblib.load('onehot_columns.joblib')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Disaster Tweet Classifier")

tweet_text = st.text_area("Enter tweet text:")
keyword = st.text_input("Enter keyword (or leave blank):")
location = st.text_input("Enter location (or leave blank):")

def preprocess_input(text, keyword, location):
    
    keyword = keyword.strip() if keyword else 'unknown'
    location = location.strip() if location else 'unknown'

    input_df = pd.DataFrame(columns=onehot_columns)
    input_df.loc[0] = 0  # initialize zeros

    kw_col = f"keyword_{keyword}"
    loc_col = f"location_{location}"

    if kw_col in onehot_columns:
        input_df.at[0, kw_col] = 1
    if loc_col in onehot_columns:
        input_df.at[0, loc_col] = 1

    embedding = embedder.encode([text])[0]

    features = np.hstack((embedding, input_df.values.flatten()))

    return features.reshape(1, -1)

if st.button("Predict"):
    if not tweet_text.strip():
        st.warning("Please enter tweet text to classify.")
    else:
        features = preprocess_input(tweet_text, keyword, location)
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][pred]

        label = "Real Disaster" if pred == 1 else "Not a Disaster"
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: {proba:.2%}")


