from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import numpy as np
from sentence_transformers import SentenceTransformer

# Make sure 'embeddings' and 'categorical_features' are defined BEFORE this block!

# Stack embeddings and categorical features
X = np.hstack((embeddings, categorical_features))

# Target variable
y = df['target'].values

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model (fix 'eval_metric' typo: 'logloss' instead of 'logless')
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train model
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Print classification report
print(classification_report(y_val, y_pred))
