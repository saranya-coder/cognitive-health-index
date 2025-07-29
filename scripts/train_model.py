import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

import shap
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Drop rows with missing values (or use imputation if preferred)
df = df.dropna()

# Encode 'Group' column (target)
le = LabelEncoder()
df["Group_encoded"] = le.fit_transform(df["Group"])

# Save label mapping for later use
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)

# Features and target
X = df[["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "M/F", "Hand"]]
X.columns = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "M_F", "Hand"]  # rename for consistency
y = df["Group_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "cognitive_model.pkl")
print("Model saved as cognitive_model.pkl")

# ========== SHAP EXPLAINABILITY ==========

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_train)

# Plot summary (bar chart) and save
shap.summary_plot(shap_values, X_train, show=False)
plt.savefig("shap_summary_plot.png")
print("SHAP summary plot saved as shap_summary_plot.png")

# Save SHAP explainer for FastAPI use
joblib.dump(explainer, "shap_explainer.pkl")
print("SHAP explainer saved as shap_explainer.pkl")
