from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

# Load model and explainer
model = joblib.load("models/cognitive_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Class label mapping
label_map = {0: "Nondemented", 1: "Demented", 2: "Converted"}

# Define input schema
class InputFeatures(BaseModel):
    Age: float
    EDUC: float
    SES: float
    MMSE: float
    CDR: float
    eTIV: float
    nWBV: float
    ASF: float
    M_F: int
    Hand: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Cognitive Health Prediction API is running."}

# Prediction endpoint
@app.post("/predict")
def predict(input: InputFeatures):
    data = np.array([[
        input.Age, input.EDUC, input.SES, input.MMSE, input.CDR,
        input.eTIV, input.nWBV, input.ASF, input.M_F, input.Hand
    ]])
    prediction = model.predict(data)[0]
    return {
        "predicted_class": int(prediction),
        "label": label_map[prediction]
    }


@app.post("/explain")
def explain(input: InputFeatures):
    input_data = input.dict()
    input_df = pd.DataFrame([input_data])
    input_array = input_df.to_numpy()

    prediction = model.predict(input_array)[0]
    shap_values = explainer.shap_values(input_array)

    if isinstance(shap_values, list):  # Multiclass
        class_shap = shap_values[int(prediction)][0]
        expected_val = explainer.expected_value[int(prediction)]
    else:
        class_shap = shap_values[0]
        expected_val = explainer.expected_value

    contribution = {
        name: float(np.ravel(val)[0]) if isinstance(val, (np.ndarray, list)) else float(val)
        for name, val in zip(input_df.columns, class_shap)
    }

    expected_val_clean = float(expected_val[0]) if isinstance(expected_val, list) else float(expected_val)

    return {
        "predicted_class": int(prediction),
        "label": label_map[int(prediction)],
        "contribution": contribution,
        "expected_value": expected_val_clean
    }




@app.post("/explain_plot")
def explain_plot(input: InputFeatures):
    input_data = input.dict()
    input_df = pd.DataFrame([input_data])
    input_array = input_df.to_numpy()

    # Predict and get SHAP values
    prediction = model.predict(input_array)[0]
    shap_values = explainer.shap_values(input_array)

    # Handle binary vs multiclass SHAP outputs
    if isinstance(shap_values, list):
        shap_val = shap_values[int(prediction)][0]  # shape: (features,)
        expected_val = explainer.expected_value[int(prediction)]
    else:
        shap_val = shap_values[0]
        expected_val = explainer.expected_value

    # Clean expected_val if needed
    if isinstance(expected_val, list):
        expected_val = expected_val[0]

    shap_val = np.array(shap_val).flatten()
    input_row = input_array[0]

    # Determine feature names
    feature_names = explainer.data.feature_names if hasattr(explainer, "data") and hasattr(explainer.data, "feature_names") else input_df.columns.tolist()

    # Truncate if SHAP thinks there are fewer features
    expected_len = min(len(shap_val), len(input_row), len(feature_names))
    shap_val = shap_val[:expected_len]
    input_row = input_row[:expected_len]
    feature_names = feature_names[:expected_len]

    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_val,
        base_values=expected_val,
        data=input_row,
        feature_names=feature_names
    )

    # Plot and save
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.savefig("plots/shap_plot.png")
    plt.close()

    return FileResponse("plots/shap_plot.png", media_type="image/png")
