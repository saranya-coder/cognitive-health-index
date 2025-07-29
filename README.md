# Cognitive Health Index - FastAPI App for Dementia Prediction and SHAP Explanations

This project builds a machine learning system to predict a person’s cognitive health status—whether they’re currently nondemented, diagnosed with dementia, or have converted from healthy to demented over time. It uses structured clinical and imaging-derived data as input, trains an XGBoost model for classification, and serves predictions through a FastAPI backend. Every prediction is also explained using SHAP, so we can see exactly which features influenced the outcome. The result is a fully functional and interpretable ML pipeline designed for real-time use.

# 1. Project Structure

        cognitive-health-index/
        │
        ├── app/                 # FastAPI app for prediction and SHAP
        │ └── app.py
        │
        ├── data/                # Raw and processed datasets
        │ ├── dementia_dataset.csv
        │ └── final_dataset.csv
        │
        ├── models/              # Saved ML model and SHAP explainer
        │ ├── cognitive_model.pkl
        │ └── shap_explainer.pkl
        │
        ├── scripts/             # Training and preprocessing scripts
        │ ├── preprocess.py
        │ ├── explore_data.py
        │ └── train_model.py
        │
        ├── plots/                # SHAP plots 
        │ ├── shap_plot.png
        │ └── shap_summary_plot.png
        │
        ├── requirements.txt      # Dependency list
        └── README.md # Project overview

# 2. Create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


# 3.Install dependencies

pip install -r requirements.txt


# 4. Run the app

uvicorn app.app:app --reload


# Example Input:

    {
    "Age": 68,
    "EDUC": 16,
    "SES": 3,
    "MMSE": 29,
    "CDR": 0.0,
    "eTIV": 1500,
    "nWBV": 0.78,
    "ASF": 1.05,
    "M_F": 1,
    "Hand": 0
    }


# Visualizations

    shap_summary_plot.png: Feature impact across the entire dataset

    shap_plot.png: Local SHAP explanation for a single prediction


