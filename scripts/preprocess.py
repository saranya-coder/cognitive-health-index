import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the original dataset
df = pd.read_csv("dementia_dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=["Subject ID", "MRI ID", "Visit", "MR Delay"])

# Fill missing values
df["SES"] = df["SES"].fillna(df["SES"].mode()[0])         # Mode for categorical SES
df["MMSE"] = df["MMSE"].fillna(df["MMSE"].median())       # Median for numerical
df["CDR"] = df["CDR"].fillna(df["CDR"].median())

# Encode categorical variables
label_encoders = {}

for col in ["M/F", "Hand", "Group"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save cleaned data
df.to_csv("final_dataset.csv", index=False)
print("Final dataset saved to final_dataset.csv")
