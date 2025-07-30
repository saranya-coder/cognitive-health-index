# src/explore_data.py

import pandas as pd

# Load the dataset
df=pd.read_csv("data/dementia_dataset.csv")

# Show the first few rows
print(" First 5 rows:")
print(df.head())

# Show data types and non-null counts
print("\n Dataset Info:")
print(df.info())

# Check for missing values
print("\n Missing Values:")
print(df.isnull().sum())

# Check class distribution in target column
print("\n Target distribution in 'Group':")
print(df['Group'].value_counts())
