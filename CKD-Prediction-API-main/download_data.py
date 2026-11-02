from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# Fetch CKD dataset from UCI
ckd = fetch_ucirepo(id=336)

# Get features and target
X = ckd.data.features
y = ckd.data.targets

# Combine them
data = pd.concat([X, y], axis=1)

# Create data folder
os.makedirs("data", exist_ok=True)

# Save as CSV
data.to_csv("data/kidney_disease.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())