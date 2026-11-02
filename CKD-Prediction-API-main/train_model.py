import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load dataset
df = pd.read_csv("data/kidney_disease.csv")

print("Dataset shape:", df.shape)


# 2. Clean column names
df.columns = df.columns.str.strip()


# 3. Replace ? with missing values
df = df.replace("?", pd.NA)


# 4. Remove extra spaces from text values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )


# 5. Separate features and target
X = df.drop("class", axis=1)
y = df["class"]


# 6. Clean target values
y = y.str.strip().str.lower()

# Convert target into numbers
y = y.map({
    "ckd": 1,
    "notckd": 0
})


# Remove rows where target is missing
valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows]


# 7. Numerical columns
numeric_columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc"
]


# 8. Categorical columns
categorical_columns = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane"
]


# 9. Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 10. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 11. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# 12. Create ML pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# 13. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 14. Train model
model_pipeline.fit(X_train, y_train)

print("\nModel training completed!")


# 15. Make predictions
y_pred = model_pipeline.predict(X_test)


# 16. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# 17. Display results
print("\nModel Evaluation")
print("------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# 18. Create model folder
os.makedirs("model", exist_ok=True)


# 19. Save model
joblib.dump(
    model_pipeline,
    "model/ckd_model.joblib"
)


print("\nModel saved successfully!")
print("Location: model/ckd_model.joblib")