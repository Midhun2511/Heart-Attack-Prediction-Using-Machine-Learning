import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(r"F:\heart\csv\heart_attack_prediction_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# -----------------------------
# Drop useless column
# -----------------------------
df = df.drop("Patient ID", axis=1)

# -----------------------------
# Encode categorical columns
# -----------------------------
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, "encoders.pkl")

# -----------------------------
# Split Features & Target
# -----------------------------
X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]

# Save columns
joblib.dump(X.columns.tolist(), "columns.pkl")

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# -----------------------------
# Best Model
# -----------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("Best Model:", best_model_name)

joblib.dump(best_model, "best_model.pkl")

print("✅ Training Completed!")