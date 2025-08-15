import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths
data_dir = "processed_data"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Load preprocessed data
X_train_scaled = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
X_test_scaled = pd.read_csv(f"{data_dir}/X_test_scaled.csv")
y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Train and save models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    model_path = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"{name} trained and saved to {model_path}.")

print("All models trained and saved successfully.")
