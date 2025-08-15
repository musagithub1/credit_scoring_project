import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ===== 1. Paths =====
data_dir = "processed_data"
model_dir = "models"

# ===== 2. Load test data =====
X_test_scaled = pd.read_csv(f"{data_dir}/X_test_scaled.csv")
y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

# ===== 3. Evaluate each model =====
for model_file in os.listdir(model_dir):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(model_dir, model_file)
        
        # Load model
        model = joblib.load(model_path)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        
        print(f"\n📊 Model: {model_file}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
