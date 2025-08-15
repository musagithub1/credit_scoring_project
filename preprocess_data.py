import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===== 1. Load dataset =====
df = pd.read_csv("credit_risk_dataset.csv")

# ===== 2. Handle unrealistic ages and make a copy to avoid warnings =====
df = df[df["person_age"] < 100].copy()

# ===== 3. Impute missing values (no inplace to avoid FutureWarning) =====
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].mean())

# ===== 4. Encode categorical features =====
encoders = {}
for column in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# ===== 5. Define features (X) and target (y) =====
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# If target is categorical text, encode it
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# ===== 6. Split data into training and testing sets =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 7. Scale numerical features =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 8. Create output folder =====
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# ===== 9. Save processed data =====
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{output_dir}/X_test_scaled.csv", index=False)
pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)

print(f"Data preprocessing complete. Files saved in '{output_dir}' folder.")
