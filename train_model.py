
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# ================= LOAD DATA =================
df = pd.read_csv("health_risk_dataset.csv")

# ================= ENCODE CATEGORICAL COLUMNS =================
categorical_cols = [
    "Gender",
    "Smoking",
    "Salt_intake",
    "Sleep_quality",
    "Alcohol_Consumption"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")
print("Label encoders saved → label_encoders.pkl")

# ================= FEATURE LIST (30 features) =================
features = [
    "Age",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "Family_History_Diabetes",
    "Family_History_Heart_Disease",
    "Existing_BP_Issues",
    "Cholesterol_mg_dL",
    "Physical_Activity_days_per_week",
    "Smoking",
    "Sleep_hours",
    "Junk_food_per_week",
    "Salt_intake",
    "Systolic_BP",
    "Diastolic_BP",
    "Shortness_of_breath",
    "Waist_circumference_cm",
    "Frequent_urination",
    "Excessive_thirst",
    "Body_fat_percent",
    "Sleep_quality",
    "Red_meat_per_week",
    "Fried_food_per_week",
    "Water_intake_liters_per_day",
    "BMI",
    "Alcohol_Consumption",
    "Blood_Sugar_mg_dL",
]

X = df[features].values.astype(np.float32)

# ================= TARGETS =================
targets = [
    "Diabetes_Risk_%",
    "Heart_Disease_Risk_%",
    "Hypertension_Risk_%",
    "Obesity_Risk_%"
]

for disease in targets:
    print(f"\n{'='*50}")
    print(f"Training model for: {disease}")
    print(f"{'='*50}")

    y = df[disease].values.reshape(-1, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = TabNetRegressor(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        verbose=10
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["val"],
        eval_metric=["rmse"],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=256
    )

    save_name = f"{disease}_tabnet_model"
    model.save_model(save_name)
    print(f"Saved → {save_name}.zip")

print("\n✅  All models trained and saved.")