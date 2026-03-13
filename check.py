import pandas as pd
import numpy as np
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from pytorch_tabnet.tab_model import TabNetRegressor
from xgboost import XGBRegressor


# ================= LOAD DATA =================
df = pd.read_csv("health_risk_dataset.csv")


# ================= ENCODE CATEGORICAL COLUMNS =================
categorical_cols = [
    "Gender","Smoking","Salt_intake","Sleep_quality","Alcohol_Consumption"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")


# ================= FEATURES =================
features = [
    "Age","Gender","Height_cm","Weight_kg",
    "Family_History_Diabetes","Family_History_Heart_Disease",
    "Existing_BP_Issues","Cholesterol_mg_dL",
    "Physical_Activity_days_per_week","Smoking",
    "Sleep_hours","Junk_food_per_week","Salt_intake",
    "Systolic_BP","Diastolic_BP","Shortness_of_breath",
    "Waist_circumference_cm","Frequent_urination",
    "Excessive_thirst","Body_fat_percent","Sleep_quality",
    "Red_meat_per_week","Fried_food_per_week",
    "Water_intake_liters_per_day","BMI",
    "Alcohol_Consumption","Blood_Sugar_mg_dL"
]

X = df[features].values.astype(np.float32)


# ================= SCALE FEATURES =================
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")


# ================= TARGETS =================
targets = [
    "Diabetes_Risk_%",
    "Heart_Disease_Risk_%",
    "Hypertension_Risk_%",
    "Obesity_Risk_%"
]


# ================= RISK CLASS FUNCTION =================
def risk_to_class(values):
    return np.where(values < 35, 0,
           np.where(values < 65, 1, 2))


# ================= STORE ACCURACIES =================
all_accuracies = {}


# ================= TRAIN MODELS =================
for disease in targets:

    print("\n==============================")
    print(f"Model Result: {disease}")
    print("==============================")

    y = df[disease].values.reshape(-1,1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # ================= TABNET MODEL =================
    tabnet = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=7,
        gamma=1.5,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.01),
        verbose=0
    )

    tabnet.fit(
        X_train, y_train,
        max_epochs=200,
        batch_size=512,
        virtual_batch_size=128
    )


    # ================= XGBOOST MODEL =================
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    xgb.fit(X_train, y_train)


    # ================= PREDICTIONS =================
    tab_pred = tabnet.predict(X_test)
    xgb_pred = xgb.predict(X_test).reshape(-1,1)


    # ================= ENSEMBLE (AVERAGE) =================
    final_pred = (tab_pred + xgb_pred) / 2


    # ================= CLASS CONVERSION =================
    y_test_class = risk_to_class(y_test.flatten())
    pred_class = risk_to_class(final_pred.flatten())


    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_test_class, pred_class)

    print("\nConfusion Matrix:")
    print(cm)


    # ================= ACCURACY =================
    acc = accuracy_score(y_test_class, pred_class)

    all_accuracies[disease] = acc

    print("\nAccuracy:", round(acc*100,2), "%")


    # ================= REPORT =================
    print("\nClassification Report:")
    print(classification_report(y_test_class, pred_class))


# ================= OVERALL ACCURACY =================
overall_accuracy = np.mean(list(all_accuracies.values()))

print("\n===================================")
print("PROJECT MODEL ACCURACY SUMMARY")
print("===================================")

for k,v in all_accuracies.items():
    print(f"{k} : {round(v*100,2)} %")

print("-----------------------------------")
print("Overall System Accuracy :", round(overall_accuracy*100,2), "%")
print("===================================")
