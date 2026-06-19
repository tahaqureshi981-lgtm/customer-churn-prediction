import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_clean(path):
    df = pd.read_csv(path)

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges dtype bug
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill nulls in TotalCharges (new customers, tenure=0)
    df["TotalCharges"].fillna(0, inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_features(df):
    df = df.copy()

    # Binary columns — map directly
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "No phone service": 0, "No internet service": 0}
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "MultipleLines", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies", "PaperlessBilling"]

    for col in binary_cols:
        df[col] = df[col].replace(binary_map)
        # Catch any remaining string values
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Multi-class categorical — one-hot encode
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df


def feature_engineer(df):
    df = df.copy()

    # Charge per month of tenure (avoid div by zero)
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # High value flag
    df["is_high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # Long-term customer flag
    df["is_long_term"] = (df["tenure"] > 24).astype(int)

    return df


def split_and_scale(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # SMOTE — fix class imbalance on training set only
    print(f"Before SMOTE — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"After  SMOTE — 0: {(y_train_res==0).sum()}, 1: {(y_train_res==1).sum()}")

    # Save scaler + feature names for inference
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_names.pkl")
    print("✅ Saved scaler.pkl and feature_names.pkl")

    return X_train_res, X_test, y_train_res, y_test


def run():
    df = load_and_clean(DATA_PATH)
    df = encode_features(df)
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test = split_and_scale(df)

    print(f"\nTrain shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")
    print(f"Features    : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run()
