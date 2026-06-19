from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn based on their profile.",
    version="1.0.0"
)

# ── Load model artifacts ──────────────────────────────────────────────────────
MODELS_DIR = "models"

try:
    model        = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    scaler       = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    model_name   = joblib.load(f"{MODELS_DIR}/best_model_name.pkl")
    print(f"✅ Loaded model: {model_name}")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


# ── Request schema ────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: int             = Field(..., description="0=Female, 1=Male")
    SeniorCitizen: int      = Field(..., description="0=No, 1=Yes")
    Partner: int            = Field(..., description="0=No, 1=Yes")
    Dependents: int         = Field(..., description="0=No, 1=Yes")
    tenure: int             = Field(..., description="Months with company")
    PhoneService: int       = Field(..., description="0=No, 1=Yes")
    MultipleLines: int      = Field(..., description="0=No, 1=Yes")
    OnlineSecurity: int     = Field(..., description="0=No, 1=Yes")
    OnlineBackup: int       = Field(..., description="0=No, 1=Yes")
    DeviceProtection: int   = Field(..., description="0=No, 1=Yes")
    TechSupport: int        = Field(..., description="0=No, 1=Yes")
    StreamingTV: int        = Field(..., description="0=No, 1=Yes")
    StreamingMovies: int    = Field(..., description="0=No, 1=Yes")
    PaperlessBilling: int   = Field(..., description="0=No, 1=Yes")
    MonthlyCharges: float   = Field(..., description="Monthly charge amount")
    TotalCharges: float     = Field(..., description="Total charges to date")
    InternetService_Fiber_optic: int = Field(..., description="0=No, 1=Yes")
    InternetService_No: int          = Field(..., description="0=No, 1=Yes")
    Contract_One_year: int           = Field(..., description="0=No, 1=Yes")
    Contract_Two_year: int           = Field(..., description="0=No, 1=Yes")
    PaymentMethod_Credit_card__automatic_: int = Field(..., description="0=No, 1=Yes")
    PaymentMethod_Electronic_check: int        = Field(..., description="0=No, 1=Yes")
    PaymentMethod_Mailed_check: int            = Field(..., description="0=No, 1=Yes")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": 1,
                "SeniorCitizen": 0,
                "Partner": 0,
                "Dependents": 0,
                "tenure": 2,
                "PhoneService": 1,
                "MultipleLines": 0,
                "OnlineSecurity": 0,
                "OnlineBackup": 0,
                "DeviceProtection": 0,
                "TechSupport": 0,
                "StreamingTV": 0,
                "StreamingMovies": 0,
                "PaperlessBilling": 1,
                "MonthlyCharges": 70.0,
                "TotalCharges": 140.0,
                "InternetService_Fiber_optic": 1,
                "InternetService_No": 0,
                "Contract_One_year": 0,
                "Contract_Two_year": 0,
                "PaymentMethod_Credit_card__automatic_": 0,
                "PaymentMethod_Electronic_check": 1,
                "PaymentMethod_Mailed_check": 0
            }
        }


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    model_used: str
    explanation: str


# ── Helper ────────────────────────────────────────────────────────────────────
def build_features(data: CustomerInput) -> pd.DataFrame:
    base = data.dict()

    # Engineered features (must match preprocess.py)
    monthly = base["MonthlyCharges"]
    tenure  = base["tenure"]
    base["charges_per_tenure"] = monthly / (tenure + 1)
    base["is_high_value"]      = int(monthly > 64.76)   # median from training
    base["is_long_term"]       = int(tenure > 24)

    df = pd.DataFrame([base])

    # Reindex to match training feature order — fill missing with 0
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numeric cols
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def get_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def get_explanation(prob: float, data: CustomerInput) -> str:
    factors = []
    if data.Contract_One_year == 0 and data.Contract_Two_year == 0:
        factors.append("month-to-month contract")
    if data.tenure < 12:
        factors.append("short tenure")
    if data.MonthlyCharges > 65:
        factors.append("high monthly charges")
    if data.OnlineSecurity == 0:
        factors.append("no online security")
    if data.TechSupport == 0:
        factors.append("no tech support")

    if not factors:
        return "No major churn risk factors detected."
    return f"Risk factors: {', '.join(factors)}."


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running", "model": model_name}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model": model_name}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        features = build_features(customer)
        pred     = int(model.predict(features)[0])
        prob     = float(model.predict_proba(features)[0][1])

        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(prob, 4),
            risk_level=get_risk_level(prob),
            model_used=model_name,
            explanation=get_explanation(prob, customer)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        "model_name": model_name,
        "n_features": len(feature_names),
        "feature_names": feature_names
    }
