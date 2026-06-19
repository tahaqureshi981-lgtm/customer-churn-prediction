import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier

from src.preprocess import load_and_clean, encode_features, feature_engineer, split_and_scale

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = "data/telco_churn.csv"


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric="logloss", verbosity=0),
    }


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stay','Churn'])}")

    return {"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc,
            "y_pred": y_pred, "y_prob": y_prob}


def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay", "Churn"],
                yticklabels=["Stay", "Churn"])
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fname = f"{REPORTS_DIR}/cm_{name.lower().replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"✅ Saved: {fname}")


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        top15 = imp.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        top15.plot(kind="barh", ax=ax, color="#4C9BE8")
        ax.set_title("Top 15 Feature Importances (XGBoost)", fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=150)
        plt.close()
        print("✅ Saved: feature_importance.png")


def plot_model_comparison(results):
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("y_pred", "y_prob")} for r in results])
    df = df.set_index("model")
    fig, ax = plt.subplots(figsize=(8, 4))
    df[["accuracy", "f1", "roc_auc"]].plot(kind="bar", ax=ax,
                                            color=["#4C9BE8", "#E8614C", "#50C878"],
                                            edgecolor="white")
    ax.set_title("Model Comparison", fontweight="bold", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticklabels(df.index, rotation=15)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print("✅ Saved: model_comparison.png")


def run():
    # Load + preprocess
    df = load_and_clean(DATA_PATH)
    df = encode_features(df)
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test = split_and_scale(df)

    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")

    models   = get_models()
    results  = []
    best_auc = 0
    best_model = None
    best_name  = ""

    for name, model in models.items():
        print(f"\n⏳ Training {name}...")
        model.fit(X_train, y_train)
        res = evaluate(name, model, X_test, y_test)
        results.append(res)
        plot_confusion_matrix(name, y_test, res["y_pred"])

        joblib.dump(model, f"{MODELS_DIR}/{name.lower()}.pkl")
        print(f"✅ Saved: models/{name.lower()}.pkl")

        if res["roc_auc"] > best_auc:
            best_auc   = res["roc_auc"]
            best_model = model
            best_name  = name

    # Feature importance for XGBoost
    xgb_model = joblib.load(f"{MODELS_DIR}/xgboost.pkl")
    plot_feature_importance(xgb_model, feature_names)

    # Comparison chart
    plot_model_comparison(results)

    # Save best model separately
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(best_name,  f"{MODELS_DIR}/best_model_name.pkl")
    print(f"\n🏆 Best model: {best_name} (ROC-AUC: {best_auc:.4f})")
    print(f"✅ Saved: models/best_model.pkl")

    return best_model, best_name


if __name__ == "__main__":
    run()
