"""Shared data loading and modelling code for the Telco churn project."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_FILE = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def load_data(path=DATA_FILE):
    """Load the Telco dataset and drop rows with a missing TotalCharges."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle "
            "(Telco Customer Churn) and place it in the 'data' folder."
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna(subset=["TotalCharges"])


def split_data(df):
    """Return a stratified 80/20 train/test split (X_train, X_test, y_train, y_test)."""
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(X):
    """Scale numeric columns and one-hot encode categorical ones."""
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns
    return ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numeric_features),
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_models(X):
    """Return both pipelines. Each gets its own preprocessor instance."""
    return {
        "Logistic regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(model, X_test, y_test):
    """Return the headline metrics for a fitted pipeline."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Churn precision": precision_score(y_test, predictions, zero_division=0),
        "Churn recall": recall_score(y_test, predictions, zero_division=0),
        "Churn F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Average precision": average_precision_score(y_test, probabilities),
    }


def train_and_evaluate(df):
    """Fit both models on the training split and return a metrics table."""
    X_train, X_test, y_train, y_test = split_data(df)
    results = {}
    for name, model in build_models(X_train).items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
    return pd.DataFrame(results).T
