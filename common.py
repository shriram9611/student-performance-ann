"""Shared constants and helpers.

Nothing in this file imports TensorFlow, so the data / metric code can be
tested quickly and reused by the notebook, train.py, the app and the API.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "student_performance.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
META_PATH = MODEL_DIR / "metadata.json"

SEED = 42
ID_COLUMN = "student_id"
TARGET = "performance_category"
CLASSES = ["Low", "Medium", "High"]          # index 0, 1, 2

# One entry per input feature (all 9 numeric columns of the dataset).
# min / max are the ranges seen in the data; they are used for input validation
# and for the sliders / number boxes in the web app.
FEATURE_SPECS = [
    dict(name="age", label="Age (years)", min=15, max=30, default=20, step=1),
    dict(name="attendance_pct", label="Attendance (%)", min=0, max=100, default=80, step=0.1),
    dict(name="study_hours_per_week", label="Study hours per week", min=0, max=60, default=15, step=0.5),
    dict(name="previous_score", label="Previous score (0-100)", min=0, max=100, default=67, step=0.1),
    dict(name="assignment_completion_pct", label="Assignment completion (%)", min=0, max=100, default=75, step=0.1),
    dict(name="internal_assessment_score", label="Internal assessment (0-100)", min=0, max=100, default=65, step=0.1),
    dict(name="participation_score", label="Participation (0-10)", min=0, max=10, default=6, step=0.1),
    dict(name="sleep_hours_per_day", label="Sleep hours per day", min=0, max=14, default=7, step=0.1),
    dict(name="extracurricular_hours_per_week", label="Extracurricular hours per week", min=0, max=40, default=5, step=0.5),
    dict(name="failed_subjects", label="Failed subjects (count)", min=0, max=10, default=1, step=1),
]
FEATURES = [f["name"] for f in FEATURE_SPECS]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def data_summary(df: pd.DataFrame) -> dict:
    """Step 1 - data understanding in one dictionary."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_student_ids": int(df[ID_COLUMN].duplicated().sum()),
        "class_distribution": df[TARGET].value_counts().to_dict(),
    }


def prepare_data(df: pd.DataFrame):
    """Step 2 - clean, encode labels, split 70/15/15 (stratified) and scale.

    The scaler is fitted on the training split ONLY, so no information from
    validation / test data leaks into training.
    """
    df = df.drop_duplicates().dropna().copy()
    X = df[FEATURES].astype("float32")
    y = df[TARGET].map({c: i for i, c in enumerate(CLASSES)}).astype("int64")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    scaler = StandardScaler().fit(X_train)
    return {
        "X_train": scaler.transform(X_train).astype("float32"),
        "X_val": scaler.transform(X_val).astype("float32"),
        "X_test": scaler.transform(X_test).astype("float32"),
        "y_train": y_train.to_numpy(), "y_val": y_val.to_numpy(), "y_test": y_test.to_numpy(),
        "scaler": scaler,
    }


def evaluate(y_true, y_pred) -> dict:
    """Step 7 - accuracy, macro precision / recall / F1, per-class scores, confusion matrix."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "per_class": {
            CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
