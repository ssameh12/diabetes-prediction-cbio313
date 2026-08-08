"""Loading and cleaning for the Pima Indians Diabetes dataset.

Single source of truth: the notebook, train.py and app.py all import from here so the
same cleaning rules are applied everywhere.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# Columns where a recorded 0 is biologically impossible and therefore means "not measured".
# Pregnancies may legitimately be 0, and DiabetesPedigreeFunction / Age are never 0 in this file.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path=DATA_PATH):
    """Read the CSV exactly as distributed, zeros and all."""
    return pd.read_csv(path)


def missing_report(df):
    """How many hidden-missing values each affected column carries."""
    counts = (df[ZERO_AS_MISSING] == 0).sum()
    return pd.DataFrame(
        {"zeros": counts, "percent": (100 * counts / len(df)).round(2)}
    ).sort_values("zeros", ascending=False)


def clean(df):
    """Replace the impossible zeros with NaN. Imputation happens inside the model
    pipeline so that the fill values are learned from training data only."""
    out = df.copy()
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].replace(0, np.nan)
    return out


def load_xy(path=DATA_PATH):
    """Cleaned feature matrix and target vector."""
    df = clean(load_raw(path))
    return df[FEATURES], df[TARGET]
