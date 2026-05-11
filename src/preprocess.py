"""
preprocess.py
--------------
Data loading, cleaning, feature selection, SMOTE balancing,
and 80:20 train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
import joblib
import os

# Columns that use 0 as a proxy for missing values in the Pima dataset
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the diabetes CSV and return a DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded dataset: {csv_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 values in clinical columns with median imputation."""
    df = df.copy()
    for col in ZERO_IS_MISSING:
        if col in df.columns:
            median_val = df.loc[df[col] != 0, col].median()
            zeros = (df[col] == 0).sum()
            df[col] = df[col].replace(0, median_val)
            if zeros > 0:
                print(f"   [FIX] {col}: imputed {zeros} zero-values -> median ({median_val:.2f})")
    return df


def feature_importance_analysis(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute Mutual Information scores for feature ranking."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    importance = pd.DataFrame({
        "Feature": X.columns,
        "MI_Score": mi_scores
    }).sort_values("MI_Score", ascending=False).reset_index(drop=True)
    print("\n[FEATURES] Feature Importance (Mutual Information):")
    for _, row in importance.iterrows():
        bar = "#" * int(row["MI_Score"] * 50)
        print(f"   {row['Feature']:>28s}: {row['MI_Score']:.4f}  {bar}")
    return importance


def balance_data(X: np.ndarray, y: np.ndarray, method: str = "smote") -> tuple:
    """Apply SMOTE or ADASYN to balance the dataset."""
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    print(f"\n[BALANCE] Class distribution BEFORE balancing: Positive={pos}, Negative={neg}")

    if method.lower() == "adasyn":
        sampler = ADASYN(random_state=42)
    else:
        sampler = SMOTE(random_state=42)

    X_bal, y_bal = sampler.fit_resample(X, y)
    pos_new = (y_bal == 1).sum()
    neg_new = (y_bal == 0).sum()
    print(f"   [OK] AFTER {method.upper()}: Positive={pos_new}, Negative={neg_new}")
    return X_bal, y_bal


def preprocess_pipeline(csv_path: str, balance_method: str = "smote"):
    """
    Full preprocessing pipeline:
      1. Load data
      2. Impute missing values
      3. Feature importance analysis
      4. 80:20 train-test split
      5. Scale features
      6. Balance training data with SMOTE/ADASYN

    Returns:
      X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("=" * 60)
    print("  DIABETES PREDICTION SYSTEM -- Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_data(csv_path)

    # 2. Impute
    df = handle_missing(df)

    # 3. Feature importance
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    feature_importance_analysis(X, y)

    # 4. 80:20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] 80:20 Split -> Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 5. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Balance training data
    X_train_balanced, y_train_balanced = balance_data(
        X_train_scaled, y_train.values, method=balance_method
    )

    print("\n[OK] Preprocessing complete.\n")

    return (
        X_train_balanced, X_test_scaled,
        y_train_balanced, y_test.values,
        scaler, FEATURE_COLS
    )


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
    preprocess_pipeline(data_path)
