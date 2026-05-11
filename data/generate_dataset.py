"""
generate_dataset.py
--------------------
Generates a realistic synthetic diabetes dataset modeled after the
Pima Indian Diabetes Dataset for development and testing.

Run this script to create `diabetes.csv` in the data/ directory.
You can replace it later with the real Pima Indian dataset from Kaggle:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Columns:
  Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
  BMI, DiabetesPedigreeFunction, Age, Outcome
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 768  # Same size as Pima dataset

# ---------- helpers ----------
def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)

# ---------- Generate features based on real Pima distributions ----------
pregnancies = clamp(np.random.poisson(3.8, N), 0, 17).astype(int)
age = clamp(np.random.normal(33, 12, N), 21, 81).astype(int)

# Glucose - bimodal: healthy vs diabetic
glucose_healthy = np.random.normal(110, 20, N)
glucose_diabetic = np.random.normal(155, 30, N)
is_high_glucose = np.random.random(N) > 0.65
glucose = np.where(is_high_glucose, glucose_diabetic, glucose_healthy)
glucose = clamp(glucose, 44, 199).astype(int)

blood_pressure = clamp(np.random.normal(72, 12, N), 24, 122).astype(int)
skin_thickness = clamp(np.random.normal(26, 10, N), 7, 99).astype(int)

# Insulin correlated with glucose
insulin = clamp(glucose * np.random.normal(1.1, 0.5, N) + np.random.normal(0, 30, N), 14, 846).astype(int)

bmi = clamp(np.random.normal(32, 7, N), 18.0, 67.1).round(1)
dpf = clamp(np.random.exponential(0.47, N), 0.078, 2.42).round(3)

# ---------- Generate outcome with realistic correlations ----------
# Higher glucose, BMI, age, pregnancies → higher chance of diabetes
risk_score = (
    (glucose - 100) / 60 * 0.40 +
    (bmi - 25) / 15 * 0.25 +
    (age - 25) / 30 * 0.15 +
    (dpf - 0.3) / 0.5 * 0.10 +
    (pregnancies - 2) / 5 * 0.10
)
probability = 1 / (1 + np.exp(-risk_score * 2.5))
outcome = (np.random.random(N) < probability).astype(int)

# ---------- Introduce some zeros to mimic missing data in Pima ----------
for col_arr, pct in [(glucose, 0.007), (blood_pressure, 0.045),
                      (skin_thickness, 0.30), (insulin, 0.49), (bmi, 0.014)]:
    mask = np.random.random(N) < pct
    col_arr[mask] = 0

df = pd.DataFrame({
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
    "Outcome": outcome,
})

output_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
df.to_csv(output_path, index=False)

positive = outcome.sum()
negative = N - positive
print(f"[OK] Generated synthetic diabetes dataset -> {output_path}")
print(f"     Samples: {N}  |  Positive: {positive} ({positive/N*100:.1f}%)  |  Negative: {negative} ({negative/N*100:.1f}%)")
