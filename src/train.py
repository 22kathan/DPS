"""
train.py
---------
Mini Project — Diabetes Prediction System
Trains lightweight ML models on the preprocessed diabetes dataset:
  - KNN, Random Forest, SVM, Naive Bayes


Saves the best model along with the scaler for deployment.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess_pipeline

# ────────────────────── Model Definitions ──────────────────────

def get_models():
    """Return dictionary of lightweight ML classifiers."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
    }





# ────────────────────── Evaluation ──────────────────────

def evaluate_model(name, model, X_test, y_test):
    """Evaluate a trained model and return a metrics dictionary."""
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "predict"):
        y_proba = model.predict(X_test).ravel()
    else:
        y_proba = None

    y_pred = (y_proba >= 0.5).astype(int) if y_proba is not None else model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba) * 100, 2) if y_proba is not None else None,
    }
    return metrics, y_pred


def print_results_table(all_metrics):
    """Pretty-print a comparison table of all models."""
    print("\n" + "=" * 80)
    print("  MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    header = f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}"
    print(header)
    print("-" * 80)
    for m in sorted(all_metrics, key=lambda x: x["Accuracy"], reverse=True):
        auc_str = f"{m['ROC-AUC']:>9.2f}%" if m["ROC-AUC"] else "     N/A"
        print(f"{m['Model']:<18} {m['Accuracy']:>9.2f}% {m['Precision']:>9.2f}% "
              f"{m['Recall']:>9.2f}% {m['F1-Score']:>9.2f}% {auc_str}")
    print("=" * 80)


# ────────────────────── Main Training Pipeline ──────────────────────

def train_all(csv_path: str, models_dir: str):
    """
    Mini-project training pipeline:
      1. Preprocess data (80:20 split, SMOTE, scaling)
      2. Train traditional classifiers
      3. Evaluate and compare
      4. Save the best model and scaler
    """
    os.makedirs(models_dir, exist_ok=True)

    # -- Preprocess --
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(csv_path)

    all_metrics = []
    trained_models = {}

    # -- Traditional Models --
    print("\n" + "-" * 60)
    print("  Training: Lightweight Classifiers")
    print("-" * 60)
    for name, model in get_models().items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        metrics, _ = evaluate_model(name, model, X_test, y_test)
        metrics["Time (s)"] = round(elapsed, 3)
        all_metrics.append(metrics)
        trained_models[name] = model
        print(f"  [OK] {name:<16} -> Accuracy: {metrics['Accuracy']:.2f}%  ({elapsed:.3f}s)")



    # -- Results --
    print_results_table(all_metrics)

    # -- Save best model --
    best = max(all_metrics, key=lambda x: x["Accuracy"])
    print(f"\n[BEST] Best Model: {best['Model']} with {best['Accuracy']:.2f}% accuracy")

    # Save scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save metadata
    meta_path = os.path.join(models_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "best_model": best["Model"],
            "feature_names": feature_names,
            "metrics": all_metrics
        }, f, indent=2)

    # Save the best model
    if best["Model"] in trained_models:
        model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(trained_models[best["Model"]], model_path)
        print(f"[SAVE] Saved best model  -> {model_path}")

    print(f"[SAVE] Saved scaler      -> {scaler_path}")
    print(f"[SAVE] Saved metadata    -> {meta_path}")
    print("\n[DONE] Training pipeline complete!")

    return all_metrics


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data", "diabetes.csv")
    models_dir = os.path.join(base_dir, "models")

    if not os.path.exists(csv_path):
        print("[WARN] Dataset not found. Generating synthetic dataset first...")
        exec(open(os.path.join(base_dir, "data", "generate_dataset.py")).read())

    train_all(csv_path, models_dir)
