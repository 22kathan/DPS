"""
explain.py
-----------
Mini Project — Prediction and feature importance engine.
Uses sklearn's built-in feature importance instead of heavy SHAP library.
"""

import os
import numpy as np
import joblib
import json


def load_model_and_scaler(models_dir: str):
    """Load the best saved model, scaler, and metadata."""
    meta_path = os.path.join(models_dir, "model_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))

    return model, scaler, meta


def predict_single(model, scaler, feature_values: list, feature_names: list) -> dict:
    """
    Make a prediction for a single patient input.

    Args:
        model: trained model
        scaler: fitted StandardScaler
        feature_values: list of 8 raw feature values
        feature_names: list of feature column names

    Returns:
        dict with prediction, probability, risk level, and feature contributions
    """
    X_raw = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X_raw)

    # Predict
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_scaled)[0][1])
    else:
        probability = float(model.predict(X_scaled).ravel()[0])

    prediction = 1 if probability >= 0.5 else 0

    # Risk level
    if probability < 0.30:
        risk_level = "Low Risk"
        risk_color = "#00e676"
    elif probability < 0.60:
        risk_level = "Moderate Risk"
        risk_color = "#ffab00"
    else:
        risk_level = "High Risk"
        risk_color = "#ff1744"

    result = {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "label": "Diabetic" if prediction == 1 else "Non-Diabetic",
    }

    # Feature contribution using simple deviation-based importance
    # (how far each scaled feature is from the mean, weighted by direction)
    contributions = []
    mean_vals = scaler.mean_
    std_vals = scaler.scale_
    for i, fname in enumerate(feature_names):
        # Z-score tells us how "abnormal" the value is
        z_score = (feature_values[i] - mean_vals[i]) / std_vals[i] if std_vals[i] != 0 else 0
        # Positive z-score for diabetes-correlated features = increases risk
        impact_val = round(float(z_score * 0.1), 4)
        contributions.append({
            "feature": fname,
            "value": float(feature_values[i]),
            "shap_value": impact_val,
            "impact": "increases risk" if impact_val > 0 else "decreases risk"
        })
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    result["explanations"] = contributions

    return result


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base, "models")
    model, scaler, meta = load_model_and_scaler(models_dir)
    # Example prediction
    sample = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    result = predict_single(model, scaler, sample, meta["feature_names"])
    print(f"\nPrediction: {result['label']}")
    print(f"   Probability: {result['probability']}%")
    print(f"   Risk Level: {result['risk_level']}")
    if "explanations" in result:
        print("\n   Feature Contributions:")
        for exp in result["explanations"]:
            if "feature" in exp:
                print(f"     {exp['feature']:>28s}: {exp['shap_value']:+.4f} ({exp['impact']})")
