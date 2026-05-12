import os
import joblib
import m2cgen as m2c
import json

base_dir = r"c:\Users\gadhi\OneDrive\Desktop\DPS"
models_dir = os.path.join(base_dir, "models")
static_dir = os.path.join(base_dir, "app", "static")

model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

# Export model to JS
code = m2c.export_to_javascript(model)

# Extract scaler info
mean_vals = list(scaler.mean_)
scale_vals = list(scaler.scale_)

js_content = f"""
// Auto-generated Machine Learning Model (m2cgen)
// Scaler Info
const scalerMean = {json.dumps(mean_vals)};
const scalerScale = {json.dumps(scale_vals)};

function scaleFeatures(features) {{
    let scaled = [];
    for (let i = 0; i < features.length; i++) {{
        let v = (features[i] - scalerMean[i]) / scalerScale[i];
        scaled.push(v);
    }}
    return scaled;
}}

{code}

function predict_single(feature_values, feature_names) {{
    const scaled = scaleFeatures(feature_values);
    
    // score function returns an array of scores for each class (for RandomForest)
    // for binary classification, it usually returns [score_class_0, score_class_1]
    const scores = score(scaled);
    
    // Softmax / Normalize to get probabilities (for Random Forest from m2cgen, 
    // the output is the raw votes, so we divide by the total number of trees)
    // Wait, m2cgen for RandomForestClassifier returns the average probability directly.
    let prob0 = scores[0];
    let prob1 = scores[1];
    
    // Normalize just in case
    let probability = prob1 / (prob0 + prob1);
    
    let prediction = probability >= 0.5 ? 1 : 0;
    
    let risk_level = "High Risk";
    let risk_color = "#ff1744";
    if (probability < 0.30) {{
        risk_level = "Low Risk";
        risk_color = "#00e676";
    }} else if (probability < 0.60) {{
        risk_level = "Moderate Risk";
        risk_color = "#ffab00";
    }}
    
    // Simple SHAP-like deviation importance
    let contributions = [];
    for (let i = 0; i < feature_names.length; i++) {{
        let z_score = (feature_values[i] - scalerMean[i]) / scalerScale[i];
        let impact_val = parseFloat((z_score * 0.1).toFixed(4));
        contributions.push({{
            feature: feature_names[i],
            value: feature_values[i],
            shap_value: impact_val,
            impact: impact_val > 0 ? "increases risk" : "decreases risk"
        }});
    }}
    contributions.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    
    return {{
        prediction: prediction,
        probability: parseFloat((probability * 100).toFixed(2)),
        risk_level: risk_level,
        risk_color: risk_color,
        label: prediction === 1 ? "Diabetic" : "Non-Diabetic",
        explanations: contributions
    }};
}}
"""

with open(os.path.join(static_dir, "model_inference.js"), "w", encoding="utf-8") as f:
    f.write(js_content)

print("Exported model to app/static/model_inference.js")
