# Diabetes Prediction System (DPS) - Mini Project

A lightweight ML-powered diabetes risk assessment tool with a web interface.

## Models Used

| Category | Models |
|---|---|
| Traditional ML | KNN, Random Forest, SVM, Naive Bayes |


## Project Structure

```
DPS/
├── data/
│   ├── generate_dataset.py   # Synthetic dataset generator
│   └── diabetes.csv           # Dataset
├── src/
│   ├── preprocess.py          # Imputation, SMOTE, 80:20 split, scaling
│   ├── train.py               # Training pipeline (4 models)
│   └── explain.py             # Prediction & feature importance
├── models/                    # Saved models & scaler
├── app/
│   ├── api.py                 # FastAPI backend
│   ├── templates/index.html   # Web interface
│   └── static/                # CSS & JS
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python data/generate_dataset.py

# 3. Train all models
python src/train.py

# 4. Launch the web app
cd app
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## Key Features

- **80:20 Train-Test Split** for robust evaluation
- **SMOTE** for handling class imbalance
- **Mutual Information** feature importance
- **Feature contribution analysis** on every prediction
- **Real-time predictions** via a web interface

## Disclaimer

This tool is for **educational purposes only**. It is not a substitute for professional medical diagnosis.
