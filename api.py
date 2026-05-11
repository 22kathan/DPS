"""
api.py
-------
FastAPI backend for the Diabetes Prediction System.
Serves the web interface and handles prediction requests.
"""

import os
import sys
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.explain import load_model_and_scaler, predict_single

# ─── Initialize ───
app = FastAPI(title="Diabetes Prediction System", version="1.0.0")

# Static files and templates
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Load model at startup
MODELS_DIR = os.path.join(BASE_DIR, "models")
model, scaler, meta = None, None, None


@app.on_event("startup")
async def startup():
    global model, scaler, meta
    try:
        model, scaler, meta = load_model_and_scaler(MODELS_DIR)
        print(f"[OK] Model loaded: {meta['best_model']}")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        print("       Run `python src/train.py` first to train the models.")


# ─── Routes ───

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main prediction interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_name": meta["best_model"] if meta else "Not Loaded",
        "metrics": meta["metrics"] if meta else [],
    })


@app.post("/predict")
async def predict(request: Request):
    """Handle prediction requests from the frontend."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Run training first."}
        )

    body = await request.json()

    feature_names = meta["feature_names"]
    try:
        feature_values = [float(body.get(f, 0)) for f in feature_names]
    except (ValueError, TypeError) as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid input: {e}"})

    result = predict_single(model, scaler, feature_values, feature_names)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": meta["best_model"] if meta else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:__name__", host="127.0.0.1", port=8000, reload=True)
