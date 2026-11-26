"""Inference API (FastAPI). Run: uvicorn pisdr.api:app --app-dir src
The API serves the trained model artifact if present; without one it returns
a clear 503 rather than a fabricated prediction."""
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PISDR Inference API", version="0.1.0")
MODEL_PATH = Path("models/model.pkl")
_bundle = None


class Shipment(BaseModel):
    scheduled_days: float
    port_congestion: float
    disruption_window: float
    order_value: float
    units: float
    month: float
    dow: float


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "No trained model artifact. Run the training pipeline first.")
        with open(MODEL_PATH, "rb") as fh:
            _bundle = pickle.load(fh)
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.post("/predict")
def predict(shipment: Shipment):
    bundle = _load()
    import pandas as pd
    X = pd.DataFrame([shipment.model_dump()])
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    uncertain = bool((prob < bundle["conformal_q"]) and ((1 - prob) < bundle["conformal_q"]))
    return {"late_probability": prob, "uncertain": uncertain,
            "evidence": {"conformal_q": bundle["conformal_q"],
                         "model": bundle["model_name"]}}
