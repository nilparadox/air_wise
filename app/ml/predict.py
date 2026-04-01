from __future__ import annotations

import joblib
import pandas as pd

from app.core.config import MODEL_PATH, METADATA_PATH


def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def predict_risk(payload: dict) -> dict:
    model, metadata = load_model()
    X = pd.DataFrame([payload], columns=metadata["features"])
    pred = model.predict(X)[0]

    risk_score = float(pred[0])
    safe_minutes = float(pred[1])
    recovery_minutes = float(pred[2])
    end_load = float(pred[3])
    end_inflammation = float(pred[4])
    irritation_prob = float(pred[5])
    oxygen_drop_pct = float(pred[6])

    risk_score = max(0.0, min(100.0, risk_score))
    safe_minutes = max(5.0, min(240.0, safe_minutes))
    recovery_minutes = max(5.0, min(1440.0, recovery_minutes))
    irritation_prob = max(0.0, min(1.0, irritation_prob))
    oxygen_drop_pct = max(0.0, min(12.0, oxygen_drop_pct))
    end_load = max(0.0, end_load)
    end_inflammation = max(0.0, end_inflammation)

    if risk_score < 25:
        band = "low"
    elif risk_score < 50:
        band = "moderate"
    elif risk_score < 75:
        band = "high"
    else:
        band = "severe"

    if risk_score < 25:
        advice = "Current conditions look manageable for most healthy adults."
    elif risk_score < 50:
        advice = "Limit long exposure and avoid unnecessary outdoor exertion."
    elif risk_score < 75:
        advice = "Avoid jogging or exercise outside. Use a protective mask and reduce exposure time."
    else:
        advice = "High-risk exposure conditions. Stay indoors if possible and minimize outdoor time."

    return {
        "risk_score": round(risk_score, 2),
        "risk_band": band,
        "safe_minutes": round(safe_minutes, 1),
        "recovery_minutes": round(recovery_minutes, 1),
        "lung_load": round(end_load, 3),
        "inflammation_score": round(end_inflammation, 3),
        "irritation_probability": round(irritation_prob, 4),
        "oxygen_drop_pct": round(oxygen_drop_pct, 2),
        "advice": advice,
    }
