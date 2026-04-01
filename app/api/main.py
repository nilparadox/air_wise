from __future__ import annotations

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import STATIC_DIR, TEMPLATES_DIR, MODEL_PATH
from app.ml.predict import predict_risk
from app.ml.profile import estimate_baseline_lung, sensitivity_label
from app.api.aqi_client import fetch_open_meteo_bundle

app = FastAPI(title="AirWise AI")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def default_form_data() -> dict:
    return {
        "city": "",
        "age": 30,
        "exposure_min": 45,
        "activity": "walk",
        "asthma": 0,
        "smoker": 0,
        "mask_type": "none",
    }


def build_chart_data(result: dict | None, air_result: dict | None, form_data: dict) -> dict | None:
    if not result or "risk_score" not in result:
        return None

    exposure_min = float(form_data.get("exposure_min", 45))
    safe_minutes = float(result.get("safe_minutes", 0))
    recovery_minutes = float(result.get("recovery_minutes", 0))
    risk_score = float(result.get("risk_score", 0))
    irritation_probability = float(result.get("irritation_probability", 0))
    lung_load = float(result.get("lung_load", 0))
    inflammation_score = float(result.get("inflammation_score", 0))
    oxygen_drop_pct = float(result.get("oxygen_drop_pct", 0))

    exposure_pct = min(100.0, (exposure_min / max(safe_minutes, 1.0)) * 100.0)
    safe_pct = min(100.0, (safe_minutes / 240.0) * 100.0)
    recovery_pct = min(100.0, (recovery_minutes / 1440.0) * 100.0)
    lung_pct = min(100.0, lung_load * 18.0)
    inflam_pct = min(100.0, inflammation_score * 55.0)
    irritation_pct = min(100.0, irritation_probability * 100.0)
    oxygen_pct = min(100.0, oxygen_drop_pct * 12.5)

    air_chart = None
    if air_result:
        pm25 = float(air_result.get("pm25") or 0)
        pm10 = float(air_result.get("pm10") or 0)
        no2 = float(air_result.get("no2") or 0)
        o3 = float(air_result.get("o3") or 0)
        air_chart = {
            "pm25_pct": min(100.0, pm25 / 2.5),
            "pm10_pct": min(100.0, pm10 / 4.0),
            "no2_pct": min(100.0, no2 / 1.5),
            "o3_pct": min(100.0, o3 / 2.0),
        }

    return {
        "risk_pct": risk_score,
        "safe_pct": safe_pct,
        "recovery_pct": recovery_pct,
        "exposure_pct": exposure_pct,
        "lung_pct": lung_pct,
        "inflam_pct": inflam_pct,
        "irritation_pct": irritation_pct,
        "oxygen_pct": oxygen_pct,
        "air_chart": air_chart,
    }


def render_home(
    request: Request,
    result=None,
    air_result=None,
    profile_result=None,
    form_data=None,
):
    if form_data is None:
        form_data = default_form_data()

    chart_data = build_chart_data(result, air_result, form_data)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result,
            "air_result": air_result,
            "profile_result": profile_result,
            "model_ready": MODEL_PATH.exists(),
            "form_data": form_data,
            "chart_data": chart_data,
        },
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_home(request)


@app.post("/predict-live", response_class=HTMLResponse)
def predict_live(
    request: Request,
    city: str = Form(...),
    age: int = Form(...),
    exposure_min: float = Form(...),
    activity: str = Form(...),
    asthma: int = Form(...),
    smoker: int = Form(...),
    mask_type: str = Form(...),
):
    form_data = {
        "city": city,
        "age": age,
        "exposure_min": exposure_min,
        "activity": activity,
        "asthma": asthma,
        "smoker": smoker,
        "mask_type": mask_type,
    }

    try:
        air = fetch_open_meteo_bundle(city)

        pm25 = float(air["pm25"] if air["pm25"] is not None else 50.0)
        pm10 = float(air["pm10"] if air["pm10"] is not None else pm25 * 1.5)
        temp_c = float(air["temp_c"] if air["temp_c"] is not None else 30.0)
        humidity = float(air["humidity"] if air["humidity"] is not None else 60.0)

        baseline_lung = estimate_baseline_lung(
            age=age,
            asthma=asthma,
            smoker=smoker,
            activity=activity,
        )

        payload = {
            "age": age,
            "pm25": pm25,
            "pm10": pm10,
            "temp_c": temp_c,
            "humidity": humidity,
            "exposure_min": exposure_min,
            "activity": activity,
            "asthma": asthma,
            "smoker": smoker,
            "mask_type": mask_type,
            "baseline_lung": baseline_lung,
        }

        result = predict_risk(payload)

        if air.get("weather_note"):
            result["advice"] = f'{result["advice"]} Note: {air["weather_note"]}'

        profile_result = {
            "baseline_lung": baseline_lung,
            "sensitivity": sensitivity_label(baseline_lung),
        }

        return render_home(
            request=request,
            result=result,
            air_result=air,
            profile_result=profile_result,
            form_data=form_data,
        )
    except Exception:
        return render_home(
            request=request,
            result={
                "advice": "Live data could not be fetched right now. Please try again in a minute."
            },
            air_result=None,
            profile_result=None,
            form_data=form_data,
        )


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
