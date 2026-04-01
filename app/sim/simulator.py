from __future__ import annotations
import numpy as np
import pandas as pd


def _clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def simulate_health_dataset(n: int = 50000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(8, 80, size=n)
    pm25 = rng.uniform(5, 350, size=n)
    pm10 = rng.uniform(10, 500, size=n)
    temp_c = rng.uniform(5, 45, size=n)
    humidity = rng.uniform(15, 95, size=n)
    exposure_min = rng.uniform(5, 360, size=n)

    activity = rng.choice(
        ["rest", "walk", "jog", "exercise"],
        size=n,
        p=[0.18, 0.42, 0.25, 0.15]
    )

    asthma = rng.binomial(1, 0.18, size=n)
    smoker = rng.binomial(1, 0.16, size=n)
    mask_type = rng.choice(
        ["none", "cloth", "surgical", "n95"],
        size=n,
        p=[0.40, 0.18, 0.22, 0.20]
    )

    baseline_lung = rng.normal(1.0, 0.12, size=n)
    baseline_lung = _clamp(baseline_lung, 0.65, 1.35)

    age_penalty = np.where(age < 18, 0.10, 0.0) + np.where(age > 60, 0.18, 0.0)
    sensitivity = 1.0 + age_penalty + 0.35 * asthma + 0.22 * smoker + rng.normal(0, 0.05, size=n)
    sensitivity = _clamp(sensitivity, 0.7, 2.2)

    activity_factor_map = {
        "rest": 0.8,
        "walk": 1.0,
        "jog": 1.45,
        "exercise": 1.95,
    }
    activity_factor = np.array([activity_factor_map[a] for a in activity])

    mask_eff_map = {
        "none": 0.00,
        "cloth": 0.18,
        "surgical": 0.38,
        "n95": 0.72,
    }
    mask_eff = np.array([mask_eff_map[m] for m in mask_type])

    humidity_factor = 1.0 + 0.0018 * np.abs(humidity - 50)
    temp_factor = 1.0 + 0.006 * np.maximum(temp_c - 28, 0) + 0.003 * np.maximum(12 - temp_c, 0)

    deposition = (
        pm25 * 1.35
        + pm10 * 0.35
    ) * activity_factor * humidity_factor * temp_factor * (1.0 - mask_eff)

    alveolar_load = deposition * (exposure_min / 60.0) * sensitivity / baseline_lung

    inflammation_index = (
        0.011 * alveolar_load
        + 0.18 * asthma
        + 0.10 * smoker
        + 0.025 * np.maximum(temp_c - 34, 0)
        + 0.010 * np.maximum(humidity - 80, 0)
        + rng.normal(0, 0.08, size=n)
    )

    irritation_prob = 1.0 / (1.0 + np.exp(-(inflammation_index - 1.8)))

    oxygen_drop_pct = (
        0.0045 * alveolar_load
        + 0.20 * asthma
        + 0.12 * smoker
        + rng.normal(0, 0.10, size=n)
    )
    oxygen_drop_pct = _clamp(oxygen_drop_pct, 0.0, 12.0)

    safe_minutes = (
        180
        / (1.0 + 0.028 * np.maximum(pm25 - 15, 0))
        / (1.0 + 0.65 * (activity_factor - 0.8))
        / (1.0 + 0.55 * asthma + 0.28 * smoker)
        / (1.0 + 0.25 * np.maximum(temp_c - 34, 0))
        / (1.0 + 0.18 * np.maximum(humidity - 80, 0))
        * (1.0 + 0.9 * mask_eff)
    )
    safe_minutes = _clamp(safe_minutes + rng.normal(0, 5, size=n), 5, 240)

    risk_score = (
        22
        + 0.17 * pm25
        + 0.03 * pm10
        + 12 * asthma
        + 8 * smoker
        + 8.5 * (activity_factor - 0.8)
        + 4.0 * np.maximum(temp_c - 34, 0)
        + 2.2 * np.maximum(humidity - 80, 0)
        + 0.045 * exposure_min
        + 15.0 * irritation_prob
        + 1.8 * oxygen_drop_pct
        - 20.0 * mask_eff
        + rng.normal(0, 4.0, size=n)
    )
    risk_score = _clamp(risk_score, 0, 100)

    risk_band = pd.cut(
        risk_score,
        bins=[-1, 25, 50, 75, 100],
        labels=["low", "moderate", "high", "severe"]
    ).astype(str)

    df = pd.DataFrame({
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
        "alveolar_load": alveolar_load,
        "inflammation_index": inflammation_index,
        "irritation_prob": irritation_prob,
        "oxygen_drop_pct": oxygen_drop_pct,
        "safe_minutes": safe_minutes,
        "risk_score": risk_score,
        "risk_band": risk_band,
    })

    return df
