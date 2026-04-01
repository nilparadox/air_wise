from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def activity_to_breathing(activity: str) -> float:
    table = {
        "rest": 11.0,
        "walk": 16.0,
        "jog": 24.0,
        "exercise": 32.0,
    }
    return table.get(activity, 16.0)


def mask_efficiency(mask_type: str) -> float:
    table = {
        "none": 0.00,
        "cloth": 0.15,
        "surgical": 0.32,
        "n95": 0.65,
    }
    return table.get(mask_type, 0.0)


def compute_sensitivity(age: float, asthma: int, smoker: int, baseline_lung: float) -> float:
    s = 1.0

    if age < 18:
        s += 0.10
    elif age > 60:
        s += 0.14

    if asthma:
        s += 0.28

    if smoker:
        s += 0.18

    s += (1.0 - baseline_lung) * 0.8
    return float(_clip(s, 0.80, 1.90))


def sample_pm25(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Realistic mixed distribution:
    many clean/moderate cases, fewer extreme cases.
    """
    bands = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    out = np.zeros(n, dtype=float)

    for i, b in enumerate(bands):
        if b == 0:
            out[i] = rng.uniform(3, 15)
        elif b == 1:
            out[i] = rng.uniform(15, 35)
        elif b == 2:
            out[i] = rng.uniform(35, 80)
        elif b == 3:
            out[i] = rng.uniform(80, 150)
        else:
            out[i] = rng.uniform(150, 260)

    return out


def sample_exposure(rng: np.random.Generator, n: int) -> np.ndarray:
    bands = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.35, 0.25, 0.10])
    out = np.zeros(n, dtype=float)

    for i, b in enumerate(bands):
        if b == 0:
            out[i] = rng.uniform(5, 20)
        elif b == 1:
            out[i] = rng.uniform(20, 45)
        elif b == 2:
            out[i] = rng.uniform(45, 90)
        else:
            out[i] = rng.uniform(90, 180)

    return out


def simulate_one_case(
    pm25: float,
    pm10: float,
    temp_c: float,
    humidity: float,
    exposure_min: float,
    age: float,
    activity: str,
    asthma: int,
    smoker: int,
    mask_type: str,
    baseline_lung: float,
    dt_min: float = 1.0,
):
    breathing = activity_to_breathing(activity)
    mask_eff = mask_efficiency(mask_type)
    sensitivity = compute_sensitivity(age, asthma, smoker, baseline_lung)

    conc = 1.10 * pm25 + 0.18 * pm10

    temp_mult = 1.0 + 0.006 * max(temp_c - 32.0, 0.0) + 0.002 * max(12.0 - temp_c, 0.0)
    hum_mult = 1.0 + 0.0018 * abs(humidity - 50.0)

    alpha = 0.00075
    beta = 0.050 / sensitivity
    gamma = 0.020 * sensitivity
    delta = 0.040 / sensitivity
    quad_clear = 0.004

    L = 0.0
    I = 0.0

    n_steps = max(1, int(round(exposure_min / dt_min)))
    load_trace = []
    inflam_trace = []

    for _ in range(n_steps):
        intake = alpha * conc * breathing * temp_mult * hum_mult * (1.0 - mask_eff) * sensitivity

        # bounded load dynamics
        dL = (intake - beta * L - quad_clear * L * L) * dt_min
        L = max(0.0, L + dL)

        # smooth bounded inflammation dynamics
        activation = gamma * (L / (1.0 + L))
        dI = (activation - delta * I) * dt_min
        I = max(0.0, I + dI)

        load_trace.append(L)
        inflam_trace.append(I)

    end_load = float(load_trace[-1])
    max_load = float(np.max(load_trace))
    mean_load = float(np.mean(load_trace))
    end_inflammation = float(inflam_trace[-1])
    max_inflammation = float(np.max(inflam_trace))

    # recovery
    recovery_min = 0.0
    Lr = L
    Ir = I
    for _ in range(24 * 60):
        dL = (-beta * Lr - quad_clear * Lr * Lr) * dt_min
        Lr = max(0.0, Lr + dL)

        activation = gamma * (Lr / (1.0 + Lr))
        dI = (activation - delta * Ir) * dt_min
        Ir = max(0.0, Ir + dI)

        recovery_min += dt_min
        if Lr < 0.08 and Ir < 0.06:
            break

    # softer physiological proxies
    z = -3.0 + 2.2 * end_inflammation + 0.55 * end_load + 0.12 * (sensitivity - 1.0)
    irritation_prob = 1.0 / (1.0 + np.exp(-z))
    irritation_prob = float(_clip(irritation_prob, 0.0, 1.0))

    oxygen_drop_pct = 0.22 * end_load + 1.2 * end_inflammation + 0.22 * asthma + 0.10 * smoker
    oxygen_drop_pct = float(_clip(oxygen_drop_pct, 0.0, 8.0))

    # calibrated risk score
    raw_risk = (
        -1.5
        + 2.6 * end_inflammation
        + 0.55 * end_load
        + 0.010 * pm25
        + 0.003 * exposure_min
        + 2.5 * asthma
        + 1.4 * smoker
        + 1.6 * max((breathing - 16.0) / 10.0, 0.0)
        - 4.0 * mask_eff
    )

    risk_score = 100.0 / (1.0 + np.exp(-raw_risk / 3.2))
    risk_score = float(_clip(risk_score, 0.0, 100.0))

    # safe exposure threshold
    Ls = 0.0
    Is = 0.0
    safe_minutes = 240.0
    threshold_load = 2.8 / sensitivity
    threshold_inflam = 0.45 / sensitivity

    for minute in range(1, 241):
        intake = alpha * conc * breathing * temp_mult * hum_mult * (1.0 - mask_eff) * sensitivity
        dL = intake - beta * Ls - quad_clear * Ls * Ls
        Ls = max(0.0, Ls + dL)

        activation = gamma * (Ls / (1.0 + Ls))
        dI = activation - delta * Is
        Is = max(0.0, Is + dI)

        if Ls >= threshold_load or Is >= threshold_inflam:
            safe_minutes = float(minute)
            break

    if risk_score < 25:
        risk_band = "low"
    elif risk_score < 50:
        risk_band = "moderate"
    elif risk_score < 75:
        risk_band = "high"
    else:
        risk_band = "severe"

    return {
        "breathing_rate": breathing,
        "mask_efficiency": mask_eff,
        "sensitivity": sensitivity,
        "end_load": end_load,
        "max_load": max_load,
        "mean_load": mean_load,
        "end_inflammation": end_inflammation,
        "max_inflammation": max_inflammation,
        "irritation_prob": irritation_prob,
        "oxygen_drop_pct": oxygen_drop_pct,
        "safe_minutes": float(_clip(safe_minutes, 5.0, 240.0)),
        "recovery_minutes": float(_clip(recovery_min, 5.0, 1440.0)),
        "risk_score": risk_score,
        "risk_band": risk_band,
    }


def generate_simulated_dataset(n: int = 60000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ages = rng.integers(8, 80, size=n)
    pm25s = sample_pm25(rng, n)
    pm10s = pm25s * rng.uniform(1.15, 1.90, size=n)
    temps = rng.uniform(10, 42, size=n)
    humidities = rng.uniform(20, 90, size=n)
    exposure_mins = sample_exposure(rng, n)

    activities = rng.choice(["rest", "walk", "jog", "exercise"], size=n, p=[0.22, 0.45, 0.22, 0.11])
    asthmas = rng.binomial(1, 0.14, size=n)
    smokers = rng.binomial(1, 0.12, size=n)
    masks = rng.choice(["none", "cloth", "surgical", "n95"], size=n, p=[0.40, 0.20, 0.24, 0.16])
    baseline_lungs = _clip(rng.normal(1.0, 0.10, size=n), 0.78, 1.12)

    rows = []
    for i in range(n):
        out = simulate_one_case(
            pm25=float(pm25s[i]),
            pm10=float(pm10s[i]),
            temp_c=float(temps[i]),
            humidity=float(humidities[i]),
            exposure_min=float(exposure_mins[i]),
            age=float(ages[i]),
            activity=str(activities[i]),
            asthma=int(asthmas[i]),
            smoker=int(smokers[i]),
            mask_type=str(masks[i]),
            baseline_lung=float(baseline_lungs[i]),
        )

        rows.append({
            "age": int(ages[i]),
            "pm25": float(pm25s[i]),
            "pm10": float(pm10s[i]),
            "temp_c": float(temps[i]),
            "humidity": float(humidities[i]),
            "exposure_min": float(exposure_mins[i]),
            "activity": str(activities[i]),
            "asthma": int(asthmas[i]),
            "smoker": int(smokers[i]),
            "mask_type": str(masks[i]),
            "baseline_lung": float(baseline_lungs[i]),
            **out,
        })

    return pd.DataFrame(rows)
