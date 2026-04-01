from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.core.config import DATA_DIR, MODELS_DIR, MODEL_PATH, METADATA_PATH
from app.sim.digital_lung import generate_simulated_dataset


FEATURES = [
    "age",
    "pm25",
    "pm10",
    "temp_c",
    "humidity",
    "exposure_min",
    "activity",
    "asthma",
    "smoker",
    "mask_type",
    "baseline_lung",
]

TARGETS = [
    "risk_score",
    "safe_minutes",
    "recovery_minutes",
    "end_load",
    "end_inflammation",
    "irritation_prob",
    "oxygen_drop_pct",
]


def train_digital_model(n_samples: int = 120000, seed: int = 42):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_simulated_dataset(n=n_samples, seed=seed)
    df.to_csv(DATA_DIR / "digital_lung_simulated_dataset.csv", index=False)

    X = df[FEATURES]
    y = df[TARGETS]

    cat_cols = ["activity", "mask_type"]
    num_cols = [c for c in FEATURES if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    base_model = RandomForestRegressor(
        n_estimators=260,
        max_depth=18,
        min_samples_split=6,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", MultiOutputRegressor(base_model)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {}
    for i, target in enumerate(TARGETS):
        metrics[target] = {
            "mae": float(mean_absolute_error(y_test.iloc[:, i], preds[:, i])),
            "r2": float(r2_score(y_test.iloc[:, i], preds[:, i])),
        }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        {
            "features": FEATURES,
            "targets": TARGETS,
            "metrics": metrics,
            "n_samples": n_samples,
            "seed": seed,
            "dataset_csv": str(DATA_DIR / "digital_lung_simulated_dataset.csv"),
        },
        METADATA_PATH,
    )

    return metrics
