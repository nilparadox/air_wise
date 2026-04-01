from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.sim.digital_lung import generate_simulated_dataset

if __name__ == "__main__":
    df = generate_simulated_dataset(n=12000, seed=42)

    print("\n=== RISK BAND DISTRIBUTION ===")
    print((df["risk_band"].value_counts(normalize=True) * 100).round(2))

    print("\n=== RISK SCORE SUMMARY ===")
    print(df["risk_score"].describe().round(3))

    print("\n=== SAFE MINUTES SUMMARY ===")
    print(df["safe_minutes"].describe().round(3))

    print("\n=== SAMPLE SANITY CASES ===")
    cols = ["pm25", "exposure_min", "activity", "asthma", "smoker", "mask_type", "risk_score", "risk_band", "safe_minutes"]
    print(df[cols].sample(10, random_state=42).to_string(index=False))
