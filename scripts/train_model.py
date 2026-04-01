from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.train import train_model

if __name__ == "__main__":
    metrics = train_model(n_samples=60000, seed=42)
    print("\n=== TRAINING METRICS ===")
    for target, vals in metrics.items():
        print(f"{target}: MAE={vals['mae']:.4f}  R2={vals['r2']:.4f}")
