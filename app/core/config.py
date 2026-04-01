from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

MODEL_PATH = MODELS_DIR / "digital_lung_model.joblib"
METADATA_PATH = MODELS_DIR / "digital_lung_model_metadata.joblib"
