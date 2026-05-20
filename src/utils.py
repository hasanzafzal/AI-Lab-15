import os
import json
import logging
from datetime import datetime

# Path Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Dataset filename
RAW_DATASET = os.path.join(DATA_DIR, "Crop_recommendation.csv")
PROCESSED_DATASET = os.path.join(DATA_DIR, "Crop_recommendation_processed.csv")

# Model artifact filenames
DECISION_TREE_MODEL = os.path.join(MODELS_DIR, "decision_tree_classifier.joblib")
KMEANS_MODEL = os.path.join(MODELS_DIR, "kmeans_clustering.joblib")
LINEAR_REG_MODEL = os.path.join(MODELS_DIR, "linear_regression.joblib")
SCALER_ARTIFACT = os.path.join(MODELS_DIR, "feature_scaler.joblib")
LABEL_ENCODER_ARTIFACT = os.path.join(MODELS_DIR, "label_encoder.joblib")


# Feature Definitions
SOIL_FEATURES = ["N", "P", "K", "ph"]
CLIMATE_FEATURES = ["temperature", "humidity", "rainfall"]
ALL_FEATURES = SOIL_FEATURES + CLIMATE_FEATURES
TARGET_CLASSIFICATION = "label"
TARGET_REGRESSION = "yield_ton_per_ha"

# Crop-Specific Yield Parameters (tons/hectare)
# Used for realistic synthetic yield generation
CROP_YIELD_PARAMS = {
    "rice":         {"base_yield": 4.5,  "max_yield": 8.0,  "optimal_temp": 25, "optimal_rain": 200},
    "maize":        {"base_yield": 5.0,  "max_yield": 10.0, "optimal_temp": 27, "optimal_rain": 120},
    "chickpea":     {"base_yield": 1.0,  "max_yield": 2.5,  "optimal_temp": 24, "optimal_rain": 80},
    "kidneybeans":  {"base_yield": 1.5,  "max_yield": 3.0,  "optimal_temp": 22, "optimal_rain": 90},
    "pigeonpeas":   {"base_yield": 1.0,  "max_yield": 2.0,  "optimal_temp": 28, "optimal_rain": 100},
    "mothbeans":    {"base_yield": 0.5,  "max_yield": 1.5,  "optimal_temp": 30, "optimal_rain": 50},
    "mungbean":     {"base_yield": 0.8,  "max_yield": 2.0,  "optimal_temp": 28, "optimal_rain": 70},
    "blackgram":    {"base_yield": 0.7,  "max_yield": 1.8,  "optimal_temp": 27, "optimal_rain": 75},
    "lentil":       {"base_yield": 1.0,  "max_yield": 2.5,  "optimal_temp": 22, "optimal_rain": 60},
    "pomegranate":  {"base_yield": 8.0,  "max_yield": 15.0, "optimal_temp": 30, "optimal_rain": 60},
    "banana":       {"base_yield": 20.0, "max_yield": 40.0, "optimal_temp": 27, "optimal_rain": 150},
    "mango":        {"base_yield": 7.0,  "max_yield": 15.0, "optimal_temp": 30, "optimal_rain": 100},
    "grapes":       {"base_yield": 10.0, "max_yield": 25.0, "optimal_temp": 25, "optimal_rain": 80},
    "watermelon":   {"base_yield": 15.0, "max_yield": 30.0, "optimal_temp": 28, "optimal_rain": 70},
    "muskmelon":    {"base_yield": 12.0, "max_yield": 25.0, "optimal_temp": 28, "optimal_rain": 65},
    "apple":        {"base_yield": 10.0, "max_yield": 20.0, "optimal_temp": 18, "optimal_rain": 100},
    "orange":       {"base_yield": 12.0, "max_yield": 25.0, "optimal_temp": 25, "optimal_rain": 110},
    "papaya":       {"base_yield": 15.0, "max_yield": 30.0, "optimal_temp": 28, "optimal_rain": 120},
    "coconut":      {"base_yield": 6.0,  "max_yield": 12.0, "optimal_temp": 27, "optimal_rain": 150},
    "cotton":       {"base_yield": 1.5,  "max_yield": 3.5,  "optimal_temp": 30, "optimal_rain": 80},
    "jute":         {"base_yield": 2.0,  "max_yield": 4.0,  "optimal_temp": 28, "optimal_rain": 170},
    "coffee":       {"base_yield": 1.5,  "max_yield": 3.0,  "optimal_temp": 23, "optimal_rain": 150},
}

# Logging
def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# Helper Functions
def ensure_dirs():
    """Create project directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_metrics(metrics: dict, filename: str):
    """Save evaluation metrics as a JSON file in the results directory."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return filepath


def load_metrics(filename: str) -> dict:
    """Load evaluation metrics from a JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "r") as f:
        return json.load(f)


def get_timestamp() -> str:
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")