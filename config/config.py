from pathlib import Path

"""
Path configurations.
"""

ROOT_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = ROOT_DIR / "saved_models"

STATISTICS_DIR = ROOT_DIR / "statistics"
DATA_STATISTICS = STATISTICS_DIR / "data_statistics"
MODEL_STATISTICS = STATISTICS_DIR / "model_statistics"
PORTFOLIO_STATISTICS = STATISTICS_DIR / "portfolio_statistics"

folders = [DATA_DIR, RAW_DATA_DIR, PREDICTIONS_DIR,
           MODELS_DIR,
           STATISTICS_DIR, DATA_STATISTICS, MODEL_STATISTICS, PORTFOLIO_STATISTICS]


for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
