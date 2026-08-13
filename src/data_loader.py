from pathlib import Path
import pandas as pd

# Get project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to raw data
RAW_DATA_PATH = BASE_DIR / "data" / "raw"

def load_cost_yield():
    return pd.read_csv(RAW_DATA_PATH / "cost_yield.csv")

if __name__ == "__main__":from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data directory
RAW_DATA_PATH = BASE_DIR / "data" / "raw"

def load_cost_yield():
    return pd.read_csv(RAW_DATA_PATH / "cost_yield.csv")

def load_production_history():
    return pd.read_csv(RAW_DATA_PATH / "production_history.csv")

def load_production_index():
    return pd.read_csv(RAW_DATA_PATH / "production_index.csv")

def load_crop_variety():
    return pd.read_csv(RAW_DATA_PATH / "crop_variety.csv")

def load_area_production_yield():
    return pd.read_csv(RAW_DATA_PATH / "area_production_yield.csv")
    df = load_cost_yield()
    print(df.head())