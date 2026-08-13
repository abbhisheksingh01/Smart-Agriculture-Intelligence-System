from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"


def clean_column_names(df):
    """
    Remove unnecessary spaces from column names.
    """
    df.columns = df.columns.str.strip()
    return df


def clean_text_columns(df):
    """
    Remove unnecessary spaces from text values.
    """
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates()


def preprocess_cost_yield():
    """
    Clean the cost and yield dataset.
    """

    file_path = RAW_DATA_PATH / "cost_yield.csv"

    df = pd.read_csv(file_path)

    print("\nBefore cleaning - Cost & Yield")
    print("Shape:", df.shape)

    # Clean column names
    df = clean_column_names(df)

    # Clean text values
    df = clean_text_columns(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Convert numerical columns
    numerical_columns = [
        "Cost of Cultivation (`/Hectare) A2+FL",
        "Cost of Cultivation (`/Hectare) C2",
        "Cost of Production (`/Quintal) C2",
        "Yield (Quintal/ Hectare)"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows where important values are missing
    df = df.dropna()

    print("After cleaning - Cost & Yield")
    print("Shape:", df.shape)

    # Save processed dataset
    output_path = PROCESSED_DATA_PATH / "cost_yield_cleaned.csv"
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return df


def preprocess_crop_variety():
    """
    Clean crop variety dataset.
    """

    file_path = RAW_DATA_PATH / "crop_variety.csv"

    df = pd.read_csv(file_path)

    print("\nBefore cleaning - Crop Variety")
    print("Shape:", df.shape)

    df = clean_column_names(df)
    df = clean_text_columns(df)
    df = remove_duplicates(df)

    # Keep rows where crop and variety are available
    df = df.dropna(subset=["Crop", "Variety"])

    print("After cleaning - Crop Variety")
    print("Shape:", df.shape)

    output_path = PROCESSED_DATA_PATH / "crop_variety_cleaned.csv"
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return df


def preprocess_production_index():
    """
    Clean production index dataset.
    """

    file_path = RAW_DATA_PATH / "production_index.csv"

    df = pd.read_csv(file_path)

    print("\nBefore cleaning - Production Index")
    print("Shape:", df.shape)

    df = clean_column_names(df)
    df = clean_text_columns(df)
    df = remove_duplicates(df)

    print("After cleaning - Production Index")
    print("Shape:", df.shape)

    output_path = PROCESSED_DATA_PATH / "production_index_cleaned.csv"
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return df


def preprocess_area_production_yield():
    """
    Clean area, production and yield dataset.
    """

    file_path = RAW_DATA_PATH / "area_production_yield.csv"

    df = pd.read_csv(file_path)

    print("\nBefore cleaning - Area Production Yield")
    print("Shape:", df.shape)

    df = clean_column_names(df)
    df = clean_text_columns(df)
    df = remove_duplicates(df)

    print("After cleaning - Area Production Yield")
    print("Shape:", df.shape)

    output_path = PROCESSED_DATA_PATH / "area_production_yield_cleaned.csv"
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return df


def preprocess_production_history():
    """
    Clean production history dataset.
    """

    file_path = RAW_DATA_PATH / "production_history.csv"

    df = pd.read_csv(file_path)

    print("\nBefore cleaning - Production History")
    print("Shape:", df.shape)

    df = clean_column_names(df)
    df = clean_text_columns(df)
    df = remove_duplicates(df)

    print("After cleaning - Production History")
    print("Shape:", df.shape)

    output_path = PROCESSED_DATA_PATH / "production_history_cleaned.csv"
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return df


if __name__ == "__main__":

    # Create processed directory if it doesn't exist
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    preprocess_cost_yield()
    preprocess_crop_variety()
    preprocess_production_index()
    preprocess_area_production_yield()
    preprocess_production_history()

    print("\n====================================")
    print("ALL DATASETS PREPROCESSED SUCCESSFULLY")
    print("====================================")