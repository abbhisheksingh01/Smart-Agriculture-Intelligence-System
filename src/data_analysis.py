from pathlib import Path
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cost_yield_cleaned.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("DETAILED DATA ANALYSIS")
print("=" * 60)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nCrop Counts:")
print(df["Crop"].value_counts())


# ============================================================
# YIELD STATISTICS
# ============================================================

yield_column = "Yield (Quintal/ Hectare)"

print("\nYield Statistics:")
print(df[yield_column].describe())


# ============================================================
# SORT BY YIELD
# ============================================================

print("\nHighest Yield Records:")
print(
    df[
        [
            "Crop",
            "State",
            yield_column
        ]
    ]
    .sort_values(
        by=yield_column,
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# LOWEST YIELD
# ============================================================

print("\nLowest Yield Records:")

print(
    df[
        [
            "Crop",
            "State",
            yield_column
        ]
    ]
    .sort_values(
        by=yield_column
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# CROP-WISE YIELD
# ============================================================

print("\nAverage Yield by Crop:")

crop_yield = (
    df.groupby("Crop")[yield_column]
    .agg(
        [
            "count",
            "mean",
            "min",
            "max"
        ]
    )
    .sort_values(
        by="mean",
        ascending=False
    )
)

print(crop_yield.to_string())


# ============================================================
# CHECK EXTREME VALUES
# ============================================================

print("\nRecords with Yield > 100:")

extreme = df[
    df[yield_column] > 100
]

print(
    extreme.to_string(index=False)
)


# ============================================================
# COST INFORMATION
# ============================================================

print("\nCost Statistics:")

cost_columns = [
    "Cost of Cultivation (`/Hectare) A2+FL",
    "Cost of Cultivation (`/Hectare) C2",
    "Cost of Production (`/Quintal) C2"
]

print(
    df[cost_columns].describe()
)


print("\n" + "=" * 60)
print("DATA ANALYSIS COMPLETED")
print("=" * 60)