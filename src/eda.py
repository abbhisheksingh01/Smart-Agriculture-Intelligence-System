from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
OUTPUT_PATH = BASE_DIR / "outputs" / "figures"

# Create output folder if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

cost_yield = pd.read_csv(
    PROCESSED_DATA_PATH / "cost_yield_cleaned.csv"
)

crop_variety = pd.read_csv(
    PROCESSED_DATA_PATH / "crop_variety_cleaned.csv"
)

production_index = pd.read_csv(
    PROCESSED_DATA_PATH / "production_index_cleaned.csv"
)

area_production_yield = pd.read_csv(
    PROCESSED_DATA_PATH / "area_production_yield_cleaned.csv"
)

production_history = pd.read_csv(
    PROCESSED_DATA_PATH / "production_history_cleaned.csv"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nCost & Yield Dataset")
print(cost_yield.info())

print("\nMissing Values:")
print(cost_yield.isnull().sum())


# ============================================================
# 1. CROP DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

cost_yield["Crop"].value_counts().plot(kind="bar")

plt.title("Number of Records by Crop")
plt.xlabel("Crop")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "crop_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. AVERAGE YIELD BY CROP
# ============================================================

average_yield = (
    cost_yield
    .groupby("Crop")["Yield (Quintal/ Hectare)"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

average_yield.plot(kind="bar")

plt.title("Average Crop Yield")
plt.xlabel("Crop")
plt.ylabel("Average Yield (Quintal/Hectare)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "average_yield_by_crop.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. AVERAGE YIELD BY STATE
# ============================================================

average_state_yield = (
    cost_yield
    .groupby("State")["Yield (Quintal/ Hectare)"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

average_state_yield.plot(kind="bar")

plt.title("Average Crop Yield by State")
plt.xlabel("State")
plt.ylabel("Average Yield (Quintal/Hectare)")
plt.xticks(rotation=60)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "average_yield_by_state.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. CULTIVATION COST VS YIELD
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=cost_yield,
    x="Cost of Cultivation (`/Hectare) C2",
    y="Yield (Quintal/ Hectare)",
    hue="Crop"
)

plt.title("Cultivation Cost vs Crop Yield")
plt.xlabel("Cultivation Cost C2")
plt.ylabel("Yield (Quintal/Hectare)")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "cultivation_cost_vs_yield.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

numeric_columns = cost_yield.select_dtypes(
    include="number"
)

correlation = numeric_columns.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Agricultural Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. PRODUCTION INDEX
# ============================================================

index_data = production_index.set_index("Crop")

plt.figure(figsize=(12, 7))

for crop in index_data.index:
    plt.plot(
        index_data.columns,
        index_data.loc[crop],
        marker="o",
        label=crop
    )

plt.title("Agricultural Production Index Trend")
plt.xlabel("Year")
plt.ylabel("Production Index")

plt.xticks(rotation=45)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "production_index_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. AREA, PRODUCTION AND YIELD
# ============================================================

print("\nArea Production Yield Dataset")
print(area_production_yield.head())


# ============================================================
# 8. TOP CROPS BY PRODUCTION
# ============================================================

production_column = "Production 2010-11"

if production_column in area_production_yield.columns:

    top_production = (
        area_production_yield[
            ["Crop", production_column]
        ]
        .sort_values(
            by=production_column,
            ascending=False
        )
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        top_production["Crop"],
        top_production[production_column]
    )

    plt.title("Top Crops by Production (2010-11)")
    plt.xlabel("Crop")
    plt.ylabel("Production")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH / "top_crop_production.png",
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGraphs saved in:")

print(OUTPUT_PATH)