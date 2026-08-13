from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cost_yield_cleaned.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_crop_yield_model.pkl"
)

OUTPUT_PATH = BASE_DIR / "outputs"

FIGURE_PATH = OUTPUT_PATH / "figures"

FIGURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)


# ============================================================
# FEATURES AND TARGET
# ============================================================

target = "Yield (Quintal/ Hectare)"

features = [
    "Crop",
    "State",
    "Cost of Cultivation (`/Hectare) A2+FL",
    "Cost of Cultivation (`/Hectare) C2",
    "Cost of Production (`/Quintal) C2"
]

X = df[features]

y = df[target]


# ============================================================
# SAME TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nModel: Decision Tree")

print("\nMAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))


# ============================================================
# ACTUAL VS PREDICTED DATA
# ============================================================

results = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": predictions
})

results["Error"] = (
    results["Actual Yield"]
    - results["Predicted Yield"]
)

print("\nActual vs Predicted:")
print(results.to_string(index=False))


# ============================================================
# SAVE PREDICTION RESULTS
# ============================================================

results_path = (
    OUTPUT_PATH
    / "prediction_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nPrediction results saved:")
print(results_path)


# ============================================================
# 1. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Actual Yield"],
    results["Predicted Yield"]
)

min_value = min(
    results["Actual Yield"].min(),
    results["Predicted Yield"].min()
)

max_value = max(
    results["Actual Yield"].max(),
    results["Predicted Yield"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.title(
    "Actual vs Predicted Crop Yield"
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH
    / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. RESIDUAL PLOT
# ============================================================

residuals = (
    results["Actual Yield"]
    - results["Predicted Yield"]
)

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Predicted Yield"],
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Yield")
plt.ylabel("Residual")

plt.title(
    "Prediction Residuals"
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH
    / "residual_plot.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. MODEL COMPARISON GRAPH
# ============================================================

comparison_path = (
    BASE_DIR
    / "models"
    / "model_comparison.csv"
)

comparison = pd.read_csv(
    comparison_path
)

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.xlabel("Model")
plt.ylabel("R² Score")

plt.title(
    "Machine Learning Model Comparison"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH
    / "model_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

# Access the Decision Tree from the pipeline
tree_model = model.named_steps["model"]

preprocessor = model.named_steps["preprocessor"]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importance = tree_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print("\nFeature Importance:")
print(
    feature_importance.head(15)
    .to_string(index=False)
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = (
    feature_importance
    .head(10)
    .sort_values(
        "Importance"
    )
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top Features Influencing Crop Yield"
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH
    / "feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print("\nGenerated files:")
print("1. prediction_results.csv")
print("2. actual_vs_predicted.png")
print("3. residual_plot.png")
print("4. model_comparison.png")
print("5. feature_importance.png")