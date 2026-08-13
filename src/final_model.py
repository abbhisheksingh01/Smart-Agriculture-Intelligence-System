from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


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

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("FINAL RANDOM FOREST MODEL")
print("=" * 65)

print("\nDataset Shape:")
print(df.shape)


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
# FEATURE TYPES
# ============================================================

categorical_features = [
    "Crop",
    "State"
]

numerical_features = [
    "Cost of Cultivation (`/Hectare) A2+FL",
    "Cost of Cultivation (`/Hectare) C2",
    "Cost of Production (`/Quintal) C2"
]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            categorical_features
        ),

        (
            "numerical",

            "passthrough",

            numerical_features
        )
    ]
)


# ============================================================
# FINAL RANDOM FOREST
# ============================================================

random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            random_forest
        )
    ]
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(
    X,
    y
)

print("Training completed successfully.")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "best_crop_yield_model.pkl"
)

joblib.dump(
    pipeline,
    model_path
)


print("\nFinal model saved at:")

print(model_path)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_model = pipeline.named_steps["model"]

processor = pipeline.named_steps["preprocessor"]

feature_names = (
    processor
    .get_feature_names_out()
)

importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop 15 Features:")

print(
    importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    MODEL_DIR
    / "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


print("\nFeature importance saved at:")

print(importance_path)


print("\n" + "=" * 65)
print("FINAL MODEL READY")
print("=" * 65)