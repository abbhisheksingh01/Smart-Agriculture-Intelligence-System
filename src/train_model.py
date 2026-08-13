from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cost_yield_cleaned.csv"

MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

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
# CATEGORICAL AND NUMERICAL FEATURES
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
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=5
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=8
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)
    print("-" * 60)

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
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

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="R2",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .sort_values(
        by="R2",
        ascending=False
    )
    .iloc[0]["Model"]
)

best_model = trained_models[best_model_name]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_file = MODEL_PATH / "best_crop_yield_model.pkl"

joblib.dump(
    best_model,
    best_model_file
)

print("\nBest model saved at:")

print(best_model_file)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = MODEL_PATH / "model_comparison.csv"

results_df.to_csv(
    results_file,
    index=False
)

print("\nModel comparison saved at:")

print(results_file)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)