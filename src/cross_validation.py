from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


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

print("=" * 65)
print("5-FOLD CROSS VALIDATION")
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
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Crop",
    "State"
]


# ============================================================
# NUMERICAL FEATURES
# ============================================================

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
# CROSS VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# ============================================================
# TRAIN AND VALIDATE
# ============================================================

for name, model in models.items():

    print("\n" + "-" * 65)
    print("Model:", name)
    print("-" * 65)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,

        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        }
    )

    mae = -scores["test_MAE"]

    rmse = -scores["test_RMSE"]

    r2 = scores["test_R2"]


    print(
        "MAE per fold:",
        np.round(mae, 3)
    )

    print(
        "RMSE per fold:",
        np.round(rmse, 3)
    )

    print(
        "R² per fold:",
        np.round(r2, 3)
    )


    print(
        "\nAverage MAE:",
        round(mae.mean(), 3)
    )

    print(
        "Average RMSE:",
        round(rmse.mean(), 3)
    )

    print(
        "Average R²:",
        round(r2.mean(), 3)
    )


    results.append({

        "Model": name,

        "MAE": mae.mean(),

        "RMSE": rmse.mean(),

        "R2": r2.mean()
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


print("\n")
print("=" * 65)
print("CROSS-VALIDATION MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    BASE_DIR
    / "models"
    / "cross_validation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\nResults saved to:")

print(output_path)


print("\n" + "=" * 65)
print("CROSS VALIDATION COMPLETED")
print("=" * 65)