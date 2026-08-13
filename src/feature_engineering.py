from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    file_path = PROCESSED_DATA_PATH / "cost_yield_cleaned.csv"

    df = pd.read_csv(file_path)

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data():

    df = load_data()

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = "Yield (Quintal/ Hectare)"

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = [
        "Crop",
        "State",
        "Cost of Cultivation (`/Hectare) A2+FL",
        "Cost of Cultivation (`/Hectare) C2",
        "Cost of Production (`/Quintal) C2"
    ]

    X = df[features]

    y = df[target]

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    print("\nFeature Shape:", X.shape)
    print("Target Shape:", y.shape)

    # --------------------------------------------------------
    # Categorical and numerical features
    # --------------------------------------------------------

    categorical_features = [
        "Crop",
        "State"
    ]

    numerical_features = [
        "Cost of Cultivation (`/Hectare) A2+FL",
        "Cost of Cultivation (`/Hectare) C2",
        "Cost of Production (`/Quintal) C2"
    ]

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data:", X_test.shape)

    # --------------------------------------------------------
    # Transform data
    # --------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(X_train)

    X_test_transformed = preprocessor.transform(X_test)

    print("\nTransformed Training Data:")
    print(X_train_transformed.shape)

    print("\nTransformed Testing Data:")
    print(X_test_transformed.shape)

    print("\nFeature engineering completed successfully.")

    return (
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_data()