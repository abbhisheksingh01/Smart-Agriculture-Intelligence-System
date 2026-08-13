from pathlib import Path
import pandas as pd
import joblib


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_crop_yield_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("SMART AGRICULTURE - CROP YIELD PREDICTION")
print("=" * 65)

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_yield(
    crop,
    state,
    cultivation_cost_a2fl,
    cultivation_cost_c2,
    production_cost_c2
):

    input_data = pd.DataFrame({

        "Crop": [crop],

        "State": [state],

        "Cost of Cultivation (`/Hectare) A2+FL": [
            cultivation_cost_a2fl
        ],

        "Cost of Cultivation (`/Hectare) C2": [
            cultivation_cost_c2
        ],

        "Cost of Production (`/Quintal) C2": [
            production_cost_c2
        ]
    })

    prediction = model.predict(input_data)

    return prediction[0]


# ============================================================
# EXAMPLE PREDICTION
# ============================================================

if __name__ == "__main__":

    crop = "SUGARCANE"

    state = "Tamil Nadu"

    cultivation_cost_a2fl = 66335.06

    cultivation_cost_c2 = 89025.27

    production_cost_c2 = 85.79


    predicted_yield = predict_yield(
        crop,
        state,
        cultivation_cost_a2fl,
        cultivation_cost_c2,
        production_cost_c2
    )


    print("\nInput Details")
    print("-" * 65)

    print("Crop:", crop)

    print("State:", state)

    print(
        "Cultivation Cost A2+FL:",
        cultivation_cost_a2fl
    )

    print(
        "Cultivation Cost C2:",
        cultivation_cost_c2
    )

    print(
        "Production Cost C2:",
        production_cost_c2
    )


    print("\nPrediction")
    print("-" * 65)

    print(
        f"Predicted Yield: {predicted_yield:.2f} "
        "Quintal/Hectare"
    )

    print("=" * 65)