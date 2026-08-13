from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_crop_yield_model.pkl"

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cost_yield_cleaned.csv"
)

CV_PATH = BASE_DIR / "models" / "cross_validation_results.csv"

IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

cv_results = pd.read_csv(CV_PATH)

feature_importance = pd.read_csv(IMPORTANCE_PATH)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Smart Agriculture Intelligence System"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    crop: str

    state: str

    cost_a2fl: float

    cost_c2: float

    production_cost: float


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
def predict_crop_yield(request: PredictionRequest):

    input_data = pd.DataFrame({

        "Crop": [request.crop],

        "State": [request.state],

        "Cost of Cultivation (`/Hectare) A2+FL": [
            request.cost_a2fl
        ],

        "Cost of Cultivation (`/Hectare) C2": [
            request.cost_c2
        ],

        "Cost of Production (`/Quintal) C2": [
            request.production_cost
        ]
    })

    prediction = model.predict(input_data)[0]

    return {
        "success": True,
        "prediction": round(float(prediction), 2),
        "crop": request.crop,
        "state": request.state
    }


# ============================================================
# DASHBOARD DATA
# ============================================================

@app.get("/api/dashboard")
def dashboard():

    best_model = (
        cv_results
        .sort_values("R2", ascending=False)
        .iloc[0]
    )

    return {

        "records": len(df),

        "crops": int(df["Crop"].nunique()),

        "states": int(df["State"].nunique()),

        "average_yield": round(
            float(
                df["Yield (Quintal/ Hectare)"].mean()
            ),
            2
        ),

        "best_model": str(
            best_model["Model"]
        ),

        "best_r2": round(
            float(best_model["R2"]),
            3
        )
    }


# ============================================================
# CROP DATA
# ============================================================

@app.get("/api/crop-analysis")
def crop_analysis():

    result = (
        df.groupby("Crop")[
            "Yield (Quintal/ Hectare)"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    return {
        "labels": result.index.tolist(),
        "values": result.round(2).tolist()
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get("/api/model-performance")
def model_performance():

    return {

        "models": cv_results["Model"].tolist(),

        "r2": cv_results["R2"].round(3).tolist(),

        "mae": cv_results["MAE"].round(3).tolist(),

        "rmse": cv_results["RMSE"].round(3).tolist()
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.get("/api/feature-importance")
def feature_importance_api():

    data = (
        feature_importance
        .head(10)
    )

    return {

        "features": data["Feature"].tolist(),

        "importance": data["Importance"].round(4).tolist()
    }


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=BASE_DIR / "frontend"
    ),
    name="static"
)


@app.get("/")
def home():

    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )