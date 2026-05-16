from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modeling import MODEL_PATH


class PredictionRequest(BaseModel):
    weather_code: int = Field(..., example=1)
    season: int = Field(..., example=1)
    is_holiday: int = Field(..., example=0)
    is_weekend: int = Field(..., example=0)
    hour: int = Field(..., ge=0, le=23, example=8)
    day_of_week: int = Field(..., ge=0, le=6, example=2)
    month: int = Field(..., ge=1, le=12, example=7)
    t1: float = Field(..., example=18.5)
    t2: float = Field(..., example=17.0)
    hum: float = Field(..., example=65.0)
    wind_speed: float = Field(..., example=12.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found: {model_path}. Run `python src/modeling.py` first."
        )

    app.state.model = joblib.load(model_path)
    yield


app = FastAPI(title="Bike Demand API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model is not loaded")

    hour_sin = float(np.sin(2 * np.pi * request.hour / 24))
    hour_cos = float(np.cos(2 * np.pi * request.hour / 24))

    features = pd.DataFrame(
        [
            {
                "weather_code": request.weather_code,
                "season": request.season,
                "is_holiday": request.is_holiday,
                "is_weekend": request.is_weekend,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "day_of_week": request.day_of_week,
                "month": request.month,
                "t1": request.t1,
                "t2": request.t2,
                "hum": request.hum,
                "wind_speed": request.wind_speed,
            }
        ]
    )

    prediction_log = float(app.state.model.predict(features)[0])
    prediction_cnt = float(np.expm1(prediction_log))

    return {"count": prediction_cnt}
