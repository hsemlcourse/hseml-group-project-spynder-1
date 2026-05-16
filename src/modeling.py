import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
BEST_PARAMS = {
    "regressor__depth": 6,
    "regressor__iterations": 500,
    "regressor__l2_leaf_reg": 5,
    "regressor__learning_rate": 0.05,
}

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "london_merged.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "catboost_pipeline.joblib"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)


def get_rmse_and_r2(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2 = r2_score(y_true_real, y_pred_real)
    return rmse, r2


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    df["cnt_log"] = np.log1p(df["cnt"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    categorical_features = ["weather_code", "season", "is_holiday", "is_weekend"]
    numerical_features = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "month",
        "t1",
        "t2",
        "hum",
        "wind_speed",
    ]

    X = df[categorical_features + numerical_features]
    y = df["cnt_log"]
    return X, y, categorical_features, numerical_features


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_val_size = int(len(X) * 0.8)
    X_train_val = X.iloc[:train_val_size].copy()
    X_test = X.iloc[train_val_size:].copy()
    y_train_val = y.iloc[:train_val_size].copy()
    y_test = y.iloc[train_val_size:].copy()
    return X_train_val, X_test, y_train_val, y_test


def build_pipeline(categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                CatBoostRegressor(
                    random_state=SEED,
                    verbose=0,
                    allow_writing_files=False,
                ),
            ),
        ]
    )
    pipeline.set_params(**BEST_PARAMS)
    return pipeline


def train_and_save() -> None:
    set_seed()

    df = load_data()
    X, y, categorical_features, _ = prepare_features(df)
    X_train_val, X_test, y_train_val, y_test = time_split(X, y)

    pipeline = build_pipeline(categorical_features)
    pipeline.fit(X_train_val, y_train_val)

    y_test_pred = pipeline.predict(X_test)
    rmse, r2 = get_rmse_and_r2(y_test, y_test_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")


if __name__ == "__main__":
    train_and_save()
