import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

DATA_PATH = "hackathon_preli_dataset.xlsx"
SHEET_NAME = "bangladesh_dengue_large_dataset"
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)

    g = df.groupby("ward_id", group_keys=False)

    # Multi-week weather lags (1 to 4 weeks)
    for col in ["rainfall_mm", "avg_temp_c", "relative_humidity_pct"]:
        for lag in [1, 2, 3, 4]:
            df[f"{col}_lag_{lag}w"] = g[col].shift(lag)

    # 2-to-4 week cumulative rainfall & mean temp/humidity lags
    df["rain_sum_2_4w_lag"] = (
        df["rainfall_mm_lag_2w"] + df["rainfall_mm_lag_3w"] + df["rainfall_mm_lag_4w"]
    )
    df["temp_mean_2_4w_lag"] = (
        df["avg_temp_c_lag_2w"] + df["avg_temp_c_lag_3w"] + df["avg_temp_c_lag_4w"]
    ) / 3.0
    df["hum_mean_2_4w_lag"] = (
        df["relative_humidity_pct_lag_2w"]
        + df["relative_humidity_pct_lag_3w"]
        + df["relative_humidity_pct_lag_4w"]
    ) / 3.0

    # Interaction between lagged rainfall and local ward vulnerability
    df["rain_x_waterlogging_lag2_4w"] = (
        df["rain_sum_2_4w_lag"] * df["water_logging_risk_index"]
    )
    df["rain_x_density_lag2_4w"] = (
        df["rain_sum_2_4w_lag"] * (df["population_density_sqkm"] / 10000.0)
    )

    # Target: cumulative dengue cases in future 2-to-4 week window (t+2, t+3, t+4)
    lead2 = g["dengue_cases"].shift(-2)
    lead3 = g["dengue_cases"].shift(-3)
    lead4 = g["dengue_cases"].shift(-4)
    df["target_2_4w_cases"] = lead2 + lead3 + lead4

    # Keep rows where lags and target are valid
    valid_mask = df["rainfall_mm_lag_4w"].notna() & df["target_2_4w_cases"].notna()
    return df.loc[valid_mask].copy()


def assign_risk_tier(pred_cases: np.ndarray) -> list[str]:
    tiers = []
    for val in pred_cases:
        if val < 5:
            tiers.append("Low")
        elif val < 15:
            tiers.append("Medium")
        elif val < 30:
            tiers.append("High")
        else:
            tiers.append("Critical")
    return tiers


def main():
    raw_df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    feat_df = build_features(raw_df)

    feature_cols = [
        "population_density_sqkm",
        "water_logging_risk_index",
        "sanitation_score",
        "avg_temp_c",
        "max_temp_c",
        "min_temp_c",
        "rainfall_mm",
        "relative_humidity_pct",
        "rainfall_mm_lag_1w",
        "rainfall_mm_lag_2w",
        "rainfall_mm_lag_3w",
        "rainfall_mm_lag_4w",
        "avg_temp_c_lag_1w",
        "avg_temp_c_lag_2w",
        "avg_temp_c_lag_3w",
        "avg_temp_c_lag_4w",
        "relative_humidity_pct_lag_1w",
        "relative_humidity_pct_lag_2w",
        "relative_humidity_pct_lag_3w",
        "relative_humidity_pct_lag_4w",
        "rain_sum_2_4w_lag",
        "temp_mean_2_4w_lag",
        "hum_mean_2_4w_lag",
        "rain_x_waterlogging_lag2_4w",
        "rain_x_density_lag2_4w",
    ]

    # Chronological split: train <= 2023, test >= 2024
    train_df = feat_df[feat_df["date"].dt.year <= 2023]
    test_df = feat_df[feat_df["date"].dt.year >= 2024]

    X_train, y_train = train_df[feature_cols], train_df["target_2_4w_cases"]
    X_test, y_test = test_df[feature_cols], test_df["target_2_4w_cases"]

    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_preds = np.clip(model.predict(X_test), 0, None)
    mae = float(mean_absolute_error(y_test, test_preds))
    rmse = float(root_mean_squared_error(y_test, test_preds))

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        ART_DIR / "dengue_model.joblib",
    )

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_mae_2_4w_cases": round(mae, 3),
        "test_rmse_2_4w_cases": round(rmse, 3),
    }
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Training complete:", metrics)


if __name__ == "__main__":
    main()