import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from train_pipeline import assign_risk_tier, build_features

st.set_page_config(page_title="Urban Dengue Early-Warning System", layout="wide")

ART_DIR = Path("artifacts")
DATA_PATH = "hackathon_preli_dataset.xlsx"
SHEET_NAME = "bangladesh_dengue_large_dataset"


@st.cache_resource
def load_bundle():
    return joblib.load(ART_DIR / "dengue_model.joblib")


@st.cache_data
def load_processed_data():
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    return build_features(raw)


@st.cache_data
def load_metrics():
    path = ART_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


st.title("Urban Dengue Outbreak Early-Warning Dashboard")
st.caption("2-to-4-Week Lag-Aware Ward Risk Prediction (EDU AI Engineering Hackathon 2026)")

bundle_path = ART_DIR / "dengue_model.joblib"
if not bundle_path.exists():
    st.error("Model artifact missing. Run `python train_pipeline.py` first.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
df = load_processed_data()
metrics = load_metrics()

# Sidebar controls
st.sidebar.header("Selection & Simulation Controls")
cities = sorted(df["city"].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

city_wards = sorted(df[df["city"] == selected_city]["ward_id"].unique())
selected_ward = st.sidebar.selectbox("Select Ward", city_wards)

ward_df = df[df["ward_id"] == selected_ward].sort_values("date")
dates = ward_df["date"].dt.strftime("%Y-%m-%d").tolist()
selected_date_str = st.sidebar.selectbox("Select Reference Week", dates, index=len(dates) - 1)

row = ward_df[ward_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0].copy()

st.sidebar.subheader("What-If Weather Adjustment")
rain_boost_pct = st.sidebar.slider("Adjust Lagged Rainfall (%)", -50, 100, 0, step=10)
temp_delta = st.sidebar.slider("Adjust Avg Temp (°C)", -3.0, 3.0, 0.0, step=0.5)

# Apply what-if adjustments
factor = 1.0 + (rain_boost_pct / 100.0)
for col in ["rainfall_mm", "rainfall_mm_lag_1w", "rainfall_mm_lag_2w", "rainfall_mm_lag_3w", "rainfall_mm_lag_4w"]:
    row[col] = max(0.0, row[col] * factor)
for col in ["avg_temp_c", "avg_temp_c_lag_1w", "avg_temp_c_lag_2w", "avg_temp_c_lag_3w", "avg_temp_c_lag_4w"]:
    row[col] = row[col] + temp_delta

row["rain_sum_2_4w_lag"] = row["rainfall_mm_lag_2w"] + row["rainfall_mm_lag_3w"] + row["rainfall_mm_lag_4w"]
row["temp_mean_2_4w_lag"] = (row["avg_temp_c_lag_2w"] + row["avg_temp_c_lag_3w"] + row["avg_temp_c_lag_4w"]) / 3.0
row["rain_x_waterlogging_lag2_4w"] = row["rain_sum_2_4w_lag"] * row["water_logging_risk_index"]
row["rain_x_density_lag2_4w"] = row["rain_sum_2_4w_lag"] * (row["population_density_sqkm"] / 10000.0)

X_input = pd.DataFrame([row[feature_cols]])
pred_cases = float(np.clip(model.predict(X_input)[0], 0, None))
risk_tier = assign_risk_tier([pred_cases])[0]

# Top KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted 2–4w Cases", f"{pred_cases:.1f}")
c2.metric("Risk Tier", risk_tier)
c3.metric("2–4w Lagged Rain (mm)", f"{row['rain_sum_2_4w_lag']:.1f}")
c4.metric("Water-Logging Index", f"{row['water_logging_risk_index']:.2f}")

# Driver breakdown
st.subheader("Key Weather & Vulnerability Drivers")
driver_df = pd.DataFrame({
    "Driver": [
        "2–4w Cumulative Lagged Rainfall (mm)",
        "Lag 2w Rainfall (mm)",
        "Lag 3w Rainfall (mm)",
        "Lag 4w Rainfall (mm)",
        "2–4w Mean Lagged Temp (°C)",
        "Relative Humidity (%)",
        "Water-Logging Risk Index",
        "Sanitation Score",
        "Population Density (/sqkm)",
    ],
    "Value": [
        round(row["rain_sum_2_4w_lag"], 2),
        round(row["rainfall_mm_lag_2w"], 2),
        round(row["rainfall_mm_lag_3w"], 2),
        round(row["rainfall_mm_lag_4w"], 2),
        round(row["temp_mean_2_4w_lag"], 2),
        round(row["relative_humidity_pct"], 2),
        round(row["water_logging_risk_index"], 2),
        round(row["sanitation_score"], 2),
        int(row["population_density_sqkm"]),
    ],
})
st.dataframe(driver_df, use_container_width=True, hide_index=True)

# Ward historical trend
st.subheader(f"Historical Weekly Cases & Rainfall — {selected_ward} ({selected_city})")
chart_df = ward_df.set_index("date")[["dengue_cases", "rainfall_mm"]].tail(52)
st.line_chart(chart_df)

# Chronological validation metrics (bonus/transparency)
if metrics:
    st.subheader("Chronological Holdout Evaluation (2024–2025)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE (2–4w Cases)", metrics.get("test_mae_2_4w_cases", "N/A"))
    m2.metric("Test RMSE (2–4w Cases)", metrics.get("test_rmse_2_4w_cases", "N/A"))
    m3.metric("Train / Test Rows", f"{metrics.get('train_rows', 0)} / {metrics.get('test_rows', 0)}")