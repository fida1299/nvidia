# Architecture Design — Urban Dengue Early-Warning System

## 1. Overview & Problem Framing
Dengue outbreaks follow a 2-to-4-week lag after rainfall and weather shifts, and local impact also depends on ward vulnerability (water logging, sanitation, and population density)[cite: 1]. This system predicts 2-to-4-week future Dengue cases per ward and assigns risk tiers: Low, Medium, High, and Critical[cite: 1, 3].

## 2. System Data Flow Diagram
hackathon_preli_dataset.xlsx
  -> train_pipeline.py (ward-wise 1w-4w weather lags + vulnerability features + 2-4w target)
  -> Chronological Split (<=2023 train, >=2024 test)
  -> HistGradientBoostingRegressor model saved to artifacts/dengue_model.joblib
  -> app.py (Streamlit dashboard loads cached model & data for fast local predictions)

## 3. Chronological Validation Split Rationale
We split data by time (<=2023 for training and >=2024 for testing) instead of random shuffling. Time-series data with seasonal patterns would leak future information if randomly split, so chronological testing checks real future early-warning performance[cite: 3].

## 4. Model Selection & Cost/Caching Efficiency
- Model: HistGradientBoostingRegressor captures non-linear lag and vulnerability patterns well and saves quickly with joblib[cite: 3].
- Risk Tiers: Predicted 2-4w cases are mapped to Low (<5), Medium (5-14.9), High (15-29.9), and Critical (>=30)[cite: 1, 3].
- Cost & Speed: Inference runs locally with zero paid API/LLM calls, and Streamlit uses @st.cache_resource and @st.cache_data to avoid repeated loading or retraining[cite: 3].