# AI for Urban Dengue Outbreak — Early-Warning Dashboard

This prototype predicts 2-to-4-week future Dengue outbreak risk across urban wards using multi-week weather lags (`rainfall_mm`, `avg_temp_c`, `relative_humidity_pct`) and local ward vulnerability indicators (`water_logging_risk_index`, `sanitation_score`, `population_density_sqkm`)[cite: 1, 3].

## Quick Start (Run Locally in <= 3 Minutes)

1. Open a terminal inside the project folder containing `hackathon_preli_dataset.xlsx`, `requirements.txt`, `train_pipeline.py`, `app.py`, and `ARCHITECTURE.md`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt