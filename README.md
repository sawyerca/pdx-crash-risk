# Portland Traffic Crash Risk Prediction

An interactive machine learning system that predicts hourly crash risk for every street in the Portland area, processing nearly 25M predictions daily with real-time weather integration.

**Live Demo: https://pdx-crash-risk.org** 

<img width="1920" height="965" alt="Screenshot 2025-10-01 at 6 48 49 PM" src="https://github.com/user-attachments/assets/575556f6-6e95-446e-89cc-c09e1018c826" />


---

## Objective 

Traffic crashes in Portland result in dozens of fatalities and many more injuries each year. The aim of this project is to predict *where* and *when* crashes are most likely to occur using readily available data like:

- Weather conditions
- Date and time
- Historical crash locations

---

## Key Features

- Real-time Predictions: Hourly crash risk forecasts updated automatically
- Interactive Map: Explore risk levels across Portland's street network with time controls
- Weather Integration: Live weather data from Open-Meteo API drives predictions
- Street-Level Granularity: Risk assessment for road segments
- Memory-Optimized: Efficient processing of 25 hours of citywide predictions

---

##  Technical Implementation

### How It Works

1. Data Pipeline: Fetches real-time weather for 25 hours across Portland weather stations
2. Feature Engineering: Combines weather, temporal, and street characteristics
3. Prediction: XGBoost model with custom objective function generates risk levels for each street segment
4. Risk Scoring: Percentile-based ranking within elevated-risk segments
5. Visualization: Interactive map shows color-coded risk levels with hourly time controls

### Tech Highlights

- Background Updates: Two-phase system (prepare + deploy) ensures fresh data without user interruption
- Memory Management: Adaptive chunking based on available RAM
- On-demand Maps: Maps generated on-demand using preprocessed segment geometry for memory efficiency
- Caching Strategy: Multi-level caching for optimal performance

### Tech Stack

- **Backend:** Python, Pandas, XGBoost, Scikit-learn
- **Web Framework:** Dash (Flask-based)
- **Frontend:** Deck.gl for WebGL map rendering
- **Deployment:** Render (containerized)

---

### Test Set Metrics (2024 Hold-out Data)

| Metric | Value | Interpretation for Crash Risk Prediction |
|--------|-------|------------------------------------------|
| **ROC-AUC** | 0.92 | Model demonstrates excellent discriminative ability between crash and non-crash segment-hours; ranks 92% of crash instances higher than non-crash instances |
| **Average Precision** | 0.69 | Strong performance given class imbalance; substantially outperforms random baseline (~0.10) and naive strategies |
| **Lift @ Top 1%** | 10.36x | Highest-scoring 1% of segment-hour predictions contain 10.36% of all crashes; model successfully concentrates risk in top predictions |
| **Precision @ 50% Recall** | 0.77 | When threshold captures half of all crashes, 77% of flagged segment-hours actually experience crashes; false positive rate = 23% |

---

## Project Structure

```
pdx-crash-risk/
├── app/
│   ├── preprocessing.py           # Real-time feature generation
│   ├── predictor.py               # Prediction engine
│   ├── app.py                     # Dash application entry
│   ├── layout.py                  # Dashboard UI components
│   ├── callbacks.py               # Interactive functionality
│   ├── bg_updater.py              # Background refresh system
│   └── config.py                  # Configuration & styling
├── data/                          # Processed datasets
│   ├── id_lookup.csv              # Weather station id lookup
│   ├── segment_stats.parquet      # Historical crash stats for road segments
│   └── street_seg.parquet         # Info and geometry for road segments
├── models/                        # Trained models & artifacts
│   └── crash_model.pkl            # Final serialized crash prediction model
├── training/                      # ML-ready training data
│   ├── ml_input_data.parquet      # Final training/test data for model
│   ├── model_training.py          # XGBoost training pipeline
│   └── model_comparison.py        # Test file for comparing model performance
└── wrangling/                     # Initial data processing and engineering
    ├── initial_processing.R       # Feature engineering and model data prep 
    ├── crashes.csv                # Full set of Oregon crash data
    ├── streets.geojson            # Road network geometry and data
    ├── weather.csv                # Full weather data for 2019-2024
    └── street_segments.R          # Road network segmentation 


```

---


## Data Sources

- **Weather:** [Open-Meteo.com](https://open-meteo.com)
- **Crashes:** [ODOT Traffic Crash Reporting](https://tvc.odot.state.or.us/tvc/)
- **Streets:** [PortlandMaps Open Data](https://gis-pdx.opendata.arcgis.com/)
- **Maps:** © [Carto](https://carto.com/), © [OpenStreetMap](https://www.openstreetmap.org/) contributors

---

**NOTE:** This project is for demonstration purposes only. The crash risk predictions and associated scores should not be used for real-world traffic safety decisions.
