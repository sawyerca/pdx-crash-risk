# Portland Crash Risk Dashboard

Real-time traffic crash risk prediction for Portland street segments using machine learning and current weather conditions.

 **Live Demo**: [pdx-crash-risk.org](https://pdx-crash-risk.org)

## Overview

Interactive web dashboard that displays hourly crash risk predictions on a map of Portland streets. Built with XGBoost model trained on 2019-2024 crash and weather data.

## Tech Stack

- **Languages & Frameworks**: R, Python, Dash
- **ML Model**: XGBoost
- **Data Services**: Open-Meteo API for real-time weather
- **Visualization**: Deck.GL for high-performance map rendering
- **Deployment**: Memory-optimized cloud hosting on Render with automated background updates

## Key Features

- Hourly risk forecasts (25 hours ahead)
- Color-coded street segments 
- Automatic updates every hour
- Memory-optimized for cloud deployment

## Architecture Sample


- **preprocessing.py**: Real-time weather data pipeline
- **predictor.py**: XGBoost model inference  
- **bg_updater.py**: Scheduled background updates
- **callbacks.py**: Interactive dashboard logic


## Data Sources
- [Open-Meteo weather API](https://open-meteo.com)
- [ODOT Crash Reporting](https://tvc.odot.state.or.us/tvc/)
- [PortlandMaps Open Data](https://gis-pdx.opendata.arcgis.com/)
