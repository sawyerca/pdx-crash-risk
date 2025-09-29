# Portland Traffic Crash Risk Prediction

 **Live Demo:** [pdx-crash-risk.org](https://pdx-crash-risk.org)

An interactive web application that predicts hourly traffic crash risk for Portland street segments using machine learning and real-time weather data.

## Key Features

- **Real-time Predictions**: Hourly crash risk forecasts updated automatically
- **Interactive Map**: Explore risk levels across Portland's street network with time controls
- **Weather Integration**: Live weather data from Open-Meteo API drives predictions
- **Street-Level Granularity**: Risk assessment for road segments
- **Memory-Optimized**: Efficient processing of 25+ hours of citywide predictions

## Technical Stack

**Machine Learning:**
- XGBoost binary classifier trained on 2019-2023 crash data
- Features: weather conditions, temporal patterns, street characteristics, historical crash statistics
- Percentile-based risk scoring with statistical filtering

**Backend:**
- Python (Pandas, Scikit-learn, XGBoost)
- Dash framework for web application
- Background data pipeline with automatic hourly updates
- Memory-optimized chunked processing

**Frontend:**
- Interactive dashboard with Deck.gl map visualization
- Real-time data refresh notifications
- Responsive design with time slider controls

## How It Works

1. **Data Pipeline**: Fetches real-time weather for 25+ hours across Portland weather stations
2. **Feature Engineering**: Combines weather, temporal, and street characteristics
3. **Prediction**: XGBoost model generates crash probabilities for each street segment
4. **Risk Scoring**: Percentile-based ranking within elevated-risk segments
5. **Visualization**: Interactive map shows color-coded risk levels with hourly time controls

## Architecture Highlights

- **Background Updates**: Two-phase system (prepare + deploy) ensures fresh data without user interruption
- **Memory Management**: Adaptive chunking based on available RAM
- **Pre-generated Maps**: All hours rendered in advance for instant user interaction
- **Caching Strategy**: Multi-level caching for optimal performance

## Data Sources

- Weather data by [Open-Meteo.com](https://open-meteo.com)
- Crash data courtesy of [ODOT Crash Reporting](https://tvc.odot.state.or.us/tvc/)
- Road data courtesy of [PortlandMaps Open Data](https://gis-pdx.opendata.arcgis.com/)
- Maps via © [Carto](https://carto.com/about-carto/), © [OpenStreetMap](http://www.openstreetmap.org/about/) contributors


