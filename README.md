# Hurricane ZERO - Machine Learning Hurricane Prediction System

**A data-driven approach to hurricane tracking and analysis using advanced machine learning techniques**

## Overview

Hurricane ZERO is a comprehensive machine learning-based hurricane prediction system that leverages historical Atlantic hurricane data to predict wind speeds and classify hurricane categories. The system employs supervised learning techniques (Random Forest) to provide real-time predictions for disaster preparedness and emergency response.

### Key Features

- **Wind Speed Prediction**: RMSE of 0.68 mph
- **Hurricane Category Classification**: 98% accuracy
- **Real-time Parameter Input**: Interactive prediction interface
- **Atlantic Region Optimized**: Models trained on NOAA HURDAT2 data
- **Comprehensive Analysis**: Includes clustering and association rule mining

## Performance Metrics

### Regression Model (Wind Speed Prediction)
- **Mean Squared Error (MSE)**: 0.47
- **Root Mean Squared Error (RMSE)**: 0.68 mph
- **R² Score**: 0.95+
- **Response Time**: < 50ms per prediction

### Classification Model (Category Prediction)
- **Accuracy**: 98%
- **Precision**: 0.98
- **Recall**: 0.98
- **F1-Score**: 0.98

## System Architecture

```
Input Parameters → Preprocessing → Model Pipeline → Output Predictions
      ↓               ↓              ↓                 ↓
[Lat, Lon,     [Normalization,  [Regression &    [Wind Speed,
 Pressure,      Validation,      Classification    Category,
 Temperature]   Feature Eng.]    Models]          Safety Tips]
```

## Installation

### System Requirements

- **Python**: 3.7+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB available space
- **OS**: Windows 10, macOS 10.15, Linux Ubuntu 18.04+

### Dependencies

```bash
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
numpy>=1.21.0
```

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/Swaygordon/Hurricane-regression-model.git
   cd Hurricane-regression-model
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv hurricane_env
   source hurricane_env/bin/activate  # Linux/Mac
   # hurricane_env\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model Files**
   Ensure these files are present:
   - `hurricane_regression_model.pkl`
   - `hurricane_classification_model.pkl`

5. **Run System Test**
   ```bash
   python Prototype1AL.py
   ```

## Usage

### Interactive Mode

```bash
python Prototype1AL.py
```

The system will prompt for input parameters:

```
Enter real-world hurricane parameters:
Latitude: 25.5
Longitude: -80.2
Pressure (mb): 965
Temperature (°C): 28.5
```

### Example Output

```
Hurricane Prediction Results
Predicted Wind Speed: 95.23 mph
Predicted Hurricane Category: Category 2
Safety Tips: Evacuate if advised. Turn off gas, electricity, and water if instructed.
```

### Programmatic Usage

```python
import joblib
import pandas as pd
from hurricane_predictor import HurricanePredictor

# Load models
predictor = HurricanePredictor()
predictor.load_models()

# Make prediction
params = {
    'latitude': 25.5,
    'longitude': -80.2,
    'pressure': 965,
    'temperature': 28.5
}

wind_speed, category, safety_tips = predictor.predict(params)
print(f"Wind Speed: {wind_speed:.2f} mph")
print(f"Category: {category}")
```

## Data Requirements

### Input Parameters

| Parameter   | Type  | Unit    | Range        | Description           |
|-------------|-------|---------|-------------|-----------------------|
| Latitude    | Float | Degrees | 7.2 to 81.0  | Geographic latitude   |
| Longitude   | Float | Degrees | -109.5 to 63.0| Geographic longitude |
| Pressure    | Float | mb      | 920 to 1020  | Atmospheric pressure  |
| Temperature | Float | °C      | Variable     | Sea surface temp      |

### Hurricane Categories

- **Tropical Depression**: < 39 mph
- **Tropical Storm**: 39-73 mph
- **Category 1**: 74-95 mph
- **Category 2**: 96-110 mph
- **Category 3**: 111-129 mph (Major Hurricane)
- **Category 4**: 130-156 mph (Major Hurricane)
- **Category 5**: 157+ mph (Major Hurricane)

## Model Details

### Random Forest Regression (Wind Speed)
- **Target**: Wind Speed (mph)
- **Features**: Latitude, Longitude, Pressure, Temperature
- **Hyperparameters**:
  ```python
  {
      'n_estimators': 200,
      'max_depth': None,
      'min_samples_split': 2,
      'min_samples_leaf': 1,
      'random_state': 42
  }
  ```

### Random Forest Classification (Category)
- **Target**: Hurricane Category
- **Features**: Latitude, Longitude, Pressure, Temperature, Wind Speed
- **Categories**: Tropical Depression through Category 5

## Data Sources

- **Primary**: NOAA HURDAT2 Database
- **Secondary**: NASA Climate Data
- **Supplementary**: Kaggle Hurricane Datasets

## Research Insights

### Feature Importance Analysis
- **Wind Speed**: 85% importance
- **Pressure**: 10% importance
- **Temperature**: 3% importance
- **Latitude, Longitude**: Negligible

### Clustering Analysis
The system identified three distinct hurricane clusters:
1. **Low Wind Speed**: Tropical storms and weak hurricanes
2. **Moderate Wind Speed**: Intensifying storms
3. **High Wind Speed**: Major hurricanes with extreme wind speeds

### Association Rule Mining
Key findings from temporal analysis:
- Morning conditions show 67% chance of medium wind speed with low pressure
- Night conditions correlate with warm temperatures and medium wind speeds
- Time-dependent relationships between storm characteristics

## Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/Swaygordon/Hurricane-regression-model.git .
EXPOSE 8000
CMD ["python", "Prototype1AL.py"]
```

## API Reference

### Core Classes

#### HurricanePredictor

**Methods:**
- `load_models()`: Load pre-trained models
- `predict(parameters: dict) -> tuple`: Make predictions
- `categorize_hurricane(wind_speed: float) -> tuple`: Categorize by wind speed

### Utility Functions

- `validate_input(parameters: dict) -> bool`: Input validation
- `preprocess_data(raw_data: dict) -> pd.DataFrame`: Data preprocessing

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Verify model files are in correct directory
   - Check file permissions and paths

2. **Invalid Input Parameters**
   - Check input ranges match specifications
   - Verify latitude/longitude are for Atlantic region

3. **Poor Prediction Accuracy**
   - Ensure input parameters are for Atlantic region
   - Verify units are correct (mb for pressure, °C for temperature)

## Contributing

This project was developed by Group 23 for COMP 358 - Artificial Intelligence:
- Elijah Ato Baiden - UEB3517922
- Mensah Jonathan - UEB3518722
- Samuel Adzaho - UEB3510522
- Gabriel Asankomah Gordon-Mensah - UEB3503522
- John Koyah - UEB3504821

## Future Enhancements

### Immediate Improvements
- Add XGBoost for improved RMSE
- Implement ensemble methods
- Enhanced feature engineering with pressure gradients

### Advanced Enhancements
- Deep Learning Integration (LSTM for time-series)
- Multimodal Learning with satellite imagery
- Physics-Informed Models
- Real-Time Learning capabilities

## Disclaimer

Hurricane ZERO is designed to assist in hurricane prediction and preparedness. Predictions should be used in conjunction with official meteorological services and emergency management guidance. The system is optimized for Atlantic region hurricanes and may have reduced accuracy for other geographic regions.

**Important Notes:**
- Always consult official weather services for emergency decisions
- This system provides supplementary information only
- Predictions are based on current atmospheric conditions and may change rapidly
- Regular model updates are essential for maintaining accuracy

## License

This project is part of academic coursework at the University of Energy and Natural Resources.

## References

- National Oceanic and Atmospheric Administration (NOAA) HURDAT2 Dataset
- NASA Climate Data
- Recent research on advanced machine learning techniques for tropical cyclone forecasting (2024)
- Google DeepMind GraphCast hurricane forecasting system (2024-2025)
