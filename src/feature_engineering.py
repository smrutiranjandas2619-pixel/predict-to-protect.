"""
Feature Engineering Module for Predict-to-Protect
Defines feature extraction, scaling, encoding, and vector conversion
for both training pipelines and real-time single-sample inference.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERICAL_FEATURES = [
    'Max_Temp', 'Min_Temp', 'Mean_Temp', 'Temp_Range',
    'RH1', 'RH2', 'Mean_RH', 'RH_Diff',
    'Rainfall', 'Rainfall_7d', 'Rainfall_14d', 'Rainfall_21d', 'Rainfall_Trend',
    'Wind_Speed', 'Sunshine_Hours', 'Evaporation',
    'Temp_7d_avg', 'Temp_14d_avg', 'Humidity_7d_avg', 'Humidity_14d_avg',
    'Pest_Value_lag1', 'Pest_Value_lag2', 'Pest_Trend',
    'Standard_Week', 'Month'
]

CATEGORICAL_FEATURES = [
    'Location', 'Season', 'Growth_Stage', 'Rice_Variety',
    'Soil_Type', 'Previous_Pest_Occurrence', 'Previous_Pest_Type'
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

SOIL_PH_MAP = {
    'Alluvial': 6.8,
    'Loamy': 6.5,
    'Clayey': 6.2,
    'Clay Loam': 6.4,
    'Sandy Loam': 6.0,
    'Black': 7.6,
    'Red': 5.8
}

STAGE_SUSCEPTIBILITY = {
    'Brownplanthopper': {'Nursery': 0.2, 'Tillering': 0.6, 'Vegetative / Stem Elongation': 0.9, 'Booting / Panicle Initiation': 1.0, 'Flowering / Heading': 0.9, 'Milking / Ripening': 0.5},
    'Yellowstemborer': {'Nursery': 0.5, 'Tillering': 0.9, 'Vegetative / Stem Elongation': 1.0, 'Booting / Panicle Initiation': 0.9, 'Flowering / Heading': 0.8, 'Milking / Ripening': 0.3},
    'Gallmidge': {'Nursery': 0.4, 'Tillering': 1.0, 'Vegetative / Stem Elongation': 0.7, 'Booting / Panicle Initiation': 0.3, 'Flowering / Heading': 0.1, 'Milking / Ripening': 0.0},
    'LeafFolder': {'Nursery': 0.3, 'Tillering': 0.8, 'Vegetative / Stem Elongation': 1.0, 'Booting / Panicle Initiation': 0.8, 'Flowering / Heading': 0.6, 'Milking / Ripening': 0.2},
    'Greenleafhopper': {'Nursery': 0.6, 'Tillering': 0.9, 'Vegetative / Stem Elongation': 0.9, 'Booting / Panicle Initiation': 0.7, 'Flowering / Heading': 0.5, 'Milking / Ripening': 0.2},
    'Whitebackedplanthopper': {'Nursery': 0.3, 'Tillering': 0.8, 'Vegetative / Stem Elongation': 0.9, 'Booting / Panicle Initiation': 0.7, 'Flowering / Heading': 0.4, 'Milking / Ripening': 0.2},
    'Caseworm': {'Nursery': 0.8, 'Tillering': 0.9, 'Vegetative / Stem Elongation': 0.4, 'Booting / Panicle Initiation': 0.1, 'Flowering / Heading': 0.0, 'Milking / Ripening': 0.0},
    'Miridbug': {'Nursery': 0.2, 'Tillering': 0.6, 'Vegetative / Stem Elongation': 0.8, 'Booting / Panicle Initiation': 0.9, 'Flowering / Heading': 0.8, 'Milking / Ripening': 0.4},
    'ZigZagleafhopper': {'Nursery': 0.4, 'Tillering': 0.8, 'Vegetative / Stem Elongation': 0.8, 'Booting / Panicle Initiation': 0.6, 'Flowering / Heading': 0.4, 'Milking / Ripening': 0.2},
    'LeafBlast': {'Nursery': 0.9, 'Tillering': 0.9, 'Vegetative / Stem Elongation': 0.8, 'Booting / Panicle Initiation': 0.5, 'Flowering / Heading': 0.4, 'Milking / Ripening': 0.2},
    'NeckBlast': {'Nursery': 0.0, 'Tillering': 0.1, 'Vegetative / Stem Elongation': 0.3, 'Booting / Panicle Initiation': 0.7, 'Flowering / Heading': 1.0, 'Milking / Ripening': 0.9}
}

def create_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def compute_rainfall_trend(recent_7d_rainfall, previous_7d_rainfall):
    """
    Calculates the rainfall trend percentage and classification.
    """
    if previous_7d_rainfall > 1.0:
        pct = ((recent_7d_rainfall - previous_7d_rainfall) / (previous_7d_rainfall + 0.1)) * 100.0
    else:
        if recent_7d_rainfall > 15.0:
            pct = 60.0
        elif recent_7d_rainfall > 5.0:
            pct = 25.0
        else:
            pct = 0.0

    pct = float(np.clip(pct, -100.0, 300.0))

    if pct > 20.0:
        label = 'Increasing'
        icon = '↑'
        badge_class = 'trend-up'
    elif pct < -20.0:
        label = 'Decreasing'
        icon = '↓'
        badge_class = 'trend-down'
    else:
        label = 'Stable'
        icon = '→'
        badge_class = 'trend-stable'

    return {
        'trend_pct': pct,
        'label': label,
        'icon': icon,
        'badge_class': badge_class,
        'display': f"{icon} {abs(pct):.1f}% {label}"
    }

def build_single_inference_dataframe(
    location,
    rice_variety,
    growth_stage,
    soil_type,
    previous_pest_occurrence,
    previous_pest_type,
    live_weather,
    standard_week=None,
    month=None,
    recent_pest_value=0.0
):
    """
    Builds a single-row DataFrame suitable for preprocessing and model inference
    from farmer inputs and auto-retrieved weather.
    """
    import datetime
    now = datetime.datetime.now()
    if standard_week is None:
        standard_week = int(now.strftime("%U")) + 1
    if month is None:
        month = now.month

    def get_season(m):
        if m in [6, 7, 8, 9, 10]:
            return 'Kharif'
        elif m in [11, 12, 1, 2, 3]:
            return 'Rabi'
        else:
            return 'Zaid'

    season = get_season(month)

    temp_max = live_weather.get('temp_max', 31.0)
    temp_min = live_weather.get('temp_min', 23.0)
    mean_temp = (temp_max + temp_min) / 2.0
    temp_range = temp_max - temp_min

    rh1 = live_weather.get('humidity_morning', live_weather.get('humidity', 80.0))
    rh2 = live_weather.get('humidity_evening', max(50.0, rh1 - 15.0))
    mean_rh = (rh1 + rh2) / 2.0
    rh_diff = rh1 - rh2

    rainfall = live_weather.get('rainfall_7d', live_weather.get('rainfall', 15.0))
    rainfall_14d = live_weather.get('rainfall_14d', rainfall * 1.8)
    rainfall_21d = live_weather.get('rainfall_21d', rainfall * 2.5)
    rainfall_trend = live_weather.get('rainfall_trend_pct', 20.0)

    wind_speed = live_weather.get('wind_speed', 10.0)
    sunshine_hours = live_weather.get('sunshine_hours', 6.5)
    evaporation = live_weather.get('evaporation', 4.2)

    temp_7d_avg = live_weather.get('temp_7d_avg', mean_temp)
    temp_14d_avg = live_weather.get('temp_14d_avg', mean_temp)
    humidity_7d_avg = live_weather.get('humidity_7d_avg', mean_rh)
    humidity_14d_avg = live_weather.get('humidity_14d_avg', mean_rh)

    pest_val_lag1 = float(recent_pest_value if previous_pest_occurrence == 1 else 0.0)
    pest_val_lag2 = float(pest_val_lag1 * 0.8)
    pest_trend = 15.0 if previous_pest_occurrence == 1 else 0.0

    row_data = {
        'Max_Temp': temp_max,
        'Min_Temp': temp_min,
        'Mean_Temp': mean_temp,
        'Temp_Range': temp_range,
        'RH1': rh1,
        'RH2': rh2,
        'Mean_RH': mean_rh,
        'RH_Diff': rh_diff,
        'Rainfall': rainfall,
        'Rainfall_7d': rainfall,
        'Rainfall_14d': rainfall_14d,
        'Rainfall_21d': rainfall_21d,
        'Rainfall_Trend': rainfall_trend,
        'Wind_Speed': wind_speed,
        'Sunshine_Hours': sunshine_hours,
        'Evaporation': evaporation,
        'Temp_7d_avg': temp_7d_avg,
        'Temp_14d_avg': temp_14d_avg,
        'Humidity_7d_avg': humidity_7d_avg,
        'Humidity_14d_avg': humidity_14d_avg,
        'Pest_Value_lag1': pest_val_lag1,
        'Pest_Value_lag2': pest_val_lag2,
        'Pest_Trend': pest_trend,
        'Standard_Week': standard_week,
        'Month': month,
        'Location': location,
        'Season': season,
        'Growth_Stage': growth_stage,
        'Rice_Variety': rice_variety,
        'Soil_Type': soil_type,
        'Previous_Pest_Occurrence': 1 if str(previous_pest_occurrence).lower() in ['1', 'yes', 'true'] else 0,
        'Previous_Pest_Type': previous_pest_type if previous_pest_type else 'None'
    }

    return pd.DataFrame([row_data])
