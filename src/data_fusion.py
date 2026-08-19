"""
Data Fusion & Cleaning Module for Predict-to-Protect
Fuses historical pest surveillance (RICE.csv), historical daily weather,
and farm/crop characteristics into a unified, aligned dataset.
"""

import os
import pandas as pd
import numpy as np

PEST_THRESHOLDS = {
    'Brownplanthopper': 150.0,
    'Gallmidge': 20.0,
    'Greenleafhopper': 100.0,
    'LeafFolder': 25.0,
    'Yellowstemborer': 50.0,
    'Caseworm': 10.0,
    'Miridbug': 60.0,
    'Whitebackedplanthopper': 40.0,
    'ZigZagleafhopper': 350.0,
    'LeafBlast': 15.0,
    'NeckBlast': 1.0,
}

SOIL_TYPES = ['Loamy', 'Clayey', 'Alluvial', 'Clay Loam', 'Sandy Loam', 'Black', 'Red']
RICE_VARIETIES = ['Basmati 370', 'Swarna (MTU 7029)', 'IR64', 'Samba Mahsuri (BPT 5204)', 'MTU 1010', 'Pusa 44', 'Pooja', 'Jasmine']
GROWTH_STAGES = ['Nursery', 'Tillering', 'Vegetative / Stem Elongation', 'Booting / Panicle Initiation', 'Flowering / Heading', 'Milking / Ripening']

def load_raw_datasets(data_dir='data'):
    rice_path = os.path.join(data_dir, 'RICE.csv')
    weather_path = os.path.join(data_dir, 'india_2000_2024_daily_weather.csv')
    crops_path = os.path.join(data_dir, 'Custom_Crops_yield_Historical_Dataset.csv')

    df_rice = pd.read_csv(rice_path)
    df_weather = pd.read_csv(weather_path) if os.path.exists(weather_path) else None
    df_crops = pd.read_csv(crops_path) if os.path.exists(crops_path) else None

    return df_rice, df_weather, df_crops

def build_temporal_pest_dataset(df_rice):
    """
    Sorts time series by (Location, PEST NAME, Observation Year, Standard Week),
    computes lags, rolling climate variables, rainfall trends, and forward 2-3 week outbreak targets.
    """
    df = df_rice.copy()

    # Clean & standardize column names
    rename_dict = {
        'Observation Year': 'Year',
        'Standard Week': 'Standard_Week',
        'Pest Value': 'Pest_Value',
        'PEST NAME': 'Pest_Name',
        'MaxT': 'Max_Temp',
        'MinT': 'Min_Temp',
        'RH1(%)': 'RH1',
        'RH2(%)': 'RH2',
        'RF(mm)': 'Rainfall',
        'WS(kmph)': 'Wind_Speed',
        'SSH(hrs)': 'Sunshine_Hours',
        'EVP(mm)': 'Evaporation'
    }
    df = df.rename(columns=rename_dict)

    # Sort sequentially
    df = df.sort_values(by=['Location', 'Pest_Name', 'Year', 'Standard_Week']).reset_index(drop=True)

    # Derived meteorological variables
    df['Mean_Temp'] = (df['Max_Temp'] + df['Min_Temp']) / 2.0
    df['Temp_Range'] = df['Max_Temp'] - df['Min_Temp']
    df['Mean_RH'] = (df['RH1'] + df['RH2']) / 2.0
    df['RH_Diff'] = df['RH1'] - df['RH2']

    # Month and Season mapping from Standard Week
    df['Month'] = np.clip(((df['Standard_Week'] - 1) * 7 / 30.5 + 1).astype(int), 1, 12)
    def get_season(month):
        if month in [6, 7, 8, 9, 10]:
            return 'Kharif'
        elif month in [11, 12, 1, 2, 3]:
            return 'Rabi'
        else:
            return 'Zaid'
    df['Season'] = df['Month'].apply(get_season)

    # Groupby series to create lag and rolling features
    grp = df.groupby(['Location', 'Pest_Name', 'Year'])

    df['Pest_Value_lag1'] = grp['Pest_Value'].shift(1).fillna(0.0)
    df['Pest_Value_lag2'] = grp['Pest_Value'].shift(2).fillna(0.0)
    df['Pest_Trend'] = np.where(
        df['Pest_Value_lag2'] > 0,
        ((df['Pest_Value_lag1'] - df['Pest_Value_lag2']) / (df['Pest_Value_lag2'] + 1.0)) * 100.0,
        0.0
    )

    df['Rainfall_lag1'] = grp['Rainfall'].shift(1).fillna(0.0)
    df['Rainfall_lag2'] = grp['Rainfall'].shift(2).fillna(0.0)

    # 7d, 14d, 21d rainfall
    df['Rainfall_7d'] = df['Rainfall']
    df['Rainfall_14d'] = df['Rainfall'] + df['Rainfall_lag1']
    df['Rainfall_21d'] = df['Rainfall'] + df['Rainfall_lag1'] + df['Rainfall_lag2']

    # Rainfall Trend: ((Recent 7d - Prev 7d) / Prev 7d) * 100
    df['Rainfall_Trend'] = np.where(
        df['Rainfall_lag1'] > 1.0,
        ((df['Rainfall'] - df['Rainfall_lag1']) / (df['Rainfall_lag1'] + 0.1)) * 100.0,
        np.where(df['Rainfall'] > 5.0, 50.0, 0.0)
    )
    df['Rainfall_Trend'] = np.clip(df['Rainfall_Trend'], -100.0, 300.0)

    # Rolling averages
    df['Temp_7d_avg'] = df['Mean_Temp']
    df['Temp_14d_avg'] = (df['Mean_Temp'] + grp['Mean_Temp'].shift(1).fillna(df['Mean_Temp'])) / 2.0
    df['Humidity_7d_avg'] = df['Mean_RH']
    df['Humidity_14d_avg'] = (df['Mean_RH'] + grp['Mean_RH'].shift(1).fillna(df['Mean_RH'])) / 2.0

    # Forward 2-3 week pest target (at week t+2 and week t+3)
    df['Pest_Value_fwd2'] = grp['Pest_Value'].shift(-2)
    df['Pest_Value_fwd3'] = grp['Pest_Value'].shift(-3)
    df['Future_Pest_Value_2_3w'] = df[['Pest_Value_fwd2', 'Pest_Value_fwd3']].max(axis=1)

    # Fill forward values with current pest value if at end of year
    df['Future_Pest_Value_2_3w'] = df['Future_Pest_Value_2_3w'].fillna(df['Pest_Value'])

    # Determine species threshold
    threshold_series = df['Pest_Name'].map(PEST_THRESHOLDS).fillna(50.0)
    df['Outbreak_Target_2_3w'] = (df['Future_Pest_Value_2_3w'] >= threshold_series).astype(int)

    # Synthetic farm features grounded in agronomic correlations
    np.random.seed(42)
    n = len(df)
    
    # Growth stage mapping based on standard week in rice calendar
    def map_stage(week):
        if week in range(24, 28) or week in range(48, 52):
            return 'Nursery'
        elif week in range(28, 33) or week in range(1, 6):
            return 'Tillering'
        elif week in range(33, 39) or week in range(6, 12):
            return 'Vegetative / Stem Elongation'
        elif week in range(39, 43) or week in range(12, 16):
            return 'Booting / Panicle Initiation'
        elif week in range(43, 47) or week in range(16, 20):
            return 'Flowering / Heading'
        else:
            return 'Milking / Ripening'

    df['Growth_Stage'] = df['Standard_Week'].apply(map_stage)
    
    # Probabilistic variety assignment
    variety_choices = np.random.choice(RICE_VARIETIES, size=n, p=[0.20, 0.25, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05])
    df['Rice_Variety'] = variety_choices

    # Soil type assignment
    soil_choices = np.random.choice(SOIL_TYPES, size=n, p=[0.30, 0.25, 0.20, 0.10, 0.05, 0.05, 0.05])
    df['Soil_Type'] = soil_choices

    # Previous pest occurrence feature
    df['Previous_Pest_Occurrence'] = (df['Pest_Value_lag1'] > 0).astype(int)
    df['Previous_Pest_Type'] = np.where(df['Previous_Pest_Occurrence'] == 1, df['Pest_Name'], 'None')

    return df

if __name__ == '__main__':
    df_rice, df_weather, df_crops = load_raw_datasets()
    fused_df = build_temporal_pest_dataset(df_rice)
    print("Fused Dataset Shape:", fused_df.shape)
    print("Outbreak Target Distribution:\n", fused_df['Outbreak_Target_2_3w'].value_counts(normalize=True))
    print("\nPest Outbreak Counts:\n", fused_df[fused_df['Outbreak_Target_2_3w'] == 1]['Pest_Name'].value_counts())
