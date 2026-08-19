"""
Automated End-to-End Integration Test for Predict-to-Protect
Validates the entire pipeline:
Location Geocoding -> Weather Retrieval -> Feature Engineering -> Ensemble Prediction -> SHAP -> Farmer Advisory
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from weather_service import fetch_live_weather, PRESET_LOCATIONS
from feature_engineering import build_single_inference_dataframe, ALL_FEATURE_COLUMNS
from explainability import explain_prediction
from advisory_engine import generate_farmer_advisory

def run_integration_test():
    print("==========================================================")
    print("RUNNING PREDICT-TO-PROTECT END-TO-END PIPELINE TEST")
    print("==========================================================")

    # 1. Test Weather Service
    test_locations = ['Cuttack, Odisha', 'Raipur, Chhattisgarh', 'Ludhiana, Punjab']
    for loc in test_locations:
        print(f"\n[Step 1] Fetching live climate for: {loc}...")
        w_data = fetch_live_weather(loc)
        assert 'current_temp' in w_data, "Missing current_temp in weather data"
        assert 'humidity' in w_data, "Missing humidity in weather data"
        assert 'rainfall_trend_pct' in w_data, "Missing rainfall_trend_pct in weather data"
        print(f"  ✓ Resolved: {w_data['location_name']} (Lat: {w_data['latitude']:.2f}, Lon: {w_data['longitude']:.2f})")
        print(f"  ✓ Temp: {w_data['current_temp']:.1f}°C, RH: {w_data['humidity']:.0f}%, 7d Rain: {w_data['rainfall_7d']:.1f}mm, Trend: {w_data['rainfall_trend_label']} ({w_data['rainfall_trend_pct']:+.1f}%)")

    # 2. Test Feature Engineering
    print("\n[Step 2] Testing feature construction for single-row inference...")
    sample_weather = fetch_live_weather('Cuttack, Odisha')
    inf_df = build_single_inference_dataframe(
        location='Cuttack',
        rice_variety='Swarna (MTU 7029)',
        growth_stage='Vegetative / Stem Elongation',
        soil_type='Loamy',
        previous_pest_occurrence=1,
        previous_pest_type='Brownplanthopper',
        live_weather=sample_weather
    )
    assert inf_df.shape[0] == 1, "Inference DataFrame should have exactly 1 row"
    assert len(ALL_FEATURE_COLUMNS) == 32, f"Expected 32 features, got {len(ALL_FEATURE_COLUMNS)}"
    print(f"  ✓ Inference DataFrame constructed with {inf_df.shape[1]} columns.")

    # 3. Test Models & Artifact Loading
    print("\n[Step 3] Loading saved ML models from models/...")
    preprocessor = joblib.load('models/preprocessor.joblib')
    rf_outbreak = joblib.load('models/rf_outbreak_model.joblib')
    xgb_outbreak = joblib.load('models/xgb_outbreak_model.joblib')
    rf_pest = joblib.load('models/rf_pest_model.joblib')
    xgb_pest = joblib.load('models/xgb_pest_model.joblib')
    pest_encoder = joblib.load('models/pest_label_encoder.joblib')
    shap_explainer = joblib.load('models/shap_explainer.joblib')
    print("  ✓ All 7 model/preprocessor artifacts loaded successfully.")

    # 4. Test Preprocessing and Inference
    print("\n[Step 4] Executing Ensemble Prediction...")
    X_inf = preprocessor.transform(inf_df[ALL_FEATURE_COLUMNS])
    
    rf_prob = float(rf_outbreak.predict_proba(X_inf)[0, 1])
    xgb_prob = float(xgb_outbreak.predict_proba(X_inf)[0, 1])
    ensemble_prob = float(0.45 * rf_prob + 0.55 * xgb_prob)

    # Multi-class pest
    rf_pest_probs = rf_pest.predict_proba(X_inf)[0]
    xgb_pest_probs = xgb_pest.predict_proba(X_inf)[0]
    ens_pest_probs = 0.5 * rf_pest_probs + 0.5 * xgb_pest_probs
    pred_pest_idx = np.argmax(ens_pest_probs)
    pred_pest_name = pest_encoder.classes_[pred_pest_idx]

    risk_tier = 'HIGH' if ensemble_prob >= 0.65 else ('MEDIUM' if ensemble_prob >= 0.38 else 'LOW')

    print(f"  ✓ RF Outbreak Prob : {rf_prob:.1%}")
    print(f"  ✓ XGB Outbreak Prob: {xgb_prob:.1%}")
    print(f"  ✓ Ensemble Risk    : {ensemble_prob:.1%} [{risk_tier} RISK]")
    print(f"  ✓ Predicted Species: {pred_pest_name} ({ens_pest_probs[pred_pest_idx]:.1%})")

    # 5. Test SHAP Explainability
    print("\n[Step 5] Computing SHAP Explainability...")
    xai = explain_prediction(xgb_outbreak, shap_explainer, X_inf, inf_df)
    assert 'top_factors' in xai, "Missing top_factors in SHAP result"
    assert 'summary' in xai, "Missing summary in SHAP result"
    print(f"  ✓ SHAP Explanation: {xai['summary']}")
    for f in xai['top_factors'][:3]:
        print(f"    - {f['name']} ({f['value_display']}): {f['percentage']:.1f}% [{f['impact']}]")

    # 6. Test Advisory Generation
    print("\n[Step 6] Generating Farmer Advisory Package...")
    advisory = generate_farmer_advisory(
        predicted_pest=pred_pest_name,
        risk_level=risk_tier,
        probability=ensemble_prob,
        farm_details={'growth_stage': 'Vegetative / Stem Elongation', 'soil_type': 'Loamy'},
        weather_details=sample_weather
    )
    assert len(advisory['action_list']) > 0, "Advisory action list cannot be empty"
    print(f"  ✓ Alert Title: {advisory['alert_title']}")
    print(f"  ✓ Common Name: {advisory['common_name']} ({advisory['scientific_name']})")
    print(f"  ✓ Action Steps Generated: {len(advisory['action_list'])} actionable IPM recommendations")

    print("\n==========================================================")
    print("🎉 ALL PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == '__main__':
    run_integration_test()
