"""
Flask Backend Server for Predict-to-Protect Web Dashboard
Serves the web UI and provides API endpoints for:
- /api/weather: Auto-geocoding & live Open-Meteo climate fetching
- /api/predict: Feature engineering, Ensemble ML prediction, SHAP attribution, and Farmer Advisory
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from weather_service import fetch_live_weather, PRESET_LOCATIONS, geocode_location
from feature_engineering import build_single_inference_dataframe, ALL_FEATURE_COLUMNS
from explainability import explain_prediction
from advisory_engine import generate_farmer_advisory, PEST_DETAILS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Load ML artifacts at startup
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
rf_outbreak = joblib.load(os.path.join(MODELS_DIR, 'rf_outbreak_model.joblib'))
xgb_outbreak = joblib.load(os.path.join(MODELS_DIR, 'xgb_outbreak_model.joblib'))
rf_pest = joblib.load(os.path.join(MODELS_DIR, 'rf_pest_model.joblib'))
xgb_pest = joblib.load(os.path.join(MODELS_DIR, 'xgb_pest_model.joblib'))
pest_encoder = joblib.load(os.path.join(MODELS_DIR, 'pest_label_encoder.joblib'))
shap_explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.joblib'))

with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
    model_metadata = json.load(f)

print("[INFO] All Predict-to-Protect models successfully loaded into Flask server.")

@app.route('/')
def index():
    return render_template('index.html', preset_locations=list(PRESET_LOCATIONS.keys()))

@app.route('/api/weather', methods=['GET'])
def get_weather():
    location = request.args.get('location', 'Cuttack, Odisha')
    weather_data = fetch_live_weather(location)
    return jsonify(weather_data)

@app.route('/api/predict', methods=['POST'])
def predict_outbreak():
    try:
        data = request.get_json(force=True)
        
        location = data.get('location', 'Cuttack, Odisha')
        rice_variety = data.get('rice_variety', 'Swarna (MTU 7029)')
        growth_stage = data.get('growth_stage', 'Vegetative / Stem Elongation')
        soil_type = data.get('soil_type', 'Loamy')
        prev_pest_occurrence = int(data.get('previous_pest_occurrence', 1))
        prev_pest_type = data.get('previous_pest_type', 'Brownplanthopper')
        
        # Weather details provided or fetched
        weather_data = data.get('weather')
        if not weather_data:
            weather_data = fetch_live_weather(location)
            
        # Build single-row feature dataframe
        inf_df = build_single_inference_dataframe(
            location=location.split(',')[0],
            rice_variety=rice_variety,
            growth_stage=growth_stage,
            soil_type=soil_type,
            previous_pest_occurrence=prev_pest_occurrence,
            previous_pest_type=prev_pest_type if prev_pest_occurrence == 1 else 'None',
            live_weather=weather_data
        )
        
        # Preprocess features
        X_inf = preprocessor.transform(inf_df[ALL_FEATURE_COLUMNS])
        
        # 1. Binary Outbreak Probabilities
        rf_prob = float(rf_outbreak.predict_proba(X_inf)[0, 1])
        xgb_prob = float(xgb_outbreak.predict_proba(X_inf)[0, 1])
        ensemble_prob = float(0.45 * rf_prob + 0.55 * xgb_prob)
        
        # Microclimate sensitivity adjustment if severe conditions co-occur
        if weather_data.get('humidity', 80) > 80 and weather_data.get('rainfall_trend_pct', 0) > 20 and prev_pest_occurrence == 1:
            ensemble_prob = min(0.96, ensemble_prob + 0.08)
            
        if ensemble_prob >= 0.65:
            risk_tier = 'HIGH'
            risk_color = '#EF4444'
            risk_icon = '🔴'
        elif ensemble_prob >= 0.38:
            risk_tier = 'MEDIUM'
            risk_color = '#F59E0B'
            risk_icon = '🟡'
        else:
            risk_tier = 'LOW'
            risk_color = '#10B981'
            risk_icon = '🟢'
            
        # 2. Multi-Class Pest Classification
        rf_pest_probs = rf_pest.predict_proba(X_inf)[0]
        xgb_pest_probs = xgb_pest.predict_proba(X_inf)[0]
        ens_pest_probs = 0.5 * rf_pest_probs + 0.5 * xgb_pest_probs
        
        if prev_pest_occurrence == 1 and prev_pest_type in pest_encoder.classes_:
            p_idx = list(pest_encoder.classes_).index(prev_pest_type)
            ens_pest_probs[p_idx] += 0.25
            ens_pest_probs = ens_pest_probs / np.sum(ens_pest_probs)
            
        pred_pest_idx = int(np.argmax(ens_pest_probs))
        predicted_pest = pest_encoder.classes_[pred_pest_idx]
        
        # Pest spectrum top 5
        pest_spectrum = []
        for idx, cls in enumerate(pest_encoder.classes_):
            pest_spectrum.append({
                'pest_name': cls,
                'probability': round(float(ens_pest_probs[idx]) * 100.0, 1)
            })
        pest_spectrum = sorted(pest_spectrum, key=lambda x: x['probability'], reverse=True)[:6]
        
        # 3. SHAP Explainability
        xai = explain_prediction(xgb_outbreak, shap_explainer, X_inf, inf_df)
        
        # 4. Actionable Farmer Advisory
        advisory = generate_farmer_advisory(
            predicted_pest=predicted_pest,
            risk_level=risk_tier,
            probability=ensemble_prob,
            farm_details={'growth_stage': growth_stage, 'soil_type': soil_type, 'variety': rice_variety},
            weather_details=weather_data
        )
        
        response = {
            'status': 'success',
            'location': location,
            'prediction_window': 'Next 2–3 Weeks (14–21 Days)',
            'outbreak_probability': round(ensemble_prob * 100.0, 1),
            'risk_level': risk_tier,
            'risk_color': risk_color,
            'risk_icon': risk_icon,
            'rf_probability': round(rf_prob * 100.0, 1),
            'xgb_probability': round(xgb_prob * 100.0, 1),
            'predicted_pest': predicted_pest,
            'scientific_name': advisory['scientific_name'],
            'common_name': advisory['common_name'],
            'pest_spectrum': pest_spectrum,
            'xai': xai,
            'advisory': advisory,
            'weather': weather_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Launching Predict-to-Protect Web Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
