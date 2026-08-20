"""
🌾 PREDICT-TO-PROTECT: AI-Powered Rice Pest Outbreak Early Warning & Farmer Advisory System
Main Streamlit Web Application
"""

import os
import sys
import json
import datetime
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from weather_service import fetch_live_weather, PRESET_LOCATIONS, geocode_location
from feature_engineering import build_single_inference_dataframe, ALL_FEATURE_COLUMNS, compute_rainfall_trend
from explainability import explain_prediction
from advisory_engine import generate_farmer_advisory, PEST_DETAILS

# Streamlit Page Config
st.set_page_config(
    page_title="Predict-to-Protect | AI Rice Pest Early Warning",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at 10% 20%, rgba(16, 37, 24, 0.05) 0%, rgba(248, 250, 252, 1) 90%);
    }
    
    /* Header Card */
    .hero-banner {
        background: linear-gradient(135deg, #064E3B 0%, #047857 50%, #059669 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.25), 0 8px 10px -6px rgba(6, 78, 59, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #D1FAE5;
        margin-top: 0.4rem;
        font-weight: 400;
        max-width: 800px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(255, 255, 255, 0.2);
        color: #ECFDF5;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Modern Glass Card */
    .glass-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.6rem;
    }
    
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Weather Grid Stat Box */
    .stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    
    .stat-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .stat-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 2px;
    }
    
    /* Trend Badges */
    .trend-up {
        color: #10B981;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    
    .trend-down {
        color: #EF4444;
        background: #FEF2F2;
        border: 1px solid #FECACA;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    
    .trend-stable {
        color: #F59E0B;
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    
    /* Risk Badges */
    .risk-high {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.35);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }
    
    .advisory-item {
        background: #F8FAFC;
        border-left: 4px solid #059669;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.45;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load Trained Models & Artifacts
@st.cache_resource
def load_all_models(models_dir='models'):
    def do_load():
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        rf_outbreak = joblib.load(os.path.join(models_dir, 'rf_outbreak_model.joblib'))
        xgb_outbreak = joblib.load(os.path.join(models_dir, 'xgb_outbreak_model.joblib'))
        rf_pest = joblib.load(os.path.join(models_dir, 'rf_pest_model.joblib'))
        xgb_pest = joblib.load(os.path.join(models_dir, 'xgb_pest_model.joblib'))
        pest_encoder = joblib.load(os.path.join(models_dir, 'pest_label_encoder.joblib'))
        shap_explainer = joblib.load(os.path.join(models_dir, 'shap_explainer.joblib'))
        
        with open(os.path.join(models_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        return {
            'preprocessor': preprocessor,
            'rf_outbreak': rf_outbreak,
            'xgb_outbreak': xgb_outbreak,
            'rf_pest': rf_pest,
            'xgb_pest': xgb_pest,
            'pest_encoder': pest_encoder,
            'shap_explainer': shap_explainer,
            'metadata': metadata,
            'status': 'loaded'
        }
    try:
        return do_load()
    except Exception as e:
        try:
            from train_models import train_and_evaluate_models
            train_and_evaluate_models(data_dir='data', output_dir=models_dir)
            return do_load()
        except Exception as retrain_e:
            return {'status': 'error', 'message': f"Failed to load: {e}. Auto cloud-retraining failed: {retrain_e}"}

models_dict = load_all_models()

# Top Hero Section
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div>
            <div class="badge-pill">🌾 AI-Powered Agricultural Decision Support System</div>
            <h1 class="hero-title">Predict-to-Protect</h1>
            <p class="hero-subtitle">
                Location-aware Rice Pest Outbreak Early Warning System. Combining real-time Open-Meteo climate intelligence, 
                historical surveillance fusion, and Random Forest + XGBoost Ensemble ML to forecast outbreaks <strong>2–3 weeks ahead</strong>.
            </p>
        </div>
        <div style="text-align: right;">
            <div class="badge-pill" style="background: rgba(16, 185, 129, 0.35); border-color: #34D399;">● LIVE ENSEMBLE ENGINE</div>
            <div style="color: #A7F3D0; font-size: 0.78rem; margin-top: 6px;">TreeSHAP Explainable AI Active</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Dashboard Layout
col_input, col_results = st.columns([1.1, 1.9], gap="large")

with col_input:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <h3 class="card-title">📍 Farm & Crop Profile</h3>
            <span class="badge-pill" style="background: #E2E8F0; color: #475569; border: none;">FARMER INPUT</span>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Location Selection
    loc_mode = st.radio(
        "Location Input Mode:",
        options=["Preset Rice Hub", "✏️ Enter Manual City / District"],
        index=0,
        horizontal=True
    )
    
    if loc_mode == "Preset Rice Hub":
        selected_location = st.selectbox(
            "🌍 Field Location (Auto-Geocoded)",
            options=list(PRESET_LOCATIONS.keys()),
            index=0,
            help="Selecting a location triggers automated real-time weather retrieval and historical rainfall trend calculation via Open-Meteo."
        )
        target_location = selected_location
    else:
        custom_loc = st.text_input(
            "✏️ Enter Manual Location (City / District / State):",
            value="Bhubaneswar, Odisha",
            help="Type any city or district name across India. System will automatically geocode and fetch live climate data."
        )
        target_location = custom_loc.strip() if custom_loc.strip() else "Cuttack, Odisha"
        
    # 2. Rice Variety Selection
    rice_variety = st.selectbox(
        "🌾 Rice Cultivar / Variety",
        options=['Swarna (MTU 7029)', 'Basmati 370', 'IR64', 'Samba Mahsuri (BPT 5204)', 'MTU 1010', 'Pusa 44', 'Pooja', 'Jasmine'],
        index=0
    )
    
    # 3. Growth Stage
    growth_stage = st.selectbox(
        "🌱 Crop Growth Stage",
        options=['Nursery', 'Tillering', 'Vegetative / Stem Elongation', 'Booting / Panicle Initiation', 'Flowering / Heading', 'Milking / Ripening'],
        index=2
    )
    
    # 4. Soil Type
    soil_type = st.selectbox(
        "🌍 Soil Texture Type",
        options=['Loamy', 'Clayey', 'Alluvial', 'Clay Loam', 'Sandy Loam', 'Black', 'Red'],
        index=0
    )
    
    # 5. Pest History
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        prev_occurrence = st.radio(
            "Prior Pest Occurrence?",
            options=["Yes", "No"],
            index=0,
            horizontal=True
        )
    with col_p2:
        prev_pest_type = st.selectbox(
            "Previously Seen Pest:",
            options=['Brownplanthopper', 'Yellowstemborer', 'Gallmidge', 'LeafFolder', 'Greenleafhopper', 'LeafBlast', 'NeckBlast', 'Caseworm', 'Whitebackedplanthopper', 'None'],
            index=0 if prev_occurrence == "Yes" else 9,
            disabled=(prev_occurrence == "No")
        )
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Automatic Weather Retrieval Panel
    with st.spinner("Fetching live climate data from Open-Meteo..."):
        weather_data = fetch_live_weather(target_location)
        
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-header">
            <h3 class="card-title">🌦️ Automated Climate Intelligence</h3>
            <span class="badge-pill" style="background: #E0F2FE; color: #0284C7; border: 1px solid #BAE6FD;">
                {weather_data.get('source', 'Open-Meteo')}
            </span>
        </div>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.75rem;">
            📍 Resolved: <strong>{weather_data['location_name']}</strong> ({weather_data['latitude']:.2f}°N, {weather_data['longitude']:.2f}°E)
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 10px;">
            <div class="stat-box">
                <div class="stat-label">🌡️ Temperature</div>
                <div class="stat-value">{weather_data['current_temp']:.1f} °C</div>
                <div class="stat-sub">Max: {weather_data['temp_max']:.1f}° | Min: {weather_data['temp_min']:.1f}°</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">💧 Humidity (RH)</div>
                <div class="stat-value">{weather_data['humidity']:.0f} %</div>
                <div class="stat-sub">Morning RH: {weather_data['humidity_morning']:.0f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">🌧️ 7-Day Rainfall</div>
                <div class="stat-value">{weather_data['rainfall_7d']:.1f} mm</div>
                <div class="stat-sub">14-Day Sum: {weather_data['rainfall_14d']:.1f} mm</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">📈 Rainfall Trend</div>
                <div class="stat-value" style="font-size: 1.15rem;">
                    <span class="{weather_data['rainfall_trend_badge']}">{weather_data['rainfall_trend_icon']} {abs(weather_data['rainfall_trend_pct']):.1f}%</span>
                </div>
                <div class="stat-sub">{weather_data['rainfall_trend_label']} vs prior 7d</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; background: #F1F5F9; padding: 6px 12px; border-radius: 8px;">
            <span>💨 Wind: <strong>{weather_data['wind_speed']:.1f} km/h</strong></span>
            <span>☀️ Sunshine: <strong>{weather_data['sunshine_hours']:.1f} hrs</strong></span>
            <span>💧 Evap: <strong>{weather_data['evaporation']:.1f} mm</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    predict_btn = st.button("🔮 Run 2–3 Week Outbreak Prediction", type="primary", use_container_width=True)

with col_results:
    # Build inference input dataframe
    inf_df = build_single_inference_dataframe(
        location=target_location.split(',')[0],
        rice_variety=rice_variety,
        growth_stage=growth_stage,
        soil_type=soil_type,
        previous_pest_occurrence=1 if prev_occurrence == "Yes" else 0,
        previous_pest_type=prev_pest_type,
        live_weather=weather_data
    )
    
    # Execute Prediction
    if models_dict.get('status') == 'loaded':
        preprocessor = models_dict['preprocessor']
        rf_outbreak = models_dict['rf_outbreak']
        xgb_outbreak = models_dict['xgb_outbreak']
        rf_pest = models_dict['rf_pest']
        xgb_pest = models_dict['xgb_pest']
        pest_encoder = models_dict['pest_encoder']
        shap_explainer = models_dict['shap_explainer']
        
        # Preprocess row
        X_inf = preprocessor.transform(inf_df[ALL_FEATURE_COLUMNS])
        
        # 1. Binary Outbreak Probabilities
        rf_prob = float(rf_outbreak.predict_proba(X_inf)[0, 1])
        xgb_prob = float(xgb_outbreak.predict_proba(X_inf)[0, 1])
        ensemble_prob = float(0.45 * rf_prob + 0.55 * xgb_prob)
        
        # Adjust ensemble probability slightly based on severe microclimate triggers if present
        if weather_data['humidity'] > 80 and weather_data['rainfall_trend_pct'] > 20 and prev_occurrence == 'Yes':
            ensemble_prob = min(0.95, ensemble_prob + 0.08)
            
        # Classify Risk Level
        if ensemble_prob >= 0.65:
            risk_tier = 'HIGH'
            risk_badge_class = 'risk-high'
            risk_icon = '🔴'
        elif ensemble_prob >= 0.38:
            risk_tier = 'MEDIUM'
            risk_badge_class = 'risk-medium'
            risk_icon = '🟡'
        else:
            risk_tier = 'LOW'
            risk_badge_class = 'risk-low'
            risk_icon = '🟢'
            
        # 2. Multi-Class Pest Species Distribution
        rf_pest_probs = rf_pest.predict_proba(X_inf)[0]
        xgb_pest_probs = xgb_pest.predict_proba(X_inf)[0]
        ens_pest_probs = 0.5 * rf_pest_probs + 0.5 * xgb_pest_probs
        
        # If user selected a previous pest, give realistic prior weight
        if prev_occurrence == 'Yes' and prev_pest_type in pest_encoder.classes_:
            p_idx = list(pest_encoder.classes_).index(prev_pest_type)
            ens_pest_probs[p_idx] += 0.25
            ens_pest_probs = ens_pest_probs / np.sum(ens_pest_probs)
            
        pred_pest_idx = np.argmax(ens_pest_probs)
        predicted_pest = pest_encoder.classes_[pred_pest_idx]
        
        # 3. Explainability (SHAP)
        xai_results = explain_prediction(xgb_outbreak, shap_explainer, X_inf, inf_df)
        
        # 4. Generate Farmer Advisory
        advisory = generate_farmer_advisory(
            predicted_pest=predicted_pest,
            risk_level=risk_tier,
            probability=ensemble_prob,
            farm_details={'growth_stage': growth_stage, 'soil_type': soil_type, 'variety': rice_variety},
            weather_details=weather_data
        )
        
        # Render Prediction Result Cards
        st.markdown(f"""
        <div class="glass-card" style="border-left: 6px solid {advisory['badge_color']};">
            <div class="card-header">
                <div>
                    <span class="badge-pill" style="background: {advisory['badge_color']}; color: white;">HORIZON: 2–3 WEEKS AHEAD</span>
                    <h2 style="margin: 6px 0 0 0; font-size: 1.6rem; color: #0F172A; font-weight: 800;">
                        {advisory['alert_title']}
                    </h2>
                </div>
                <div style="text-align: right;">
                    <span class="{risk_badge_class}">{risk_icon} {risk_tier} RISK</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 15px; margin-top: 1rem; align-items: center;">
                <div style="background: #F8FAFC; padding: 1rem; border-radius: 12px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748B;">PRIMARY TARGET SPECIES</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A; margin-top: 2px;">{advisory['common_name']}</div>
                    <div style="font-size: 0.78rem; font-style: italic; color: #475569;">({advisory['scientific_name']})</div>
                </div>
                <div style="background: #F8FAFC; padding: 1rem; border-radius: 12px; border: 1px solid #E2E8F0; text-align: center;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748B;">OUTBREAK PROBABILITY</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {advisory['badge_color']};">{ensemble_prob:.1%}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Soft Calibrated Ensemble</div>
                </div>
                <div style="background: #F8FAFC; padding: 1rem; border-radius: 12px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B;">MODEL CONSENSUS</div>
                    <div style="font-size: 0.85rem; color: #334155; margin-top: 4px;">🌳 RF: <strong>{rf_prob:.1%}</strong></div>
                    <div style="font-size: 0.85rem; color: #334155;">⚡ XGB: <strong>{xgb_prob:.1%}</strong></div>
                </div>
            </div>
            <p style="margin-top: 1rem; font-size: 0.95rem; color: #334155; line-height: 1.5;">
                {advisory['summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Charts Grid
        c_gauge, c_pest_dist = st.columns([1, 1.2])
        
        with c_gauge:
            st.markdown('<div class="glass-card"><h4 class="card-title">🎯 Outbreak Probability Gauge</h4>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ensemble_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "%", 'font': {'size': 36, 'family': 'Plus Jakarta Sans', 'color': '#0F172A'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': advisory['badge_color'], 'thickness': 0.28},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 38], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [38, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "#DC2626", 'width': 3},
                        'thickness': 0.75,
                        'value': 65
                    }
                }
            ))
            fig_gauge.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=25, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_pest_dist:
            st.markdown('<div class="glass-card"><h4 class="card-title">🐛 Pest Species Probability Spectrum</h4>', unsafe_allow_html=True)
            
            # Format pest species df
            pest_df = pd.DataFrame({
                'Pest': pest_encoder.classes_,
                'Probability': ens_pest_probs * 100.0
            }).sort_values('Probability', ascending=True).tail(6)
            
            fig_pest = px.bar(
                pest_df,
                x='Probability',
                y='Pest',
                orientation='h',
                text=pest_df['Probability'].apply(lambda x: f"{x:.1f}%"),
                color='Probability',
                color_continuous_scale=['#CBD5E1', '#F59E0B', '#EF4444']
            )
            fig_pest.update_layout(
                height=240,
                margin=dict(l=10, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="",
                yaxis_title="",
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9', range=[0, max(pest_df['Probability']) * 1.25]),
                yaxis=dict(tickfont=dict(size=11, family='Plus Jakarta Sans'))
            )
            fig_pest.update_traces(textposition='outside', marker_line_color='#FFFFFF', marker_line_width=1)
            st.plotly_chart(fig_pest, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Explainable AI (SHAP) Section
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <h3 class="card-title">🔍 Explainable AI: Why did the Model Predict this?</h3>
                <span class="badge-pill" style="background: #EEF2FF; color: #4F46E5; border: 1px solid #C7D2FE;">SHAP ATTRIBUTION</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Agronomic Translation**: {xai_results['summary']}")
        
        # SHAP Bar Chart
        shap_df = pd.DataFrame(xai_results['top_factors'])
        fig_shap = px.bar(
            shap_df,
            x='percentage',
            y='name',
            orientation='h',
            color='impact',
            color_discrete_map={'Increases Risk': '#EF4444', 'Decreases Risk': '#10B981'},
            text=shap_df.apply(lambda r: f"{r['value_display']} ({r['percentage']:.1f}%)", axis=1)
        )
        fig_shap.update_layout(
            height=220,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Relative Feature Importance Contribution (%)",
            yaxis_title="",
            yaxis=dict(autorange="reversed", tickfont=dict(size=12, family='Plus Jakarta Sans')),
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_shap.update_traces(textposition='outside')
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actionable Farmer Advisory Section
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-header">
                <h3 class="card-title">👨‍🌾 Recommended Preventive Action Checklist (IPM)</h3>
                <span class="badge-pill" style="background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0;">ACTION PLAN</span>
            </div>
            <div style="margin-bottom: 0.75rem; font-size: 0.88rem; color: #475569;">
                <strong>Vulnerable Crop Stage:</strong> {advisory['vulnerable_stage']} | 
                <strong>Favorable Conditions:</strong> {advisory['favorable_conditions']}
            </div>
        """, unsafe_allow_html=True)
        
        for idx, action in enumerate(advisory['action_list'], 1):
            st.markdown(f'<div class="advisory-item"><strong>Step {idx}:</strong> {action}</div>', unsafe_allow_html=True)
            
        if advisory['contextual_notes']:
            st.markdown("<div style='margin-top: 10px; font-weight: 600; color: #1E293B;'>📌 Micro-Climate & Field Context:</div>", unsafe_allow_html=True)
            for note in advisory['contextual_notes']:
                st.markdown(f"<div style='font-size: 0.85rem; color: #475569; margin-left: 8px; margin-bottom: 4px;'>{note}</div>", unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.error(f"Models could not be loaded: {models_dict.get('message', 'Unknown error')}. Please make sure models are trained.")
