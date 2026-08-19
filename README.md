---
title: Predict-To-Protect
emoji: 🌾
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.60.0
app_file: app.py
pinned: false
---

# 🌾 PREDICT-TO-PROTECT
### Location-Aware AI for Rice Pest Outbreak Early Warning, Explainable Risk Assessment & Farmer Advisory

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost%20%7C%20SHAP-green.svg)]()
[![App](https://img.shields.io/badge/UI-Streamlit%20%2B%20Plotly-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## 📌 Executive Summary & Problem Solved
Traditional pest management in agriculture is **reactive**: farmers notice visible crop damage only after pest populations reach critical destructive levels, leading to severe yield losses (up to 30–70%) and indiscriminate pesticide spraying.

**Predict-to-Protect** transforms pest management into a **proactive, location-aware early warning copilot**:
1. The farmer enters basic farm and crop parameters (location, rice cultivar, growth stage, soil type, previous pest history).
2. The system **automatically geocodes the location and queries real-time Open-Meteo meteorological data** (temperature, morning/evening relative humidity, precipitation, wind speed, sunshine hours).
3. The system computes dynamic temporal metrics like **Rainfall Trend %** (`((Recent 7d - Prev 7d) / Prev 7d) * 100`) and rolling climate averages.
4. An **Ensemble Machine Learning Architecture (Random Forest + XGBoost)** forecasts the outbreak risk **2–3 weeks (14–21 days) ahead** and classifies the specific target rice pest species among 11 major pests.
5. **TreeSHAP Explainable AI (XAI)** identifies the exact microclimatic and crop factors driving the risk score and translates them into plain-language agricultural explanations.
6. A dynamic **Integrated Pest Management (IPM) Advisory Engine** generates actionable, prioritized preventive guidelines tailored to the predicted pest species and risk tier.

---

## 🏗️ End-to-End System Architecture

```
                 📍 Location Selection (Farmer)
                               │
                               ▼
                 🌍 Geocoding & Open-Meteo Weather Service
                 (Auto-retrieves Temp, RH, Rain, Wind)
                               │
                               ▼
                     ┌──────────────────┐
                     │ LIVE CLIMATE DATA│
                     │ Temp, RH, Rain,  │
                     │ Rainfall Trend % │
                     └────────┬─────────┘
                              │
                              ▼
                 ⚙️ Multi-Source Feature Engineering
                 (Lag Features, Rolling Averages, Crop Susceptibility)
                              │
                              ▼
                 🧠 Calibrated Soft-Voting Ensemble
                     ┌────────┴────────┐
                     ▼                 ▼
             🌳 Random Forest      ⚡ XGBoost
                     └────────┬────────┘
                              ▼
                 🎯 2-3 Week Outbreak Probability &
                    Pest Species Classification (11 Pests)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             🔍 TreeSHAP XAI      👨‍🌾 Dynamic IPM Farmer
             Feature Attribution      Advisory Engine
                    └─────────┬─────────┘
                              ▼
                 🌾 Interactive Web Dashboard (Streamlit)
```

---

## 📂 Multi-Dataset Intelligence Fusion
The project unifies three complementary historical datasets without naive concatenation:
- **`data/RICE.csv` (19,404 records, 1959–2011)**: Core pest intelligence dataset spanning 6 major Indian agricultural research stations (*Cuttack, Raipur, Ludhiana, Maruteru, Palampur, Rajendranagar*) and 11 distinct rice pest types.
- **`data/india_2000_2024_daily_weather.csv` (91,320 records)**: High-resolution historical weather patterns across Indian agro-climatic zones.
- **`data/Custom_Crops_yield_Historical_Dataset.csv` (50,765 records, 311 districts)**: Crop phenology, soil pH, nutrient profiles, and yield correlations.
- **Live Open-Meteo API**: Live weather observations and short-term microclimate forecasts.

---

## 🏆 Supported Rice Pest Species (11 Classes)
1. **Brown Planthopper (BPH)** (*Nilaparvata lugens*)
2. **Yellow Stem Borer (YSB)** (*Scirpophaga incertulas*)
3. **Rice Gall Midge** (*Orseolia oryzae*)
4. **Rice Leaf Folder** (*Cnaphalocrocis medinalis*)
5. **Green Leafhopper (GLH)** (*Nephotettix virescens*)
6. **Rice Leaf Blast** (*Magnaporthe oryzae*)
7. **Rice Neck Blast** (*Magnaporthe oryzae*)
8. **Rice Caseworm** (*Nymphula depunctalis*)
9. **White-Backed Planthopper (WBPH)** (*Sogatella furcifera*)
10. **Mirid Bug** (*Cyrtorhinus lividipennis*)
11. **Zigzag Leafhopper** (*Recilia dorsalis*)

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost shap plotly requests joblib
```

### 2. Train Models (Automated Pipeline)
```bash
python src/train_models.py
```
*Evaluates using a strict temporal split (Years <= 2008 for training, > 2008 for future testing) to ensure zero data leakage.*

### 3. Run Pipeline Verification Test
```bash
python test_pipeline.py
```

### 4. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Summary
- **Ensemble ROC-AUC**: `0.9059`
- **Ensemble Recall (Outbreak Detection)**: `87.07%`
- **Ensemble Accuracy**: `80.57%`
- **Brier Probability Calibration Score**: `0.1259`
