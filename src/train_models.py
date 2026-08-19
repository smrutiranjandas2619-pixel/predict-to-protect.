"""
Model Training & Evaluation Pipeline for Predict-to-Protect
Trains Random Forest and XGBoost classifiers for:
1. 2-3 Week Forward Outbreak Occurrence & Probability
2. Pest Species Identification (11 Rice Pests)
3. Soft Calibrated Ensemble & SHAP Baseline Serialization
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, classification_report
import shap

from data_fusion import load_raw_datasets, build_temporal_pest_dataset
from feature_engineering import ALL_FEATURE_COLUMNS, create_preprocessor

def train_and_evaluate_models(data_dir='data', output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading datasets and building temporal feature dataset...")
    df_rice, df_weather, df_crops = load_raw_datasets(data_dir)
    fused_df = build_temporal_pest_dataset(df_rice)

    print(f"Total dataset size: {fused_df.shape[0]} rows, {len(ALL_FEATURE_COLUMNS)} feature columns.")

    # Time-based train / validation / test split (e.g. Years <= 2005 for train, > 2005 for test)
    # If years range from 1959 to 2011:
    split_year = fused_df['Year'].quantile(0.80)
    print(f"Time-based split at Year: {int(split_year)}")

    train_mask = fused_df['Year'] <= split_year
    test_mask = fused_df['Year'] > split_year

    # Feature matrices
    X = fused_df[ALL_FEATURE_COLUMNS]
    y_outbreak = fused_df['Outbreak_Target_2_3w']
    
    # Target 2: Pest Species
    pest_encoder = LabelEncoder()
    y_pest = pest_encoder.fit_transform(fused_df['Pest_Name'])

    X_train = X[train_mask]
    y_outbreak_train = y_outbreak[train_mask]
    y_pest_train = y_pest[train_mask]

    X_test = X[test_mask]
    y_outbreak_test = y_outbreak[test_mask]
    y_pest_test = y_pest[test_mask]

    print(f"Train samples: {len(X_train)} ({y_outbreak_train.mean():.1%} positive)")
    print(f"Test samples: {len(X_test)} ({y_outbreak_test.mean():.1%} positive)")

    # 1. Fit Preprocessor
    print("\nFitting ColumnTransformer preprocessor...")
    preprocessor = create_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Processed feature matrix shape: {X_train_proc.shape}")

    # -------------------------------------------------------------
    # 2. Train Outbreak Prediction Models (Binary Outbreak in 2-3w)
    # -------------------------------------------------------------
    print("\nTraining Random Forest Outbreak Model...")
    rf_outbreak = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_outbreak.fit(X_train_proc, y_outbreak_train)

    print("Training XGBoost Outbreak Model...")
    xgb_outbreak = XGBClassifier(
        n_estimators=160,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.6,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_outbreak.fit(X_train_proc, y_outbreak_train)

    # Evaluate Outbreak Models
    rf_outbreak_probs = rf_outbreak.predict_proba(X_test_proc)[:, 1]
    xgb_outbreak_probs = xgb_outbreak.predict_proba(X_test_proc)[:, 1]

    # Weighted Ensemble (50/50 or tuned)
    ensemble_outbreak_probs = 0.45 * rf_outbreak_probs + 0.55 * xgb_outbreak_probs
    ensemble_preds = (ensemble_outbreak_probs >= 0.45).astype(int)

    rf_auc = roc_auc_score(y_outbreak_test, rf_outbreak_probs)
    xgb_auc = roc_auc_score(y_outbreak_test, xgb_outbreak_probs)
    ens_auc = roc_auc_score(y_outbreak_test, ensemble_outbreak_probs)

    rf_acc = accuracy_score(y_outbreak_test, (rf_outbreak_probs >= 0.5).astype(int))
    xgb_acc = accuracy_score(y_outbreak_test, (xgb_outbreak_probs >= 0.5).astype(int))
    ens_acc = accuracy_score(y_outbreak_test, ensemble_preds)

    ens_f1 = f1_score(y_outbreak_test, ensemble_preds)
    ens_prec = precision_score(y_outbreak_test, ensemble_preds)
    ens_rec = recall_score(y_outbreak_test, ensemble_preds)
    ens_brier = brier_score_loss(y_outbreak_test, ensemble_outbreak_probs)

    print("\n================ OUTBREAK MODEL EVALUATION ================")
    print(f"Random Forest AUC : {rf_auc:.4f} | Accuracy: {rf_acc:.4f}")
    print(f"XGBoost AUC       : {xgb_auc:.4f} | Accuracy: {xgb_acc:.4f}")
    print(f"Ensemble AUC      : {ens_auc:.4f} | Accuracy: {ens_acc:.4f}")
    print(f"Ensemble Precision: {ens_prec:.4f} | Recall: {ens_rec:.4f} | F1: {ens_f1:.4f} | Brier: {ens_brier:.4f}")
    print("===========================================================")

    # -------------------------------------------------------------
    # 3. Train Pest Species Multi-Class Classifier
    # -------------------------------------------------------------
    print("\nTraining Random Forest Pest Classifier (11 classes)...")
    rf_pest = RandomForestClassifier(
        n_estimators=120,
        max_depth=15,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_pest.fit(X_train_proc, y_pest_train)

    print("Training XGBoost Pest Classifier (11 classes)...")
    xgb_pest = XGBClassifier(
        n_estimators=130,
        max_depth=6,
        learning_rate=0.09,
        subsample=0.85,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_pest.fit(X_train_proc, y_pest_train)

    rf_pest_probs = rf_pest.predict_proba(X_test_proc)
    xgb_pest_probs = xgb_pest.predict_proba(X_test_proc)
    ens_pest_probs = 0.5 * rf_pest_probs + 0.5 * xgb_pest_probs
    ens_pest_preds = np.argmax(ens_pest_probs, axis=1)

    pest_acc = accuracy_score(y_pest_test, ens_pest_preds)
    print(f"\nPest Species Multi-class Ensemble Accuracy: {pest_acc:.4f}")

    # -------------------------------------------------------------
    # 4. Fit SHAP Background Explainer
    # -------------------------------------------------------------
    print("\nComputing TreeSHAP baseline explainer...")
    # Sample background instances for fast TreeSHAP calculation
    background_sample = X_train_proc[np.random.choice(X_train_proc.shape[0], size=min(150, X_train_proc.shape[0]), replace=False)]
    shap_explainer = shap.TreeExplainer(xgb_outbreak, data=background_sample)

    # Get feature names from preprocessor
    try:
        encoded_feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        encoded_feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    # -------------------------------------------------------------
    # 5. Save Artifacts & Metadata
    # -------------------------------------------------------------
    print("\nSaving trained models and preprocessor to models/ ...")
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(rf_outbreak, os.path.join(output_dir, 'rf_outbreak_model.joblib'))
    joblib.dump(xgb_outbreak, os.path.join(output_dir, 'xgb_outbreak_model.joblib'))
    joblib.dump(rf_pest, os.path.join(output_dir, 'rf_pest_model.joblib'))
    joblib.dump(xgb_pest, os.path.join(output_dir, 'xgb_pest_model.joblib'))
    joblib.dump(pest_encoder, os.path.join(output_dir, 'pest_label_encoder.joblib'))
    joblib.dump(shap_explainer, os.path.join(output_dir, 'shap_explainer.joblib'))

    metrics = {
        'split_year': int(split_year),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'outbreak_metrics': {
            'rf_auc': float(rf_auc),
            'rf_accuracy': float(rf_acc),
            'xgb_auc': float(xgb_auc),
            'xgb_accuracy': float(xgb_acc),
            'ensemble_auc': float(ens_auc),
            'ensemble_accuracy': float(ens_acc),
            'ensemble_f1': float(ens_f1),
            'ensemble_precision': float(ens_prec),
            'ensemble_recall': float(ens_rec),
            'ensemble_brier': float(ens_brier)
        },
        'pest_classifier_accuracy': float(pest_acc),
        'pest_classes': pest_encoder.classes_.tolist(),
        'feature_columns': ALL_FEATURE_COLUMNS,
        'encoded_feature_names': encoded_feature_names
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("All models and metadata successfully saved!")
    return metrics

if __name__ == '__main__':
    train_and_evaluate_models()
