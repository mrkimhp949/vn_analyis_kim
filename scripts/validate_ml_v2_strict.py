#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict validation - ensure no data leakage
Uses completely unseen symbols for final test
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def validate_on_unseen_symbols():
    """Test on symbols NOT used in training"""
    from src.data.loader import load_data
    from src.ml.features.enhanced_v2 import add_enhanced_features_v2, get_feature_columns_v2

    print("\n" + "=" * 70)
    print("🔬 STRICT VALIDATION - UNSEEN SYMBOLS")
    print("=" * 70)

    # These symbols were NOT in training set
    test_symbols = ["REE", "DGC", "PNJ", "VRE", "NVL", "POW", "GVR", "BCM"]

    all_data = []
    feature_cols = get_feature_columns_v2()

    print(f"\n📊 Loading UNSEEN symbols for testing...")

    for symbol in test_symbols:
        try:
            df = load_data(symbol, lookback=300)
            if df is None or len(df) < 100:
                print(f"   ⚠️ {symbol}: Insufficient data")
                continue

            try:
                index_df = load_data("VNINDEX", lookback=300, is_index=True)
            except:
                index_df = None

            df = add_enhanced_features_v2(df, index_df, target_type="multi_horizon")
            df = df.dropna(subset=feature_cols + ["target"])

            if len(df) >= 30:
                df["symbol"] = symbol
                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df)} rows")

        except Exception as e:
            print(f"   ❌ {symbol}: {e}")

    if not all_data:
        print("❌ No valid test data")
        return

    combined = pd.concat(all_data, ignore_index=True)

    X_test = combined[feature_cols]
    y_test = combined["target"]

    print(f"\n📊 Test samples: {len(X_test)}")
    print(f"📊 Class distribution:")
    print(f"   Class 0: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"   Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

    # Load scaler and models
    scaler = joblib.load("models/scaler_v2.pkl")
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("📈 RESULTS ON COMPLETELY UNSEEN DATA")
    print("=" * 70)

    models = ["rf_v2", "xgb_v2", "lgb_v2", "ensemble_v2"]

    for model_name in models:
        model_path = f"models/{model_name}.pkl"
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        edge = (bal_acc - 0.5) * 100

        print(f"\n📈 {model_name}:")
        print(f"   Balanced Accuracy: {bal_acc:.4f}")
        print(f"   AUC:               {auc:.4f}")
        print(f"   F1 Score:          {f1:.4f}")
        print(f"   Edge vs Random:    {edge:+.2f}%")

    print("\n" + "=" * 70)


def validate_forward_looking():
    """
    Most strict test: Train on old data, test on recent data
    Simulates real trading scenario
    """
    from src.data.loader import load_data
    from src.ml.features.enhanced_v2 import add_enhanced_features_v2, get_feature_columns_v2
    from sklearn.ensemble import RandomForestClassifier

    print("\n" + "=" * 70)
    print("🔬 FORWARD-LOOKING VALIDATION (Most Realistic)")
    print("=" * 70)

    symbols = ["VNM", "FPT", "VIC", "VHM", "HPG", "MWG", "VCB", "TCB"]
    feature_cols = get_feature_columns_v2()

    all_data = []

    print(f"\n📊 Loading data...")

    for symbol in symbols:
        try:
            df = load_data(symbol, lookback=500)
            if df is None or len(df) < 200:
                continue

            try:
                index_df = load_data("VNINDEX", lookback=500, is_index=True)
            except:
                index_df = None

            df = add_enhanced_features_v2(df, index_df, target_type="multi_horizon")
            df = df.dropna(subset=feature_cols + ["target"])

            if len(df) >= 100:
                df["symbol"] = symbol
                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df)} rows")

        except Exception as e:
            print(f"   ❌ {symbol}: {e}")

    combined = pd.concat(all_data, ignore_index=True)

    if "time" in combined.columns:
        combined = combined.sort_values("time").reset_index(drop=True)

    # Split: 80% train, 20% test (most recent)
    split_idx = int(len(combined) * 0.8)

    train_data = combined.iloc[:split_idx]
    test_data = combined.iloc[split_idx:]

    X_train = train_data[feature_cols]
    y_train = train_data["target"]
    X_test = test_data[feature_cols]
    y_test = test_data["target"]

    print(f"\n📊 Train: {len(X_train)} samples (older data)")
    print(f"📊 Test:  {len(X_test)} samples (recent data)")
    print(f"📊 Test class distribution: {y_test.value_counts(normalize=True).to_dict()}")

    # Train fresh model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RF with same params as V2
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    edge = (bal_acc - 0.5) * 100

    print("\n" + "=" * 70)
    print("📈 FORWARD-LOOKING RESULTS (Train on past, test on future)")
    print("=" * 70)
    print(f"\n   Balanced Accuracy: {bal_acc:.4f}")
    print(f"   AUC:               {auc:.4f}")
    print(f"   F1 Score:          {f1:.4f}")
    print(f"   Edge vs Random:    {edge:+.2f}%")

    if edge >= 8:
        print("\n✅ CONFIRMED: Model has real predictive power!")
    elif edge >= 5:
        print(f"\n⚠️ MODERATE: Edge is {edge:.1f}% (acceptable)")
    else:
        print(f"\n❌ WEAK: Edge is only {edge:.1f}%")

    return edge


if __name__ == "__main__":
    validate_on_unseen_symbols()
    edge = validate_forward_looking()

    print("\n" + "=" * 70)
    print("📋 FINAL VERDICT")
    print("=" * 70)

    if edge >= 8:
        print("✅ ML V2 model achieves target 58-62% accuracy (8-12% edge)")
        print("   Ready for production use")
    else:
        print(f"⚠️ Current edge: {edge:.1f}%")
        print("   May need further tuning")
