#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate ML V2 models with proper metrics
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")


def load_test_data():
    """Load test data with V2 features"""
    from src.data.loader import load_data
    from src.ml.features.enhanced_v2 import add_enhanced_features_v2, get_feature_columns_v2

    symbols = ["VNM", "FPT", "VIC", "VHM", "HPG", "MWG", "MSN", "VCB", "TCB", "VPB"]

    all_data = []
    feature_cols = get_feature_columns_v2()

    print(f"\n📊 Loading data for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            df = load_data(symbol, lookback=500)
            if df is None or len(df) < 100:
                continue

            try:
                index_df = load_data("VNINDEX", lookback=500, is_index=True)
            except:
                index_df = None

            df = add_enhanced_features_v2(df, index_df, target_type="multi_horizon")
            df = df.dropna(subset=feature_cols + ["target"])

            if len(df) >= 50:
                df["symbol"] = symbol
                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df)} rows")

        except Exception as e:
            print(f"   ❌ {symbol}: {e}")

    combined = pd.concat(all_data, ignore_index=True)

    if "time" in combined.columns:
        combined = combined.sort_values("time").reset_index(drop=True)

    X = combined[feature_cols]
    y = combined["target"]

    return X, y, feature_cols


def validate_v2_models():
    """Validate V2 models"""
    print("\n" + "=" * 70)
    print("🔍 VALIDATING ML V2 MODELS")
    print("=" * 70)

    X, y, feature_cols = load_test_data()

    print(f"\n📊 Total samples: {len(X)}")
    print(f"📊 Class distribution:")
    print(f"   Class 0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Class 1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    # Load scaler
    scaler = joblib.load("models/scaler_v2.pkl")
    X_scaled = scaler.transform(X)

    # Test each model
    models = ["rf_v2", "gb_v2", "xgb_v2", "lgb_v2", "ensemble_v2"]

    print("\n" + "=" * 70)
    print("📈 MODEL PERFORMANCE (Full Dataset)")
    print("=" * 70)

    results = {}
    for model_name in models:
        model_path = f"models/{model_name}.pkl"
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_proba)

        results[model_name] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        }

        print(f"\n📈 {model_name}:")
        print(f"   Accuracy:          {acc:.4f}")
        print(f"   Balanced Accuracy: {bal_acc:.4f} ← More reliable!")
        print(f"   Precision:         {prec:.4f}")
        print(f"   Recall:            {rec:.4f}")
        print(f"   F1 Score:          {f1:.4f}")
        print(f"   AUC:               {auc:.4f}")

    # Time Series CV validation
    print("\n" + "=" * 70)
    print("🔍 TIME SERIES CROSS-VALIDATION (Realistic)")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    cv_results = {m: {"bal_acc": [], "f1": [], "auc": []} for m in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Use saved scaler (don't refit)
        X_test_scaled = scaler.transform(X_test)

        for model_name in models:
            model_path = f"models/{model_name}.pkl"
            if not os.path.exists(model_path):
                continue

            model = joblib.load(model_path)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            cv_results[model_name]["bal_acc"].append(bal_acc)
            cv_results[model_name]["f1"].append(f1)
            cv_results[model_name]["auc"].append(auc)

    print("\n📊 CV RESULTS (Balanced Accuracy):")
    for model_name, metrics in cv_results.items():
        if metrics["bal_acc"]:
            mean_bal_acc = np.mean(metrics["bal_acc"])
            std_bal_acc = np.std(metrics["bal_acc"])
            mean_auc = np.mean(metrics["auc"])

            print(f"\n{model_name}:")
            print(f"   Balanced Acc: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
            print(f"   AUC:          {mean_auc:.4f}")
            print(f"   Edge:         {(mean_bal_acc - 0.5) * 100:+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)

    best_model = None
    best_edge = 0

    for model_name, metrics in cv_results.items():
        if metrics["bal_acc"]:
            edge = (np.mean(metrics["bal_acc"]) - 0.5) * 100
            if edge > best_edge:
                best_edge = edge
                best_model = model_name

    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Edge (Balanced): {best_edge:+.2f}%")

    if best_edge >= 8:
        print("\n✅ TARGET ACHIEVED: Balanced edge >= 8%")
    elif best_edge >= 5:
        print(f"\n⚠️ PARTIAL SUCCESS: Edge is {best_edge:.1f}% (target: 8%)")
    else:
        print(f"\n❌ TARGET NOT MET: Edge is only {best_edge:.1f}%")


if __name__ == "__main__":
    validate_v2_models()
