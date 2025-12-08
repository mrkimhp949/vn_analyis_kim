#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML Model Performance Validation Script
Xác minh accuracy thực tế của ML models
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data():
    """Load data và prepare features"""
    from src.data.loader import load_data
    from src.ml.features.enhanced_v2 import add_enhanced_features, get_feature_columns

    # Test với nhiều symbols
    symbols = ["VNM", "FPT", "VIC", "VHM", "HPG", "MWG", "MSN", "VCB", "TCB", "VPB"]

    all_data = []
    feature_cols = get_feature_columns()

    print(f"\n📊 Loading data for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            df = load_data(symbol, lookback=500)
            if df is None or len(df) < 100:
                print(f"   ⚠️ {symbol}: Insufficient data")
                continue

            # Load index for RS calculation
            try:
                index_df = load_data("VNINDEX", lookback=500, is_index=True)
            except:
                index_df = None

            # Add features
            df = add_enhanced_features(df, index_df)

            # Filter valid rows
            df = df.dropna(subset=feature_cols + ["target"])

            if len(df) >= 50:
                df["symbol"] = symbol
                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df)} rows")
            else:
                print(f"   ⚠️ {symbol}: Only {len(df)} valid rows")

        except Exception as e:
            print(f"   ❌ {symbol}: {e}")

    if not all_data:
        return None, None, None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total samples: {len(combined)}")

    X = combined[feature_cols]
    y = combined["target"]

    return X, y, feature_cols


def validate_saved_models(X, y):
    """Validate các models đã save"""
    import joblib

    print("\n" + "=" * 70)
    print("🔍 VALIDATING SAVED MODELS")
    print("=" * 70)

    models_dir = "models"
    results = {}

    # Load scaler
    scaler_path = os.path.join(models_dir, "scaler_enhanced.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        print(f"✅ Loaded scaler from {scaler_path}")
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("⚠️ No saved scaler, using new StandardScaler")

    # Test each model
    model_files = [
        ("rf_enhanced.pkl", "Random Forest"),
        ("xgb_enhanced.pkl", "XGBoost"),
        ("lgb_enhanced.pkl", "LightGBM"),
        ("ensemble_gb.pkl", "Gradient Boosting"),
        ("random_forest.pkl", "RF (old)"),
        ("xgboost.pkl", "XGB (old)"),
    ]

    for model_file, model_name in model_files:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            continue

        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_scaled)

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)

            results[model_name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

            print(f"\n📈 {model_name}:")
            print(f"   Accuracy:  {acc:.4f} ({(acc-0.5)*100:+.1f}% vs random)")
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall:    {rec:.4f}")
            print(f"   F1 Score:  {f1:.4f}")

        except Exception as e:
            print(f"\n❌ {model_name}: {e}")

    return results


def validate_with_time_series_cv(X, y, feature_cols):
    """Validate với Time Series Cross-Validation (realistic)"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("🔍 TIME SERIES CROSS-VALIDATION (More Realistic)")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train simple RF
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f"   Fold {fold+1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    print(f"\n📊 AVERAGE RESULTS (Time Series CV):")
    print(f"   Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"   Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"   Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"   F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    edge = (np.mean(accuracies) - 0.5) * 100
    print(f"\n🎯 EDGE OVER RANDOM: {edge:+.2f}%")

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "edge": edge,
    }


def check_class_distribution(y):
    """Check class balance"""
    print("\n" + "=" * 70)
    print("📊 CLASS DISTRIBUTION")
    print("=" * 70)

    counts = y.value_counts()
    total = len(y)

    print(f"   Class 0 (Down/Hold): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"   Class 1 (Up):        {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"   Total:               {total}")

    # Check if balanced
    ratio = counts.get(1, 0) / total
    if 0.45 <= ratio <= 0.55:
        print("   ✅ Classes are balanced")
    else:
        print(f"   ⚠️ Classes are imbalanced (ratio: {ratio:.2f})")


def main():
    print("\n" + "=" * 70)
    print("🔬 ML MODEL PERFORMANCE VALIDATION")
    print("=" * 70)

    # Load data
    X, y, feature_cols = load_and_prepare_data()

    if X is None:
        print("❌ Failed to load data")
        return

    # Check class distribution
    check_class_distribution(y)

    # Validate saved models
    saved_results = validate_saved_models(X, y)

    # Time series CV (more realistic)
    cv_results = validate_with_time_series_cv(X, y, feature_cols)

    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)

    if saved_results:
        best_model = max(saved_results.items(), key=lambda x: x[1]["accuracy"])
        print(f"\n🏆 Best Saved Model: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   Edge: {(best_model[1]['accuracy']-0.5)*100:+.2f}%")

    print(f"\n📈 Time Series CV Results:")
    print(f"   Accuracy: {cv_results['accuracy']:.4f}")
    print(f"   Edge: {cv_results['edge']:+.2f}%")

    # Verdict
    print("\n" + "=" * 70)
    if cv_results["edge"] < 5:
        print("❌ VERDICT: Model has NO significant edge over random")
        print("   Recommendation: Review features and target definition")
    elif cv_results["edge"] < 10:
        print("⚠️ VERDICT: Model has WEAK edge ({:.1f}%)".format(cv_results["edge"]))
        print("   Recommendation: May not be profitable after transaction costs")
    else:
        print("✅ VERDICT: Model has REASONABLE edge ({:.1f}%)".format(cv_results["edge"]))
    print("=" * 70)


if __name__ == "__main__":
    main()
