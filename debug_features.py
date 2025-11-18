#!/usr/bin/env python3
"""
Debug feature generation to understand ML scaling issues
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.loader import load_data
from src.ml.features.technical import add_ml_features, get_feature_columns

def main():
    print("🔍 Debugging feature generation...")
    
    try:
        # Load data for a test stock
        symbol = "VNM"
        print(f"📊 Loading data for {symbol}...")
        
        df = load_data(symbol, lookback=100)
        if df.empty:
            print("❌ No data loaded")
            return
            
        print(f"✅ Loaded {len(df)} rows")
        print(f"📅 Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Add ML features
        print("🔧 Adding ML features...")
        enriched_df = add_ml_features(df)
        
        # Get expected feature columns
        feature_cols = get_feature_columns()
        print(f"📋 Expected {len(feature_cols)} features:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        # Check which features are missing or have issues
        print("\n🔍 Feature analysis:")
        missing_features = []
        nan_features = []
        inf_features = []
        
        for col in feature_cols:
            if col not in enriched_df.columns:
                missing_features.append(col)
            else:
                nan_count = enriched_df[col].isna().sum()
                inf_count = np.isinf(enriched_df[col]).sum()
                
                if nan_count > 0:
                    nan_features.append((col, nan_count))
                if inf_count > 0:
                    inf_features.append((col, inf_count))
                    
                print(f"  ✅ {col}: {len(enriched_df[col])} values, {nan_count} NaN, {inf_count} inf")
        
        if missing_features:
            print(f"\n❌ Missing features: {missing_features}")
        
        if nan_features:
            print(f"\n⚠️ Features with NaN values:")
            for col, count in nan_features:
                print(f"  - {col}: {count} NaN values")
        
        if inf_features:
            print(f"\n⚠️ Features with infinite values:")
            for col, count in inf_features:
                print(f"  - {col}: {count} inf values")
        
        # Test feature extraction for ML
        print("\n🧪 Testing feature extraction for ML...")
        
        # Get the last 50 rows (like in backtesting)
        test_data = enriched_df.iloc[-50:].copy()
        
        # Extract features
        X = test_data[feature_cols].fillna(0)
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Expected shape: ({len(test_data)}, {len(feature_cols)})")
        
        # Check data types first
        print(f"📊 Data types in X:")
        for col in X.columns:
            print(f"  - {col}: {X[col].dtype}")
        
        # Convert to numeric and check for problematic values
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        nan_count = np.isnan(X_numeric.values).sum()
        inf_count = np.isinf(X_numeric.values).sum()
        
        print(f"📊 NaN values in matrix: {nan_count}")
        print(f"📊 Inf values in matrix: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ Feature matrix looks good!")
        else:
            print("⚠️ Feature matrix has issues that need fixing")
            
        # Show sample of the last row
        print(f"\n📋 Sample features (last row):")
        last_row = X_numeric.iloc[-1]
        for i, (col, val) in enumerate(zip(feature_cols, last_row), 1):
            if pd.isna(val):
                print(f"  {i:2d}. {col}: NaN")
            else:
                print(f"  {i:2d}. {col}: {val:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()