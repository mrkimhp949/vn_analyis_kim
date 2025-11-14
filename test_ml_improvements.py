# -*- coding: utf-8 -*-
"""
Test script for ML improvements
Chạy: python test_ml_improvements.py
"""

import numpy as np
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*70)
    print("📦 TESTING IMPORTS")
    print("="*70)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'shap': 'shap',
        'ta': 'ta',
        'joblib': 'joblib',
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All packages installed!")
    return True


def test_features():
    """Test enhanced features"""
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED FEATURES")
    print("="*70)
    
    try:
        from features_enhanced import add_enhanced_features, get_feature_columns
        
        # Create dummy data
        n = 200
        df = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=n),
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 102,
            'low': np.random.randn(n).cumsum() + 98,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        
        # Add features
        df_enhanced = add_enhanced_features(df)
        
        # Check features
        feature_cols = get_feature_columns()
        print(f"   Expected features: {len(feature_cols)}")
        print(f"   Available features: {sum(col in df_enhanced.columns for col in feature_cols)}")
        
        # Check for NaN
        nan_count = df_enhanced[feature_cols].isna().sum().sum()
        print(f"   NaN values: {nan_count}")
        
        if nan_count > 0:
            print("   ⚠️ Warning: Some NaN values found")
        
        print("\n✅ Features test passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test enhanced models"""
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED MODELS")
    print("="*70)
    
    try:
        from ml_models_enhanced import EnhancedMLPredictor
        
        # Create dummy data
        n_samples = 1000
        n_features = 28
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        
        X_test = np.random.randn(100, n_features)
        
        # Train models
        print("   Training models...")
        predictor = EnhancedMLPredictor()
        predictor.train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        # Predict
        print("   Testing predictions...")
        predictions = predictor.predict(X_test, use_ensemble=True)
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Check feature importance
        if predictor.feature_importance:
            print(f"   Feature importance calculated: ✅")
        
        print("\n✅ Models test passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_generator():
    """Test enhanced signal generator"""
    print("\n" + "="*70)
    print("🧪 TESTING SIGNAL GENERATOR")
    print("="*70)
    
    try:
        from ml_signals_enhanced import EnhancedMLSignalGenerator
        from features_enhanced import add_enhanced_features
        
        # Create dummy data
        n = 200
        df = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=n),
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 102,
            'low': np.random.randn(n).cumsum() + 98,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        
        # Add features
        df = add_enhanced_features(df)
        
        # Generate signal
        print("   Generating signal...")
        generator = EnhancedMLSignalGenerator()
        signal = generator.analyze(df, explain=False)
        
        # Check result
        print(f"   Signal: {signal['signal']}")
        print(f"   Confidence: {signal['confidence']}%")
        print(f"   ML Score: {signal['ml_score']:.4f}")
        
        assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal['confidence'] <= 100
        assert 0 <= signal['ml_score'] <= 1
        
        print("\n✅ Signal generator test passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Signal generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration"""
    print("\n" + "="*70)
    print("🧪 TESTING FULL INTEGRATION")
    print("="*70)
    
    try:
        # Test with real data if available
        try:
            from data_loader import load_data
            
            print("   Loading real data...")
            df = load_data("VNM", lookback=200)
            index_df = load_data("VNINDEX", lookback=200, is_index=True)
            
            from ml_signals_enhanced import EnhancedMLSignalGenerator
            
            generator = EnhancedMLSignalGenerator()
            signal = generator.analyze(df, index_df, explain=False)
            
            print(f"   Real data signal: {signal['signal']} ({signal['confidence']}%)")
            
        except Exception as e:
            print(f"   ⚠️ Could not test with real data: {e}")
            print("   Using dummy data instead...")
            
            # Use dummy data
            n = 200
            df = pd.DataFrame({
                'time': pd.date_range('2023-01-01', periods=n),
                'open': np.random.randn(n).cumsum() + 100,
                'high': np.random.randn(n).cumsum() + 102,
                'low': np.random.randn(n).cumsum() + 98,
                'close': np.random.randn(n).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, n)
            })
            
            from features_enhanced import add_enhanced_features
            from ml_signals_enhanced import EnhancedMLSignalGenerator
            
            df = add_enhanced_features(df)
            generator = EnhancedMLSignalGenerator()
            signal = generator.analyze(df, explain=False)
            
            print(f"   Dummy data signal: {signal['signal']} ({signal['confidence']}%)")
        
        print("\n✅ Integration test passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 ML IMPROVEMENTS TEST SUITE")
    print("="*70)
    
    results = {
        'imports': test_imports(),
        'features': test_features(),
        'models': test_models(),
        'signal_generator': test_signal_generator(),
        'integration': test_integration(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
