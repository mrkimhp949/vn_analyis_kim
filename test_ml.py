# Tạo file test_ml.py
from ml_signals import MLSignalGenerator
from data_loader import load_data
import pandas as pd

print("🤖 Testing ML Model...")
ml = MLSignalGenerator()

symbols = ['ACB', 'VNM', 'VCB']
for symbol in symbols:
    try:
        df = load_data(symbol, 100)
        result = ml.analyze(df)
        print(f"📊 {symbol}: {result['signal']} ({result['confidence']}%) - {result['reason']}")
    except Exception as e:
        print(f"❌ {symbol}: {e}")