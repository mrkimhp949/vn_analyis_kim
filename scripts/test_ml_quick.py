#!/usr/bin/env python3
"""Quick test for ML signal generation"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.data.loader import load_data

df = load_data("VNM", lookback=200)
gen = EnhancedMLSignalGenerator()
result = gen.analyze(df, symbol="VNM")
print(f"Signal: {result.get('signal')}, ML Score: {result.get('ml_score', 0):.3f}")
