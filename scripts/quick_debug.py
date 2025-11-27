"""Quick debug for one ticker"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.WARNING)

from src.data.loader import load_data
from src.strategies.entry_logic import ImprovedEntryLogic
from src.ml.signals.enhanced import EnhancedMLSignalGenerator

ticker = "FPT"
df = load_data(ticker, lookback=250, required_bars=50)
print(f"Loaded {len(df)} bars for {ticker}")

ml_gen = EnhancedMLSignalGenerator()
ml_signal = ml_gen.analyze(df)
print(
    f"\nML Signal: {ml_signal['signal']} | Conf: {ml_signal['confidence']}% | Score: {ml_signal['ml_score']:.3f}"
)

entry = ImprovedEntryLogic(min_confidence=35, min_risk_reward=1.0)
signal = entry.analyze_entry(df, ml_signal, symbol=ticker)

print(f"\n=== ENTRY RESULT ===")
print(f"Should Enter: {signal.should_enter}")
print(f"Signal Type: {signal.signal_type}")
print(f"Confidence: {signal.confidence}")
print(f"Entry Price: {signal.entry_price}")
print(f"Stop Loss: {signal.stop_loss}")

print(f"\nReasons: {signal.reasons}")
print(f"Warnings: {signal.warnings}")
