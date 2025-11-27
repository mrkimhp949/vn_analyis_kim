"""Check actual ML scores for tickers"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.WARNING)

from src.data.loader import load_data
from src.ml.signals.enhanced import EnhancedMLSignalGenerator

tickers = ["FPT", "VNM", "VCB", "HPG", "MWG", "VIC", "VHM", "TCB", "ACB", "MBB"]

print("=" * 70)
print("ML SCORE ANALYSIS")
print("=" * 70)

ml_gen = EnhancedMLSignalGenerator()

for ticker in tickers:
    try:
        df = load_data(ticker, lookback=250, use_cache=True, required_bars=50)
        if df is None or df.empty:
            print(f"{ticker}: No data")
            continue

        result = ml_gen.analyze(df)

        ml_score = result.get("ml_score", 0.5)
        signal = result.get("signal", "N/A")
        confidence = result.get("confidence", 0)
        reason = result.get("reason", "")

        # Highlight BUY signals
        marker = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")

        print(
            f"{marker} {ticker}: ML={ml_score:.3f} | Signal={signal} | Conf={confidence}% | {reason[:50]}"
        )

    except Exception as e:
        print(f"{ticker}: Error - {e}")

print("\n" + "=" * 70)
print("ANALYSIS:")
print("- ML score > 0.55 → BUY signal")
print("- ML score < 0.45 → SELL signal")
print("- ML score 0.45-0.55 → HOLD (neutral)")
print("=" * 70)
