"""Quick test for technical signal after fix"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.disable(logging.WARNING)

from src.data.loader import load_data
from src.strategies.technical_scorers import TechnicalScorer

print("Testing FPT...")
df = load_data("FPT", lookback=200)
if df is not None and len(df) > 50:
    print(f"Data loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    scorer = TechnicalScorer()
    signal = scorer.get_technical_signal(df)
    conf = scorer.calculate_technical_confidence(df)

    print(f"Signal: {signal}")
    print(f"Confidence: {conf:.1f}%")
else:
    print("No data loaded")
