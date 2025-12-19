"""Test multiple tickers with relaxed conditions"""

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
from src.strategies.entry_logic import ImprovedEntryLogic

print("=" * 60)
print("TESTING MULTIPLE TICKERS AFTER FIX")
print("=" * 60)

tickers = ["FPT", "VNM", "MWG", "HPG", "TCB", "ACB", "MBB", "SSI"]
scorer = TechnicalScorer()

buy_signals = []
buy_entries = []

for ticker in tickers:
    df = load_data(ticker, lookback=200)
    if df is None or len(df) < 50:
        print(f"{ticker}: No data")
        continue

    # Get technical signal and confidence
    signal = scorer.get_technical_signal(df)
    conf = scorer.calculate_technical_confidence(df)

    # Try entry logic
    entry_logic = ImprovedEntryLogic(
        min_confidence=40,  # Relaxed
        min_risk_reward=1.5,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )
    entry = entry_logic.analyze_entry(df, symbol=ticker)

    status = "BUY" if signal == "BUY" else ("SELL" if signal == "SELL" else "HOLD")
    entry_status = "ENTRY" if entry.should_enter else "NO_ENTRY"

    if signal == "BUY":
        buy_signals.append(ticker)
    if entry.should_enter:
        buy_entries.append(ticker)

    print(
        f"{ticker}: {status} (conf={conf:.1f}%) | EntryLogic: {entry_status} (conf={entry.confidence})"
    )
    if entry.should_enter:
        print(f"    Entry: {entry.entry_price:,.0f}, SL: {entry.stop_loss:,.0f}")
    elif entry.warnings:
        print(f"    Warnings: {entry.warnings[:2]}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Technical BUY signals: {len(buy_signals)}/{len(tickers)}")
print(f"  Tickers: {buy_signals}")
print(f"Entry logic BUY entries: {len(buy_entries)}/{len(tickers)}")
print(f"  Tickers: {buy_entries}")
