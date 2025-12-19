"""
Deep debug script: Analyze exactly why technical scorer doesn't generate BUY signals
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.disable(logging.WARNING)

from src.data.loader import load_data
from src.strategies.technical_scorers import TechnicalScorer
from utils.dataframe_utils import safe_get_latest


def debug_technical_signal(ticker: str, df):
    """Debug the get_technical_signal method step by step"""
    print(f"\n{'='*60}")
    print(f"DEEP DEBUG: {ticker}")
    print(f"{'='*60}")

    if df is None or df.empty or len(df) < 50:
        print("  Insufficient data")
        return

    scorer = TechnicalScorer()

    # Get raw data
    current_price = safe_get_latest(df, "close", 0)
    print(f"\nCurrent Price: {current_price:,.0f}")

    # Calculate EMAs if not present
    if "ema20" not in df.columns:
        df["ema20"] = df["close"].ewm(span=20).mean()
    if "ema50" not in df.columns:
        df["ema50"] = df["close"].ewm(span=50).mean()
    if "rsi" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
    if "macd" not in df.columns:
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

    ema20 = safe_get_latest(df, "ema20", 0)
    ema50 = safe_get_latest(df, "ema50", 0)
    rsi = safe_get_latest(df, "rsi", 50)
    current_volume = safe_get_latest(df, "volume", 0)
    avg_volume = df["volume"].tail(20).mean()
    macd = safe_get_latest(df, "macd", 0)
    macd_signal = safe_get_latest(df, "macd_signal", 0)

    print(f"\nINDICATORS:")
    print(f"  EMA20: {ema20:,.0f}")
    print(f"  EMA50: {ema50:,.0f}")
    print(f"  RSI: {rsi:.1f}")
    print(f"  Volume: {current_volume:,.0f} (Avg: {avg_volume:,.0f})")
    print(f"  MACD: {macd:.2f}, Signal: {macd_signal:.2f}")

    # Simulate scoring logic from get_technical_signal
    print(f"\nSCORING (need >= 3 for BUY):")
    score = 0

    # 1. EMA alignment (2 points)
    if ema20 > ema50:
        score += 2
        print(f"  [+2] EMA20 > EMA50 (bullish trend)")
    elif ema20 < ema50 * 0.98:
        score -= 2
        print(f"  [-2] EMA20 < EMA50*0.98 (bearish)")
    else:
        print(f"  [ 0] EMA alignment neutral")

    # 2. RSI (1 point)
    if 30 <= rsi <= 65:
        score += 1
        print(f"  [+1] RSI {rsi:.1f} in healthy zone (30-65)")
    elif rsi > 70:
        score -= 1
        print(f"  [-1] RSI {rsi:.1f} overbought (>70)")
    else:
        print(f"  [ 0] RSI {rsi:.1f} not in zone")

    # 3. Price vs EMA20 (1 point)
    if current_price > ema20:
        score += 1
        print(f"  [+1] Price > EMA20")
    elif current_price < ema20 * 0.98:
        score -= 1
        print(f"  [-1] Price < EMA20*0.98")
    else:
        print(f"  [ 0] Price near EMA20")

    # 4. Volume (1 point)
    vol_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    if vol_ratio > 1.2:
        score += 1
        print(f"  [+1] Volume {vol_ratio:.1f}x avg (> 1.2x)")
    elif vol_ratio < 0.5:
        score -= 1
        print(f"  [-1] Volume {vol_ratio:.1f}x avg (< 0.5x)")
    else:
        print(f"  [ 0] Volume {vol_ratio:.1f}x avg (normal)")

    # 5. MACD (1 point)
    if macd > macd_signal and macd > 0:
        score += 1
        print(f"  [+1] MACD > Signal AND MACD > 0")
    elif macd < macd_signal and macd < 0:
        score -= 1
        print(f"  [-1] MACD < Signal AND MACD < 0")
    else:
        print(f"  [ 0] MACD neutral")

    print(f"\n  TOTAL SCORE: {score}")
    print(f"  SIGNAL: {'BUY' if score >= 3 else 'SELL' if score <= -3 else 'HOLD'}")

    # Confidence score
    all_scores = scorer.get_all_scores(df)
    print(f"\n  Technical Confidence: {all_scores['confidence']:.1f}%")
    print(f"  Individual scores: {all_scores['scores']}")

    # Show what's needed for BUY
    if score < 3:
        needed = 3 - score
        print(f"\n  NEEDS +{needed} more points for BUY signal")
        print(f"  Suggestions to get BUY:")
        if ema20 < ema50:
            print(f"    - Wait for EMA20 to cross above EMA50 (+2 points)")
        if rsi < 30 or rsi > 65:
            print(f"    - RSI needs to be 30-65 (+1 point)")
        if current_price <= ema20:
            print(f"    - Price needs to be above EMA20 (+1 point)")
        if vol_ratio <= 1.2:
            print(f"    - Volume needs to increase above 1.2x average (+1 point)")
        if not (macd > macd_signal and macd > 0):
            print(f"    - MACD needs to be positive and above signal (+1 point)")


def main():
    print("=" * 60)
    print("DEEP DEBUG: TECHNICAL SCORING ANALYSIS")
    print("=" * 60)

    tickers = ["FPT", "VNM", "VCB", "HPG", "MWG", "TCB", "ACB", "MBB"]

    buy_count = 0
    for ticker in tickers:
        df = load_data(ticker, lookback=200)
        if df is not None and not df.empty:
            debug_technical_signal(ticker, df)

            # Check if it would be BUY
            scorer = TechnicalScorer()
            signal = scorer.get_technical_signal(df)
            if signal == "BUY":
                buy_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BUY signals: {buy_count}/{len(tickers)}")

    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    print(
        """
The scoring system requires >= 3 points for BUY signal:
  +2: EMA20 > EMA50 (trend alignment)
  +1: RSI 30-65 (healthy zone)  
  +1: Price > EMA20
  +1: Volume > 1.2x average
  +1: MACD > Signal AND MACD > 0

PROBLEM: In sideways or correcting market, it's very hard to get 3+ points!

SOLUTION OPTIONS:
1. Lower BUY threshold from 3 to 2 points
2. Make EMA alignment worth +1 instead of +2
3. Add more positive signals (like momentum, RSI turning up)
4. Use ML signals as primary instead of technical-only
"""
    )


if __name__ == "__main__":
    main()
