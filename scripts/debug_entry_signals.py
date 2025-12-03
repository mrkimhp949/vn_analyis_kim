"""
Debug script to analyze why no entry signals are generated
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.DEBUG)

from src.data.loader import load_data
from src.strategies.entry_logic import ImprovedEntryLogic
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.market.regime_detector import MarketRegimeDetector as MarketRegimeAnalyzer


def debug_ticker(ticker: str):
    """Debug entry signal for a single ticker"""
    print(f"\n{'='*70}")
    print(f"🔍 DEBUGGING: {ticker}")
    print(f"{'='*70}")

    # Load data
    df = load_data(ticker, lookback=250, use_cache=False, required_bars=50)
    if df is None or df.empty:
        print(f"❌ No data for {ticker}")
        return

    print(f"✅ Loaded {len(df)} bars")
    print(f"   Latest price: {df.iloc[-1]['close']:,.0f}")
    print(f"   Avg volume (20d): {df['volume'].tail(20).mean():,.0f}")

    # Get market regime
    regime_analyzer = MarketRegimeAnalyzer()
    market_regime = regime_analyzer.analyze_market_regime()
    print(f"\n📊 Market Regime: {market_regime.get('regime', 'UNKNOWN')}")
    print(f"   Tradeable: {market_regime.get('tradeable', False)}")
    print(f"   Confidence: {market_regime.get('confidence', 0)}%")

    # Get ML signal
    print(f"\n🤖 ML Signal Analysis...")
    try:
        ml_generator = EnhancedMLSignalGenerator()
        ml_signal = ml_generator.analyze(df)
        print(f"   Signal: {ml_signal.get('signal', 'N/A')}")
        print(f"   ML Confidence: {ml_signal.get('confidence', 0)}%")
        print(f"   ML Score: {ml_signal.get('ml_score', 0):.3f}")
    except Exception as e:
        print(f"   ❌ ML Error: {e}")
        ml_signal = {"signal": "HOLD", "confidence": 0}

    # Analyze entry with LOWER threshold for testing
    print(f"\n📈 Entry Logic Analysis (min_confidence=35)...")
    entry_logic = ImprovedEntryLogic(
        min_confidence=35,  # Lowered for testing
        min_risk_reward=1.0,  # Lowered for testing - allow R:R >= 1.0
        soft_filter_mode=True,
    )

    signal = entry_logic.analyze_entry(
        df=df,
        ml_signal=ml_signal,
        market_regime=market_regime,
        symbol=ticker,
    )

    print(f"\n📋 RESULT:")
    print(f"   Should Enter: {signal.should_enter}")
    print(f"   Signal Type: {signal.signal_type}")
    print(f"   Final Confidence: {signal.confidence}%")
    print(f"   Strength: {signal.strength}")

    if signal.reasons:
        print(f"\n   ✅ Reasons:")
        for r in signal.reasons:
            print(f"      - {r}")

    if signal.warnings:
        print(f"\n   ⚠️ Warnings ({len(signal.warnings)}):")
        for w in signal.warnings:
            print(f"      - {w}")

    if signal.should_enter:
        print(f"\n   💰 Entry Details:")
        print(f"      Entry Price: {signal.entry_price:,.0f}")
        print(f"      Stop Loss: {signal.stop_loss:,.0f}")
        print(f"      Take Profits: {signal.take_profit_targets}")


def main():
    # Test a few popular tickers
    tickers = ["FPT", "VNM", "VCB", "HPG", "MWG", "VIC", "VHM", "TCB", "ACB", "MBB"]

    print("=" * 70)
    print("🔍 DEBUG ENTRY SIGNALS")
    print("=" * 70)

    buy_signals = []

    for ticker in tickers:
        try:
            debug_ticker(ticker)

            # Quick check if it would generate BUY
            df = load_data(ticker, lookback=250, use_cache=True, required_bars=50)
            if df is not None and not df.empty:
                ml_gen = EnhancedMLSignalGenerator()
                ml_signal = ml_gen.analyze(df)

                entry_logic = ImprovedEntryLogic(min_confidence=35, min_risk_reward=1.0)
                signal = entry_logic.analyze_entry(df, ml_signal, symbol=ticker)

                if signal.should_enter:
                    buy_signals.append(ticker)

        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Tickers tested: {len(tickers)}")
    print(f"BUY signals: {len(buy_signals)}")
    if buy_signals:
        print(f"Tickers with BUY: {buy_signals}")
    else:
        print("⚠️ NO BUY SIGNALS GENERATED!")
        print("\nPossible reasons:")
        print("1. ML model not generating BUY signals")
        print("2. Confidence too low")
        print("3. R:R ratio not met")
        print("4. Liquidity filters blocking")
        print("5. Market regime not tradeable")


if __name__ == "__main__":
    main()
