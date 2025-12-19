"""
Debug script: Analyze why bot doesn't find any BUY signals after 10+ days
"""

import os
import sys
import io

# Fix Unicode encoding issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress logs for cleaner output
import logging

logging.disable(logging.WARNING)

from src.data.loader import load_data
from src.strategies.entry_logic import ImprovedEntryLogic
from src.config.constants import (
    TECH_ONLY_MIN_CONFIDENCE,
    ENTRY_QUALITY_REJECT,
    DEFAULT_MIN_RISK_REWARD,
)


def analyze_signal_failure(ticker: str, df, entry_logic: ImprovedEntryLogic) -> dict:
    """Analyze why a ticker doesn't generate a BUY signal"""
    signal = entry_logic.analyze_entry(df, symbol=ticker)

    result = {
        "ticker": ticker,
        "should_enter": signal.should_enter,
        "confidence": signal.confidence,
        "signal_type": signal.signal_type,
        "reasons": signal.reasons,
        "warnings": signal.warnings,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "risk_reward": getattr(signal, "risk_reward", 0),
    }

    # Determine failure reason
    if signal.should_enter:
        result["failure_reason"] = None
    elif signal.confidence < entry_logic.min_confidence:
        result["failure_reason"] = (
            f"Confidence too low: {signal.confidence} < {entry_logic.min_confidence}"
        )
    elif signal.signal_type != "BUY":
        result["failure_reason"] = f"Not a BUY signal: {signal.signal_type}"
    elif len(signal.warnings) > 5:
        result["failure_reason"] = f"Too many warnings: {len(signal.warnings)}"
    else:
        result["failure_reason"] = "Unknown (check telemetry)"

    return result


def main():
    print("=" * 80)
    print("ANALYSIS: WHY NO BUY SIGNALS AFTER 10+ DAYS?")
    print("=" * 80)

    # Show current thresholds
    print("\n📊 CURRENT THRESHOLDS:")
    print(f"   TECH_ONLY_MIN_CONFIDENCE: {TECH_ONLY_MIN_CONFIDENCE}")
    print(f"   DEFAULT_MIN_RISK_REWARD: {DEFAULT_MIN_RISK_REWARD}")
    print(f"   ENTRY_QUALITY_REJECT: {ENTRY_QUALITY_REJECT}")

    # Test tickers
    test_tickers = [
        "FPT",
        "VNM",
        "VCB",
        "HPG",
        "MWG",
        "VIC",
        "VHM",
        "TCB",
        "ACB",
        "MBB",
        "VND",
        "SSI",
        "HCM",
        "VCI",
        "MSN",
    ]

    print(f"\n📈 Testing {len(test_tickers)} tickers...")
    print("-" * 80)

    # Standard config (current settings)
    standard_logic = ImprovedEntryLogic(
        min_confidence=55,
        min_risk_reward=2.0,
        require_trend_alignment=True,
        require_volume_confirmation=False,
    )

    # Relaxed config
    relaxed_logic = ImprovedEntryLogic(
        min_confidence=40,
        min_risk_reward=1.5,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )

    standard_results = []
    relaxed_results = []

    for ticker in test_tickers:
        df = load_data(ticker, lookback=200)
        if df is None or df.empty or len(df) < 50:
            print(f"  ⚠️ {ticker}: Insufficient data")
            continue

        # Test standard config
        std_result = analyze_signal_failure(ticker, df, standard_logic)
        standard_results.append(std_result)

        # Test relaxed config
        rel_result = analyze_signal_failure(ticker, df, relaxed_logic)
        relaxed_results.append(rel_result)

        # Print result
        price = df.iloc[-1]["close"]
        std_emoji = "✅" if std_result["should_enter"] else "❌"
        rel_emoji = "✅" if rel_result["should_enter"] else "❌"

        print(f"\n{ticker} (price: {price:,.0f})")
        print(f"   Standard (conf>=55, R:R>=2.0): {std_emoji} conf={std_result['confidence']}")
        if not std_result["should_enter"]:
            print(f"      Reason: {std_result['failure_reason']}")
            if std_result["warnings"]:
                print(
                    f"      Warnings ({len(std_result['warnings'])}): {std_result['warnings'][:2]}"
                )

        print(f"   Relaxed  (conf>=40, R:R>=1.5): {rel_emoji} conf={rel_result['confidence']}")
        if rel_result["should_enter"]:
            print(
                f"      Entry: {rel_result['entry_price']:,.0f}, SL: {rel_result['stop_loss']:,.0f}"
            )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    std_buys = sum(1 for r in standard_results if r["should_enter"])
    rel_buys = sum(1 for r in relaxed_results if r["should_enter"])

    print(f"\n📊 Standard config (current): {std_buys}/{len(standard_results)} BUY signals")
    print(f"📊 Relaxed config: {rel_buys}/{len(relaxed_results)} BUY signals")

    # Analyze failure reasons
    failure_reasons = {}
    for r in standard_results:
        if not r["should_enter"]:
            reason = r["failure_reason"]
            key = reason.split(":")[0] if ":" in reason else reason
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

    print(f"\n📉 Failure reasons (standard config):")
    for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count} tickers")

    # Average confidence
    avg_conf = (
        sum(r["confidence"] for r in standard_results) / len(standard_results)
        if standard_results
        else 0
    )
    print(f"\n📈 Average confidence: {avg_conf:.1f}%")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    if avg_conf < 55:
        print(
            """
1. LOWER min_confidence threshold:
   - Current: 55%
   - Suggested: 45-50%
   - Edit: src/config/trading_config.py -> min_confidence
   
2. LOWER TECH_ONLY_MIN_CONFIDENCE:
   - Current: 60%
   - Suggested: 50-55%
   - Edit: src/config/constants.py -> TECH_ONLY_MIN_CONFIDENCE

3. REDUCE min_risk_reward requirement:
   - Current: 2.0 (R:R >= 2:1)
   - Suggested: 1.5 (R:R >= 1.5:1)
   - Edit: src/config/trading_config.py -> min_risk_reward

4. RELAX entry filters:
   - Disable require_trend_alignment temporarily
   - Lower ENTRY_QUALITY_REJECT from 0.40 to 0.30
   - Edit: src/services/entry_service.py
"""
        )

    if std_buys == 0 and rel_buys > 0:
        print(
            """
✅ The relaxed config DOES find signals!
This confirms the thresholds are too strict.

Quick fix - Edit src/config/trading_config.py:
   min_confidence: int = 50  # Changed from 55
   min_risk_reward: float = 1.5  # Changed from 2.0
"""
        )


if __name__ == "__main__":
    main()
