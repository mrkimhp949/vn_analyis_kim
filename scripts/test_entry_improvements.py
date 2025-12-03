#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Entry Logic Improvements
Kiểm tra các cải thiện entry logic mới
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_test_dataframe(
    days: int = 100, base_price: float = 50000, trend: str = "up", volatility: float = 0.02
) -> pd.DataFrame:
    """Create test DataFrame with OHLCV data"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Generate price series
    np.random.seed(42)
    returns = np.random.normal(0.001 if trend == "up" else -0.001, volatility, days)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, days)),
            "close": prices,
            "volume": np.random.uniform(100000, 500000, days),
        }
    )

    # Add indicators
    df["rsi"] = 50 + np.random.uniform(-20, 20, days)
    df["atr"] = prices * 0.02  # 2% ATR

    return df.set_index("date")


def test_vietnam_price_limits():
    """Test _check_vietnam_price_limits method"""
    print("\n" + "=" * 60)
    print("TEST 1: Vietnam Price Limits Check")
    print("=" * 60)

    from src.strategies.entry_logic import ImprovedEntryLogic

    entry_logic = ImprovedEntryLogic()

    # Test case 1: Normal price (not near limits)
    df = create_test_dataframe(days=50, base_price=50000)
    current_price = df["close"].iloc[-1]

    result = entry_logic._check_vietnam_price_limits(df, current_price)
    print(f"\n✅ Test 1.1: Normal price")
    print(f"   Current: {current_price:,.0f}")
    print(f"   Reference: {result['reference_price']:,.0f}")
    print(f"   Ceiling: {result['ceiling_price']:,.0f}")
    print(f"   Floor: {result['floor_price']:,.0f}")
    print(f"   Near limit: {result['near_limit']}")
    assert not result["near_limit"], "Should not be near limit"

    # Test case 2: Price near ceiling
    df_ceiling = df.copy()
    reference = df_ceiling["close"].iloc[-2]
    ceiling_price = reference * 1.069  # Just below ceiling
    df_ceiling.iloc[-1, df_ceiling.columns.get_loc("close")] = ceiling_price

    result = entry_logic._check_vietnam_price_limits(df_ceiling, ceiling_price)
    print(f"\n⚠️ Test 1.2: Price near ceiling")
    print(f"   Current: {ceiling_price:,.0f}")
    print(f"   Near limit: {result['near_limit']}")
    print(f"   Limit type: {result['limit_type']}")
    print(f"   Warning: {result['warning']}")
    assert result["near_limit"], "Should be near ceiling"
    assert result["limit_type"] == "CEILING", "Should be CEILING type"

    # Test case 3: Price near floor
    df_floor = df.copy()
    reference = df_floor["close"].iloc[-2]
    floor_price = reference * 0.931  # Just above floor
    df_floor.iloc[-1, df_floor.columns.get_loc("close")] = floor_price

    result = entry_logic._check_vietnam_price_limits(df_floor, floor_price)
    print(f"\n⚠️ Test 1.3: Price near floor")
    print(f"   Current: {floor_price:,.0f}")
    print(f"   Near limit: {result['near_limit']}")
    print(f"   Limit type: {result['limit_type']}")
    print(f"   Warning: {result['warning']}")
    assert result["near_limit"], "Should be near floor"
    assert result["limit_type"] == "FLOOR", "Should be FLOOR type"

    print("\n✅ All Vietnam Price Limits tests passed!")


def test_vietnam_market_liquidity():
    """Test _check_vietnam_market_liquidity method"""
    print("\n" + "=" * 60)
    print("TEST 2: Vietnam Market Liquidity Check")
    print("=" * 60)

    from src.strategies.entry_logic import ImprovedEntryLogic

    entry_logic = ImprovedEntryLogic()

    # Test case 1: Good liquidity (>2B VND)
    df_good = create_test_dataframe(days=50, base_price=50000)
    df_good["volume"] = 100000  # 100K shares * 50K = 5B VND

    result = entry_logic._check_vietnam_market_liquidity(df_good)
    print(f"\n✅ Test 2.1: Good liquidity")
    print(f"   Avg daily value: {result['avg_daily_value']/1e9:.2f}B VND")
    print(f"   Sufficient: {result['sufficient']}")
    print(f"   Reason: {result['reason']}")
    assert result["sufficient"], "Should have sufficient liquidity"

    # Test case 2: Low liquidity (500M - 2B VND)
    df_low = create_test_dataframe(days=50, base_price=50000)
    df_low["volume"] = 20000  # 20K shares * 50K = 1B VND

    result = entry_logic._check_vietnam_market_liquidity(df_low)
    print(f"\n⚠️ Test 2.2: Low liquidity")
    print(f"   Avg daily value: {result['avg_daily_value']/1e9:.2f}B VND")
    print(f"   Sufficient: {result['sufficient']}")
    print(f"   Reason: {result['reason']}")
    # Low liquidity is allowed but with warning
    assert result["sufficient"], "Should still be sufficient (with warning)"

    # Test case 3: Critical liquidity (<500M VND)
    df_critical = create_test_dataframe(days=50, base_price=10000)
    df_critical["volume"] = 10000  # 10K shares * 10K = 100M VND

    result = entry_logic._check_vietnam_market_liquidity(df_critical)
    print(f"\n🚫 Test 2.3: Critical liquidity")
    print(f"   Avg daily value: {result['avg_daily_value']/1e9:.2f}B VND")
    print(f"   Sufficient: {result['sufficient']}")
    print(f"   Reason: {result['reason']}")
    assert not result["sufficient"], "Should NOT have sufficient liquidity"

    print("\n✅ All Vietnam Market Liquidity tests passed!")


def test_entry_signal_flow():
    """Test complete entry signal flow with improvements"""
    print("\n" + "=" * 60)
    print("TEST 3: Complete Entry Signal Flow")
    print("=" * 60)

    from src.strategies.entry_logic import ImprovedEntryLogic

    entry_logic = ImprovedEntryLogic(
        min_confidence=50,
        min_risk_reward=1.5,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )

    # Create good setup
    df = create_test_dataframe(days=100, base_price=50000, trend="up")
    df["volume"] = 200000  # Good liquidity
    df["rsi"] = 45  # Good RSI

    ml_signal = {
        "signal": "BUY",
        "confidence": 65,
    }

    market_regime = {
        "regime": "BULL",
        "tradeable": True,
        "confidence": 75,
    }

    result = entry_logic.analyze_entry(
        df=df,
        ml_signal=ml_signal,
        market_regime=market_regime,
        symbol="TEST",
    )

    print(f"\n📊 Entry Signal Result:")
    print(f"   Should enter: {result.should_enter}")
    print(f"   Signal type: {result.signal_type}")
    print(f"   Confidence: {result.confidence}%")
    print(f"   Strength: {result.strength}")
    print(f"   Entry price: {result.entry_price:,.0f}")
    print(f"   Stop loss: {result.stop_loss:,.0f}")
    print(f"   Take profits: {[f'{tp:,.0f}' for tp in result.take_profit_targets]}")
    print(f"   Position multiplier: {result.position_size_multiplier:.2f}")
    print(f"\n   Reasons:")
    for reason in result.reasons:
        print(f"      {reason}")
    print(f"\n   Warnings:")
    for warning in result.warnings:
        print(f"      {warning}")

    if result.telemetry:
        print(f"\n   Telemetry:")
        print(f"      Base confidence: {result.telemetry.get('base_confidence', 'N/A')}")
        print(f"      Signal source: {result.telemetry.get('signal_source', 'N/A')}")

    print("\n✅ Entry signal flow test completed!")


def test_filter_tracking():
    """Test filter performance tracking"""
    print("\n" + "=" * 60)
    print("TEST 4: Filter Performance Tracking")
    print("=" * 60)

    from src.strategies.entry_logic import ImprovedEntryLogic

    entry_logic = ImprovedEntryLogic()

    # Track some filter results
    entry_logic._track_filter("trend_alignment", True, "TEST")
    entry_logic._track_filter("trend_alignment", True, "TEST")
    entry_logic._track_filter("trend_alignment", False, "TEST")
    entry_logic._track_filter("liquidity", True, "TEST")
    entry_logic._track_filter("rsi", False, "TEST")

    print("\n✅ Filter tracking test completed!")
    print("   (Check filter_performance.json for results)")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ENTRY LOGIC IMPROVEMENTS - TEST SUITE")
    print("=" * 60)

    try:
        test_vietnam_price_limits()
        test_vietnam_market_liquidity()
        test_entry_signal_flow()
        test_filter_tracking()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
