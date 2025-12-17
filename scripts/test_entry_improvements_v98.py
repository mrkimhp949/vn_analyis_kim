#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Entry Logic v9.8 Improvements

Tests:
1. Filter Priority System
2. Dynamic Intraday Momentum Threshold (ATR-based)
3. Dynamic Gap Threshold (ATR-based)
4. Weighted Consecutive Loss Protection
5. Entry Quality Score
6. Fast Path for High Confidence Signals
7. Time-of-Day Optimization
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import entry logic
from src.strategies.entry_logic import ImprovedEntryLogic
from src.config.constants import (
    FILTER_PRIORITY_CRITICAL,
    FILTER_PRIORITY_IMPORTANT,
    FILTER_PRIORITY_OPTIONAL,
    ENTRY_QUALITY_EXCELLENT,
    ENTRY_QUALITY_GOOD,
    ENTRY_QUALITY_ACCEPTABLE,
    CONSECUTIVE_LOSS_WEIGHTED_LIMIT,
    INTRADAY_MOMENTUM_ATR_MULTIPLIER,
    GAP_ATR_MULTIPLIER,
)


def create_test_dataframe(days: int = 50, volatility: float = 0.02) -> pd.DataFrame:
    """Create test DataFrame with OHLCV data."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    base_price = 50000  # 50,000 VND

    # Generate price data
    returns = np.random.normal(0, volatility, days)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, days)),
            "close": prices,
            "volume": np.random.randint(100000, 500000, days),
        }
    )

    # Add ATR
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_14"] = df["atr"]

    return df


def test_filter_priority_system():
    """Test 1: Filter Priority System"""
    print("\n" + "=" * 60)
    print("TEST 1: Filter Priority System")
    print("=" * 60)

    el = ImprovedEntryLogic()

    # Test critical filters
    for f in ["price_limit", "liquidity", "margin_check"]:
        priority = el.get_filter_priority(f)
        assert priority == "CRITICAL", f"Expected CRITICAL for {f}, got {priority}"
        print(f"  ✅ {f}: {priority}")

    # Test important filters
    for f in ["trend", "rsi", "foreign_flow"]:
        priority = el.get_filter_priority(f)
        assert priority == "IMPORTANT", f"Expected IMPORTANT for {f}, got {priority}"
        print(f"  ✅ {f}: {priority}")

    # Test optional filters
    for f in ["session_timing", "gap_analysis", "pre_holiday"]:
        priority = el.get_filter_priority(f)
        assert priority == "OPTIONAL", f"Expected OPTIONAL for {f}, got {priority}"
        print(f"  ✅ {f}: {priority}")

    print("  ✅ Filter Priority System: PASSED")


def test_dynamic_momentum_threshold():
    """Test 2: Dynamic Intraday Momentum Threshold"""
    print("\n" + "=" * 60)
    print("TEST 2: Dynamic Intraday Momentum Threshold (ATR-based)")
    print("=" * 60)

    el = ImprovedEntryLogic()

    # Test with low volatility stock
    df_low_vol = create_test_dataframe(volatility=0.01)
    current_price = df_low_vol["close"].iloc[-1]
    threshold_low = el._get_dynamic_momentum_threshold(df_low_vol, current_price)
    print(f"  Low volatility stock: threshold = {threshold_low*100:.2f}%")

    # Test with high volatility stock
    df_high_vol = create_test_dataframe(volatility=0.04)
    current_price = df_high_vol["close"].iloc[-1]
    threshold_high = el._get_dynamic_momentum_threshold(df_high_vol, current_price)
    print(f"  High volatility stock: threshold = {threshold_high*100:.2f}%")

    # High vol should have higher threshold
    assert threshold_high > threshold_low, "High vol should have higher threshold"
    print(
        f"  ✅ High vol threshold ({threshold_high*100:.2f}%) > Low vol ({threshold_low*100:.2f}%)"
    )

    # Check bounds
    assert 0.025 <= threshold_low <= 0.06, f"Threshold out of bounds: {threshold_low}"
    assert 0.025 <= threshold_high <= 0.06, f"Threshold out of bounds: {threshold_high}"
    print("  ✅ Thresholds within bounds [2.5%, 6%]")

    print("  ✅ Dynamic Momentum Threshold: PASSED")


def test_dynamic_gap_threshold():
    """Test 3: Dynamic Gap Threshold"""
    print("\n" + "=" * 60)
    print("TEST 3: Dynamic Gap Threshold (ATR-based)")
    print("=" * 60)

    el = ImprovedEntryLogic()

    # Test with different volatility
    df_low = create_test_dataframe(volatility=0.01)
    df_high = create_test_dataframe(volatility=0.04)

    threshold_low = el._get_dynamic_gap_threshold(df_low, df_low["close"].iloc[-1])
    threshold_high = el._get_dynamic_gap_threshold(df_high, df_high["close"].iloc[-1])

    print(f"  Low volatility: gap threshold = {threshold_low*100:.2f}%")
    print(f"  High volatility: gap threshold = {threshold_high*100:.2f}%")

    assert threshold_high > threshold_low, "High vol should have higher gap threshold"
    print(f"  ✅ High vol gap threshold > Low vol gap threshold")

    # Check bounds [4%, 8%]
    assert 0.04 <= threshold_low <= 0.08, f"Gap threshold out of bounds: {threshold_low}"
    assert 0.04 <= threshold_high <= 0.08, f"Gap threshold out of bounds: {threshold_high}"
    print("  ✅ Gap thresholds within bounds [4%, 8%]")

    print("  ✅ Dynamic Gap Threshold: PASSED")


def test_weighted_loss_protection():
    """Test 4: Weighted Consecutive Loss Protection"""
    print("\n" + "=" * 60)
    print("TEST 4: Weighted Consecutive Loss Protection")
    print("=" * 60)

    el = ImprovedEntryLogic()
    symbol = "TEST"

    # Reset tracking
    el.reset_loss_tracking(symbol)

    # Record small loss (< 3%)
    el.record_trade_result(symbol, is_win=False, loss_pct=0.02)
    weighted = el._weighted_losses.get(symbol, 0)
    print(f"  After 2% loss: weighted = {weighted} (expected 0.5)")
    assert weighted == 0.5, f"Expected 0.5, got {weighted}"

    # Record medium loss (3-6%)
    el.record_trade_result(symbol, is_win=False, loss_pct=0.04)
    weighted = el._weighted_losses.get(symbol, 0)
    print(f"  After 4% loss: weighted = {weighted} (expected 1.5)")
    assert weighted == 1.5, f"Expected 1.5, got {weighted}"

    # Record large loss (> 6%)
    el.record_trade_result(symbol, is_win=False, loss_pct=0.08)
    weighted = el._weighted_losses.get(symbol, 0)
    print(f"  After 8% loss: weighted = {weighted} (expected 3.5)")
    assert weighted == 3.5, f"Expected 3.5, got {weighted}"

    # Check if blocked (limit is 3.0)
    result = el._check_consecutive_losses(symbol)
    assert result["blocked"] == True, "Should be blocked after weighted sum >= 3.0"
    print(f"  ✅ Entry blocked: {result['reason'][:50]}...")

    # Test win resets
    el.record_trade_result(symbol, is_win=True)
    weighted = el._weighted_losses.get(symbol, 0)
    assert weighted == 0, "Win should reset weighted losses"
    print("  ✅ Win resets weighted losses to 0")

    print("  ✅ Weighted Loss Protection: PASSED")


def test_entry_quality_score():
    """Test 5: Entry Quality Score"""
    print("\n" + "=" * 60)
    print("TEST 5: Entry Quality Score")
    print("=" * 60)

    el = ImprovedEntryLogic()

    # Test with good entry (many reasons, few warnings)
    el._filter_scores = {
        "price_limit": 1.0,
        "trend": 1.0,
        "liquidity": 1.0,
        "rsi": 0.8,
        "volatility": 0.7,
    }

    reasons = ["✅ Good trend", "✅ Good liquidity", "✅ RSI optimal"]
    warnings = ["⚠️ Minor warning"]
    adjustments = [10, 5, -5]

    score = el._calculate_entry_quality_score(reasons, warnings, adjustments, None)
    label = el.get_entry_quality_label(score)
    print(f"  Good entry: score = {score:.2f}, label = {label}")
    assert score >= ENTRY_QUALITY_GOOD, f"Expected GOOD or better, got {score}"

    # Test with poor entry (many warnings)
    el._filter_scores = {
        "price_limit": 0.5,
        "trend": 0.3,
        "liquidity": 0.4,
    }

    reasons = ["✅ One positive"]
    warnings = ["⚠️ Warning 1", "⚠️ Warning 2", "⚠️ Warning 3", "⚠️ Warning 4"]
    adjustments = [-10, -10, -5]

    score = el._calculate_entry_quality_score(reasons, warnings, adjustments, None)
    label = el.get_entry_quality_label(score)
    print(f"  Poor entry: score = {score:.2f}, label = {label}")

    print("  ✅ Entry Quality Score: PASSED")


def test_fast_path():
    """Test 6: Fast Path for High Confidence Signals"""
    print("\n" + "=" * 60)
    print("TEST 6: Fast Path for High Confidence Signals")
    print("=" * 60)

    el = ImprovedEntryLogic()
    df = create_test_dataframe()
    current_price = df["close"].iloc[-1]

    # Test with high confidence (should use fast path)
    # Note: Fast path requires confidence >= 80 AND good R:R
    use_fast = el._should_use_fast_path(85, df, current_price)
    print(f"  Confidence 85%: fast_path = {use_fast}")

    # Test with low confidence (should not use fast path)
    use_fast = el._should_use_fast_path(60, df, current_price)
    print(f"  Confidence 60%: fast_path = {use_fast}")
    assert use_fast == False, "Low confidence should not use fast path"

    print("  ✅ Fast Path: PASSED")


def test_time_of_day_optimization():
    """Test 7: Time-of-Day Optimization"""
    print("\n" + "=" * 60)
    print("TEST 7: Time-of-Day Optimization")
    print("=" * 60)

    el = ImprovedEntryLogic()

    # Test simple time check (fallback)
    result = el._check_time_of_day_simple()
    print(f"  Current time check: adjustment = {result.get('adjustment', 0)}")
    print(f"  Note: {result.get('note', 'N/A')}")

    # Verify structure
    assert "blocked" in result
    assert "warning" in result
    assert "positive" in result
    assert "adjustment" in result
    assert "position_multiplier" in result

    print("  ✅ Time-of-Day Optimization: PASSED")


def test_constants_imported():
    """Test that all new constants are properly imported"""
    print("\n" + "=" * 60)
    print("TEST 8: New Constants Verification")
    print("=" * 60)

    # Check filter priority constants
    assert len(FILTER_PRIORITY_CRITICAL) > 0, "CRITICAL filters should not be empty"
    assert len(FILTER_PRIORITY_IMPORTANT) > 0, "IMPORTANT filters should not be empty"
    assert len(FILTER_PRIORITY_OPTIONAL) > 0, "OPTIONAL filters should not be empty"
    print(f"  ✅ FILTER_PRIORITY_CRITICAL: {FILTER_PRIORITY_CRITICAL}")
    print(f"  ✅ FILTER_PRIORITY_IMPORTANT: {FILTER_PRIORITY_IMPORTANT}")
    print(f"  ✅ FILTER_PRIORITY_OPTIONAL: {FILTER_PRIORITY_OPTIONAL}")

    # Check entry quality thresholds
    assert ENTRY_QUALITY_EXCELLENT > ENTRY_QUALITY_GOOD > ENTRY_QUALITY_ACCEPTABLE
    print(
        f"  ✅ Entry quality thresholds: EXCELLENT={ENTRY_QUALITY_EXCELLENT}, GOOD={ENTRY_QUALITY_GOOD}, ACCEPTABLE={ENTRY_QUALITY_ACCEPTABLE}"
    )

    # Check ATR multipliers
    assert INTRADAY_MOMENTUM_ATR_MULTIPLIER > 0
    assert GAP_ATR_MULTIPLIER > 0
    print(
        f"  ✅ ATR multipliers: momentum={INTRADAY_MOMENTUM_ATR_MULTIPLIER}, gap={GAP_ATR_MULTIPLIER}"
    )

    print("  ✅ Constants Verification: PASSED")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ENTRY LOGIC v9.8 IMPROVEMENTS TEST SUITE")
    print("=" * 60)

    tests = [
        test_filter_priority_system,
        test_dynamic_momentum_threshold,
        test_dynamic_gap_threshold,
        test_weighted_loss_protection,
        test_entry_quality_score,
        test_fast_path,
        test_time_of_day_optimization,
        test_constants_imported,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
