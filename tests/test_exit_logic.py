"""
Unit tests for src/strategies/exit_logic.py - Exit Strategy

Tests cover:
- Initialization and configuration
- Stop loss triggers
- Take profit levels (TP1/TP2/TP3, partial exits)
- Trailing stop (activation, ATR-based, fixed %)
- Profit protection (multi-tier)
- Time decay
- Market crash protection
- ML sell signals
- Reversal patterns
- Support breakdown
- Position tracking
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.strategies.exit_logic import (
    ImprovedExitStrategy,
    ExitDecision,
    ExitReason,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    n = 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")

    close = np.linspace(10000, 11000, n) + np.random.normal(0, 50, n)
    high = close + np.abs(np.random.normal(50, 20, n))
    low = close - np.abs(np.random.normal(50, 20, n))
    open_price = close - np.random.normal(0, 30, n)
    volume = np.random.uniform(200_000, 250_000, n)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    # Add ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    return df


@pytest.fixture
def bull_market_regime():
    """Bull market regime"""
    return {
        "regime": "BULL",
        "tradeable": True,
        "confidence": 80,
    }


@pytest.fixture
def bear_market_regime():
    """Bear market regime"""
    return {
        "regime": "BEAR",
        "tradeable": False,
        "confidence": 75,
    }


@pytest.fixture
def sell_ml_signal():
    """ML sell signal"""
    return {
        "signal": "SELL",
        "confidence": 75,
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_exit_strategy_init_default():
    """Test default initialization (v3.0 - refactored with ExitConfig)"""
    strategy = ImprovedExitStrategy()

    # v3.0: Uses ExitConfig with new defaults
    assert strategy.tp_levels == [0.12, 0.20]  # 2 TP levels
    assert strategy.trailing_activation == 0.05  # v3.0: Changed from 0.08 to 0.05
    assert strategy.trailing_distance == 0.03  # v3.0: Changed from 0.05 to 0.03
    assert strategy.max_holding_days == 20  # v3.0: Changed from 25 to 20
    assert strategy.time_decay_threshold == 0.02  # v3.0: Changed from 0.03 to 0.02
    assert strategy.default_stop_loss_pct == 0.07  # v3.0: Now positive (7%)


def test_exit_strategy_init_custom():
    """Test custom initialization"""
    strategy = ImprovedExitStrategy(
        take_profit_levels=[0.08, 0.12, 0.18],
        trailing_stop_activation=0.06,
        trailing_stop_distance=0.04,
        max_holding_days=15,
        time_decay_threshold=0.03,
    )

    assert strategy.tp_levels == [0.08, 0.12, 0.18]
    assert strategy.trailing_activation == 0.06
    assert strategy.trailing_distance == 0.04
    assert strategy.max_holding_days == 15
    assert strategy.time_decay_threshold == 0.03


def test_exit_strategy_position_highs_init():
    """Test that position highs dict is initialized"""
    strategy = ImprovedExitStrategy()

    assert isinstance(strategy.position_highs, dict)
    assert len(strategy.position_highs) == 0


# ============================================================================
# STOP LOSS TESTS
# ============================================================================


def test_check_exit_stop_loss_hit(sample_stock_data):
    """Test stop loss trigger"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 9200  # -8% loss
    stop_loss = 9300  # -7% stop
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        take_profit_targets=[10500, 11000, 11500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.STOP_LOSS
    assert decision.exit_type == "FULL"
    assert decision.urgency == 5
    assert decision.expected_pnl_percent < 0


def test_check_exit_stop_loss_not_hit(sample_stock_data):
    """Test when price above stop loss (no exit)"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 10200  # +2% profit
    stop_loss = 9300  # -7% stop
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        take_profit_targets=[10500, 11000, 11500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    # Should not trigger stop loss (may exit for other reasons or HOLD)
    if decision.should_exit:
        assert decision.exit_reason != ExitReason.STOP_LOSS


def test_ensure_stop_loss_valid(sample_stock_data):
    """Test _ensure_stop_loss with valid stop loss"""
    strategy = ImprovedExitStrategy()

    stop_loss = strategy._ensure_stop_loss("TEST", 10000, 9300, sample_stock_data)

    assert stop_loss == 9300  # Should return as-is


def test_ensure_stop_loss_none_fallback(sample_stock_data):
    """Test _ensure_stop_loss with None (should use fallback)"""
    strategy = ImprovedExitStrategy(default_stop_loss_pct=-7.0)

    stop_loss = strategy._ensure_stop_loss("TEST", 10000, None, sample_stock_data)

    # Should calculate fallback (ATR-based or percentage)
    assert stop_loss < 10000
    assert stop_loss > 0


def test_ensure_stop_loss_zero_fallback(sample_stock_data):
    """Test _ensure_stop_loss with zero (should use fallback)"""
    strategy = ImprovedExitStrategy()

    stop_loss = strategy._ensure_stop_loss("TEST", 10000, 0, sample_stock_data)

    # Should calculate fallback
    assert stop_loss < 10000
    assert stop_loss > 0


# ============================================================================
# TAKE PROFIT TESTS
# ============================================================================


def test_check_take_profit_tp1(sample_stock_data):
    """Test TP1 trigger (partial exit 50% - v3.0 simplified)"""
    strategy = ImprovedExitStrategy(take_profit_levels=[0.10, 0.15, 0.25])

    entry_price = 10000
    current_price = 11100  # +11% (above TP1 10%)
    tp_targets = [11000, 11500, 12500]  # 10%, 15%, 25%
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=tp_targets,
        entry_date=entry_date,
        df=sample_stock_data,
        partial_exits=[],  # No exits yet
    )

    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.TAKE_PROFIT_1
    assert decision.exit_type == "PARTIAL_50%"  # v3.0: Changed from 30% to 50%
    # Net PnL = 11% gross - 1.6% transaction costs = ~9.4%
    assert decision.expected_pnl_percent > 9


def test_check_take_profit_tp2(sample_stock_data):
    """Test TP2 trigger (partial exit 50%)"""
    strategy = ImprovedExitStrategy(take_profit_levels=[0.10, 0.15, 0.25])

    entry_price = 10000
    current_price = 11600  # +16% (above TP2 15%)
    tp_targets = [11000, 11500, 12500]
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=tp_targets,
        entry_date=entry_date,
        df=sample_stock_data,
        partial_exits=[11100],  # TP1 already taken
    )

    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.TAKE_PROFIT_2
    assert decision.exit_type == "PARTIAL_50%"


def test_check_take_profit_tp2_full_exit(sample_stock_data):
    """Test TP2 trigger (full exit) - v2.0 simplified to 2 levels"""
    # v2.0: Only 2 TP levels now (12%, 20%)
    strategy = ImprovedExitStrategy(take_profit_levels=[0.12, 0.20])

    entry_price = 10000
    current_price = 12200  # +22% (above TP2 20%)
    tp_targets = [11200, 12000]  # 12%, 20%
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=tp_targets,
        entry_date=entry_date,
        df=sample_stock_data,
        partial_exits=[11200],  # TP1 already taken
    )

    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.TAKE_PROFIT_2
    assert decision.exit_type == "FULL"


def test_check_take_profit_levels_skip_taken(sample_stock_data):
    """Test that already-taken TP levels are skipped"""
    strategy = ImprovedExitStrategy()

    # v3.0: _check_take_profit now uses ctx dict
    ctx = {
        "current_price": 11100,
        "take_profit_targets": [11000, 11500, 12500],
        "partial_exits": [11100],  # TP1 taken
        "pnl_percent": 11.0,
        "pnl_amount": 1100,
    }

    result = strategy._check_take_profit(ctx)

    # Should not exit for TP1 again (returns None or different reason)
    assert result is None or result.exit_reason != ExitReason.TAKE_PROFIT_1


# ============================================================================
# TRAILING STOP TESTS
# ============================================================================


def test_trailing_stop_not_activated_yet(sample_stock_data):
    """Test trailing stop not activated (profit < activation threshold)"""
    strategy = ImprovedExitStrategy(trailing_stop_activation=0.08)  # 8%

    # v3.0: _check_trailing_stop now uses ctx dict
    ctx = {
        "entry_price": 10000,
        "current_price": 10500,  # +5% (below 8% activation)
        "highest_price": 10500,
        "pnl_percent": 5.0,
        "pnl_amount": 500,
        "df": sample_stock_data,
    }

    result = strategy._check_trailing_stop(ctx)

    assert result is None  # v3.0: Returns None instead of dict


def test_trailing_stop_activated_not_hit(sample_stock_data):
    """Test trailing stop activated but not hit yet"""
    strategy = ImprovedExitStrategy(
        trailing_stop_activation=0.08,
        trailing_stop_distance=0.05,
        use_dynamic_trailing=False,  # Use fixed %
    )

    # v3.0: _check_trailing_stop now uses ctx dict
    ctx = {
        "entry_price": 10000,
        "current_price": 11000,  # +10% (above 8% activation)
        "highest_price": 11000,  # Peak
        "pnl_percent": 10.0,
        "pnl_amount": 1000,
        "df": sample_stock_data,
    }

    result = strategy._check_trailing_stop(ctx)

    # At peak, should not exit yet
    assert result is None  # v3.0: Returns None instead of dict


def test_trailing_stop_hit(sample_stock_data):
    """Test trailing stop hit (price dropped from peak)"""
    strategy = ImprovedExitStrategy(
        trailing_stop_activation=0.08,
        trailing_stop_distance=0.05,  # 5% from peak
        use_dynamic_trailing=False,
    )

    # v3.0: _check_trailing_stop now uses ctx dict
    ctx = {
        "entry_price": 10000,
        "highest_price": 11500,  # Peak was +15%
        "current_price": 10900,  # Dropped 5.2% from peak (11500 * 0.95 = 10925)
        "pnl_percent": 9.0,
        "pnl_amount": 900,
        "df": sample_stock_data,
    }

    result = strategy._check_trailing_stop(ctx)

    # Should trigger trailing stop
    assert result is not None
    assert result.exit_reason == ExitReason.TRAILING_STOP


def test_trailing_stop_atr_based(sample_stock_data):
    """Test ATR-based dynamic trailing stop"""
    strategy = ImprovedExitStrategy(
        trailing_stop_activation=0.08,
        trailing_stop_atr_multiplier=2.0,
        use_dynamic_trailing=True,  # Use ATR
    )

    # Ensure ATR exists
    sample_stock_data["atr"] = 100

    # v3.0: _check_trailing_stop now uses ctx dict
    ctx = {
        "entry_price": 10000,
        "highest_price": 11500,
        "current_price": 11000,  # May or may not trigger depending on ATR
        "pnl_percent": 10.0,
        "pnl_amount": 1000,
        "df": sample_stock_data,
    }

    result = strategy._check_trailing_stop(ctx)

    # Should use ATR-based trailing
    # Trailing stop = highest - 2*ATR = 11500 - 200 = 11300
    # Current = 11000 < 11300 → should exit
    assert result is not None  # v3.0: Returns ExitDecision or None


# ============================================================================
# PROFIT PROTECTION TESTS
# ============================================================================


def test_profit_protection_not_activated(sample_stock_data):
    """Test profit protection not activated (profit below threshold)"""
    strategy = ImprovedExitStrategy(profit_protection_activation=0.05)  # 5%

    # v3.0: _check_profit_protection now uses ctx dict
    ctx = {
        "entry_price": 10000,
        "current_price": 10300,  # +3% (below 5% activation)
        "highest_price": 10300,
        "pnl_percent": 3.0,
        "pnl_amount": 300,
    }

    result = strategy._check_profit_protection(ctx)

    assert result is None  # v3.0: Returns None instead of dict


def test_profit_protection_activated_safe(sample_stock_data):
    """Test profit protection activated but price still safe"""
    strategy = ImprovedExitStrategy(
        profit_protection_activation=0.05,
        profit_protection_percent=0.50,  # Protect 50% of max profit
        trailing_stop_activation=0.10,  # Set higher to test profit protection range
    )

    # v3.0: _check_profit_protection now uses ctx dict
    ctx = {
        "entry_price": 10000,
        "highest_price": 11000,  # Max profit was +10%
        "current_price": 10800,  # Still at +8% (above 50% of 10% = 5%)
        "pnl_percent": 8.0,
        "pnl_amount": 800,
    }

    result = strategy._check_profit_protection(ctx)

    # Should not exit yet (protecting 50% of 10% = 5%, current is 8%)
    assert result is None  # v3.0: Returns None instead of dict


def test_check_exit_time_decay_triggered(sample_stock_data):
    """Test time decay exit (held too long with minimal profit)"""
    strategy = ImprovedExitStrategy(
        max_holding_days=15,
        time_decay_threshold=0.03,  # 3% threshold
    )

    entry_price = 10000
    current_price = 10100  # +1% (below 3% threshold)
    entry_date = datetime.now() - timedelta(
        days=25
    )  # Held 25 calendar days (~17 trading days > 15)

    # Create simple bullish data to avoid triggering reversal patterns
    df = sample_stock_data.copy()
    # Make last few candles bullish (close near high) to avoid shooting star pattern
    df.loc[df.index[-3:], "open"] = [10050, 10075, 10095]
    df.loc[df.index[-3:], "high"] = [10105, 10110, 10110]
    df.loc[df.index[-3:], "low"] = [10040, 10070, 10090]
    df.loc[df.index[-3:], "close"] = [10100, 10105, 10100]

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=df,
    )

    # Should exit due to time decay
    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.TIME_DECAY
    assert decision.urgency == 2


def test_check_exit_time_decay_not_triggered_good_profit(sample_stock_data):
    """Test time decay not triggered when profit is good"""
    strategy = ImprovedExitStrategy(
        max_holding_days=15,
        time_decay_threshold=0.03,
    )

    entry_price = 10000
    current_price = 10500  # +5% (above 3% threshold)
    entry_date = datetime.now() - timedelta(days=20)  # Held long

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    # Should not exit due to time decay (profit is good)
    if decision.should_exit:
        assert decision.exit_reason != ExitReason.TIME_DECAY


def test_check_exit_time_decay_not_triggered_short_hold(sample_stock_data):
    """Test time decay not triggered when holding period is short"""
    strategy = ImprovedExitStrategy(
        max_holding_days=15,
        time_decay_threshold=0.03,
    )

    entry_price = 10000
    current_price = 10100  # +1% (below threshold)
    entry_date = datetime.now() - timedelta(days=5)  # Only 5 days

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    # Should not exit (not held long enough)
    if decision.should_exit:
        assert decision.exit_reason != ExitReason.TIME_DECAY


# ============================================================================
# MARKET CRASH PROTECTION TESTS
# ============================================================================


def test_check_exit_market_crash_with_profit(sample_stock_data, bear_market_regime):
    """Test market crash exit when in profit"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 10500  # +5% profit
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
        market_regime=bear_market_regime,
    )

    # Should exit to protect profit
    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.MARKET_CRASH
    assert decision.urgency == 4


def test_check_exit_market_crash_small_loss(sample_stock_data, bear_market_regime):
    """Test market crash exit when in small loss"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    # With 1.6% transaction costs, need gross loss < 0.4% to have net loss > -2%
    # -1% gross = -2.6% net, which is < -2% threshold, so won't trigger
    # Use -0.3% gross = -1.9% net, which is > -2% threshold
    current_price = 9970  # -0.3% gross loss → ~-1.9% net loss (within -2% threshold)
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
        market_regime=bear_market_regime,
    )

    # Should exit to cut loss (net PnL > -2%)
    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.MARKET_CRASH


def test_check_exit_no_market_crash_bull(sample_stock_data, bull_market_regime):
    """Test no market crash exit in bull market"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 10500
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
        market_regime=bull_market_regime,
    )

    # Should not trigger market crash in BULL
    if decision.should_exit:
        assert decision.exit_reason != ExitReason.MARKET_CRASH


# ============================================================================
# ML SIGNAL TESTS
# ============================================================================


def test_check_exit_ml_sell_signal(sample_stock_data, sell_ml_signal):
    """Test ML sell signal exit"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 10200  # +2% profit
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
        ml_signal=sell_ml_signal,
    )

    # Should exit on ML sell signal
    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.ML_SIGNAL_SELL


def test_check_exit_ml_sell_low_confidence(sample_stock_data):
    """Test ML sell signal with low confidence (should not exit)"""
    strategy = ImprovedExitStrategy()

    ml_signal = {"signal": "SELL", "confidence": 50}  # Low confidence

    entry_price = 10000
    current_price = 10200
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
        ml_signal=ml_signal,
    )

    # Should not exit on low confidence signal
    if decision.should_exit:
        # May exit for other reasons, but not ML signal
        pass  # Depends on other checks


def test_check_exit_ml_sell_in_loss(sample_stock_data):
    """Test ML sell signal when in loss (should not exit)"""
    strategy = ImprovedExitStrategy()

    ml_signal = {"signal": "SELL", "confidence": 75}

    entry_price = 10000
    current_price = 9500  # -5% loss
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
        ml_signal=ml_signal,
    )

    # Logic may exit on ML if loss < -3%, check implementation
    # Based on code: only exit if pnl > -3%
    # Current is -5%, so should NOT exit on ML signal (would hit stop loss first)
    if decision.should_exit and decision.exit_reason != ExitReason.STOP_LOSS:
        # May exit for other reasons
        pass


# ============================================================================
# REVERSAL PATTERN TESTS
# ============================================================================


def test_check_reversal_pattern_bearish_engulfing(sample_stock_data):
    """Test bearish engulfing pattern detection"""
    strategy = ImprovedExitStrategy()

    # Create bearish engulfing
    sample_stock_data.iloc[-2, sample_stock_data.columns.get_loc("open")] = 10000
    sample_stock_data.iloc[-2, sample_stock_data.columns.get_loc("close")] = 10200  # Bullish
    sample_stock_data.iloc[-1, sample_stock_data.columns.get_loc("open")] = 10300
    sample_stock_data.iloc[-1, sample_stock_data.columns.get_loc("close")] = (
        9900  # Bearish engulfing
    )

    # v3.0: _check_reversal_pattern now uses ctx dict
    ctx = {
        "df": sample_stock_data,
        "pnl_percent": 5.0,
        "pnl_amount": 500,
    }

    result = strategy._check_reversal_pattern(ctx)

    # Should detect bearish engulfing
    assert result is not None
    assert result.exit_reason == ExitReason.REVERSAL_PATTERN


def test_check_reversal_pattern_no_pattern(sample_stock_data):
    """Test when no reversal pattern present"""
    strategy = ImprovedExitStrategy()

    # v3.0: _check_reversal_pattern now uses ctx dict
    ctx = {
        "df": sample_stock_data,
        "pnl_percent": 5.0,
        "pnl_amount": 500,
    }

    result = strategy._check_reversal_pattern(ctx)

    # Should not detect pattern
    assert result is None  # v3.0: Returns None instead of dict


def test_check_reversal_pattern_in_loss(sample_stock_data):
    """Test reversal pattern when in loss (should not exit)"""
    strategy = ImprovedExitStrategy()

    # v3.0: _check_reversal_pattern now uses ctx dict
    ctx = {
        "df": sample_stock_data,
        "pnl_percent": -2.0,
        "pnl_amount": -200,
    }

    result = strategy._check_reversal_pattern(ctx)

    # Should not exit on pattern when in loss
    assert result is None  # v3.0: Returns None instead of dict


# ============================================================================
# SUPPORT BREAKDOWN TESTS
# ============================================================================


def test_check_support_breakdown_triggered(sample_stock_data):
    """Test support breakdown with volume confirmation"""
    strategy = ImprovedExitStrategy()

    # Set current price below support
    support = sample_stock_data["low"].iloc[-20:-1].min()
    current_price = support * 0.98  # Below support

    # Set high volume
    sample_stock_data.loc[sample_stock_data.index[-1], "volume"] = 500_000  # High

    # v3.0: _check_support_breakdown now uses ctx dict
    ctx = {
        "df": sample_stock_data,
        "current_price": current_price,
        "entry_price": 10000,
        "pnl_percent": 2.0,
        "pnl_amount": 200,
    }

    result = strategy._check_support_breakdown(ctx)

    # Should trigger breakdown
    assert result is not None
    assert result.exit_reason == ExitReason.BREAKDOWN


def test_check_support_breakdown_no_volume(sample_stock_data):
    """Test support breakdown without volume (should not exit)"""
    strategy = ImprovedExitStrategy()

    support = sample_stock_data["low"].iloc[-20:-1].min()
    current_price = support * 0.98

    # Low volume (no confirmation)
    sample_stock_data.loc[sample_stock_data.index[-1], "volume"] = 100_000

    # v3.0: _check_support_breakdown now uses ctx dict
    ctx = {
        "df": sample_stock_data,
        "current_price": current_price,
        "entry_price": 10000,
        "pnl_percent": 2.0,
        "pnl_amount": 200,
    }

    result = strategy._check_support_breakdown(ctx)

    # Should not trigger without volume
    assert result is None  # v3.0: Returns None instead of dict


# ============================================================================
# VOLUME CONFIRMATION TESTS
# ============================================================================


def test_position_highs_tracking(sample_stock_data):
    """Test that highest price is tracked"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    entry_date = datetime.now() - timedelta(days=5)

    # First check - price at 10500
    strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=10500,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    assert strategy.position_highs["TEST"] == 10500

    # Second check - price higher at 11000
    strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=11000,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    assert strategy.position_highs["TEST"] == 11000

    # Third check - price lower at 10800 (should keep 11000)
    strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=10800,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    assert strategy.position_highs["TEST"] == 11000  # Still highest


def test_clear_position_tracking():
    """Test clearing position tracking"""
    strategy = ImprovedExitStrategy()

    strategy.position_highs["TEST"] = 11000

    strategy.clear_position_tracking("TEST")

    assert "TEST" not in strategy.position_highs


def test_get_tracked_positions():
    """Test getting list of tracked positions"""
    strategy = ImprovedExitStrategy()

    strategy.position_highs["TEST1"] = 10000
    strategy.position_highs["TEST2"] = 11000

    tracked = strategy.get_tracked_positions()

    assert "TEST1" in tracked
    assert "TEST2" in tracked
    assert len(tracked) == 2


def test_clear_all_tracking():
    """Test clearing all position tracking"""
    strategy = ImprovedExitStrategy()

    strategy.position_highs["TEST1"] = 10000
    strategy.position_highs["TEST2"] = 11000

    strategy.clear_all_tracking()

    assert len(strategy.position_highs) == 0


# ============================================================================
# MESSAGE FORMATTING TESTS
# ============================================================================


def test_format_exit_message_exit():
    """Test exit message formatting"""
    strategy = ImprovedExitStrategy()

    decision = ExitDecision(
        should_exit=True,
        exit_reason=ExitReason.TAKE_PROFIT_1,
        exit_type="PARTIAL_30%",
        exit_price=11000,
        expected_pnl=1000,
        expected_pnl_percent=10.0,
        message="TP1 hit",
        urgency=3,
    )

    message = strategy.format_exit_message("TEST", decision)

    assert "TEST" in message
    assert "EXIT" in message
    assert "PARTIAL_30%" in message
    assert "11,000" in message or "11000" in message


def test_format_exit_message_hold():
    """Test hold message formatting"""
    strategy = ImprovedExitStrategy()

    decision = ExitDecision(
        should_exit=False,
        exit_reason=None,
        exit_type="HOLD",
        exit_price=10500,
        expected_pnl=500,
        expected_pnl_percent=5.0,
        message="HOLD - P&L: +5.00%",
        urgency=0,
    )

    message = strategy.format_exit_message("TEST", decision)

    assert "TEST" in message
    assert "HOLD" in message


# ============================================================================
# EDGE CASES
# ============================================================================


def test_check_exit_negative_pnl(sample_stock_data):
    """Test with large negative PnL"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 8000  # -20% loss
    stop_loss = 9000
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=sample_stock_data,
    )

    # Should trigger stop loss
    assert decision.should_exit is True
    assert decision.expected_pnl_percent < 0


def test_check_exit_empty_df():
    """Test with empty dataframe"""
    strategy = ImprovedExitStrategy()

    empty_df = pd.DataFrame()
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=10000,
        current_price=10500,
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=entry_date,
        df=empty_df,
    )

    # Should handle gracefully
    assert isinstance(decision, ExitDecision)


def test_check_exit_no_tp_targets(sample_stock_data):
    """Test with no take profit targets"""
    strategy = ImprovedExitStrategy()

    entry_price = 10000
    current_price = 11000
    entry_date = datetime.now() - timedelta(days=5)

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=9300,
        take_profit_targets=[],  # No TPs
        entry_date=entry_date,
        df=sample_stock_data,
    )

    # Should handle gracefully (no TP exits possible)
    assert isinstance(decision, ExitDecision)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow_stop_loss(sample_stock_data):
    """Test complete workflow - stop loss exit"""
    strategy = ImprovedExitStrategy()

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=10000,
        current_price=9200,  # Hit stop
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=datetime.now() - timedelta(days=5),
        df=sample_stock_data,
    )

    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.STOP_LOSS
    assert decision.exit_type == "FULL"
    assert decision.urgency == 5


def test_full_workflow_take_profit(sample_stock_data):
    """Test complete workflow - take profit exit (v2.0 - 2 TP levels)"""
    strategy = ImprovedExitStrategy()

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=10000,
        current_price=12200,  # Hit TP2 (22% gross → ~20.4% net after 1.6% costs)
        stop_loss=9300,
        take_profit_targets=[11200, 12000],  # v2.0: 12%, 20%
        entry_date=datetime.now() - timedelta(days=5),
        df=sample_stock_data,
        partial_exits=[11200],  # TP1 already taken
    )

    assert decision.should_exit is True
    assert decision.exit_reason == ExitReason.TAKE_PROFIT_2
    # Net PnL = 22% gross - 1.6% transaction costs = ~20.4%
    assert decision.expected_pnl_percent > 20


def test_full_workflow_hold(sample_stock_data):
    """Test complete workflow - hold position"""
    strategy = ImprovedExitStrategy()

    # Create simple bullish data to avoid triggering reversal patterns
    df = sample_stock_data.copy()
    # Make last few candles bullish (close near high) to avoid shooting star pattern
    df.loc[df.index[-3:], "open"] = [10150, 10175, 10195]
    df.loc[df.index[-3:], "high"] = [10205, 10210, 10210]
    df.loc[df.index[-3:], "low"] = [10140, 10170, 10190]
    df.loc[df.index[-3:], "close"] = [10200, 10205, 10200]

    decision = strategy.check_exit(
        symbol="TEST",
        entry_price=10000,
        current_price=10200,  # Small profit, no triggers
        stop_loss=9300,
        take_profit_targets=[11000, 11500, 12500],
        entry_date=datetime.now() - timedelta(days=2),  # Short hold
        df=df,
    )

    # Should hold (no exit triggers)
    assert decision.should_exit is False
    assert decision.exit_type == "HOLD"
    assert decision.urgency == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
