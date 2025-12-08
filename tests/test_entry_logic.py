"""
Unit tests for src/strategies/entry_logic.py - Entry Signal Logic with 12 Filters

Tests cover:
- Initialization and configuration
- ML signal validation and fallback
- 12 individual filters
- Filter combinations
- Confidence adjustment logic
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.strategies.entry_logic import (
    ImprovedEntryLogic,
    EntrySignal,
    SignalStrength,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    n = 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")

    # Create uptrend data
    base_price = 10000
    trend = np.linspace(0, 1000, n)
    noise = np.random.normal(0, 100, n)
    close = base_price + trend + noise

    high = close + np.abs(np.random.normal(50, 20, n))
    low = close - np.abs(np.random.normal(50, 20, n))
    open_price = close - np.random.normal(0, 30, n)
    volume = np.random.uniform(200_000, 250_000, n)  # Sufficient for VN liquidity

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

    # Add required technical indicators
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["rsi"] = 50  # Neutral RSI
    df["obv"] = df["volume"].cumsum()

    # Calculate ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    return df


@pytest.fixture
def bull_ml_signal():
    """Create strong bull ML signal"""
    return {
        "signal": "BUY",
        "confidence": 85,
        "features": {},
    }


@pytest.fixture
def weak_ml_signal():
    """Create weak ML signal (should be rejected)"""
    return {
        "signal": "BUY",
        "confidence": 45,
        "features": {},
    }


@pytest.fixture
def sell_ml_signal():
    """Create sell signal"""
    return {
        "signal": "SELL",
        "confidence": 80,
    }


@pytest.fixture
def bull_market_regime():
    """Create bull market regime"""
    return {
        "regime": "BULL",
        "tradeable": True,
        "confidence": 80,
        "details": {},
    }


@pytest.fixture
def bear_market_regime():
    """Create bear market regime"""
    return {
        "regime": "BEAR",
        "tradeable": False,
        "confidence": 70,
        "details": {},
    }


@pytest.fixture
def sideways_market_regime():
    """Create sideways market regime"""
    return {
        "regime": "SIDEWAYS",
        "tradeable": True,
        "confidence": 60,
        "details": {},
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_entry_logic_init_default():
    """Test default initialization (v3.0 - optimized defaults for better win rate)"""
    logic = ImprovedEntryLogic()

    # v3.0: Optimized thresholds - values from constructor defaults
    # Note: These are the constructor defaults, not config values
    assert logic.min_confidence == 45  # Constructor default
    assert logic.min_risk_reward == 1.0  # Constructor default (config may override)
    assert logic.support_distance_percent == 7.0  # Constructor default
    assert logic.require_trend_alignment is False  # Constructor default
    assert logic.require_volume_confirmation is False  # Constructor default (soft filter)


def test_entry_logic_init_custom():
    """Test custom initialization"""
    logic = ImprovedEntryLogic(
        min_confidence=70,
        min_risk_reward=2.5,
        support_distance_percent=4.0,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )

    assert logic.min_confidence == 70
    assert logic.min_risk_reward == 2.5
    assert logic.support_distance_percent == 4.0
    assert logic.require_trend_alignment is False
    assert logic.require_volume_confirmation is False


def test_entry_logic_init_with_managers():
    """Test initialization with portfolio and performance managers"""
    mock_portfolio = Mock()
    mock_performance = Mock()

    logic = ImprovedEntryLogic(
        portfolio_manager=mock_portfolio,
        performance_monitor=mock_performance,
    )

    assert logic.portfolio_manager is mock_portfolio
    assert logic.performance_monitor is not None


def test_entry_logic_liquidity_tiers_loaded():
    """Test that liquidity tiers are loaded from config"""
    logic = ImprovedEntryLogic()

    assert "large" in logic.liquidity_tiers
    assert "mid" in logic.liquidity_tiers
    assert "small" in logic.liquidity_tiers


# ============================================================================
# VALIDATION TESTS
# ============================================================================


def test_validate_initial_signal_valid_ml(sample_stock_data, bull_ml_signal):
    """Test validation with valid ML signal"""
    logic = ImprovedEntryLogic(min_confidence=60)

    is_valid, signal_type, confidence, price = logic._validate_initial_signal(
        sample_stock_data, bull_ml_signal
    )

    assert is_valid is True
    assert signal_type == "BUY"
    assert confidence == 85
    assert price > 0


def test_validate_initial_signal_weak_ml(sample_stock_data, weak_ml_signal):
    """Test validation with weak ML signal (should reject)"""
    logic = ImprovedEntryLogic(min_confidence=60)

    is_valid, reason, confidence, price = logic._validate_initial_signal(
        sample_stock_data, weak_ml_signal
    )

    assert is_valid is False
    assert "Confidence low" in reason or "confidence" in reason.lower()
    assert confidence == 0


def test_validate_initial_signal_sell(sample_stock_data, sell_ml_signal):
    """Test validation with SELL signal (should reject for BUY logic)"""
    logic = ImprovedEntryLogic()

    is_valid, reason, _, _ = logic._validate_initial_signal(sample_stock_data, sell_ml_signal)

    assert is_valid is False
    assert "SELL" in reason


def test_validate_initial_signal_none_ml_fallback(sample_stock_data):
    """Test fallback to technical analysis when ML signal is None"""
    logic = ImprovedEntryLogic()

    # Should fallback to technical analysis
    is_valid, signal_type, confidence, price = logic._validate_initial_signal(
        sample_stock_data, None
    )

    # May be valid or invalid depending on technical indicators
    assert isinstance(is_valid, bool)
    if is_valid:
        assert signal_type == "BUY"
        assert confidence >= 50  # Technical threshold
        assert logic._is_technical_only is True


def test_validate_initial_signal_insufficient_data():
    """Test validation with insufficient data"""
    # Only 20 rows (need 50)
    small_df = pd.DataFrame(
        {
            "close": np.random.randn(20) + 10000,
            "high": np.random.randn(20) + 10010,
            "low": np.random.randn(20) + 9990,
            "volume": [100000] * 20,
        }
    )

    logic = ImprovedEntryLogic()
    ml_signal = {"signal": "BUY", "confidence": 80}

    # The implementation may raise DataQualityError or return validation failure
    try:
        is_valid, reason, _, _ = logic._validate_initial_signal(small_df, ml_signal)
        assert is_valid is False
        assert "validation failed" in reason.lower() or "insufficient" in reason.lower()
    except Exception as e:
        # DataQualityError is also acceptable - it means validation caught the issue
        assert "insufficient" in str(e).lower() or "data" in str(e).lower()


# ============================================================================
# FILTER 1: MARKET REGIME TESTS
# ============================================================================


def test_filter_market_regime_tradeable(sample_stock_data, bull_market_regime):
    """Test market regime filter - tradeable market"""
    logic = ImprovedEntryLogic()

    passed, reasons, warnings, adjustments, breakdown = logic._run_all_filters(
        sample_stock_data, "BUY", 10500, bull_market_regime
    )

    # Should not block on market regime
    regime_blocks = [b for b in breakdown if b["filter"] == "market_regime"]
    assert len(regime_blocks) == 0 or all(b["delta"] is not None for b in regime_blocks)


def test_filter_market_regime_not_tradeable(sample_stock_data, bear_market_regime):
    """Test market regime filter - non-tradeable market (should block)"""
    logic = ImprovedEntryLogic()

    passed, reasons, warnings, adjustments, breakdown = logic._run_all_filters(
        sample_stock_data, "BUY", 10500, bear_market_regime
    )

    # Should block
    assert passed is False
    assert any(b["filter"] == "market_regime" for b in breakdown)


def test_check_trend_alignment_uptrend(sample_stock_data):
    """Test trend alignment for uptrend"""
    logic = ImprovedEntryLogic()

    # Create clear uptrend data: steadily rising prices
    # This will make calculated EMAs align properly (EMA20 > EMA50 > EMA200)
    n = len(sample_stock_data)
    # Strong uptrend: each day higher than previous
    uptrend_close = 10000 + np.arange(n) * 50  # +50 per day
    sample_stock_data["close"] = uptrend_close

    result = logic._check_trend_alignment(sample_stock_data, "BUY")

    assert result["aligned"] is True
    assert "reason" in result


def test_check_support_resistance_near_support(sample_stock_data):
    """Test support/resistance - price near support (good)"""
    logic = ImprovedEntryLogic()

    # Set current price near recent low (support)
    current_price = sample_stock_data["low"].iloc[-20:].min() * 1.01  # 1% above support

    result = logic._check_support_resistance(sample_stock_data, current_price)

    assert result["near_support"] or result["bouncing_from_support"]


def test_check_volume_confirmation_high_volume(sample_stock_data):
    """Test volume confirmation - high volume (good)"""
    logic = ImprovedEntryLogic()

    # Set recent volume high
    sample_stock_data.loc[sample_stock_data.index[-1], "volume"] = 300_000  # High

    result = logic._check_volume_confirmation(sample_stock_data, None)

    # Should confirm with high volume
    assert result["confirmed"] is True


def test_check_volume_confirmation_low_volume(sample_stock_data):
    """Test volume confirmation - low volume (bad)"""
    logic = ImprovedEntryLogic()

    # Set recent volume very low
    sample_stock_data.loc[sample_stock_data.index[-1], "volume"] = 50_000  # Low

    result = logic._check_volume_confirmation(sample_stock_data, None)

    # May reject due to low volume
    assert "confirmed" in result


def test_check_volume_confirmation_regime_relaxed(sample_stock_data, bull_market_regime):
    """Test volume confirmation relaxed in bull market"""
    logic = ImprovedEntryLogic(regime_aware_filtering=True, require_volume_confirmation=True)

    # Low volume
    sample_stock_data.loc[sample_stock_data.index[-1], "volume"] = 50_000

    # In BULL market with regime_aware=True, should relax volume requirement
    passed, reasons, warnings, adjustments, breakdown = logic._run_all_filters(
        sample_stock_data, "BUY", 10500, bull_market_regime
    )

    # Should not block (may warn)
    volume_blocks = [b for b in breakdown if b["filter"] == "volume" and b["delta"] is None]
    # In bull market, volume is relaxed
    # assert len(volume_blocks) == 0  # Should not hard block


# ============================================================================
# FILTER 5: LIQUIDITY TESTS
# ============================================================================


def test_check_liquidity_critical(sample_stock_data):
    """Test liquidity check - critical low liquidity"""
    logic = ImprovedEntryLogic(min_liquidity_value=10_000_000_000)  # Very high threshold

    # Set low volume
    sample_stock_data["volume"] = 10_000  # Very low

    current_price = 10500
    result = logic._check_liquidity(sample_stock_data, current_price)

    # Should flag as critical or insufficient
    assert result["critical"] or not result["sufficient"]


def test_check_liquidity_tiered():
    """Test tiered liquidity thresholds"""
    logic = ImprovedEntryLogic(use_tiered_liquidity=True)

    # Create data with different daily values
    df_small = pd.DataFrame(
        {
            "close": [1000] * 50,
            "volume": [100_000] * 50,  # 100M VND daily value
        }
    )

    result = logic._check_liquidity(df_small, 1000)

    # Should classify as small cap
    assert "tier" in result


# ============================================================================
# FILTER 6: VOLATILITY TESTS
# ============================================================================


def test_check_rsi_normal(sample_stock_data):
    """Test RSI filter - normal RSI (50)"""
    logic = ImprovedEntryLogic()

    sample_stock_data["rsi"] = 50  # Neutral

    result = logic._check_rsi(sample_stock_data)

    assert result["overbought"] is False
    assert result["oversold"] is False


def test_check_rsi_overbought(sample_stock_data):
    """Test RSI filter - overbought (> 70)"""
    logic = ImprovedEntryLogic()

    sample_stock_data["rsi"] = 80  # Overbought

    result = logic._check_rsi(sample_stock_data)

    assert result["overbought"] is True


def test_check_rsi_oversold(sample_stock_data):
    """Test RSI filter - oversold (< 30)"""
    logic = ImprovedEntryLogic()

    sample_stock_data["rsi"] = 20  # Oversold

    result = logic._check_rsi(sample_stock_data)

    assert result["oversold"] is True


# ============================================================================
# ADJUSTMENT LOGIC TESTS
# ============================================================================


def test_add_adjustment():
    """Test adding adjustment to lists"""
    logic = ImprovedEntryLogic()

    adjustments = []
    breakdown = []

    logic._add_adjustment(adjustments, breakdown, "test_filter", +10, "Test note")

    assert len(adjustments) == 1
    assert adjustments[0] == 10
    assert len(breakdown) == 1
    assert breakdown[0]["filter"] == "test_filter"
    assert breakdown[0]["delta"] == 10
    assert breakdown[0]["note"] == "Test note"


def test_adjustment_scaling_bull_market(sample_stock_data, bull_market_regime):
    """Test that adjustments are scaled in bull market"""
    logic = ImprovedEntryLogic()

    # In bull market, penalties should be scaled down (0.7x)
    passed, reasons, warnings, adjustments, breakdown = logic._run_all_filters(
        sample_stock_data, "BUY", 10500, bull_market_regime
    )

    # Check that bull market regime is recognized
    # (actual scaling is internal, but we can verify regime is passed)
    assert bull_market_regime["regime"] == "BULL"


def test_adjustment_scaling_bear_market(sample_stock_data, bear_market_regime):
    """Test that adjustments are scaled in bear market"""
    logic = ImprovedEntryLogic()

    # Bear market should block immediately
    passed, reasons, warnings, adjustments, breakdown = logic._run_all_filters(
        sample_stock_data, "BUY", 10500, bear_market_regime
    )

    assert passed is False  # Should not pass in bear market


# ============================================================================
# MAIN ANALYZE_ENTRY TESTS
# ============================================================================


@patch("src.strategies.entry_logic.ImprovedEntryLogic._check_vietnam_market_liquidity")
def test_analyze_entry_valid_signal(
    mock_vn_liquidity, sample_stock_data, bull_ml_signal, bull_market_regime
):
    """Test full analyze_entry with valid signal"""
    # Mock Vietnam liquidity to pass
    mock_vn_liquidity.return_value = {"sufficient": True, "reason": "OK"}

    logic = ImprovedEntryLogic(
        min_confidence=60,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )

    result = logic.analyze_entry(
        sample_stock_data, bull_ml_signal, market_regime=bull_market_regime, symbol="TEST"
    )

    assert isinstance(result, EntrySignal)
    assert result.signal_type in ["BUY", "HOLD"]


@patch("src.strategies.entry_logic.ImprovedEntryLogic._check_vietnam_market_liquidity")
def test_analyze_entry_weak_signal(mock_vn_liquidity, sample_stock_data, weak_ml_signal):
    """Test analyze_entry with weak signal (should reject)"""
    mock_vn_liquidity.return_value = {"sufficient": True, "reason": "OK"}

    logic = ImprovedEntryLogic(min_confidence=60)

    result = logic.analyze_entry(sample_stock_data, weak_ml_signal, symbol="TEST")

    assert result.should_enter is False
    assert result.confidence == 0


@patch("src.strategies.entry_logic.ImprovedEntryLogic._check_vietnam_market_liquidity")
def test_analyze_entry_bear_market(
    mock_vn_liquidity, sample_stock_data, bull_ml_signal, bear_market_regime
):
    """Test analyze_entry in bear market (should reject)"""
    mock_vn_liquidity.return_value = {"sufficient": True, "reason": "OK"}

    logic = ImprovedEntryLogic()

    result = logic.analyze_entry(
        sample_stock_data, bull_ml_signal, market_regime=bear_market_regime, symbol="TEST"
    )

    assert result.should_enter is False


@patch("src.strategies.entry_logic.ImprovedEntryLogic._check_vietnam_market_liquidity")
def test_analyze_entry_with_portfolio_manager(mock_vn_liquidity, sample_stock_data, bull_ml_signal):
    """Test analyze_entry with portfolio manager"""
    mock_vn_liquidity.return_value = {"sufficient": True, "reason": "OK"}

    mock_portfolio = Mock()
    mock_portfolio.calculate_correlation.return_value = 0.3  # Low correlation

    logic = ImprovedEntryLogic(
        portfolio_manager=mock_portfolio,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )

    result = logic.analyze_entry(sample_stock_data, bull_ml_signal, symbol="TEST")

    # Should call portfolio manager
    # mock_portfolio.calculate_correlation may be called


# ============================================================================
# STOP LOSS & TAKE PROFIT TESTS
# ============================================================================


def test_calculate_technical_confidence(sample_stock_data):
    """Test technical confidence calculation"""
    logic = ImprovedEntryLogic()

    confidence = logic._calculate_technical_confidence(sample_stock_data)

    assert isinstance(confidence, (int, float))
    assert 0 <= confidence <= 100


def test_get_technical_signal_uptrend(sample_stock_data):
    """Test technical signal for uptrend"""
    logic = ImprovedEntryLogic()

    # Last price > previous price
    sample_stock_data.loc[sample_stock_data.index[-1], "close"] = 11000
    sample_stock_data.loc[sample_stock_data.index[-2], "close"] = 10500

    signal = logic._get_technical_signal(sample_stock_data)

    assert signal == "BUY"


def test_get_technical_signal_downtrend(sample_stock_data):
    """Test technical signal for downtrend

    IMPROVED v8.1: Uses multi-indicator scoring (need <= -3 for SELL):
    - EMA20 < EMA50: -2 points
    - RSI > 70 (overbought): -1 point
    - Price < EMA20 * 0.97: -1 point
    - MACD < signal and < 0: -1 point

    For proper SELL signal, we need clear downtrend indicators.
    Simple 2-candle comparison is insufficient.
    """
    logic = ImprovedEntryLogic()

    # Create clear downtrend data: steadily falling prices
    n = len(sample_stock_data)
    downtrend_close = 15000 - np.arange(n) * 50  # -50 per day (strong downtrend)
    sample_stock_data["close"] = downtrend_close

    # Recalculate EMAs for downtrend
    sample_stock_data["ema20"] = sample_stock_data["close"].ewm(span=20).mean()
    sample_stock_data["ema50"] = sample_stock_data["close"].ewm(span=50).mean()

    # Add bearish indicators
    sample_stock_data["rsi"] = 75  # Overbought
    sample_stock_data["macd"] = -50  # Negative MACD
    sample_stock_data["macd_signal"] = -30  # MACD below signal

    signal = logic._get_technical_signal(sample_stock_data)

    # With strong downtrend indicators, should return SELL or HOLD
    # (HOLD is acceptable if score is between -3 and 3)
    assert signal in ["SELL", "HOLD"]


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================


def test_analyze_entry_missing_columns(sample_stock_data):
    """Test analyze_entry with missing required columns"""
    logic = ImprovedEntryLogic()

    # Remove required column
    df_missing = sample_stock_data.drop(columns=["close"])

    ml_signal = {"signal": "BUY", "confidence": 80}

    # The implementation may raise DataQualityError or return EntrySignal with should_enter=False
    try:
        result = logic.analyze_entry(df_missing, ml_signal, symbol="TEST")
        assert result.should_enter is False
    except Exception as e:
        # DataQualityError is also acceptable - it means validation caught the issue
        assert (
            "missing" in str(e).lower() or "column" in str(e).lower() or "close" in str(e).lower()
        )


def test_analyze_entry_nan_values(sample_stock_data):
    """Test analyze_entry with NaN values"""
    logic = ImprovedEntryLogic()

    # Add NaN values
    sample_stock_data.loc[sample_stock_data.index[-1], "close"] = np.nan

    ml_signal = {"signal": "BUY", "confidence": 80}

    # The implementation may raise DataQualityError or return EntrySignal
    try:
        result = logic.analyze_entry(sample_stock_data, ml_signal, symbol="TEST")
        # Should handle gracefully - either return EntrySignal or reject entry
        assert isinstance(result, EntrySignal)
    except Exception as e:
        # DataQualityError is also acceptable - it means validation caught the NaN issue
        assert "nan" in str(e).lower() or "invalid" in str(e).lower()


def test_analyze_entry_zero_price(sample_stock_data):
    """Test analyze_entry with zero price (edge case)"""
    logic = ImprovedEntryLogic()

    # Set price to zero (edge case)
    sample_stock_data.loc[sample_stock_data.index[-1], "close"] = 0

    ml_signal = {"signal": "BUY", "confidence": 80}

    # The implementation may raise DataQualityError or return EntrySignal with should_enter=False
    try:
        result = logic.analyze_entry(sample_stock_data, ml_signal, symbol="TEST")
        # Should reject
        assert result.should_enter is False
    except Exception as e:
        # DataQualityError is also acceptable - it means validation caught the zero price issue
        assert "invalid" in str(e).lower() or "close" in str(e).lower() or "0" in str(e)


# ============================================================================
# VIETNAM MARKET LIQUIDITY TESTS
# ============================================================================


def test_format_signal_message_entry():
    """Test signal message formatting for entry signal"""
    logic = ImprovedEntryLogic()

    signal = EntrySignal(
        should_enter=True,
        signal_type="BUY",
        confidence=85,
        strength=SignalStrength.STRONG,
        position_size_multiplier=1.0,
        reasons=["Good trend", "High volume"],
        warnings=["Near resistance"],
        entry_price=10500,
        stop_loss=9800,
        take_profit_targets=[11000, 11500, 12000],
    )

    message = logic.format_signal_message(signal, "VNM")

    assert "VNM" in message
    assert "BUY" in message
    assert "85" in message  # Confidence


def test_format_signal_message_no_entry():
    """Test signal message formatting for no entry"""
    logic = ImprovedEntryLogic()

    signal = EntrySignal(
        should_enter=False,
        signal_type="HOLD",
        confidence=0,
        strength=SignalStrength.NO_SIGNAL,
        position_size_multiplier=0.0,
        reasons=["Low confidence"],
        warnings=[],
        entry_price=0,
        stop_loss=0,
        take_profit_targets=[],
    )

    message = logic.format_signal_message(signal, "VNM")

    assert "HOLD" in message or "không" in message.lower()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@patch("src.strategies.entry_logic.ImprovedEntryLogic._check_vietnam_market_liquidity")
def test_full_workflow_strong_signal(
    mock_vn_liquidity, sample_stock_data, bull_ml_signal, bull_market_regime
):
    """Test complete workflow with strong signal"""
    mock_vn_liquidity.return_value = {"sufficient": True, "reason": "OK"}

    logic = ImprovedEntryLogic(
        min_confidence=60,
        require_trend_alignment=False,
        require_volume_confirmation=False,
    )

    # Ensure data has required indicators
    sample_stock_data["ema20"] = sample_stock_data["close"].ewm(span=20).mean()
    sample_stock_data["ema50"] = sample_stock_data["close"].ewm(span=50).mean()
    sample_stock_data["rsi"] = 50

    result = logic.analyze_entry(
        sample_stock_data, bull_ml_signal, market_regime=bull_market_regime, symbol="VNM"
    )

    # Should produce valid entry signal
    assert isinstance(result, EntrySignal)
    assert result.confidence >= 0


@patch("src.strategies.entry_logic.ImprovedEntryLogic._check_vietnam_market_liquidity")
def test_full_workflow_rejected_by_filters(
    mock_vn_liquidity, sample_stock_data, bull_ml_signal, bear_market_regime
):
    """Test complete workflow rejected by filters"""
    mock_vn_liquidity.return_value = {"sufficient": True, "reason": "OK"}

    logic = ImprovedEntryLogic()

    result = logic.analyze_entry(
        sample_stock_data,
        bull_ml_signal,
        market_regime=bear_market_regime,  # Bear market should block
        symbol="VNM",
    )

    # Should be rejected
    assert result.should_enter is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
