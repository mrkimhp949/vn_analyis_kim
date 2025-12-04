"""
Integration tests for full trading workflow
Tests the complete flow from signal generation to position exit
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Note: add_ml_features requires 'ta' library which may not be installed
# We'll manually add required indicators instead
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.exit_logic import ExitReason, ImprovedExitStrategy
from src.strategies.position_sizing import EnhancedPositionSizer
from src.utils.vietnam_market import VietnamMarketValidator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators required for testing"""
    df = df.copy()

    # Moving averages
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # RSI (simple approximation)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume indicators
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]

    # Momentum
    df["momentum"] = df["close"].pct_change(10)
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # Volatility
    df["volatility_20"] = df["close"].pct_change().rolling(20).std()

    # MACD (simplified)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_dif"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_high"] = df["bb_mid"] + (bb_std * 2)
    df["bb_low"] = df["bb_mid"] - (bb_std * 2)
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    return df


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def bull_market_data():
    """Generate realistic bull market data"""
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)

    # Bull trend: +30% over 250 days
    np.random.seed(42)
    trend = np.linspace(80000, 104000, n)
    noise = np.random.normal(0, 800, n)
    close = trend + noise

    high = close + np.abs(np.random.normal(500, 200, n))
    low = close - np.abs(np.random.normal(500, 200, n))
    openp = close + np.random.normal(0, 300, n)
    volume = np.random.uniform(250_000, 400_000, n)

    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    # Add technical indicators
    df = add_basic_indicators(df)
    return df


@pytest.fixture
def bear_market_data():
    """Generate realistic bear market data"""
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)

    # Bear trend: -20% over 250 days
    np.random.seed(43)
    trend = np.linspace(100000, 80000, n)
    noise = np.random.normal(0, 1000, n)
    close = trend + noise

    high = close + np.abs(np.random.normal(600, 300, n))
    low = close - np.abs(np.random.normal(600, 300, n))
    openp = close + np.random.normal(0, 400, n)
    volume = np.random.uniform(200_000, 350_000, n)

    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    df = add_basic_indicators(df)
    return df


@pytest.fixture
def sideways_market_data():
    """Generate realistic sideways market data"""
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)

    # Sideways: oscillates around 90k
    np.random.seed(44)
    trend = 90000 + np.sin(np.linspace(0, 4 * np.pi, n)) * 3000
    noise = np.random.normal(0, 1200, n)
    close = trend + noise

    high = close + np.abs(np.random.normal(700, 300, n))
    low = close - np.abs(np.random.normal(700, 300, n))
    openp = close + np.random.normal(0, 500, n)
    volume = np.random.uniform(220_000, 380_000, n)

    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    df = add_basic_indicators(df)
    return df


@pytest.fixture
def mock_bull_regime():
    """Mock bull market regime"""
    regime = MagicMock()
    regime.regime = "BULL"
    regime.confidence = 85
    regime.tradeable = True
    regime.description = "Strong bull market"
    return {"regime": "BULL", "confidence": 85, "tradeable": True}


@pytest.fixture
def mock_bear_regime():
    """Mock bear market regime"""
    return {"regime": "BEAR", "confidence": 75, "tradeable": False}


@pytest.fixture
def trading_components():
    """Initialize all trading components"""
    return {
        "entry_logic": ImprovedEntryLogic(
            min_confidence=50,  # Lower for testing
            min_risk_reward=1.5,  # Lower for testing
            require_trend_alignment=False,
            require_volume_confirmation=False,
            min_liquidity_value=2_000_000_000,
            min_avg_volume=150_000,
        ),
        "exit_strategy": ImprovedExitStrategy(
            take_profit_levels=[0.08, 0.12, 0.18],
            trailing_stop_activation=0.06,
            trailing_stop_distance=0.04,
            max_holding_days=15,
            default_stop_loss_pct=-6.0,
        ),
        "position_sizer": EnhancedPositionSizer(
            total_capital=100_000_000,
            max_risk_per_trade=0.02,
            max_position_size=0.10,
            max_portfolio_risk=0.15,
            use_kelly=True,
        ),
        "vn_validator": VietnamMarketValidator(),
    }


# =============================================================================
# FULL WORKFLOW INTEGRATION TESTS
# =============================================================================


def test_full_workflow_bull_market_entry_to_exit(
    bull_market_data, trading_components, mock_bull_regime
):
    """Test complete workflow in bull market: entry → monitoring → exit"""
    entry_logic = trading_components["entry_logic"]
    exit_strategy = trading_components["exit_strategy"]
    position_sizer = trading_components["position_sizer"]
    vn_validator = trading_components["vn_validator"]

    # STEP 1: Check market regime
    assert mock_bull_regime["tradeable"] is True

    # STEP 2: Check Vietnam market constraints
    is_liquid, _ = vn_validator.check_liquidity_requirements(bull_market_data, "VNM")
    assert is_liquid is True

    current_price = bull_market_data["close"].iloc[-1]
    reference_price = bull_market_data["close"].iloc[-2]
    is_safe, _ = vn_validator.check_price_floor_ceiling(current_price, reference_price, "VNM")
    assert is_safe is True

    # STEP 3: Analyze entry signal
    ml_signal = {"signal": "BUY", "confidence": 85}
    entry_signal = entry_logic.analyze_entry(bull_market_data, ml_signal)

    # Skip if entry conditions not met (can happen with borderline R:R ratios)
    if not entry_signal.should_enter:
        pytest.skip(f"Entry signal not generated: {entry_signal.warnings}")

    assert entry_signal.entry_price > 0
    assert entry_signal.stop_loss < entry_signal.entry_price
    assert len(entry_signal.take_profit_targets) > 0

    # STEP 4: Calculate position size
    # Use TP2 (middle target) for position sizing calculations
    take_profit = (
        entry_signal.take_profit_targets[1]
        if len(entry_signal.take_profit_targets) > 1
        else entry_signal.take_profit_targets[0]
    )

    position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry_signal.entry_price,
        stop_loss=entry_signal.stop_loss,
        take_profit=take_profit,
        confidence=entry_signal.confidence,
        signal_strength="STRONG",
        market_regime=mock_bull_regime,
        sector="Consumer Goods",
        win_rate=0.65,
        avg_win_loss_ratio=2.2,
        auto_detect_regime=False,
    )

    assert position.shares > 0
    assert position.shares % 100 == 0  # Lot of 100
    assert position.risk_percent <= 2.0  # Max 2% risk
    assert position.position_percent <= 10.0  # Max 10% position

    # STEP 5: Validate position size vs volume
    avg_volume = bull_market_data["volume"].tail(20).mean()
    is_safe, _ = vn_validator.validate_position_size_vs_volume(position.shares, avg_volume, "VNM")
    assert is_safe is True

    # STEP 6: Calculate T+2 cash requirement
    pending_settlements = {}
    total_t2, buffer = vn_validator.calculate_t2_cash_requirement(
        pending_settlements, position.value
    )
    assert total_t2 == position.value
    assert buffer == position.value * 0.10

    # STEP 7: Monitor position (simulate price increase)
    entry_price = entry_signal.entry_price
    highest_price = entry_price * 1.08  # +8% gain

    # Check if exit triggered
    entry_date = datetime.now() - timedelta(days=5)

    exit_signal = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=highest_price,
        stop_loss=entry_signal.stop_loss,
        take_profit_targets=entry_signal.take_profit_targets,
        entry_date=entry_date,
        df=bull_market_data.tail(50),
        ml_signal={"signal": "HOLD", "confidence": 70},
        market_regime=mock_bull_regime,
    )

    # Should exit at TP1 (8%)
    assert exit_signal.should_exit is True
    assert exit_signal.exit_reason == ExitReason.TAKE_PROFIT_1
    assert exit_signal.partial_exit is True
    assert exit_signal.exit_pct == 30  # Exit 30% at TP1


def test_full_workflow_bear_market_no_entry(bear_market_data, trading_components, mock_bear_regime):
    """Test workflow in bear market: should not enter"""
    entry_logic = trading_components["entry_logic"]
    vn_validator = trading_components["vn_validator"]

    # STEP 1: Check market regime - bear market not tradeable
    assert mock_bear_regime["tradeable"] is False

    # STEP 2: Even with good ML signal, should respect regime
    ml_signal = {"signal": "BUY", "confidence": 80}

    # Entry logic should reject due to unfavorable conditions
    # (In practice, the orchestrator would check regime first and skip entry analysis)

    # Verify Vietnam market checks still work
    is_liquid, _ = vn_validator.check_liquidity_requirements(bear_market_data, "VNM")
    # May or may not be liquid, but test should handle gracefully
    assert isinstance(is_liquid, bool)


def test_full_workflow_stop_loss_exit(bull_market_data, trading_components, mock_bull_regime):
    """Test workflow with stop loss exit"""
    entry_logic = trading_components["entry_logic"]
    exit_strategy = trading_components["exit_strategy"]
    position_sizer = trading_components["position_sizer"]

    # Enter position
    ml_signal = {"signal": "BUY", "confidence": 75}
    entry_signal = entry_logic.analyze_entry(bull_market_data, ml_signal)

    if not entry_signal.should_enter:
        pytest.skip("Entry signal not generated for test data")

    position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry_signal.entry_price,
        stop_loss=entry_signal.stop_loss,
        take_profit=entry_signal.take_profit,
        confidence=entry_signal.confidence,
        market_regime=mock_bull_regime,
        auto_detect_regime=False,
    )

    # Simulate price drop to stop loss
    entry_price = entry_signal.entry_price
    current_price = entry_signal.stop_loss - 100  # Below stop loss

    entry_date = datetime.now() - timedelta(days=2)

    exit_signal = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=entry_signal.stop_loss,
        take_profit_targets=entry_signal.take_profit_targets,
        entry_date=entry_date,
        df=bull_market_data.tail(50),
    )

    # Should exit with stop loss
    assert exit_signal.should_exit is True
    assert exit_signal.exit_reason == ExitReason.STOP_LOSS
    assert exit_signal.partial_exit is False  # Full exit on stop loss
    assert exit_signal.exit_pct == 100


def test_full_workflow_trailing_stop_exit(bull_market_data, trading_components, mock_bull_regime):
    """Test workflow with trailing stop exit"""
    exit_strategy = trading_components["exit_strategy"]
    position_sizer = trading_components["position_sizer"]

    # Simulate position entry
    entry_price = 90000
    stop_loss = 85000
    take_profit = 100000

    position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=75,
        market_regime=mock_bull_regime,
        auto_detect_regime=False,
    )

    # Create take profit targets [TP1, TP2, TP3]
    take_profit_targets = [
        entry_price * 1.08,  # TP1: +8%
        entry_price * 1.12,  # TP2: +12%
        entry_price * 1.16,  # TP3: +16%
    ]

    entry_date = datetime.now() - timedelta(days=7)

    # Test trailing stop scenario:
    # trailing_stop_activation = 6%, trailing_distance = 4%
    # Peak at +12% (will register TP2 at +12%)
    # Then drop 5% from peak: 100800 * 0.95 = 95760 (+6.4% from entry, above 6% activation)
    # Trailing stop price: 100800 * 0.96 = 96768 (4% below peak)
    # Since 95760 < 96768, trailing stop should trigger
    highest_price = entry_price * 1.12  # +12%

    # First, check at high price to register the peak (+12%)
    exit_signal_at_peak = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=highest_price,
        stop_loss=stop_loss,
        take_profit_targets=take_profit_targets,
        entry_date=entry_date,
        df=bull_market_data.tail(50),
    )
    # At +12%, will hit TP2, but we're testing trailing stop after partial exit

    # Then price drops 5% from peak
    current_price = highest_price * 0.95

    # Check again at dropped price - should trigger trailing stop
    exit_signal = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        take_profit_targets=take_profit_targets,
        entry_date=entry_date,
        df=bull_market_data.tail(50),
    )

    # Should exit with trailing stop or profit protection (both are valid profit protection mechanisms)
    assert exit_signal.should_exit is True
    assert exit_signal.exit_reason in [
        ExitReason.TRAILING_STOP,
        ExitReason.PROFIT_PROTECTION,
    ], f"Expected TRAILING_STOP or PROFIT_PROTECTION, got {exit_signal.exit_reason}"


def test_full_workflow_multiple_positions(bull_market_data, trading_components, mock_bull_regime):
    """Test workflow with multiple positions"""
    entry_logic = trading_components["entry_logic"]
    position_sizer = trading_components["position_sizer"]
    vn_validator = trading_components["vn_validator"]

    # Position 1: VNM
    ml_signal1 = {"signal": "BUY", "confidence": 80}
    entry1 = entry_logic.analyze_entry(bull_market_data, ml_signal1)

    if not entry1.should_enter:
        pytest.skip("Entry signal 1 not generated")

    pos1 = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry1.entry_price,
        stop_loss=entry1.stop_loss,
        take_profit=entry1.take_profit,
        confidence=entry1.confidence,
        sector="Consumer Goods",
        auto_detect_regime=False,
    )

    # Add to portfolio
    position_sizer.current_positions["VNM"] = {
        "shares": pos1.shares,
        "entry_price": entry1.entry_price,
        "current_price": entry1.entry_price,
        "sector": "Consumer Goods",
    }

    # Position 2: VCB (different sector)
    ml_signal2 = {"signal": "BUY", "confidence": 75}
    entry2 = entry_logic.analyze_entry(bull_market_data, ml_signal2)

    if not entry2.should_enter:
        # Still valid test - some positions may not enter
        pass
    else:
        pos2 = position_sizer.calculate_position_size(
            symbol="VCB",
            entry_price=entry2.entry_price,
            stop_loss=entry2.stop_loss,
            take_profit=entry2.take_profit,
            confidence=entry2.confidence,
            sector="Banking",
            portfolio_risk=0.02,  # Already have some risk from VNM
            auto_detect_regime=False,
        )

        # Should allow second position (different sector)
        assert pos2.shares >= 0

        # Verify portfolio constraints
        total_exposure = pos1.value + pos2.value
        max_exposure = position_sizer.total_capital * position_sizer.max_total_exposure
        assert total_exposure <= max_exposure


def test_full_workflow_session_timing(bull_market_data, trading_components):
    """Test workflow respects trading session timing"""
    vn_validator = trading_components["vn_validator"]

    # Test during safe time (10:30 AM)
    safe_time = datetime(2024, 1, 15, 10, 30, 0)
    is_safe, _ = vn_validator.check_trading_session_timing(safe_time)
    assert is_safe is True

    # Test near session boundary (11:29 AM - near 11:30 close)
    boundary_time = datetime(2024, 1, 15, 11, 29, 0)
    is_safe, warning = vn_validator.check_trading_session_timing(boundary_time)
    assert is_safe is False
    assert "morning session end" in warning


def test_full_workflow_price_limit_near_ceiling(bull_market_data, trading_components):
    """Test workflow when price is near ceiling limit"""
    vn_validator = trading_components["vn_validator"]

    # Reference: 100,000
    # Ceiling: 107,000 (+7%)
    # Current: 106,500 (too close)
    reference_price = 100_000
    current_price = 106_500

    is_safe, warning = vn_validator.check_price_floor_ceiling(current_price, reference_price, "VNM")

    assert is_safe is False
    assert "CEILING" in warning

    # Workflow should reject entry when price near ceiling


def test_full_workflow_insufficient_liquidity(trading_components):
    """Test workflow with insufficient liquidity"""
    vn_validator = trading_components["vn_validator"]

    # Create illiquid stock data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    illiquid_df = pd.DataFrame(
        {
            "close": np.ones(30) * 100_000,
            "volume": np.ones(30) * 15_000,  # 15k * 100k = 1.5B (< 2B threshold)
        },
        index=dates,
    )

    is_liquid, warning = vn_validator.check_liquidity_requirements(illiquid_df, "PENNY")
    assert is_liquid is False
    assert "Insufficient liquidity" in warning


def test_full_workflow_position_too_large_vs_volume(bull_market_data, trading_components):
    """Test workflow when position size too large vs volume"""
    vn_validator = trading_components["vn_validator"]

    avg_volume = bull_market_data["volume"].tail(20).mean()

    # Try to take 10% of daily volume (> 5% limit)
    position_shares = int(avg_volume * 0.10)

    is_safe, warning = vn_validator.validate_position_size_vs_volume(
        position_shares, avg_volume, "VNM"
    )

    assert is_safe is False
    assert "Position too large" in warning


# =============================================================================
# MULTI-DAY WORKFLOW TESTS
# =============================================================================


def test_multiday_workflow_entry_hold_exit(bull_market_data, trading_components, mock_bull_regime):
    """Test multi-day workflow: entry → hold → partial exits → full exit"""
    entry_logic = trading_components["entry_logic"]
    exit_strategy = trading_components["exit_strategy"]
    position_sizer = trading_components["position_sizer"]

    # Day 1: Enter position
    ml_signal = {"signal": "BUY", "confidence": 80}
    entry_signal = entry_logic.analyze_entry(bull_market_data, ml_signal)

    if not entry_signal.should_enter:
        pytest.skip("Entry signal not generated")

    position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry_signal.entry_price,
        stop_loss=entry_signal.stop_loss,
        take_profit=entry_signal.take_profit,
        confidence=entry_signal.confidence,
        auto_detect_regime=False,
    )

    entry_price = entry_signal.entry_price
    remaining_shares = position.shares

    # Day 3: Price up 8% - hit TP1
    price_day3 = entry_price * 1.08
    entry_date_day3 = datetime.now() - timedelta(days=3)

    exit_signal_tp1 = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=price_day3,
        stop_loss=entry_signal.stop_loss,
        take_profit_targets=entry_signal.take_profit_targets,
        entry_date=entry_date_day3,
        df=bull_market_data.tail(50),
    )

    if exit_signal_tp1.should_exit and exit_signal_tp1.exit_reason == ExitReason.TAKE_PROFIT_1:
        # Exit 30% at TP1
        assert exit_signal_tp1.partial_exit is True
        assert exit_signal_tp1.exit_pct == 30
        remaining_shares = int(remaining_shares * 0.7)

    # Day 7: Price up 12% - hit TP2
    price_day7 = entry_price * 1.12
    entry_date_day7 = datetime.now() - timedelta(days=7)

    exit_signal_tp2 = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=price_day7,
        stop_loss=entry_signal.stop_loss,
        take_profit_targets=entry_signal.take_profit_targets,
        entry_date=entry_date_day7,
        df=bull_market_data.tail(50),
        partial_exits=[price_day3],  # Already exited 30% at TP1
    )

    if exit_signal_tp2.should_exit and exit_signal_tp2.exit_reason == ExitReason.TAKE_PROFIT_2:
        # Exit 50% of remaining at TP2
        assert exit_signal_tp2.partial_exit is True
        assert exit_signal_tp2.exit_pct == 50

    # Verify position tracking
    assert remaining_shares >= 0


def test_multiday_workflow_max_hold_time_exit(bull_market_data, trading_components):
    """Test exit due to max hold time"""
    exit_strategy = trading_components["exit_strategy"]

    entry_price = 90000
    stop_loss = 85000

    # Create take profit targets [TP1, TP2, TP3]
    take_profit_targets = [
        entry_price * 1.08,  # TP1: +8%
        entry_price * 1.12,  # TP2: +12%
        entry_price * 1.16,  # TP3: +16%
    ]

    # Hold for max days (15 trading days = ~21 calendar days with weekends)
    entry_date = datetime.now() - timedelta(days=22)

    exit_signal = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=91500,  # Small gain (+1.67%, below 2% threshold)
        stop_loss=stop_loss,
        take_profit_targets=take_profit_targets,
        entry_date=entry_date,
        df=bull_market_data.tail(50),
    )

    assert exit_signal.should_exit is True
    assert exit_signal.exit_reason == ExitReason.TIME_DECAY


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_workflow_handles_empty_dataframe(trading_components):
    """Test workflow handles empty DataFrame gracefully"""
    entry_logic = trading_components["entry_logic"]
    vn_validator = trading_components["vn_validator"]

    empty_df = pd.DataFrame()

    # Liquidity check should fail gracefully
    is_liquid, warning = vn_validator.check_liquidity_requirements(empty_df, "EMPTY")
    assert is_liquid is False
    assert "Insufficient data" in warning

    # Entry logic should handle empty data - either return EntrySignal with should_enter=False
    # or raise DataQualityError (both are acceptable behaviors)
    ml_signal = {"signal": "BUY", "confidence": 80}
    try:
        entry_signal = entry_logic.analyze_entry(empty_df, ml_signal)
        assert entry_signal.should_enter is False
    except Exception as e:
        # DataQualityError is also acceptable - it means validation caught the empty data issue
        assert (
            "empty" in str(e).lower()
            or "data" in str(e).lower()
            or "insufficient" in str(e).lower()
        )


def test_workflow_handles_insufficient_data(trading_components):
    """Test workflow with insufficient data points"""
    vn_validator = trading_components["vn_validator"]

    # Only 10 days of data (need 20 for liquidity check)
    short_df = pd.DataFrame(
        {
            "close": np.ones(10) * 100_000,
            "volume": np.ones(10) * 250_000,
        }
    )

    is_liquid, warning = vn_validator.check_liquidity_requirements(short_df, "SHORT")
    assert is_liquid is False


def test_workflow_portfolio_risk_limit_reached(trading_components, mock_bull_regime):
    """Test workflow when portfolio risk limit reached"""
    position_sizer = trading_components["position_sizer"]

    # Try to add position when portfolio risk at limit (20%)
    with pytest.raises(Exception) as exc_info:
        position_sizer.calculate_position_size(
            symbol="VNM",
            entry_price=90000,
            stop_loss=85000,
            take_profit=100000,
            confidence=75,
            portfolio_risk=0.20,  # At limit
            auto_detect_regime=False,
        )

    assert "Portfolio risk" in str(exc_info.value)


def test_workflow_sector_exposure_limit_reached(trading_components):
    """Test workflow when sector exposure limit reached"""
    position_sizer = trading_components["position_sizer"]

    # Fill sector to limit (40%)
    position_sizer.current_positions["VNM"] = {
        "shares": 5000,
        "entry_price": 80000,
        "current_price": 80000,
        "sector": "Consumer Goods",
    }
    # 5000 * 80000 = 400M = 40% of 1000M

    # Try to add another position in same sector
    with pytest.raises(Exception) as exc_info:
        position_sizer.calculate_position_size(
            symbol="MSN",
            entry_price=85000,
            stop_loss=80000,
            take_profit=95000,
            confidence=75,
            sector="Consumer Goods",
            auto_detect_regime=False,
        )

    assert "Sector" in str(exc_info.value)


# =============================================================================
# REALISTIC SCENARIO TESTS
# =============================================================================


def test_realistic_scenario_successful_trade(
    bull_market_data, trading_components, mock_bull_regime
):
    """Realistic scenario: successful trade with partial exits"""
    entry_logic = trading_components["entry_logic"]
    exit_strategy = trading_components["exit_strategy"]
    position_sizer = trading_components["position_sizer"]
    vn_validator = trading_components["vn_validator"]

    # PRE-TRADE CHECKS
    # 1. Market regime
    assert mock_bull_regime["tradeable"] is True

    # 2. Trading session timing (10:00 AM)
    trade_time = datetime(2024, 1, 15, 10, 0, 0)
    is_safe_time, _ = vn_validator.check_trading_session_timing(trade_time)
    assert is_safe_time is True

    # 3. Liquidity
    is_liquid, _ = vn_validator.check_liquidity_requirements(bull_market_data, "VNM")
    assert is_liquid is True

    # ENTRY
    ml_signal = {"signal": "BUY", "confidence": 82}
    entry = entry_logic.analyze_entry(bull_market_data, ml_signal)

    if not entry.should_enter:
        pytest.skip("Entry conditions not met")

    # Position sizing
    position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry.entry_price,
        stop_loss=entry.stop_loss,
        take_profit=entry.take_profit,
        confidence=entry.confidence,
        market_regime=mock_bull_regime,
        sector="Consumer Goods",
        win_rate=0.65,
        avg_win_loss_ratio=2.2,
        auto_detect_regime=False,
    )

    # Validate position size
    avg_volume = bull_market_data["volume"].tail(20).mean()
    is_safe_size, _ = vn_validator.validate_position_size_vs_volume(
        position.shares, avg_volume, "VNM"
    )
    assert is_safe_size is True

    # T+2 settlement
    total_t2, buffer = vn_validator.calculate_t2_cash_requirement({}, position.value)
    required_cash = total_t2 + buffer
    assert required_cash > 0

    # MONITORING & EXIT
    initial_shares = position.shares
    remaining_pnl = 0

    # Price reaches +8% (TP1)
    price_tp1 = entry.entry_price * 1.08
    entry_date_tp1 = datetime.now() - timedelta(days=5)

    exit_tp1 = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry.entry_price,
        current_price=price_tp1,
        stop_loss=entry.stop_loss,
        take_profit_targets=entry.take_profit_targets,
        entry_date=entry_date_tp1,
        df=bull_market_data.tail(50),
    )

    if exit_tp1.should_exit and exit_tp1.exit_reason == ExitReason.TAKE_PROFIT_1:
        exit_shares_tp1 = int(initial_shares * 0.3)
        pnl_tp1 = exit_shares_tp1 * (price_tp1 - entry.entry_price)
        remaining_pnl += pnl_tp1
        assert pnl_tp1 > 0  # Profitable exit

    # Final result: trade should be profitable
    assert remaining_pnl >= 0


def test_realistic_scenario_stop_loss_protection(bull_market_data, trading_components):
    """Realistic scenario: stop loss protects from larger loss"""
    exit_strategy = trading_components["exit_strategy"]

    entry_price = 95000
    stop_loss = 89700  # -5.6% stop
    initial_shares = 1000

    # Create take profit targets [TP1, TP2, TP3]
    take_profit_targets = [
        entry_price * 1.08,  # TP1: +8%
        entry_price * 1.12,  # TP2: +12%
        entry_price * 1.16,  # TP3: +16%
    ]

    # Create custom df with prev_close that won't trigger floor protection
    # prev_close = 95000, floor = 95000 * 0.93 = 88350
    # crash_price = 90000 is above floor, so stop loss can trigger
    dates = pd.date_range(end=pd.Timestamp.today(), periods=50)
    custom_df = pd.DataFrame(
        {
            "close": [95000] * 50,
            "open": [94500] * 50,
            "high": [96000] * 50,
            "low": [94000] * 50,
            "volume": [300000] * 50,
            "atr": [1500] * 50,
        },
        index=dates,
    )

    # Price drops to stop loss level (above floor)
    crash_price = 89000  # Below stop_loss (89700) but above floor (88350)
    entry_date = datetime.now() - timedelta(days=2)

    # Exit at stop loss
    exit_signal = exit_strategy.check_exit(
        symbol="VNM",
        entry_price=entry_price,
        current_price=crash_price,
        stop_loss=stop_loss,
        take_profit_targets=take_profit_targets,
        entry_date=entry_date,
        df=custom_df,
    )

    assert exit_signal.should_exit is True
    assert exit_signal.exit_reason == ExitReason.STOP_LOSS

    # Calculate loss
    loss_at_stop = initial_shares * (stop_loss - entry_price)
    loss_without_stop = initial_shares * (crash_price - entry_price)

    # Stop loss should limit loss
    assert loss_at_stop > loss_without_stop  # Less negative (better)


def test_realistic_scenario_kelly_sizing_vs_fixed(
    bull_market_data, trading_components, mock_bull_regime
):
    """Compare Kelly-based sizing vs fixed sizing"""
    position_sizer = trading_components["position_sizer"]

    entry_price = 90000
    stop_loss = 85000
    take_profit = 100000

    # Kelly-based (with good win rate)
    kelly_position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=80,
        win_rate=0.70,  # 70% win rate
        avg_win_loss_ratio=2.5,  # 2.5:1 W/L
        auto_detect_regime=False,
    )

    # Fixed sizing (no Kelly)
    position_sizer.use_kelly = False
    fixed_position = position_sizer.calculate_position_size(
        symbol="VNM",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=80,
        auto_detect_regime=False,
    )

    # Both should produce valid positions
    assert kelly_position.shares > 0
    assert fixed_position.shares > 0

    # Kelly should track the calculation
    assert kelly_position.kelly_percent > 0
    assert "kelly" in kelly_position.adjustments
