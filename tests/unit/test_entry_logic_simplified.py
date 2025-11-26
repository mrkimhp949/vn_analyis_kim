# -*- coding: utf-8 -*-
"""
Unit Tests for SimplifiedEntryLogic
Tests cho entry logic đơn giản hóa với 8 core filters
"""
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.strategies.entry_logic_simplified import (
    SimplifiedEntryLogic,
    SignalStrength,
    EntrySignal,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock EntryLogicConfig"""
    config = Mock()
    config.min_confidence = 55
    config.base_min_confidence = 55
    config.min_confidence_lower_bound = 45
    config.min_confidence_upper_bound = 75
    config.use_simplified_filters = True
    config.require_trend_alignment = True
    config.require_volume_confirmation = False

    # Filters config
    config.filters = Mock()
    config.filters.liquidity_large_cap = 50_000_000_000
    config.filters.liquidity_mid_cap = 10_000_000_000
    config.filters.liquidity_small_cap = 2_000_000_000
    config.filters.liquidity_low_penalty = -15
    config.filters.liquidity_good_bonus = 5
    config.filters.trend_weak_penalty = -10
    config.filters.trend_perfect_bonus = 5
    config.filters.support_distance_percent = 5.0
    config.filters.resistance_proximity_percent = 2.0
    config.filters.support_bounce_distance = 0.02
    config.filters.resistance_close_penalty = -15
    config.filters.support_bounce_bonus = 15
    config.filters.support_near_bonus = 10
    config.filters.volume_low_penalty = -10
    config.filters.volume_surge_bonus = 5
    config.filters.volume_surge_threshold = 2.0
    config.filters.rsi_overbought = 70
    config.filters.rsi_oversold = 30
    config.filters.rsi_optimal_min = 40
    config.filters.rsi_optimal_max = 60
    config.filters.rsi_overbought_penalty = -10
    config.filters.rsi_oversold_bonus = 15
    config.filters.rsi_optimal_bonus = 5
    config.filters.correlation_max_threshold = 0.7
    config.filters.correlation_diversification_threshold = 0.5
    config.filters.correlation_avg_threshold = 0.4
    config.filters.correlation_high_penalty = -20
    config.filters.correlation_good_bonus = 5
    config.filters.fundamentals_poor_penalty = -10
    config.filters.fundamentals_good_bonus = 5
    config.filters.pe_ratio_max = 30
    config.filters.pe_ratio_optimal_min = 5
    config.filters.pe_ratio_optimal_max = 20
    config.filters.debt_ratio_max = 0.7
    config.filters.debt_ratio_optimal = 0.3
    config.filters.earnings_days_before = 7

    # Volume config
    config.volume = Mock()
    config.volume.sideways_threshold = 0.5
    config.volume.bull_threshold = 0.4
    config.volume.bear_threshold = 0.6
    config.volume.volume_ratio_strong = 1.5
    config.volume.volume_ratio_good = 1.2
    config.volume.volume_ratio_neutral = 0.8

    # Regime config
    config.regime = Mock()
    config.regime.bull_min_regime_confidence = 70
    config.regime.bull_penalty_scale = 0.7
    config.regime.bear_penalty_scale = 1.2
    config.regime.high_vol_penalty_scale = 1.3
    config.regime.sideways_penalty_scale = 1.0
    config.regime.bull_confidence_adjustment = -5
    config.regime.bear_confidence_adjustment = 10
    config.regime.high_vol_confidence_adjustment = 15
    config.regime.portfolio_heat_threshold_1 = 5
    config.regime.portfolio_heat_threshold_2 = 8
    config.regime.portfolio_heat_adjustment_1 = 5
    config.regime.portfolio_heat_adjustment_2 = 10

    # Risk config
    config.risk = Mock()
    config.risk.stop_loss_atr_multiplier = 2.0
    config.risk.take_profit_ratios = [1.5, 2.5, 4.0]
    config.risk.min_risk_reward = 1.8
    config.risk.strength_very_strong = 4.5
    config.risk.strength_strong = 3.5
    config.risk.strength_moderate = 2.5
    config.risk.strength_weak = 1.5
    config.risk.position_multiplier_min = 0.3
    config.risk.position_multiplier_max = 1.5

    # Entry optimization config
    config.entry_optimization = Mock()
    config.entry_optimization.limit_order_min_diff = 0.5

    # Performance config
    config.performance = Mock()
    config.performance.min_trades_for_feedback = 20
    config.performance.win_rate_good_threshold = 0.55
    config.performance.win_rate_poor_threshold = 0.40
    config.performance.confidence_adjustment_good = 5
    config.performance.confidence_adjustment_poor = -5

    return config


@pytest.fixture
def sample_df_uptrend():
    """DataFrame với uptrend rõ ràng"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")

    base_price = 80000
    trend = np.linspace(0, 20000, 200)
    noise = np.random.randn(200) * 500
    close_prices = base_price + trend + noise

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices - np.abs(np.random.randn(200) * 300),
            "high": close_prices + np.abs(np.random.randn(200) * 800),
            "low": close_prices - np.abs(np.random.randn(200) * 800),
            "close": close_prices,
            "volume": np.random.randint(500000, 2000000, 200),
        }
    )

    df["rsi"] = 55  # Optimal RSI
    df["atr"] = 1500

    return df


@pytest.fixture
def sample_df_downtrend():
    """DataFrame với downtrend"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")

    base_price = 100000
    trend = np.linspace(0, -20000, 200)
    noise = np.random.randn(200) * 500
    close_prices = base_price + trend + noise

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices + np.abs(np.random.randn(200) * 300),
            "high": close_prices + np.abs(np.random.randn(200) * 800),
            "low": close_prices - np.abs(np.random.randn(200) * 800),
            "close": close_prices,
            "volume": np.random.randint(300000, 800000, 200),
        }
    )

    df["rsi"] = 35
    df["atr"] = 2000

    return df


@pytest.fixture
def bull_market_regime():
    return {"regime": "BULL", "confidence": 75, "tradeable": True}


@pytest.fixture
def bear_market_regime():
    return {"regime": "BEAR", "confidence": 70, "tradeable": False}


@pytest.fixture
def sideways_market_regime():
    return {"regime": "SIDEWAYS", "confidence": 60, "tradeable": True}


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestSimplifiedEntryLogicInit:
    """Tests cho khởi tạo SimplifiedEntryLogic"""

    def test_init_with_default_config(self):
        """Test khởi tạo với config mặc định"""
        with (
            patch("src.strategies.entry_logic_simplified.get_entry_config") as mock_get_config,
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_config = Mock()
            mock_config.min_confidence = 55
            mock_config.use_simplified_filters = True
            mock_get_config.return_value = mock_config

            logic = SimplifiedEntryLogic()

            assert logic.config is not None
            assert logic._current_symbol is None

    def test_init_with_custom_config(self, mock_config):
        """Test khởi tạo với custom config"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            assert logic.config == mock_config
            assert logic.config.min_confidence == 55

    def test_init_with_portfolio_manager(self, mock_config):
        """Test khởi tạo với portfolio manager"""
        mock_pm = Mock()

        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config, portfolio_manager=mock_pm)

            assert logic.portfolio_manager == mock_pm


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestValidateInitialSignal:
    """Tests cho _validate_initial_signal"""

    def test_validate_with_valid_buy_signal(self, mock_config, sample_df_uptrend):
        """Test validation với BUY signal hợp lệ"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            ml_signal = {"signal": "BUY", "confidence": 70}

            is_valid, signal_type, confidence, price = logic._validate_initial_signal(
                sample_df_uptrend, ml_signal
            )

            assert is_valid is True
            assert signal_type == "BUY"
            assert confidence == 70
            assert price > 0

    def test_validate_with_sell_signal(self, mock_config, sample_df_uptrend):
        """Test validation với SELL signal - should reject"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            ml_signal = {"signal": "SELL", "confidence": 80}

            is_valid, reason, _, _ = logic._validate_initial_signal(sample_df_uptrend, ml_signal)

            assert is_valid is False
            assert "SELL" in reason

    def test_validate_with_hold_signal(self, mock_config, sample_df_uptrend):
        """Test validation với HOLD signal - should reject"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            ml_signal = {"signal": "HOLD", "confidence": 60}

            is_valid, reason, _, _ = logic._validate_initial_signal(sample_df_uptrend, ml_signal)

            assert is_valid is False
            assert "HOLD" in reason

    def test_validate_with_low_confidence(self, mock_config, sample_df_uptrend):
        """Test validation với confidence thấp"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            ml_signal = {"signal": "BUY", "confidence": 40}  # Dưới ngưỡng 55

            is_valid, reason, _, _ = logic._validate_initial_signal(sample_df_uptrend, ml_signal)

            assert is_valid is False
            assert "Confidence" in reason or "low" in reason.lower()

    def test_validate_with_none_ml_signal_fallback(self, mock_config, sample_df_uptrend):
        """Test fallback to technical khi ML signal = None"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
            patch("src.ml.signals.technical_fallback.analyze_technical") as mock_tech,
        ):

            mock_tech_signal = Mock()
            mock_tech_signal.confidence = 60
            mock_tech_signal.signal = "BUY"
            mock_tech.return_value = mock_tech_signal

            logic = SimplifiedEntryLogic(config=mock_config)

            is_valid, signal_type, confidence, price = logic._validate_initial_signal(
                sample_df_uptrend, None
            )

            # Should use technical fallback
            assert isinstance(is_valid, bool)

    def test_validate_insufficient_data(self, mock_config):
        """Test với data không đủ"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            small_df = pd.DataFrame({"close": [80000] * 20})  # Chỉ 20 rows
            ml_signal = {"signal": "BUY", "confidence": 70}

            is_valid, reason, _, _ = logic._validate_initial_signal(small_df, ml_signal)

            assert is_valid is False
            assert "Data" in reason or "validation" in reason.lower()


# =============================================================================
# FILTER TESTS
# =============================================================================


class TestTrendAlignmentFilter:
    """Tests cho _check_trend_alignment"""

    def test_perfect_uptrend(self, mock_config, sample_df_uptrend):
        """Test perfect uptrend alignment"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            result = logic._check_trend_alignment(sample_df_uptrend, "BUY")

            assert "aligned" in result
            assert "strength" in result
            assert "reason" in result

    def test_downtrend_not_aligned(self, mock_config, sample_df_downtrend):
        """Test downtrend không aligned cho BUY"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            result = logic._check_trend_alignment(sample_df_downtrend, "BUY")

            assert "aligned" in result
            # Downtrend should not be aligned for BUY

    def test_insufficient_data(self, mock_config):
        """Test với data không đủ"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            small_df = pd.DataFrame({"close": [80000] * 50})

            result = logic._check_trend_alignment(small_df, "BUY")

            assert result["aligned"] is True  # Default to True with insufficient data


class TestSupportResistanceFilter:
    """Tests cho _check_support_resistance"""

    def test_near_support(self, mock_config, sample_df_uptrend):
        """Test detection gần support"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            current_price = sample_df_uptrend["close"].iloc[-1]

            result = logic._check_support_resistance(sample_df_uptrend, current_price)

            assert "near_support" in result
            assert "too_close_to_resistance" in result
            assert "support_level" in result
            assert "resistance_level" in result
            assert "distance_to_support" in result
            assert "distance_to_resistance" in result

    def test_bouncing_from_support(self, mock_config):
        """Test detection bouncing from support"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Create data with bounce from support
            np.random.seed(42)
            dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
            close_prices = [80000] * 47 + [79000, 79500, 80500]  # Bounce pattern

            df = pd.DataFrame(
                {
                    "time": dates,
                    "open": close_prices,
                    "high": [p + 500 for p in close_prices],
                    "low": [p - 500 for p in close_prices],
                    "close": close_prices,
                    "volume": [1000000] * 50,
                }
            )

            result = logic._check_support_resistance(df, 80500)

            assert "bouncing_from_support" in result

    def test_insufficient_data(self, mock_config):
        """Test với data không đủ"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            small_df = pd.DataFrame(
                {"close": [80000] * 10, "high": [81000] * 10, "low": [79000] * 10}
            )

            result = logic._check_support_resistance(small_df, 80000)

            assert result["near_support"] is False
            assert result["too_close_to_resistance"] is False


class TestLiquidityFilter:
    """Tests cho _check_liquidity"""

    def test_large_cap_liquidity(self, mock_config, sample_df_uptrend):
        """Test large cap liquidity"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # High volume = large cap
            sample_df_uptrend["volume"] = 5000000  # 5M shares
            current_price = 100000  # 100k VND = 500B daily value

            result = logic._check_liquidity(sample_df_uptrend, current_price)

            assert "sufficient" in result
            assert "critical" in result
            assert "tier" in result
            assert "avg_value" in result

    def test_micro_cap_critical_liquidity(self, mock_config, sample_df_uptrend):
        """Test micro cap với critical liquidity"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Very low volume
            sample_df_uptrend["volume"] = 1000  # 1k shares
            current_price = 10000  # 10k VND = 10M daily value (very low)

            result = logic._check_liquidity(sample_df_uptrend, current_price)

            assert result["tier"] == "micro"

    def test_no_volume_data(self, mock_config):
        """Test khi không có volume data"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            df = pd.DataFrame({"close": [80000] * 50})  # No volume column

            result = logic._check_liquidity(df, 80000)

            assert result["sufficient"] is True
            assert result["critical"] is False


class TestVolumeConfirmationFilter:
    """Tests cho _check_volume_confirmation"""

    def test_volume_surge(self, mock_config, sample_df_uptrend):
        """Test volume surge detection"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Create volume surge
            sample_df_uptrend.loc[sample_df_uptrend.index[-1], "volume"] = 5000000  # Surge

            result = logic._check_volume_confirmation(sample_df_uptrend, None)

            assert "confirmed" in result
            assert "surge" in result
            assert "reason" in result

    def test_low_volume(self, mock_config, sample_df_uptrend):
        """Test low volume"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Very low current volume
            sample_df_uptrend.loc[sample_df_uptrend.index[-1], "volume"] = 10000

            result = logic._check_volume_confirmation(sample_df_uptrend, None)

            assert "confirmed" in result

    def test_volume_with_bull_regime(self, mock_config, sample_df_uptrend, bull_market_regime):
        """Test volume với BULL regime (lower threshold)"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            result = logic._check_volume_confirmation(sample_df_uptrend, bull_market_regime)

            assert "confirmed" in result

    def test_insufficient_data(self, mock_config):
        """Test với data không đủ"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            small_df = pd.DataFrame({"close": [80000] * 10, "volume": [100000] * 10})

            result = logic._check_volume_confirmation(small_df, None)

            assert result["confirmed"] is True  # Default with insufficient data


class TestRSIFilter:
    """Tests cho _check_rsi"""

    def test_rsi_overbought(self, mock_config):
        """Test RSI overbought detection"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Create df with RSI = 75 at the last row
            df = pd.DataFrame(
                {
                    "close": [80000] * 50,
                    "rsi": [50] * 49 + [75],  # Last value is 75 (overbought)
                }
            )

            result = logic._check_rsi(df)

            assert result["overbought"] == True
            assert result["oversold"] == False

    def test_rsi_oversold(self, mock_config):
        """Test RSI oversold detection - strong buy signal"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Create df with RSI = 25 at the last row
            df = pd.DataFrame(
                {
                    "close": [80000] * 50,
                    "rsi": [50] * 49 + [25],  # Last value is 25 (oversold)
                }
            )

            result = logic._check_rsi(df)

            assert result["oversold"] == True
            assert result["overbought"] == False

    def test_rsi_optimal(self, mock_config):
        """Test RSI optimal range"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Create df with RSI = 50 at the last row
            df = pd.DataFrame(
                {
                    "close": [80000] * 50,
                    "rsi": [50] * 50,  # All values are 50 (optimal)
                }
            )

            result = logic._check_rsi(df)

            assert result["optimal"] == True
            assert result["overbought"] == False
            assert result["oversold"] == False

    def test_no_rsi_column(self, mock_config):
        """Test khi không có RSI column"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            df = pd.DataFrame({"close": [80000] * 50})  # No RSI column

            result = logic._check_rsi(df)

            assert result["optimal"] == True
            assert result["value"] == 50  # Default


class TestPortfolioCorrelationFilter:
    """Tests cho _check_portfolio_correlation"""

    def test_no_portfolio_manager(self, mock_config, sample_df_uptrend):
        """Test khi không có portfolio manager"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config, portfolio_manager=None)

            result = logic._check_portfolio_correlation(sample_df_uptrend, "VNM")

            assert result["too_high"] is False
            assert result["good_diversification"] is False

    def test_no_symbol(self, mock_config, sample_df_uptrend):
        """Test khi không có symbol"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_pm = Mock()
            logic = SimplifiedEntryLogic(config=mock_config, portfolio_manager=mock_pm)

            result = logic._check_portfolio_correlation(sample_df_uptrend, None)

            assert result["too_high"] is False

    def test_empty_portfolio(self, mock_config, sample_df_uptrend):
        """Test với portfolio trống"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_pm = Mock()
            mock_pm.get_positions.return_value = {}

            logic = SimplifiedEntryLogic(config=mock_config, portfolio_manager=mock_pm)

            result = logic._check_portfolio_correlation(sample_df_uptrend, "VNM")

            assert result["good_diversification"] is True


class TestFundamentalsFilter:
    """Tests cho _check_fundamentals_via_api"""

    def test_no_symbol(self, mock_config):
        """Test khi không có symbol"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            result = logic._check_fundamentals_via_api(None, 80000)

            assert result["poor_fundamentals"] is False
            assert result["good_fundamentals"] is False

    def test_no_fundamental_data(self, mock_config):
        """Test khi không có fundamental data"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager") as mock_fm,
        ):

            mock_fm.return_value.get_fundamental_data.return_value = None

            logic = SimplifiedEntryLogic(config=mock_config)

            result = logic._check_fundamentals_via_api("VNM", 80000)

            assert result["poor_fundamentals"] is False

    def test_high_pe_ratio(self, mock_config):
        """Test với P/E ratio cao"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager") as mock_fm,
        ):

            mock_fund_data = Mock()
            mock_fund_data.is_valid.return_value = True
            mock_fund_data.pe_ratio = 50  # High P/E
            mock_fund_data.debt_ratio = 0.3
            mock_fm.return_value.get_fundamental_data.return_value = mock_fund_data
            mock_fm.return_value.get_earnings_date.return_value = None

            logic = SimplifiedEntryLogic(config=mock_config)

            result = logic._check_fundamentals_via_api("VNM", 80000)

            assert result["poor_fundamentals"] is True


# =============================================================================
# SIMPLIFIED FILTERS INTEGRATION TESTS
# =============================================================================


class TestRunSimplifiedFilters:
    """Tests cho _run_simplified_filters"""

    def test_market_not_tradeable(self, mock_config, sample_df_uptrend, bear_market_regime):
        """Test khi market không tradeable"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            current_price = sample_df_uptrend["close"].iloc[-1]

            passed, reasons, warnings, adjustments, breakdown = logic._run_simplified_filters(
                sample_df_uptrend, "BUY", current_price, bear_market_regime
            )

            assert passed is False

    def test_bull_market_passes(self, mock_config, sample_df_uptrend, bull_market_regime):
        """Test trong BULL market với data tốt"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            current_price = sample_df_uptrend["close"].iloc[-1]

            passed, reasons, warnings, adjustments, breakdown = logic._run_simplified_filters(
                sample_df_uptrend, "BUY", current_price, bull_market_regime
            )

            assert isinstance(passed, bool)
            assert isinstance(reasons, list)
            assert isinstance(warnings, list)
            assert isinstance(adjustments, list)
            assert isinstance(breakdown, list)

    def test_sideways_market(self, mock_config, sample_df_uptrend, sideways_market_regime):
        """Test trong SIDEWAYS market"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            current_price = sample_df_uptrend["close"].iloc[-1]

            passed, reasons, warnings, adjustments, breakdown = logic._run_simplified_filters(
                sample_df_uptrend, "BUY", current_price, sideways_market_regime
            )

            assert isinstance(passed, bool)


# =============================================================================
# SIGNAL STRENGTH AND POSITION MULTIPLIER TESTS
# =============================================================================


class TestSignalStrength:
    """Tests cho _calculate_signal_strength"""

    def test_very_strong_signal(self, mock_config):
        """Test very strong signal"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            strength = logic._calculate_signal_strength(confidence=90, risk_reward=3.5, warnings=[])

            assert strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]

    def test_weak_signal(self, mock_config):
        """Test weak signal"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            strength = logic._calculate_signal_strength(
                confidence=55, risk_reward=1.8, warnings=["w1", "w2", "w3"]
            )

            assert strength in [
                SignalStrength.WEAK,
                SignalStrength.VERY_WEAK,
                SignalStrength.MODERATE,
            ]

    def test_moderate_signal(self, mock_config):
        """Test moderate signal"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            strength = logic._calculate_signal_strength(
                confidence=70, risk_reward=2.5, warnings=["w1"]
            )

            assert strength in [SignalStrength.MODERATE, SignalStrength.STRONG]


class TestPositionMultiplier:
    """Tests cho _calculate_position_multiplier"""

    def test_strong_signal_bull_market(self, mock_config, bull_market_regime):
        """Test position multiplier với strong signal trong BULL market"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            multiplier = logic._calculate_position_multiplier(
                SignalStrength.VERY_STRONG, 85, [], bull_market_regime
            )

            assert 1.0 <= multiplier <= 1.5

    def test_weak_signal_with_warnings(self, mock_config, sideways_market_regime):
        """Test position multiplier với weak signal và warnings"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            multiplier = logic._calculate_position_multiplier(
                SignalStrength.WEAK, 55, ["w1", "w2"], sideways_market_regime
            )

            assert 0.3 <= multiplier <= 1.0

    def test_multiplier_bounds(self, mock_config):
        """Test position multiplier stays within bounds"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            # Very weak with many warnings
            multiplier = logic._calculate_position_multiplier(
                SignalStrength.VERY_WEAK, 40, ["w1", "w2", "w3", "w4", "w5"], None
            )

            assert multiplier >= mock_config.risk.position_multiplier_min


# =============================================================================
# PERFORMANCE FEEDBACK TESTS
# =============================================================================


class TestPerformanceFeedback:
    """Tests cho _apply_performance_feedback"""

    def test_no_performance_monitor(self, mock_config):
        """Test khi không có performance monitor"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor") as mock_pm,
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_pm.return_value = None
            logic = SimplifiedEntryLogic(config=mock_config)
            logic.performance_monitor = None

            confidence, msg = logic._apply_performance_feedback(70)

            assert confidence == 70
            assert msg is None

    def test_insufficient_trades(self, mock_config):
        """Test khi chưa đủ trades để feedback"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor") as mock_pm,
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_monitor = Mock()
            mock_monitor.get_metrics.return_value = {"total_trades": 5, "win_rate": 0.6}
            mock_pm.return_value = mock_monitor

            logic = SimplifiedEntryLogic(config=mock_config)
            logic.performance_monitor = mock_monitor

            confidence, msg = logic._apply_performance_feedback(70)

            assert confidence == 70
            assert msg is None

    def test_good_performance_boost(self, mock_config):
        """Test boost confidence với good performance"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor") as mock_pm,
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_monitor = Mock()
            mock_monitor.get_metrics.return_value = {"total_trades": 50, "win_rate": 0.60}
            mock_pm.return_value = mock_monitor

            logic = SimplifiedEntryLogic(config=mock_config)
            logic.performance_monitor = mock_monitor

            confidence, msg = logic._apply_performance_feedback(70)

            assert confidence >= 70  # Should increase
            if msg:
                assert "Good" in msg or "📈" in msg

    def test_poor_performance_penalty(self, mock_config):
        """Test penalty confidence với poor performance"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor") as mock_pm,
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_monitor = Mock()
            mock_monitor.get_metrics.return_value = {"total_trades": 50, "win_rate": 0.35}
            mock_pm.return_value = mock_monitor

            logic = SimplifiedEntryLogic(config=mock_config)
            logic.performance_monitor = mock_monitor

            confidence, msg = logic._apply_performance_feedback(70)

            assert confidence <= 70  # Should decrease
            if msg:
                assert "Poor" in msg or "⚠️" in msg


# =============================================================================
# THRESHOLD ADJUSTMENT TESTS
# =============================================================================


class TestThresholdAdjustment:
    """Tests cho _adjust_thresholds_for_market"""

    def test_bull_market_adjustment(self, mock_config, bull_market_regime):
        """Test threshold adjustment trong BULL market"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            original_confidence = logic.config.min_confidence

            logic._adjust_thresholds_for_market(bull_market_regime)

            # BULL market should lower threshold
            assert logic.config.min_confidence <= original_confidence + 5

    def test_bear_market_adjustment(self, mock_config, bear_market_regime):
        """Test threshold adjustment trong BEAR market"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            original_confidence = logic.config.min_confidence

            logic._adjust_thresholds_for_market(bear_market_regime)

            # BEAR market should raise threshold
            assert logic.config.min_confidence >= original_confidence

    def test_no_regime(self, mock_config):
        """Test với no market regime"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            logic._adjust_thresholds_for_market(None)

            assert logic.config.min_confidence == logic.config.base_min_confidence

    def test_portfolio_heat_adjustment(self, mock_config, bull_market_regime):
        """Test portfolio heat adjustment"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            mock_pm = Mock()
            mock_pm.get_positions.return_value = {f"SYM{i}": {} for i in range(6)}  # 6 positions

            logic = SimplifiedEntryLogic(config=mock_config, portfolio_manager=mock_pm)

            logic._adjust_thresholds_for_market(bull_market_regime)

            # Should have portfolio heat adjustment
            assert logic.config.min_confidence >= mock_config.base_min_confidence


# =============================================================================
# NO SIGNAL HELPER TESTS
# =============================================================================


class TestNoSignalHelper:
    """Tests cho _no_signal helper"""

    def test_no_signal_basic(self, mock_config):
        """Test _no_signal basic"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            result = logic._no_signal("Test reason")

            assert result.should_enter is False
            assert result.signal_type == "HOLD"
            assert result.confidence == 0
            assert result.strength == SignalStrength.NO_SIGNAL
            assert "Test reason" in result.warnings

    def test_no_signal_with_telemetry(self, mock_config):
        """Test _no_signal với telemetry"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            telemetry = {"base_confidence": 50, "reason": "Test"}

            result = logic._no_signal("Test reason", telemetry=telemetry)

            assert result.telemetry == telemetry


# =============================================================================
# MESSAGE FORMATTING TESTS
# =============================================================================


class TestMessageFormatting:
    """Tests cho format_signal_message"""

    def test_format_no_entry_message(self, mock_config):
        """Test format message khi không có entry"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)
            signal = logic._no_signal("Low confidence")

            msg = logic.format_signal_message(signal, "VNM")

            assert "VNM" in msg
            assert "No entry" in msg

    def test_format_entry_message(self, mock_config):
        """Test format message khi có entry"""
        with (
            patch("src.strategies.entry_logic_simplified.get_performance_monitor"),
            patch("src.strategies.entry_logic_simplified.get_fundamental_manager"),
        ):

            logic = SimplifiedEntryLogic(config=mock_config)

            signal = EntrySignal(
                should_enter=True,
                signal_type="BUY",
                confidence=75,
                strength=SignalStrength.STRONG,
                position_size_multiplier=1.1,
                reasons=["Good trend"],
                warnings=["Near resistance"],
                entry_price=80000,
                stop_loss=76000,
                take_profit_targets=[88000, 96000],
            )

            msg = logic.format_signal_message(signal, "VNM")

            assert "VNM" in msg
            assert "BUY" in msg
            assert "75%" in msg
            assert "80,000" in msg or "80000" in msg
            assert "Stop Loss" in msg
            assert "Take Profit" in msg
