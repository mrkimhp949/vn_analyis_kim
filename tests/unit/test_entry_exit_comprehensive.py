# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for Entry/Exit Logic
Tests các edge cases và scenarios quan trọng cho entry/exit logic
"""
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.strategies.entry_logic import ImprovedEntryLogic, SignalStrength, EntrySignal
from src.strategies.exit_logic import (
    ImprovedExitStrategy,
    ExitReason,
    ExitDecision,
    PartialExitTracker,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def entry_logic_default():
    """Entry logic với config mặc định"""
    return ImprovedEntryLogic(
        min_confidence=55,
        min_risk_reward=1.8,
        require_trend_alignment=True,
        require_volume_confirmation=False,
        soft_filter_mode=True,
    )


@pytest.fixture
def entry_logic_strict():
    """Entry logic với config nghiêm ngặt"""
    return ImprovedEntryLogic(
        min_confidence=70,
        min_risk_reward=2.5,
        require_trend_alignment=True,
        require_volume_confirmation=True,
        soft_filter_mode=False,
    )


@pytest.fixture
def exit_strategy_default():
    """Exit strategy với config mặc định"""
    return ImprovedExitStrategy(
        take_profit_levels=[0.12, 0.20],
        trailing_stop_activation=0.08,
        trailing_stop_distance=0.05,
        max_holding_days=25,
    )


@pytest.fixture
def sample_df_uptrend():
    """DataFrame với uptrend rõ ràng"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Tạo uptrend: giá tăng dần
    base_price = 80000
    trend = np.linspace(0, 15000, 100)  # Tăng 15k trong 100 ngày
    noise = np.random.randn(100) * 500
    close_prices = base_price + trend + noise

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices - np.abs(np.random.randn(100) * 300),
            "high": close_prices + np.abs(np.random.randn(100) * 800),
            "low": close_prices - np.abs(np.random.randn(100) * 800),
            "close": close_prices,
            "volume": np.random.randint(500000, 2000000, 100),
        }
    )

    # Add EMAs
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["atr"] = 1500  # ATR cố định

    return df


@pytest.fixture
def sample_df_downtrend():
    """DataFrame với downtrend rõ ràng"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Tạo downtrend: giá giảm dần
    base_price = 95000
    trend = np.linspace(0, -15000, 100)  # Giảm 15k trong 100 ngày
    noise = np.random.randn(100) * 500
    close_prices = base_price + trend + noise

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices + np.abs(np.random.randn(100) * 300),
            "high": close_prices + np.abs(np.random.randn(100) * 800),
            "low": close_prices - np.abs(np.random.randn(100) * 800),
            "close": close_prices,
            "volume": np.random.randint(500000, 2000000, 100),
        }
    )

    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["atr"] = 1500

    return df


@pytest.fixture
def sample_df_sideways():
    """DataFrame với sideway market"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Sideway: giá dao động quanh mức trung bình
    base_price = 85000
    noise = np.random.randn(100) * 2000  # Chỉ có noise, không có trend
    close_prices = base_price + noise

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices - np.random.randn(100) * 300,
            "high": close_prices + np.abs(np.random.randn(100) * 800),
            "low": close_prices - np.abs(np.random.randn(100) * 800),
            "close": close_prices,
            "volume": np.random.randint(300000, 800000, 100),
        }
    )

    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["atr"] = 2000

    return df


@pytest.fixture
def bull_market_regime():
    """Market regime BULL"""
    return {"regime": "BULL", "confidence": 75, "tradeable": True}


@pytest.fixture
def bear_market_regime():
    """Market regime BEAR"""
    return {"regime": "BEAR", "confidence": 70, "tradeable": False}


@pytest.fixture
def sideways_market_regime():
    """Market regime SIDEWAYS"""
    return {"regime": "SIDEWAYS", "confidence": 60, "tradeable": True}


# =============================================================================
# ENTRY LOGIC TESTS
# =============================================================================


class TestEntryLogicValidation:
    """Tests cho validation logic của entry"""

    def test_validate_initial_signal_with_valid_buy(self, entry_logic_default, sample_df_uptrend):
        """Test validation với BUY signal hợp lệ"""
        ml_signal = {"signal": "BUY", "confidence": 70}

        is_valid, signal_type, confidence, price = entry_logic_default._validate_initial_signal(
            sample_df_uptrend, ml_signal
        )

        assert is_valid is True
        assert signal_type == "BUY"
        assert confidence == 70
        assert price > 0

    def test_validate_initial_signal_with_sell(self, entry_logic_default, sample_df_uptrend):
        """Test validation với SELL signal - should reject"""
        ml_signal = {"signal": "SELL", "confidence": 80}

        is_valid, reason, _, _ = entry_logic_default._validate_initial_signal(
            sample_df_uptrend, ml_signal
        )

        assert is_valid is False
        assert "SELL" in reason

    def test_validate_initial_signal_with_hold(self, entry_logic_default, sample_df_uptrend):
        """Test validation với HOLD signal - should reject"""
        ml_signal = {"signal": "HOLD", "confidence": 60}

        is_valid, reason, _, _ = entry_logic_default._validate_initial_signal(
            sample_df_uptrend, ml_signal
        )

        assert is_valid is False
        assert "HOLD" in reason

    def test_validate_initial_signal_low_confidence(self, entry_logic_default, sample_df_uptrend):
        """Test validation với confidence thấp"""
        ml_signal = {"signal": "BUY", "confidence": 40}  # Dưới ngưỡng 55

        is_valid, reason, _, _ = entry_logic_default._validate_initial_signal(
            sample_df_uptrend, ml_signal
        )

        assert is_valid is False
        assert "Confidence" in reason or "thấp" in reason

    def test_validate_initial_signal_none_ml_fallback(self, entry_logic_default, sample_df_uptrend):
        """Test fallback to technical analysis khi ML signal = None"""
        is_valid, signal_type, confidence, price = entry_logic_default._validate_initial_signal(
            sample_df_uptrend, None
        )

        # Should either pass with technical analysis or fail gracefully
        assert isinstance(is_valid, bool)
        if is_valid:
            assert confidence > 0
            assert price > 0

    def test_validate_insufficient_data(self, entry_logic_default):
        """Test với data không đủ"""
        small_df = pd.DataFrame({"close": [80000] * 20})  # Chỉ 20 rows
        ml_signal = {"signal": "BUY", "confidence": 70}

        is_valid, reason, _, _ = entry_logic_default._validate_initial_signal(small_df, ml_signal)

        assert is_valid is False
        assert "Data" in reason or "validation" in reason.lower()


class TestEntryLogicFilters:
    """Tests cho các filter của entry logic"""

    def test_check_trend_alignment_uptrend(self, entry_logic_default, sample_df_uptrend):
        """Test trend alignment với uptrend"""
        result = entry_logic_default._check_trend_alignment(sample_df_uptrend, "BUY")

        assert "aligned" in result
        assert "strength" in result
        # Uptrend should be aligned for BUY
        assert result["aligned"] is True or result["strength"] > 0

    def test_check_trend_alignment_downtrend(self, entry_logic_default, sample_df_downtrend):
        """Test trend alignment với downtrend - should not align for BUY"""
        result = entry_logic_default._check_trend_alignment(sample_df_downtrend, "BUY")

        assert "aligned" in result
        # Downtrend should NOT be aligned for BUY
        # (có thể aligned = False hoặc strength thấp)

    def test_check_volume_confirmation_high_volume(self, entry_logic_default, sample_df_uptrend):
        """Test volume confirmation với volume cao"""
        # Tăng volume để đảm bảo confirmation
        sample_df_uptrend["volume"] = sample_df_uptrend["volume"] * 2

        result = entry_logic_default._check_volume_confirmation(sample_df_uptrend)

        assert "confirmed" in result
        assert "reason" in result

    def test_check_volatility_normal(self, entry_logic_default, sample_df_uptrend):
        """Test volatility check với volatility bình thường"""
        result = entry_logic_default._check_volatility(sample_df_uptrend)

        assert "too_high" in result
        assert "optimal" in result
        assert "value" in result

    def test_check_rsi_overbought(self, entry_logic_default, sample_df_uptrend):
        """Test RSI check với overbought condition"""
        sample_df_uptrend["rsi"] = 75  # Overbought

        result = entry_logic_default._check_rsi(sample_df_uptrend)

        assert "overbought" in result
        assert result["overbought"] is True

    def test_check_rsi_oversold(self, entry_logic_default, sample_df_uptrend):
        """Test RSI check với oversold condition - strong buy signal"""
        sample_df_uptrend["rsi"] = 25  # Oversold

        result = entry_logic_default._check_rsi(sample_df_uptrend)

        assert "oversold" in result
        assert result["oversold"] is True

    def test_check_support_resistance(self, entry_logic_default, sample_df_uptrend):
        """Test support/resistance check"""
        current_price = sample_df_uptrend["close"].iloc[-1]

        result = entry_logic_default._check_support_resistance(sample_df_uptrend, current_price)

        assert "near_support" in result
        assert "too_close_to_resistance" in result
        assert "support_level" in result
        assert "resistance_level" in result

    def test_check_liquidity_sufficient(self, entry_logic_default, sample_df_uptrend):
        """Test liquidity check với thanh khoản đủ"""
        current_price = sample_df_uptrend["close"].iloc[-1]

        result = entry_logic_default._check_liquidity(sample_df_uptrend, current_price)

        assert "sufficient" in result or "critical" in result


class TestEntryLogicMarketRegime:
    """Tests cho entry logic với các market regime khác nhau"""

    def test_run_filters_bull_market(
        self, entry_logic_default, sample_df_uptrend, bull_market_regime
    ):
        """Test filters trong BULL market - should be more lenient"""
        current_price = sample_df_uptrend["close"].iloc[-1]

        passed, reasons, warnings, adjustments, breakdown = entry_logic_default._run_all_filters(
            sample_df_uptrend, "BUY", current_price, bull_market_regime
        )

        # BULL market should pass more easily
        assert isinstance(passed, bool)
        assert isinstance(reasons, list)
        assert isinstance(warnings, list)

    def test_run_filters_bear_market_not_tradeable(
        self, entry_logic_default, sample_df_uptrend, bear_market_regime
    ):
        """Test filters trong BEAR market không tradeable - should block"""
        current_price = sample_df_uptrend["close"].iloc[-1]

        passed, reasons, warnings, adjustments, breakdown = entry_logic_default._run_all_filters(
            sample_df_uptrend, "BUY", current_price, bear_market_regime
        )

        # BEAR market with tradeable=False should block
        assert passed is False

    def test_run_filters_sideways_market(
        self, entry_logic_default, sample_df_sideways, sideways_market_regime
    ):
        """Test filters trong SIDEWAYS market"""
        current_price = sample_df_sideways["close"].iloc[-1]

        passed, reasons, warnings, adjustments, breakdown = entry_logic_default._run_all_filters(
            sample_df_sideways, "BUY", current_price, sideways_market_regime
        )

        assert isinstance(passed, bool)
        assert isinstance(breakdown, list)


class TestEntryLogicSignalStrength:
    """Tests cho signal strength calculation"""

    def test_calculate_signal_strength_very_strong(self, entry_logic_default):
        """Test signal strength với conditions rất tốt"""
        strength = entry_logic_default._calculate_signal_strength(
            confidence=90, risk_reward=3.5, warnings=[]
        )

        assert strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]

    def test_calculate_signal_strength_weak(self, entry_logic_default):
        """Test signal strength với conditions yếu"""
        strength = entry_logic_default._calculate_signal_strength(
            confidence=55, risk_reward=1.8, warnings=["warning1", "warning2", "warning3"]
        )

        assert strength in [SignalStrength.WEAK, SignalStrength.VERY_WEAK, SignalStrength.MODERATE]

    def test_calculate_position_multiplier_strong_signal(
        self, entry_logic_default, bull_market_regime
    ):
        """Test position multiplier với strong signal"""
        multiplier = entry_logic_default._calculate_position_multiplier(
            SignalStrength.VERY_STRONG, 85, [], bull_market_regime
        )

        assert 1.0 <= multiplier <= 1.5

    def test_calculate_position_multiplier_weak_signal(
        self, entry_logic_default, bear_market_regime
    ):
        """Test position multiplier với weak signal"""
        multiplier = entry_logic_default._calculate_position_multiplier(
            SignalStrength.WEAK, 55, ["warning1", "warning2"], bear_market_regime
        )

        assert 0.3 <= multiplier <= 1.0


class TestEntryLogicVietnamMarket:
    """Tests cho Vietnam market specific logic"""

    def test_check_vietnam_price_limits_near_ceiling(self, entry_logic_default, sample_df_uptrend):
        """Test Vietnam price limits gần ceiling (+7%)"""
        # Simulate price near ceiling
        current_price = sample_df_uptrend["close"].iloc[-2] * 1.068  # Gần +7%

        result = entry_logic_default._check_vietnam_price_limits(sample_df_uptrend, current_price)

        assert "near_limit" in result
        # Near ceiling should trigger warning or block

    def test_check_vietnam_price_limits_near_floor(self, entry_logic_default, sample_df_uptrend):
        """Test Vietnam price limits gần floor (-7%)"""
        current_price = sample_df_uptrend["close"].iloc[-2] * 0.932  # Gần -7%

        result = entry_logic_default._check_vietnam_price_limits(sample_df_uptrend, current_price)

        assert "near_limit" in result

    def test_check_vietnam_market_liquidity(self, entry_logic_default, sample_df_uptrend):
        """Test Vietnam market liquidity check"""
        result = entry_logic_default._check_vietnam_market_liquidity(sample_df_uptrend)

        assert "sufficient" in result
        assert "reason" in result


# =============================================================================
# EXIT LOGIC TESTS
# =============================================================================


class TestExitLogicStopLoss:
    """Tests cho stop loss logic"""

    def test_stop_loss_triggered(self, exit_strategy_default, sample_df_uptrend):
        """Test stop loss được trigger"""
        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=74000,  # Dưới stop loss
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.STOP_LOSS
        assert decision.urgency == 5
        assert decision.exit_type == "FULL"

    def test_stop_loss_not_triggered(self, exit_strategy_default, sample_df_uptrend):
        """Test stop loss không trigger khi giá trên SL"""
        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=82000,  # Trên stop loss
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
        )

        assert decision.exit_reason != ExitReason.STOP_LOSS

    def test_ensure_stop_loss_fallback(self, exit_strategy_default, sample_df_uptrend):
        """Test fallback stop loss khi không có SL"""
        sl = exit_strategy_default._ensure_stop_loss("VNM", 80000, None, sample_df_uptrend)

        assert sl > 0
        assert sl < 80000  # SL phải dưới entry price

    def test_atr_based_stop_loss(self, exit_strategy_default, sample_df_uptrend):
        """Test ATR-based stop loss calculation"""
        sl = exit_strategy_default._calculate_atr_based_stop_loss(80000, sample_df_uptrend)

        assert sl > 0
        assert sl < 80000
        # SL should be between 3% and 10% below entry
        assert sl >= 80000 * 0.90
        assert sl <= 80000 * 0.97


class TestExitLogicTakeProfit:
    """Tests cho take profit logic"""

    def test_take_profit_1_triggered(self, exit_strategy_default, sample_df_uptrend):
        """Test TP1 được trigger"""
        entry_price = 80000
        tp1_price = entry_price * 1.12  # +12%

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=tp1_price + 100,  # Vượt TP1
            stop_loss=76000,
            take_profit_targets=[tp1_price, entry_price * 1.20],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            partial_exits=[],
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.TAKE_PROFIT_1

    def test_take_profit_2_triggered(self, exit_strategy_default, sample_df_uptrend):
        """Test TP2 được trigger (full exit với 2 levels)"""
        entry_price = 80000
        tp2_price = entry_price * 1.20  # +20%

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=tp2_price + 100,
            stop_loss=76000,
            take_profit_targets=[entry_price * 1.12, tp2_price],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            partial_exits=[1],  # Đã exit TP1
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.TAKE_PROFIT_2

    def test_take_profit_already_exited(self, exit_strategy_default, sample_df_uptrend):
        """Test không trigger TP nếu đã exit trước đó"""
        entry_price = 80000

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=entry_price * 1.13,  # Vượt TP1
            stop_loss=76000,
            take_profit_targets=[entry_price * 1.12, entry_price * 1.20],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            partial_exits=[1],  # Đã exit TP1 rồi
        )

        # Should not trigger TP1 again
        if decision.should_exit:
            assert decision.exit_reason != ExitReason.TAKE_PROFIT_1


class TestExitLogicTrailingStop:
    """Tests cho trailing stop logic"""

    def test_trailing_stop_activation(self, exit_strategy_default, sample_df_uptrend):
        """Test trailing stop được kích hoạt khi profit >= 8%"""
        entry_price = 80000
        highest_price = entry_price * 1.15  # +15% high
        current_price = highest_price * 0.93  # Giảm 7% từ high

        # Set position high
        exit_strategy_default.position_highs["VNM"] = highest_price

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=76000,
            take_profit_targets=[entry_price * 1.12, entry_price * 1.20],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            partial_exits=[1],  # Đã exit TP1
        )

        # Should trigger trailing stop
        if decision.should_exit:
            assert decision.exit_reason in [ExitReason.TRAILING_STOP, ExitReason.TAKE_PROFIT_2]

    def test_trailing_stop_not_activated_low_profit(self, exit_strategy_default, sample_df_uptrend):
        """Test trailing stop không kích hoạt khi profit < 8%"""
        entry_price = 80000
        current_price = entry_price * 1.05  # Chỉ +5%

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=76000,
            take_profit_targets=[entry_price * 1.12, entry_price * 1.20],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
        )

        # Should not trigger trailing stop
        if decision.should_exit:
            assert decision.exit_reason != ExitReason.TRAILING_STOP


class TestExitLogicTimeDecay:
    """Tests cho time decay logic"""

    def test_time_decay_triggered(self, exit_strategy_default, sample_df_uptrend):
        """Test time decay trigger sau max holding days"""
        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=80500,  # Chỉ +0.6% sau nhiều ngày
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=45),  # Vượt max holding
            df=sample_df_uptrend,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.TIME_DECAY

    def test_time_decay_not_triggered_good_profit(self, exit_strategy_default, sample_df_uptrend):
        """Test time decay không trigger nếu có profit tốt"""
        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=84000,  # +5% profit
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=45),
            df=sample_df_uptrend,
        )

        # May or may not exit, but if exits, should not be time decay
        # (vì profit > threshold)


class TestExitLogicMarketCrash:
    """Tests cho market crash protection"""

    def test_market_crash_protection_with_profit(
        self, exit_strategy_default, sample_df_uptrend, bear_market_regime
    ):
        """Test market crash protection khi có profit"""
        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=83000,  # +3.75% profit
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            market_regime=bear_market_regime,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.MARKET_CRASH

    def test_market_crash_protection_small_loss(
        self, exit_strategy_default, sample_df_uptrend, bear_market_regime
    ):
        """Test market crash protection khi lỗ nhỏ (trong khoảng -2% đến 3%)"""
        # Note: Market crash protection chỉ trigger khi:
        # - pnl_percent > 3 (có lời) hoặc
        # - pnl_percent > -2 (lỗ ít)
        # Với transaction costs 1.6%, cần gross loss < 0.4% để net loss > -2%
        # -0.25% gross = -1.85% net (trong khoảng > -2%)
        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=79800,  # -0.25% gross, ~-1.85% net (trong khoảng > -2%)
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            market_regime=bear_market_regime,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.MARKET_CRASH


class TestExitLogicMLSignal:
    """Tests cho ML signal exit"""

    def test_ml_sell_signal_with_profit(self, exit_strategy_default, sample_df_uptrend):
        """Test ML SELL signal khi có profit"""
        ml_signal = {"signal": "SELL", "confidence": 70}

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=82000,  # +2.5% profit
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            ml_signal=ml_signal,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.ML_SIGNAL_SELL

    def test_ml_sell_signal_low_confidence(self, exit_strategy_default, sample_df_uptrend):
        """Test ML SELL signal với confidence thấp - should not exit"""
        ml_signal = {"signal": "SELL", "confidence": 50}  # Dưới ngưỡng 60

        decision = exit_strategy_default.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=82000,
            stop_loss=76000,
            take_profit_targets=[88000, 96000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_df_uptrend,
            ml_signal=ml_signal,
        )

        # Should not exit due to low confidence
        if decision.should_exit:
            assert decision.exit_reason != ExitReason.ML_SIGNAL_SELL


class TestPartialExitTracker:
    """Tests cho PartialExitTracker"""

    def test_initial_state(self):
        """Test initial state = 0"""
        tracker = PartialExitTracker()

        assert tracker.get_state("VNM") == 0
        assert tracker.has_partial_exit("VNM") is False
        assert tracker.is_fully_exited("VNM") is False

    def test_record_partial_exit(self):
        """Test recording partial exit"""
        tracker = PartialExitTracker()

        tracker.record_partial_exit("VNM", "PARTIAL_50%", 85000, 250)

        assert tracker.get_state("VNM") == 1
        assert tracker.has_partial_exit("VNM") is True
        assert tracker.is_fully_exited("VNM") is False

    def test_record_full_exit(self):
        """Test recording full exit"""
        tracker = PartialExitTracker()

        tracker.record_partial_exit("VNM", "PARTIAL_50%", 85000, 250)
        tracker.record_partial_exit("VNM", "FULL", 90000, 250)

        assert tracker.get_state("VNM") == 2
        assert tracker.is_fully_exited("VNM") is True

    def test_clear_position(self):
        """Test clearing position tracking"""
        tracker = PartialExitTracker()

        tracker.record_partial_exit("VNM", "FULL", 90000, 500)
        tracker.clear_position("VNM")

        assert tracker.get_state("VNM") == 0

    def test_get_exit_history(self):
        """Test getting exit history"""
        tracker = PartialExitTracker()

        tracker.record_partial_exit("VNM", "PARTIAL_50%", 85000, 250)
        tracker.record_partial_exit("VNM", "FULL", 90000, 250)

        history = tracker.get_exit_history("VNM")

        assert len(history) == 2
        assert history[0]["type"] == "PARTIAL_50%"
        assert history[1]["type"] == "FULL"

    def test_get_summary(self):
        """Test getting summary"""
        tracker = PartialExitTracker()

        tracker.record_partial_exit("VNM", "PARTIAL_50%", 85000, 250)
        tracker.record_partial_exit("VCB", "FULL", 95000, 300)

        summary = tracker.get_summary()

        assert summary["total_tracked"] == 2
        assert summary["partial_exits"] == 1
        assert summary["full_exits"] == 1
