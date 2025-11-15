# -*- coding: utf-8 -*-
"""
Unit tests for ImprovedEntryLogic
"""
import pandas as pd
import pytest
from src.strategies.entry_logic import ImprovedEntryLogic, SignalStrength


class TestImprovedEntryLogic:

    @pytest.fixture
    def entry_logic(self):
        return ImprovedEntryLogic(
            min_confidence=60,
            min_risk_reward=2.0,
            require_trend_alignment=True,
            require_volume_confirmation=False,
        )

    def test_initialization(self, entry_logic):
        """Test entry logic initialization"""
        assert entry_logic.min_confidence == 60
        assert entry_logic.min_risk_reward == 2.0
        assert entry_logic.require_trend_alignment is True

    def test_analyze_entry_insufficient_data(self, entry_logic):
        """Test with insufficient data"""
        df = pd.DataFrame({"close": [80000] * 10})
        ml_signal = {"signal": "BUY", "confidence": 70}

        result = entry_logic.analyze_entry(df, ml_signal)

        assert result.should_enter is False
        assert "Data validation" in result.warnings[0] or len(df) < 50

    def test_analyze_entry_non_buy_signal(self, entry_logic, sample_ohlcv_data):
        """Test with non-BUY signal"""
        ml_signal = {"signal": "SELL", "confidence": 70}

        result = entry_logic.analyze_entry(sample_ohlcv_data, ml_signal)

        assert result.should_enter is False
        assert result.signal_type == "HOLD"

    def test_analyze_entry_low_confidence(self, entry_logic, sample_ohlcv_data):
        """Test with low confidence signal"""
        ml_signal = {"signal": "BUY", "confidence": 50}

        result = entry_logic.analyze_entry(sample_ohlcv_data, ml_signal)

        assert result.should_enter is False
        assert "Confidence thấp" in result.warnings[0]

    def test_check_trend_alignment_uptrend(self, entry_logic, sample_ohlcv_data):
        """Test trend alignment check with uptrend"""
        # Add EMAs
        sample_ohlcv_data["ema20"] = sample_ohlcv_data["close"].ewm(span=20).mean()
        sample_ohlcv_data["ema50"] = sample_ohlcv_data["close"].ewm(span=50).mean()

        result = entry_logic._check_trend_alignment(sample_ohlcv_data, "BUY")

        assert "aligned" in result
        assert isinstance(result["strength"], (int, float))

    def test_check_support_resistance(self, entry_logic, sample_ohlcv_data):
        """Test support/resistance check"""
        current_price = sample_ohlcv_data["close"].iloc[-1]

        result = entry_logic._check_support_resistance(sample_ohlcv_data, current_price)

        assert "near_support" in result
        assert "too_close_to_resistance" in result
        assert "support_level" in result
        assert "resistance_level" in result

    def test_check_volume_confirmation(self, entry_logic, sample_ohlcv_data):
        """Test volume confirmation"""
        result = entry_logic._check_volume_confirmation(sample_ohlcv_data)

        assert "confirmed" in result
        assert "reason" in result
        assert "surge" in result

    def test_check_volatility(self, entry_logic, sample_ohlcv_data):
        """Test volatility check"""
        # Add ATR
        sample_ohlcv_data["atr"] = 2000

        result = entry_logic._check_volatility(sample_ohlcv_data)

        assert "too_high" in result
        assert "optimal" in result
        assert "value" in result

    def test_calculate_signal_strength(self, entry_logic):
        """Test signal strength calculation"""
        # High confidence, high R:R, no warnings
        strength = entry_logic._calculate_signal_strength(85, 3.0, [])
        assert strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]

        # Low confidence, low R:R, warnings
        strength = entry_logic._calculate_signal_strength(
            60, 2.0, ["warning1", "warning2"]
        )
        assert strength in [SignalStrength.WEAK, SignalStrength.MODERATE]

    def test_calculate_position_multiplier(self, entry_logic):
        """Test position multiplier calculation"""
        market_regime = {"regime": "BULL", "tradeable": True}

        # Strong signal in bull market
        multiplier = entry_logic._calculate_position_multiplier(
            SignalStrength.VERY_STRONG, 85, [], market_regime
        )
        assert 1.0 <= multiplier <= 1.5

        # Weak signal with warnings
        multiplier = entry_logic._calculate_position_multiplier(
            SignalStrength.WEAK, 60, ["warning1", "warning2"], market_regime
        )
        assert 0.3 <= multiplier <= 1.0

    def test_no_signal_helper(self, entry_logic):
        """Test _no_signal helper"""
        result = entry_logic._no_signal("Test reason")

        assert result.should_enter is False
        assert result.signal_type == "HOLD"
        assert result.confidence == 0
        assert result.strength == SignalStrength.NO_SIGNAL
        assert "Test reason" in result.warnings
