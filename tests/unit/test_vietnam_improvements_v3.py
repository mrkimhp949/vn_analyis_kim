# -*- coding: utf-8 -*-
"""
Test Vietnam Market Improvements v3.0
Kiểm tra các cải tiến cho thị trường Việt Nam
"""

import pytest
from datetime import datetime, time
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


class TestStrategyConfigV3:
    """Test Strategy Config improvements"""

    def test_entry_config_tightened(self):
        """Test entry config has been tightened"""
        from src.config.strategy_config import get_strategy_config

        config = get_strategy_config()

        # Verify tightened thresholds
        assert config.entry.min_confidence_ml >= 60, "ML confidence should be >= 60%"
        assert config.entry.min_confidence_technical >= 55, "Technical confidence should be >= 55%"
        assert config.entry.min_risk_reward >= 2.0, "R:R should be >= 2.0"
        assert config.entry.max_correlation <= 0.70, "Max correlation should be <= 0.70"
        assert config.entry.max_rsi_for_entry <= 70, "Max RSI should be <= 70"
        assert (
            config.entry.require_volume_confirmation == True
        ), "Volume confirmation should be required"
        assert config.entry.require_trend_alignment == True, "Trend alignment should be required"

    def test_exit_config_tightened(self):
        """Test exit config has been tightened"""
        from src.config.strategy_config import get_strategy_config

        config = get_strategy_config()

        # Verify tightened exit thresholds
        assert config.exit.default_stop_loss_pct >= -7.0, "Stop loss should be >= -7%"
        assert config.exit.trailing_activation <= 0.06, "Trailing activation should be <= 6%"
        assert config.exit.trailing_distance <= 0.04, "Trailing distance should be <= 4%"
        assert config.exit.partial_exit_tp1 >= 0.40, "TP1 partial exit should be >= 40%"

    def test_circuit_breaker_config_tightened(self):
        """Test circuit breaker config has been tightened"""
        from src.config.strategy_config import get_strategy_config

        config = get_strategy_config()

        # Verify tightened circuit breaker
        assert config.circuit_breaker.max_trades_per_day <= 10, "Max trades should be <= 10"
        assert config.circuit_breaker.max_loss_per_day_pct <= 0.05, "Max daily loss should be <= 5%"
        assert (
            config.circuit_breaker.max_consecutive_losses <= 5
        ), "Max consecutive losses should be <= 5"
        assert (
            config.circuit_breaker.max_portfolio_heat <= 0.70
        ), "Max portfolio heat should be <= 70%"

    def test_liquidity_tiers_tightened(self):
        """Test liquidity tiers have been tightened"""
        from src.config.strategy_config import get_strategy_config

        config = get_strategy_config()
        tiers = config.entry.liquidity_tiers

        # Verify tightened liquidity requirements
        assert tiers.large_cap["min_value"] >= 3_000_000_000, "Large cap min value should be >= 3B"
        assert tiers.mid_cap["min_value"] >= 1_000_000_000, "Mid cap min value should be >= 1B"
        assert tiers.small_cap["min_value"] >= 500_000_000, "Small cap min value should be >= 500M"


class TestEntryConfigV3:
    """Test Entry Config improvements"""

    def test_filter_thresholds_tightened(self):
        """Test filter thresholds have been tightened"""
        from src.config.entry_config import get_entry_config

        config = get_entry_config()

        # Verify tightened filter thresholds
        assert (
            config.filters.regime_confidence_threshold >= 60
        ), "Regime confidence should be >= 60%"
        assert config.filters.volume_ratio_threshold >= 1.0, "Volume ratio should be >= 1.0"
        assert config.filters.rsi_overbought <= 75, "RSI overbought should be <= 75"
        assert config.filters.correlation_max_threshold <= 0.80, "Max correlation should be <= 0.80"

    def test_risk_management_tightened(self):
        """Test risk management has been tightened"""
        from src.config.entry_config import get_entry_config

        config = get_entry_config()

        # Verify tightened risk management
        assert config.risk.min_risk_reward >= 1.5, "Min R:R should be >= 1.5"
        assert config.risk.stop_loss_max_percent <= 10.0, "Max stop loss should be <= 10%"
        assert config.risk.max_portfolio_drawdown <= 0.20, "Max drawdown should be <= 20%"

    def test_entry_logic_config_tightened(self):
        """Test entry logic config has been tightened"""
        from src.config.entry_config import get_entry_config

        config = get_entry_config()

        # Verify tightened entry logic
        assert config.min_confidence >= 50, "Min confidence should be >= 50%"
        assert config.min_confidence_lower_bound >= 35, "Lower bound should be >= 35%"


class TestVietnamIndicators:
    """Test Vietnam Market Indicators"""

    def test_price_limits_ceiling(self):
        """Test price limit detection near ceiling"""
        from src.market.vietnam_indicators import VietnamMarketIndicators

        indicators = VietnamMarketIndicators()

        # Create test data with price near ceiling
        df = pd.DataFrame(
            {
                "open": [100000, 105000],
                "high": [102000, 107000],
                "low": [99000, 104000],
                "close": [100000, 106000],  # +6% from previous close
                "volume": [1000000, 1500000],
            }
        )

        result = indicators.check_price_limits(df, 106000)

        assert result["near_limit"] == True
        assert result["limit_type"] == "CEILING"
        assert result["recommendation"] == "AVOID_BUY"

    def test_price_limits_floor(self):
        """Test price limit detection near floor"""
        from src.market.vietnam_indicators import VietnamMarketIndicators

        indicators = VietnamMarketIndicators()

        # Create test data with price near floor
        df = pd.DataFrame(
            {
                "open": [100000, 94000],
                "high": [102000, 95000],
                "low": [99000, 93000],
                "close": [100000, 94000],  # -6% from previous close
                "volume": [1000000, 1500000],
            }
        )

        result = indicators.check_price_limits(df, 94000)

        assert result["near_limit"] == True
        assert result["limit_type"] == "FLOOR"
        assert result["recommendation"] == "CAUTION"

    def test_intraday_volatility_high(self):
        """Test intraday volatility detection"""
        from src.market.vietnam_indicators import VietnamMarketIndicators

        indicators = VietnamMarketIndicators()

        # Create test data with high intraday volatility
        df = pd.DataFrame(
            {
                "open": [100000],
                "high": [107000],  # +7% from open
                "low": [99000],
                "close": [106000],  # +6% from open
                "volume": [1000000],
            }
        )

        result = indicators.check_intraday_volatility(df)

        assert result["safe"] == False
        assert len(result["warnings"]) > 0

    def test_session_timing(self):
        """Test session timing detection"""
        from src.market.vietnam_indicators import VietnamMarketIndicators

        indicators = VietnamMarketIndicators()

        result = indicators.check_session_timing()

        assert "session" in result
        assert "is_risky_period" in result
        assert "recommendation" in result

    def test_optimal_entry_time(self):
        """Test optimal entry time detection"""
        from src.market.vietnam_indicators import VietnamMarketIndicators

        indicators = VietnamMarketIndicators()

        result = indicators.get_optimal_entry_time()

        assert "is_optimal" in result
        assert "period" in result
        assert "recommendation" in result


class TestEnhancedPositionSizer:
    """Test Enhanced Position Sizer"""

    def test_basic_position_sizing(self):
        """Test basic position sizing calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80000,
            stop_loss=76000,
            take_profit=88000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert result.shares > 0
        assert result.shares % 100 == 0  # Must be multiple of lot size
        assert result.position_percent <= 15  # Max position size
        assert result.risk_percent <= 2  # Max risk per trade

    def test_kelly_criterion(self):
        """Test Kelly Criterion calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(
            total_capital=100_000_000,
            use_kelly=True,
        )

        # Test Kelly calculation
        kelly = sizer._calculate_kelly(
            win_rate=0.6,  # 60% win rate
            avg_win_loss_ratio=2.0,  # 2:1 win/loss ratio
        )

        assert kelly > 0
        assert kelly <= 0.25  # Max Kelly fraction

    def test_regime_adjustment(self):
        """Test market regime adjustment"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        # Test bull market multiplier (positional args: confidence, signal_strength, regime_info)
        bull_mult = sizer._calculate_risk_multiplier(
            70,  # confidence
            "MODERATE",  # signal_strength
            {"regime": "BULL", "confidence": 80},  # regime_info
        )

        # Test bear market multiplier
        bear_mult = sizer._calculate_risk_multiplier(
            70,  # confidence
            "MODERATE",  # signal_strength
            {"regime": "BEAR", "confidence": 80},  # regime_info
        )

        # Bull market should have higher multiplier
        assert bull_mult > bear_mult

    def test_tightened_parameters(self):
        """Test that parameters have been tightened"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        # Verify tightened parameters
        assert sizer.max_risk_per_trade <= 0.02, "Max risk should be <= 2%"
        assert sizer.max_position_size <= 0.15, "Max position should be <= 15%"
        assert sizer.max_sector_exposure <= 0.40, "Max sector exposure should be <= 40%"
        assert sizer.max_portfolio_risk <= 0.20, "Max portfolio risk should be <= 20%"


class TestConstantsV3:
    """Test Constants improvements"""

    def test_risk_constants_tightened(self):
        """Test risk constants have been tightened"""
        from src.config.constants import (
            DEFAULT_RISK_PER_TRADE,
            DEFAULT_MAX_POSITION_SIZE,
            DEFAULT_MAX_DRAWDOWN,
            DEFAULT_MIN_RISK_REWARD,
        )

        assert DEFAULT_RISK_PER_TRADE <= 0.02, "Risk per trade should be <= 2%"
        assert DEFAULT_MAX_POSITION_SIZE <= 0.15, "Max position should be <= 15%"
        assert DEFAULT_MAX_DRAWDOWN <= 0.15, "Max drawdown should be <= 15%"
        assert DEFAULT_MIN_RISK_REWARD >= 1.5, "Min R:R should be >= 1.5"

    def test_market_regime_constants(self):
        """Test market regime constants"""
        from src.config.constants import (
            BULL_MARKET_CONFIDENCE,
            BEAR_MARKET_CONFIDENCE,
            BULL_MARKET_EXPOSURE,
            BEAR_MARKET_EXPOSURE,
        )

        # Bear market should require higher confidence
        assert BEAR_MARKET_CONFIDENCE > BULL_MARKET_CONFIDENCE

        # Bear market should have lower exposure
        assert BEAR_MARKET_EXPOSURE < BULL_MARKET_EXPOSURE


class TestCircuitBreakerV3:
    """Test Circuit Breaker improvements"""

    def test_circuit_breaker_tightened(self):
        """Test circuit breaker has been tightened"""
        from src.risk.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        # Verify tightened thresholds
        assert breaker.max_trades_per_day <= 10
        assert breaker.max_loss_per_day_pct <= 0.05
        assert breaker.max_consecutive_losses <= 5
        assert breaker.max_portfolio_heat <= 0.70

    def test_gradual_response(self):
        """Test gradual response levels"""
        from src.risk.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        # Warning threshold should be less severe than caution
        assert breaker.warning_threshold > breaker.caution_threshold

        # Caution threshold should be less severe than trip
        assert breaker.caution_threshold > breaker.vnindex_drop_threshold

    def test_drawdown_protection(self):
        """Test drawdown protection"""
        from src.risk.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(total_capital=100_000_000)

        # Update with new peak
        result = breaker.update_portfolio_value(110_000_000)
        assert result["new_peak"] == True

        # Update with drawdown
        result = breaker.update_portfolio_value(95_000_000)
        assert result["new_peak"] == False
        assert result["current_drawdown"] > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
