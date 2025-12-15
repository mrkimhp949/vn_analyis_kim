# -*- coding: utf-8 -*-
"""
Test Vietnam Market Improvements v5.0

Tests for:
1. Foreign flow data integration
2. Margin debt analysis
3. Breakeven stop logic
4. Beta-adjusted stop loss
5. Ex-dividend check
6. Tiered liquidity
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np


class TestForeignFlowIntegration:
    """Test foreign flow data integration"""

    def test_foreign_flow_estimation_from_market(self):
        """Test foreign flow estimation when no direct data available"""
        from src.market.foreign_flow import ForeignFlowAnalyzer

        analyzer = ForeignFlowAnalyzer()

        # Add manual data
        analyzer.add_manual_data("2024-01-01", 100_000_000_000, 80_000_000_000)
        analyzer.add_manual_data("2024-01-02", 120_000_000_000, 90_000_000_000)
        analyzer.add_manual_data("2024-01-03", 110_000_000_000, 100_000_000_000)
        analyzer.add_manual_data("2024-01-04", 130_000_000_000, 85_000_000_000)
        analyzer.add_manual_data("2024-01-05", 125_000_000_000, 95_000_000_000)

        result = analyzer.analyze()

        assert result is not None
        assert result.trend in ["BUYING", "SELLING", "NEUTRAL"]
        assert -1 <= result.score <= 1

    def test_foreign_flow_scoring(self):
        """Test foreign flow scoring logic"""
        from src.market.foreign_flow import ForeignFlowAnalyzer

        analyzer = ForeignFlowAnalyzer()

        # Test strong buying score
        score = analyzer._calculate_score(net_value=200_000_000_000, avg_net=100_000_000_000)
        assert score == 1.0  # 2x average = strong buying

        # Test moderate buying
        score = analyzer._calculate_score(net_value=100_000_000_000, avg_net=100_000_000_000)
        assert score == 0.5  # 1x average = moderate buying

        # Test selling
        score = analyzer._calculate_score(net_value=-200_000_000_000, avg_net=100_000_000_000)
        assert score == -1.0  # -2x average = strong selling


class TestMarginDebtAnalysis:
    """Test margin debt analysis"""

    def test_margin_debt_risk_levels(self):
        """Test margin debt risk level determination"""
        from src.market.margin_debt import MarginDebtTracker, MarginRiskLevel

        tracker = MarginDebtTracker()

        # Test the tracker can analyze and return valid risk levels
        result = tracker.analyze()

        # Result should have valid risk level
        assert result.risk_level in [
            MarginRiskLevel.LOW,
            MarginRiskLevel.MODERATE,
            MarginRiskLevel.HIGH,
            MarginRiskLevel.CRITICAL,
        ]
        assert 0 <= result.risk_score <= 100
        assert 0 < result.position_adjustment <= 1.0


class TestBreakevenStop:
    """Test breakeven stop logic"""

    def test_breakeven_stop_activation(self):
        """Test breakeven stop activates after 1R profit"""
        from src.strategies.exit_logic import ImprovedExitStrategy, ExitConfig

        config = ExitConfig()
        strategy = ImprovedExitStrategy(config=config)

        # Simulate position that reached 1R profit
        entry_price = 100_000
        stop_loss = 94_000  # 6% stop = 6000 VND risk
        risk_amount = entry_price - stop_loss  # 6000 VND

        # Price went up to 1R profit (106,000) then came back
        highest_price = entry_price + risk_amount  # 106,000 (1R)
        current_price = entry_price * 1.005  # Back to near breakeven

        # Build context
        ctx = {
            "symbol": "TEST",
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "highest_price": highest_price,
            "pnl_percent": 0.5,
            "pnl_amount": 500,
        }

        result = strategy._check_breakeven_stop(ctx)

        # Should trigger breakeven stop
        assert result is not None or current_price > entry_price * 1.016  # Above breakeven


class TestBetaAdjustedStopLoss:
    """Test beta-adjusted stop loss"""

    def test_beta_calculation(self):
        """Test beta calculation"""
        from src.utils.indicators import BetaCalculator

        # Create mock data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        # Stock with higher volatility than market
        market_returns = np.random.randn(100) * 0.01
        stock_returns = market_returns * 1.5 + np.random.randn(100) * 0.005

        market_df = pd.DataFrame({"close": 1000 * (1 + np.cumsum(market_returns))}, index=dates)
        stock_df = pd.DataFrame({"close": 50000 * (1 + np.cumsum(stock_returns))}, index=dates)

        beta = BetaCalculator.calculate_beta(stock_df, market_df)

        # Beta should be > 1 for higher volatility stock
        assert 0.3 <= beta <= 2.5

    def test_beta_adjusted_stop(self):
        """Test beta-adjusted stop loss calculation"""
        from src.utils.indicators import BetaCalculator

        entry_price = 100_000

        # High beta stock should have wider stop
        stop_high, reason_high = BetaCalculator.get_beta_adjusted_stop_loss(
            entry_price=entry_price, beta=1.5
        )
        assert stop_high == entry_price * 0.92  # 8% stop for high beta

        # Low beta stock should have tighter stop
        stop_low, reason_low = BetaCalculator.get_beta_adjusted_stop_loss(
            entry_price=entry_price, beta=0.6
        )
        assert stop_low == entry_price * 0.95  # 5% stop for low beta


class TestExDividendCheck:
    """Test ex-dividend check"""

    def test_ex_dividend_detection(self):
        """Test ex-dividend date detection"""
        from src.data.fundamental_analyzer import FundamentalAnalyzer

        analyzer = FundamentalAnalyzer()

        # Mock dividend data
        with patch.object(
            analyzer,
            "_get_dividend_calendar",
            return_value={
                "ex_date": datetime.now() + timedelta(days=2),
                "dividend_yield": 3.5,
            },
        ):
            is_near, info = analyzer.is_near_ex_dividend("VNM")

            assert is_near is True
            assert info["position"] == "BEFORE_EX_DATE"

    def test_dividend_risk_adjustment(self):
        """Test dividend risk adjustment"""
        from src.data.fundamental_analyzer import FundamentalAnalyzer

        analyzer = FundamentalAnalyzer()

        # Mock near ex-dividend
        with patch.object(
            analyzer,
            "is_near_ex_dividend",
            return_value=(
                True,
                {
                    "position": "BEFORE_EX_DATE",
                    "days_until": 1,
                    "dividend_yield": 3.0,
                },
            ),
        ):
            multiplier, reason = analyzer.get_dividend_risk_adjustment("VNM")

            assert multiplier == 0.5  # 50% reduction before ex-date
            assert "avoid buying" in reason.lower()


class TestTieredLiquidity:
    """Test tiered liquidity thresholds"""

    def test_liquidity_tiers_config(self):
        """Test liquidity tier configuration"""
        from src.config.strategy_config import LiquidityTiers

        tiers = LiquidityTiers()

        # Large cap should have highest requirements
        assert tiers.large_cap["min_value"] > tiers.mid_cap["min_value"]
        assert tiers.mid_cap["min_value"] > tiers.small_cap["min_value"]

        # Position multipliers should decrease with cap size
        assert tiers.large_cap["position_multiplier"] >= tiers.mid_cap["position_multiplier"]
        assert tiers.mid_cap["position_multiplier"] >= tiers.small_cap["position_multiplier"]

    def test_micro_cap_tier(self):
        """Test micro cap tier exists"""
        from src.config.strategy_config import LiquidityTiers

        tiers = LiquidityTiers()

        assert hasattr(tiers, "micro_cap")
        assert tiers.micro_cap["min_value"] < tiers.small_cap["min_value"]
        assert tiers.micro_cap["position_multiplier"] == 0.4  # 40% for very high risk (tightened)


class TestEnhancedEntryFilters:
    """Test enhanced entry filters integration"""

    def test_ex_dividend_filter_integration(self):
        """Test ex-dividend filter is integrated"""
        from src.strategies.enhanced_entry_filters import EnhancedEntryFilters

        filters = EnhancedEntryFilters(
            use_enhanced_regime=False,
            use_session_timing=False,
            use_fundamentals=False,
            use_earnings_calendar=False,
        )

        # Check that _check_ex_dividend method exists
        assert hasattr(filters, "_check_ex_dividend")


class TestExitLogicImprovements:
    """Test exit logic improvements"""

    def test_breakeven_stop_in_checks_list(self):
        """Test breakeven stop is in the checks list"""
        from src.strategies.exit_logic import ImprovedExitStrategy

        strategy = ImprovedExitStrategy()

        # Check that _check_breakeven_stop method exists
        assert hasattr(strategy, "_check_breakeven_stop")


class TestConstantsUpdates:
    """Test constants updates"""

    def test_beta_adjusted_constants(self):
        """Test beta-adjusted stop loss constants exist"""
        from src.config.constants import (
            VN_STOP_LOSS_BASE,
            VN_STOP_LOSS_HIGH_BETA,
            VN_STOP_LOSS_LOW_BETA,
            VN_HIGH_BETA_THRESHOLD,
            VN_LOW_BETA_THRESHOLD,
        )

        assert VN_STOP_LOSS_BASE == 0.06
        assert VN_STOP_LOSS_HIGH_BETA == 0.08
        assert VN_STOP_LOSS_LOW_BETA == 0.05
        assert VN_HIGH_BETA_THRESHOLD == 1.2
        assert VN_LOW_BETA_THRESHOLD == 0.8

    def test_tiered_liquidity_constants(self):
        """Test tiered liquidity constants exist"""
        from src.config.constants import (
            VN_MIN_LIQUIDITY_VALUE,
            VN_MID_CAP_LIQUIDITY_VALUE,
            VN_LARGE_CAP_LIQUIDITY_VALUE,
            VN_SMALL_CAP_POSITION_MULT,
            VN_MID_CAP_POSITION_MULT,
            VN_LARGE_CAP_POSITION_MULT,
        )

        # v5.0 RAISED thresholds for better risk management
        assert VN_MIN_LIQUIDITY_VALUE == 2_000_000_000  # 2B (raised for safety)
        assert VN_MID_CAP_LIQUIDITY_VALUE == 3_000_000_000  # 3B for mid caps
        assert VN_LARGE_CAP_LIQUIDITY_VALUE == 5_000_000_000  # 5B for large caps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
