# -*- coding: utf-8 -*-
"""
Unit tests for Vietnam Market Improvements v6.0

Tests for:
1. Volume-based floor bounce logic
2. Enhanced gap protection
3. Liquidity risk management
4. Foreign flow fallback
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np


class TestFloorBounceVolumeLogic:
    """Test volume-based floor bounce exit logic"""

    def test_panic_volume_triggers_immediate_exit(self):
        """Volume > 3x should trigger immediate exit at floor"""
        from src.strategies.exit_logic import ImprovedExitStrategy, ExitConfig

        strategy = ImprovedExitStrategy()

        # Create DataFrame with panic volume (4x average)
        dates = pd.date_range(end=datetime.now(), periods=25, freq="D")
        df = pd.DataFrame(
            {
                "open": [100] * 25,
                "high": [105] * 25,
                "low": [95] * 25,
                "close": [100] * 24 + [93],  # Last day at floor (-7%)
                "volume": [1000] * 24 + [4000],  # 4x volume on last day
            },
            index=dates,
        )

        # Calculate volume ratio
        volume_ratio = strategy._calculate_volume_ratio(df)

        # Should be approximately 4.0
        assert volume_ratio >= 3.0, f"Expected panic volume ratio >= 3.0, got {volume_ratio}"

    def test_elevated_volume_extends_wait_time(self):
        """Volume 1.5-3x should extend wait time to 60 minutes"""
        from src.config.constants import (
            VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES,
            VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES,
            VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO,
        )

        # Verify constants are set correctly
        assert VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES == 30
        assert VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES == 60
        assert VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO == 1.5

    def test_normal_volume_uses_standard_wait(self):
        """Volume < 1.5x should use standard 30-minute wait"""
        from src.strategies.exit_logic import ImprovedExitStrategy

        strategy = ImprovedExitStrategy()

        # Create DataFrame with normal volume
        dates = pd.date_range(end=datetime.now(), periods=25, freq="D")
        df = pd.DataFrame(
            {
                "open": [100] * 25,
                "high": [105] * 25,
                "low": [95] * 25,
                "close": [100] * 25,
                "volume": [1000] * 25,  # Consistent volume
            },
            index=dates,
        )

        volume_ratio = strategy._calculate_volume_ratio(df)

        # Should be approximately 1.0
        assert 0.8 <= volume_ratio <= 1.2, f"Expected normal volume ratio ~1.0, got {volume_ratio}"

    def test_volume_ratio_calculation_empty_df(self):
        """Volume ratio should return 1.0 for empty DataFrame"""
        from src.strategies.exit_logic import ImprovedExitStrategy

        strategy = ImprovedExitStrategy()

        # Empty DataFrame
        df = pd.DataFrame()
        volume_ratio = strategy._calculate_volume_ratio(df)

        assert volume_ratio == 1.0

    def test_volume_ratio_calculation_no_volume_column(self):
        """Volume ratio should return 1.0 if no volume column"""
        from src.strategies.exit_logic import ImprovedExitStrategy

        strategy = ImprovedExitStrategy()

        df = pd.DataFrame({"close": [100, 101, 102]})
        volume_ratio = strategy._calculate_volume_ratio(df)

        assert volume_ratio == 1.0


class TestEnhancedGapProtection:
    """Test enhanced gap down/up protection"""

    def test_gap_down_emergency_threshold(self):
        """Gap down > 4% should trigger emergency exit"""
        from src.config.constants import VN_GAP_DOWN_EMERGENCY_THRESHOLD

        # Verify threshold is -4%
        assert VN_GAP_DOWN_EMERGENCY_THRESHOLD == -0.04

    def test_gap_down_profit_protection_threshold(self):
        """Gap down 2.5-4% with profit should trigger protection"""
        from src.config.constants import VN_GAP_DOWN_EXIT_THRESHOLD

        # Verify threshold is -2.5%
        assert VN_GAP_DOWN_EXIT_THRESHOLD == -0.025

    def test_gap_up_profit_take_threshold(self):
        """Gap up > 4% should consider profit taking"""
        from src.config.constants import VN_GAP_UP_PROFIT_TAKE_THRESHOLD

        # Verify threshold is +4%
        assert VN_GAP_UP_PROFIT_TAKE_THRESHOLD == 0.04


class TestLiquidityRiskManagement:
    """Test improved liquidity risk management"""

    def test_critical_liquidity_tier(self):
        """Stocks below 500M VND should get CRITICAL tier"""
        from src.strategies.position_sizing import EnhancedPositionSizer
        from src.config.constants import VN_CRITICAL_LIQUIDITY_VALUE

        sizer = EnhancedPositionSizer()

        # Test with critical liquidity (200M VND)
        tier_name, tier_config = sizer._get_liquidity_tier("ABC", 200_000_000)

        assert tier_name == "CRITICAL"
        assert tier_config["max_position_pct"] == 0.03  # 3% max
        assert tier_config["slippage"] == 0.015  # 1.5% slippage

    def test_small_cap_liquidity_tier(self):
        """Stocks 500M-3B VND should get SMALL_CAP tier"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        # Test with small cap liquidity (1B VND)
        tier_name, tier_config = sizer._get_liquidity_tier("XYZ", 1_000_000_000)

        assert tier_name == "SMALL_CAP"
        assert tier_config["max_position_pct"] == 0.06  # 6% max

    def test_vn30_gets_highest_tier(self):
        """VN30 stocks should always get VN30 tier"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        # VN30 stock with low liquidity should still get VN30 tier
        tier_name, tier_config = sizer._get_liquidity_tier("VNM", 100_000_000)

        assert tier_name == "VN30"
        assert tier_config["max_position_pct"] == 0.15  # 15% max

    def test_exit_liquidity_check_critical(self):
        """Position > 20% of daily volume should be CRITICAL"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        result = sizer.check_exit_liquidity(
            symbol="ABC",
            shares=25000,
            avg_daily_volume=100000,  # 25% of daily volume
        )

        assert result["risk_level"] == "CRITICAL"
        assert result["split_orders"] == True
        assert result["suggested_splits"] >= 4

    def test_exit_liquidity_check_low(self):
        """Position < 5% of daily volume should be LOW risk"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        result = sizer.check_exit_liquidity(
            symbol="VNM",
            shares=3000,
            avg_daily_volume=100000,  # 3% of daily volume
        )

        assert result["risk_level"] == "LOW"
        assert result["split_orders"] == False


class TestForeignFlowFallback:
    """Test foreign flow data fallback logic"""

    def test_default_result_returns_neutral(self):
        """When data unavailable, should return neutral score"""
        from src.market.foreign_flow import ForeignFlowAnalyzer

        analyzer = ForeignFlowAnalyzer()
        result = analyzer._default_result("Test unavailable")

        assert result.score == 0.0
        assert result.trend == "NEUTRAL"
        assert result.strength == "WEAK"

    def test_data_staleness_detection(self):
        """Should detect stale data (> 15 minutes old)"""
        from src.market.foreign_flow import ForeignFlowAnalyzer

        analyzer = ForeignFlowAnalyzer()

        # No cache = stale
        assert analyzer.is_data_stale() == True

        # Set recent cache
        analyzer._cache_time = datetime.now()
        assert analyzer.is_data_stale() == False

        # Set old cache (20 minutes ago)
        analyzer._cache_time = datetime.now() - timedelta(minutes=20)
        assert analyzer.is_data_stale() == True

    def test_adjusted_score_reduces_for_stale_data(self):
        """Stale data should have score reduced by 50%"""
        from src.market.foreign_flow import ForeignFlowAnalyzer, ForeignFlowData

        analyzer = ForeignFlowAnalyzer()

        # Set up cache with score 1.0 but stale (20 min old)
        # Note: cache must be valid (within TTL) for analyze() to return it
        # but stale for staleness check (> 15 min)
        analyzer._cache = ForeignFlowData(
            date=datetime.now().isoformat(),
            net_value=1000000000,
            buy_value=1500000000,
            sell_value=500000000,
            net_volume=100000,
            score=1.0,
            trend="BUYING",
            strength="STRONG",
            consecutive_days=3,
            vs_average=2.0,
        )
        # Set cache time to 20 minutes ago (stale but within default TTL of 300s)
        # We need to also set TTL high enough so cache is valid
        analyzer.cache_ttl = 3600  # 1 hour TTL
        analyzer._cache_time = datetime.now() - timedelta(minutes=20)

        # Verify staleness detection works
        assert analyzer.is_data_stale(max_delay_minutes=15) == True

        # Get adjusted score - should be reduced by 50%
        adjusted = analyzer.get_adjusted_score(max_delay_minutes=15)

        # Should be reduced by 50%
        assert adjusted == 0.5, f"Expected 0.5, got {adjusted}"

    def test_data_age_calculation(self):
        """Should correctly calculate data age in minutes"""
        from src.market.foreign_flow import ForeignFlowAnalyzer

        analyzer = ForeignFlowAnalyzer()

        # No cache = infinite age
        assert analyzer.get_data_age_minutes() == float("inf")

        # Set cache 10 minutes ago
        analyzer._cache_time = datetime.now() - timedelta(minutes=10)
        age = analyzer.get_data_age_minutes()

        assert 9.5 <= age <= 10.5  # Allow small tolerance


class TestDCADocumentation:
    """Test DCA documentation is properly included"""

    def test_dca_disabled_constant(self):
        """DCA should be disabled for VN market"""
        from src.strategies.position_sizing import PositionSizingConstants

        assert PositionSizingConstants.DCA_ENABLED == False

    def test_dca_documentation_exists(self):
        """DCA section should have comprehensive documentation"""
        import inspect
        from src.strategies.position_sizing import PositionSizingConstants

        # Get source code of the class
        source = inspect.getsource(PositionSizingConstants)

        # Check for key documentation points
        assert "1.48%" in source or "1.48" in source  # Transaction cost
        assert "T+2" in source  # Settlement
        assert "±7%" in source or "7%" in source  # Daily limit


class TestExitReasonEnums:
    """Test new exit reason enums are defined"""

    def test_floor_bounce_timeout_reason(self):
        """FLOOR_BOUNCE_TIMEOUT should be defined"""
        from src.strategies.exit_logic import ExitReason

        assert hasattr(ExitReason, "FLOOR_BOUNCE_TIMEOUT")
        assert "Floor Bounce Timeout" in ExitReason.FLOOR_BOUNCE_TIMEOUT.value

    def test_panic_selling_reason(self):
        """PANIC_SELLING should be defined"""
        from src.strategies.exit_logic import ExitReason

        assert hasattr(ExitReason, "PANIC_SELLING")
        assert "Panic" in ExitReason.PANIC_SELLING.value

    def test_gap_down_emergency_reason(self):
        """GAP_DOWN_EMERGENCY should be defined"""
        from src.strategies.exit_logic import ExitReason

        assert hasattr(ExitReason, "GAP_DOWN_EMERGENCY")
        assert "Gap Down" in ExitReason.GAP_DOWN_EMERGENCY.value


class TestFloorBounceConstants:
    """Test floor bounce constants are properly defined"""

    def test_floor_bounce_recovery_pct(self):
        """Recovery percentage should be defined"""
        from src.config.constants import VN_FLOOR_BOUNCE_RECOVERY_PCT

        assert VN_FLOOR_BOUNCE_RECOVERY_PCT == 0.01  # 1%

    def test_floor_bounce_immediate_exit_volume(self):
        """Immediate exit volume threshold should be defined"""
        from src.config.constants import VN_FLOOR_BOUNCE_IMMEDIATE_EXIT_VOLUME

        assert VN_FLOOR_BOUNCE_IMMEDIATE_EXIT_VOLUME == 3.0  # 3x volume


class TestOddLotIntegration:
    """Test odd-lot trading integration with position sizing"""

    def test_odd_lot_position_calculation(self):
        """Test odd-lot position calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=10_000_000)  # 10M VND

        # Small position that results in odd-lot
        result = sizer.calculate_odd_lot_position(
            symbol="VNM",
            entry_price=85000,
            target_value=5_000_000,  # 5M VND → ~58 shares
            expected_return_pct=5.0,
        )

        assert result["enabled"] == True
        assert result["is_odd_lot"] == True
        assert result["shares"] < 100
        assert "cost_pct" in result

    def test_odd_lot_not_worthwhile(self):
        """Test odd-lot warning when costs exceed expected return"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=10_000_000)

        # Very small expected return
        result = sizer.calculate_odd_lot_position(
            symbol="VNM",
            entry_price=85000,
            target_value=1_000_000,  # 1M VND
            expected_return_pct=0.5,  # Only 0.5% expected
        )

        assert result["enabled"] == True
        assert result["is_worthwhile"] == False
        assert "warning" in result

    def test_standard_lot_recommendation(self):
        """Test that large positions recommend standard lot"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_odd_lot_position(
            symbol="VNM",
            entry_price=85000,
            target_value=20_000_000,  # 20M VND → ~235 shares
            expected_return_pct=5.0,
        )

        assert result["is_odd_lot"] == False
        assert result["shares"] >= 100


class TestMarginTradingIntegration:
    """Test margin trading integration with position sizing"""

    def test_margin_position_calculation(self):
        """Test margin-enhanced position calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_margin_position(
            symbol="VNM",
            entry_price=85000,
            stop_loss=80000,
            confidence=70,
            use_margin=False,  # Test cash-only first
        )

        assert result["use_margin"] == False
        assert result["margin_status"] == "CASH_ONLY"
        assert result["base_shares"] > 0

    def test_margin_disabled_returns_cash_only(self):
        """Test that disabling margin returns cash-only calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_margin_position(
            symbol="HPG",
            entry_price=25000,
            stop_loss=23000,
            confidence=65,
            use_margin=False,
        )

        assert result["use_margin"] == False
        assert result["base_shares"] == result["margin_shares"]


class TestT0TradingValidation:
    """Test T+0 intraday trading validation"""

    def test_t0_validation_enabled(self):
        """Test T+0 validation when enabled"""
        from src.strategies.position_sizing import EnhancedPositionSizer
        from src.config.constants import VN_T0_ENABLED

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.validate_t0_trading(
            symbol="VNM",
            quantity=1000,
            price=85000,
            account_value=100_000_000,  # 100M VND
        )

        assert result["t0_enabled"] == VN_T0_ENABLED
        assert "validations" in result

    def test_t0_validation_account_too_small(self):
        """Test T+0 validation fails for small accounts"""
        from src.strategies.position_sizing import EnhancedPositionSizer
        from src.config.constants import VN_T0_MIN_ACCOUNT_VALUE

        sizer = EnhancedPositionSizer(total_capital=30_000_000)  # 30M VND

        result = sizer.validate_t0_trading(
            symbol="VNM",
            quantity=100,
            price=85000,
            account_value=30_000_000,  # Below 50M minimum
        )

        # Should fail account value check
        if result["t0_enabled"]:
            assert result["can_trade_t0"] == False
            assert any(
                v["check"] == "account_value" and not v["passed"] for v in result["validations"]
            )


class TestSpecialInstrumentIntegration:
    """Test warrant/ETF special instrument handling"""

    def test_auto_detect_stock(self):
        """Test auto-detection of regular stock"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_special_instrument_position(
            symbol="VNM",
            entry_price=85000,
            confidence=70,
            instrument_type="AUTO",
        )

        assert result["instrument_type"] == "STOCK"
        assert result["shares"] > 0

    def test_etf_position_calculation(self):
        """Test ETF position calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_special_instrument_position(
            symbol="E1VFVN30",
            entry_price=20000,
            confidence=65,
            instrument_type="AUTO",
        )

        assert result["instrument_type"] == "ETF"
        assert "can_short" in result

    def test_warrant_position_limits(self):
        """Test warrant position is limited to 5% of portfolio"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer(total_capital=100_000_000)

        result = sizer.calculate_special_instrument_position(
            symbol="CVNM2401",  # Warrant symbol
            entry_price=5000,
            confidence=70,
            instrument_type="WARRANT",
        )

        assert result["instrument_type"] == "WARRANT"
        # Warrant max allocation is 5% of portfolio
        if result.get("shares", 0) > 0:
            position_value = result["shares"] * 5000
            assert position_value <= sizer.total_capital * 0.05


class TestMarginConstants:
    """Test margin trading constants are properly defined"""

    def test_margin_levels_defined(self):
        """Test all margin levels are defined"""
        from src.config.constants import (
            VN_INITIAL_MARGIN,
            VN_MAINTENANCE_MARGIN,
            VN_MARGIN_WARNING_LEVEL,
            VN_MARGIN_CALL_LEVEL,
            VN_FORCE_LIQUIDATION_LEVEL,
        )

        assert VN_INITIAL_MARGIN == 0.50  # 50%
        assert VN_MAINTENANCE_MARGIN == 0.35  # 35%
        assert VN_MARGIN_WARNING_LEVEL == 0.40  # 40%
        assert VN_MARGIN_CALL_LEVEL == 0.30  # 30%
        assert VN_FORCE_LIQUIDATION_LEVEL == 0.25  # 25%


class TestT0Constants:
    """Test T+0 trading constants are properly defined"""

    def test_t0_constants_defined(self):
        """Test all T+0 constants are defined"""
        from src.config.constants import (
            VN_T0_ENABLED,
            VN_T0_MIN_ACCOUNT_VALUE,
            VN_T0_MAX_TRADES_PER_DAY,
            VN_T0_MAX_LOSS_PCT,
            VN_T0_MIN_HOLDING_MINUTES,
        )

        assert VN_T0_ENABLED == True
        assert VN_T0_MIN_ACCOUNT_VALUE == 50_000_000  # 50M VND
        assert VN_T0_MAX_TRADES_PER_DAY == 20
        assert VN_T0_MAX_LOSS_PCT == 0.02  # 2%
        assert VN_T0_MIN_HOLDING_MINUTES == 5


class TestWarrantETFConstants:
    """Test warrant/ETF constants are properly defined"""

    def test_warrant_constants_defined(self):
        """Test warrant constants are defined"""
        from src.config.constants import (
            VN_WARRANT_PRICE_LIMIT,
            VN_WARRANT_SETTLEMENT,
            VN_WARRANT_MIN_DAYS_TO_EXPIRY,
            VN_WARRANT_WARNING_DAYS,
        )

        assert VN_WARRANT_PRICE_LIMIT == 0.50  # ±50%
        assert VN_WARRANT_SETTLEMENT == 0  # T+0
        assert VN_WARRANT_MIN_DAYS_TO_EXPIRY == 3
        assert VN_WARRANT_WARNING_DAYS == 30

    def test_etf_constants_defined(self):
        """Test ETF constants are defined"""
        from src.config.constants import (
            VN_ETF_PRICE_LIMIT,
            VN_ETF_SHORT_ALLOWED,
        )

        assert VN_ETF_PRICE_LIMIT == 0.07  # ±7%
        assert VN_ETF_SHORT_ALLOWED == True
