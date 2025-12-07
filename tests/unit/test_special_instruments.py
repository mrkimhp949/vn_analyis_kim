# -*- coding: utf-8 -*-
"""
Tests for Special Instruments Trading Logic

Tests:
- Warrant Trading (±50% limit)
- ETF Trading (short selling)
- Odd-lot Trading
"""

import pytest
from datetime import datetime, timedelta

from src.strategies.special_instruments import (
    InstrumentType,
    WarrantInfo,
    WarrantTradingLogic,
    ETFInfo,
    ETFTradingLogic,
    OddLotTradingLogic,
    SpecialInstrumentHandler,
    get_warrant_logic,
    get_etf_logic,
    get_odd_lot_logic,
    get_instrument_handler,
    WARRANT_PRICE_LIMIT,
    ETF_PRICE_LIMIT,
)


class TestWarrantTradingLogic:
    """Tests for Warrant Trading Logic"""

    def test_is_warrant(self):
        """Test warrant detection"""
        logic = WarrantTradingLogic()

        # Valid warrants
        assert logic.is_warrant("CVNM") is True
        assert logic.is_warrant("CFPT") is True
        assert logic.is_warrant("CHPG") is True

        # Not warrants
        assert logic.is_warrant("VNM") is False
        assert logic.is_warrant("FPT") is False
        assert logic.is_warrant("AB") is False

    def test_get_underlying(self):
        """Test getting underlying stock"""
        logic = WarrantTradingLogic()

        assert logic.get_underlying("CVNM") == "VNM"
        assert logic.get_underlying("CFPT") == "FPT"
        assert logic.get_underlying("VNM") is None

    def test_price_limits(self):
        """Test warrant price limits (±50%)"""
        logic = WarrantTradingLogic()

        limits = logic.get_price_limits(10000)

        assert limits["ceiling"] == 15000  # +50%
        assert limits["floor"] == 5000  # -50%
        assert limits["limit_pct"] == 50

    def test_check_tradeable_valid(self):
        """Test tradeable check for valid warrant"""
        logic = WarrantTradingLogic()

        warrant = WarrantInfo(
            symbol="CVNM",
            underlying="VNM",
            issuer="SSI",
            exercise_price=80000,
            exercise_ratio=1.0,
            expiry_date=datetime.now() + timedelta(days=60),
            warrant_type="CALL",
        )

        is_tradeable, warnings = logic.check_tradeable(warrant, 10000)

        assert is_tradeable is True

    def test_check_tradeable_near_expiry(self):
        """Test tradeable check for warrant near expiry"""
        logic = WarrantTradingLogic()

        warrant = WarrantInfo(
            symbol="CVNM",
            underlying="VNM",
            issuer="SSI",
            exercise_price=80000,
            exercise_ratio=1.0,
            expiry_date=datetime.now() + timedelta(days=2),  # Too close
            warrant_type="CALL",
        )

        is_tradeable, warnings = logic.check_tradeable(warrant, 10000)

        assert is_tradeable is False
        assert len(warnings) > 0

    def test_calculate_position_size(self):
        """Test position size calculation"""
        logic = WarrantTradingLogic(max_position_pct=0.05)

        size = logic.calculate_position_size(
            portfolio_value=100_000_000, warrant_price=10000, confidence=80
        )

        # Max 5% of portfolio, adjusted by confidence
        assert size > 0
        assert size % 100 == 0  # Multiple of lot size
        assert size <= 500  # Max ~5M / 10K = 500 shares

    def test_calculate_stop_loss(self):
        """Test stop loss calculation"""
        logic = WarrantTradingLogic()

        warrant = WarrantInfo(
            symbol="CVNM",
            underlying="VNM",
            issuer="SSI",
            exercise_price=80000,
            exercise_ratio=1.0,
            expiry_date=datetime.now() + timedelta(days=60),
            warrant_type="CALL",
        )

        stop = logic.calculate_stop_loss(10000, warrant)

        # Should be 15% below for >30 days to expiry
        assert stop == 8500


class TestETFTradingLogic:
    """Tests for ETF Trading Logic"""

    def test_is_etf(self):
        """Test ETF detection"""
        logic = ETFTradingLogic()

        assert logic.is_etf("E1VFVN30") is True
        assert logic.is_etf("FUEVFVND") is True
        assert logic.is_etf("VNM") is False

    def test_can_short(self):
        """Test short selling check"""
        logic = ETFTradingLogic()

        assert logic.can_short("E1VFVN30") is True
        assert logic.can_short("FUESSVFL") is False
        assert logic.can_short("VNM") is False

    def test_get_etf_info(self):
        """Test getting ETF info"""
        logic = ETFTradingLogic()

        info = logic.get_etf_info("E1VFVN30")

        assert info is not None
        assert info["index"] == "VN30"
        assert info["short_allowed"] is True

    def test_check_tradeable(self):
        """Test ETF tradeable check"""
        logic = ETFTradingLogic()

        etf = ETFInfo(
            symbol="E1VFVN30",
            name="VFMVN30 ETF",
            underlying_index="VN30",
            fund_type="INDEX",
            nav=20000,
            premium_discount=0.01,  # 1% premium
            expense_ratio=0.005,
            avg_volume=500000,
            avg_spread=0.002,
        )

        is_tradeable, warnings = logic.check_tradeable(etf)

        assert is_tradeable is True

    def test_calculate_short_position(self):
        """Test short position calculation"""
        logic = ETFTradingLogic()

        size = logic.calculate_short_position(
            portfolio_value=100_000_000, etf_price=20000, confidence=80, max_short_pct=0.10
        )

        assert size > 0
        assert size % 100 == 0

    def test_short_position_low_confidence(self):
        """Test short position with low confidence"""
        logic = ETFTradingLogic()

        size = logic.calculate_short_position(
            portfolio_value=100_000_000,
            etf_price=20000,
            confidence=60,  # Below 70 threshold
            max_short_pct=0.10,
        )

        assert size == 0


class TestOddLotTradingLogic:
    """Tests for Odd-lot Trading Logic"""

    def test_is_odd_lot(self):
        """Test odd-lot detection"""
        logic = OddLotTradingLogic()

        assert logic.is_odd_lot(50) is True
        assert logic.is_odd_lot(99) is True
        assert logic.is_odd_lot(1) is True
        assert logic.is_odd_lot(100) is False
        assert logic.is_odd_lot(0) is False

    def test_calculate_effective_cost(self):
        """Test effective cost calculation"""
        logic = OddLotTradingLogic()

        costs = logic.calculate_effective_cost(quantity=50, price=85000, commission_rate=0.0025)

        assert costs["gross_value"] == 4_250_000
        assert costs["commission"] >= 11_000  # Minimum commission
        assert costs["spread_cost"] > 0
        assert costs["total_cost"] > 0
        assert costs["cost_pct"] > 0

    def test_is_worth_trading_profitable(self):
        """Test worth trading check - profitable"""
        logic = OddLotTradingLogic()

        is_worth, reason = logic.is_worth_trading(
            quantity=50, price=85000, expected_return_pct=5.0  # 5% expected return
        )

        assert is_worth is True

    def test_is_worth_trading_not_profitable(self):
        """Test worth trading check - not profitable"""
        logic = OddLotTradingLogic()

        is_worth, reason = logic.is_worth_trading(
            quantity=50, price=85000, expected_return_pct=0.1  # 0.1% expected return < costs
        )

        assert is_worth is False

    def test_is_worth_trading_too_small(self):
        """Test worth trading check - value too small"""
        logic = OddLotTradingLogic()

        is_worth, reason = logic.is_worth_trading(
            quantity=5,
            price=10000,  # Only 50K value
            expected_return_pct=50.0,  # High return to pass cost check
        )

        assert is_worth is False
        assert "too small" in reason.lower()

    def test_optimize_odd_lot_exit_profit(self):
        """Test odd-lot exit optimization - profitable"""
        logic = OddLotTradingLogic()

        result = logic.optimize_odd_lot_exit(
            remaining_shares=50, current_price=90000, avg_cost=80000
        )

        assert result["action"] == "SELL"
        assert result["gross_pnl"] > 0

    def test_optimize_odd_lot_exit_loss(self):
        """Test odd-lot exit optimization - loss"""
        logic = OddLotTradingLogic()

        result = logic.optimize_odd_lot_exit(
            remaining_shares=50, current_price=75000, avg_cost=80000
        )

        assert result["action"] == "SELL"
        assert result["gross_pnl"] < 0

    def test_optimize_odd_lot_exit_too_small(self):
        """Test odd-lot exit optimization - too small to sell"""
        logic = OddLotTradingLogic()

        result = logic.optimize_odd_lot_exit(
            remaining_shares=5, current_price=75000, avg_cost=80000
        )

        assert result["action"] == "HOLD"


class TestSpecialInstrumentHandler:
    """Tests for unified instrument handler"""

    def test_detect_instrument_type(self):
        """Test instrument type detection"""
        handler = SpecialInstrumentHandler()

        assert handler.detect_instrument_type("VNM") == InstrumentType.STOCK
        assert handler.detect_instrument_type("CVNM") == InstrumentType.WARRANT
        assert handler.detect_instrument_type("E1VFVN30") == InstrumentType.ETF

    def test_get_price_limits_stock(self):
        """Test price limits for stock"""
        handler = SpecialInstrumentHandler()

        limits = handler.get_price_limits("VNM", 100000)

        # Stock limit is 7% (0.07) - use approximate comparison for floats
        assert abs(limits["limit_pct"] - 7.0) < 0.01  # ±7% for stocks
        assert abs(limits["ceiling"] - 107000.0) < 1
        assert abs(limits["floor"] - 93000.0) < 1

    def test_get_price_limits_warrant(self):
        """Test price limits for warrant"""
        handler = SpecialInstrumentHandler()

        limits = handler.get_price_limits("CVNM", 10000)

        assert limits["limit_pct"] == 50  # ±50% for warrants
        assert limits["ceiling"] == 15000
        assert limits["floor"] == 5000

    def test_validate_order_valid(self):
        """Test order validation - valid"""
        handler = SpecialInstrumentHandler()

        is_valid, warnings = handler.validate_order("VNM", 100, 85000)

        assert is_valid is True

    def test_validate_order_invalid_lot(self):
        """Test order validation - invalid lot size"""
        handler = SpecialInstrumentHandler()

        is_valid, warnings = handler.validate_order("VNM", 150, 85000)

        assert is_valid is False

    def test_validate_order_odd_lot(self):
        """Test order validation - odd lot"""
        handler = SpecialInstrumentHandler()

        is_valid, warnings = handler.validate_order("VNM", 50, 85000)

        assert is_valid is True
        assert any("odd-lot" in w.lower() for w in warnings)

    def test_validate_order_warrant(self):
        """Test order validation - warrant warning"""
        handler = SpecialInstrumentHandler()

        is_valid, warnings = handler.validate_order("CVNM", 100, 10000)

        assert is_valid is True
        assert any("warrant" in w.lower() for w in warnings)


class TestSingletonInstances:
    """Test singleton pattern for logic instances"""

    def test_warrant_logic_singleton(self):
        """Test warrant logic singleton"""
        logic1 = get_warrant_logic()
        logic2 = get_warrant_logic()

        assert logic1 is logic2

    def test_etf_logic_singleton(self):
        """Test ETF logic singleton"""
        logic1 = get_etf_logic()
        logic2 = get_etf_logic()

        assert logic1 is logic2

    def test_odd_lot_logic_singleton(self):
        """Test odd-lot logic singleton"""
        logic1 = get_odd_lot_logic()
        logic2 = get_odd_lot_logic()

        assert logic1 is logic2

    def test_instrument_handler_singleton(self):
        """Test instrument handler singleton"""
        handler1 = get_instrument_handler()
        handler2 = get_instrument_handler()

        assert handler1 is handler2
