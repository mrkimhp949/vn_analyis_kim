# -*- coding: utf-8 -*-
"""
Tests for new trading modules:
- Margin Trading
- Sector Rotation Strategy
- Dividend Capture Strategy
- Order Book Integration

Author: Trading Bot Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

# =============================================================================
# MARGIN TRADING TESTS
# =============================================================================


class TestMarginTrading:
    """Tests for margin trading module"""

    @pytest.fixture
    def margin_manager(self):
        from src.strategies.margin_trading import MarginTradingManager, MarginConfig

        config = MarginConfig()
        return MarginTradingManager(config)

    def test_margin_tier_vn30(self, margin_manager):
        """Test VN30 stocks get Tier 1 classification"""
        from src.strategies.margin_trading import MarginTier

        vn30_symbols = ["VCB", "FPT", "VNM", "HPG"]
        for symbol in vn30_symbols:
            tier = margin_manager.get_margin_tier(symbol)
            assert tier == MarginTier.TIER_1, f"{symbol} should be Tier 1"

    def test_initial_margin_calculation(self, margin_manager):
        """Test initial margin requirement calculation"""
        # VN30 stocks should have 50% initial margin
        vn30_margin = margin_manager.get_initial_margin_requirement("VCB")
        assert vn30_margin == 0.50

    def test_max_buyable_shares(self, margin_manager):
        """Test max buyable shares calculation with margin"""
        shares, loan, equity = margin_manager.calculate_max_buyable_shares(
            symbol="VNM", price=80000, available_cash=100_000_000, use_margin=True
        )

        # With 50% margin and 80% utilization, should buy ~2x cash
        assert shares > 0
        assert shares % 100 == 0  # Must be multiple of lot size
        assert loan > 0  # Should use some margin
        assert equity > 0

    def test_margin_status_safe(self, margin_manager):
        """Test margin status when position is healthy"""
        from src.strategies.margin_trading import MarginStatus

        # Add a healthy position (60% equity ratio)
        margin_manager.add_position(
            symbol="VNM",
            shares=1000,
            avg_cost=80000,
            current_price=80000,
            loan_amount=32_000_000,  # 40% loan-to-value
        )

        status, details = margin_manager.check_margin_status("VNM")
        assert status == MarginStatus.SAFE

    def test_margin_call_trigger(self, margin_manager):
        """Test margin call is triggered when equity drops"""
        from src.strategies.margin_trading import MarginStatus

        # Add position with 60% equity
        margin_manager.add_position(
            symbol="HPG",
            shares=2000,
            avg_cost=30000,
            current_price=20000,  # Price dropped 33%
            loan_amount=35_000_000,  # High loan amount
        )

        status, details = margin_manager.check_margin_status("HPG")
        # Should be at least WARNING or MARGIN_CALL
        assert status in [
            MarginStatus.WARNING,
            MarginStatus.MARGIN_CALL,
            MarginStatus.FORCE_SELL,
            MarginStatus.CRITICAL,
        ]

    def test_reduction_suggestion(self, margin_manager):
        """Test position reduction suggestion"""
        # Add position with low equity
        margin_manager.add_position(
            symbol="TEST",
            shares=1000,
            avg_cost=100000,
            current_price=80000,
            loan_amount=50_000_000,  # High loan
        )

        reduction = margin_manager.suggest_reduction("TEST", target_equity_ratio=0.50)

        assert "shares_to_sell" in reduction
        assert reduction["shares_to_sell"] > 0

    def test_margin_health_score(self, margin_manager):
        """Test margin health score calculation"""
        # Empty portfolio should be 100
        score = margin_manager.get_margin_health_score()
        assert score == 100.0

        # Add healthy position
        margin_manager.add_position(
            symbol="VNM",
            shares=1000,
            avg_cost=80000,
            current_price=85000,  # Profit
            loan_amount=20_000_000,  # Low loan
        )

        score = margin_manager.get_margin_health_score()
        assert score > 50  # Should still be healthy


# =============================================================================
# SECTOR ROTATION TESTS
# =============================================================================


class TestSectorRotation:
    """Tests for sector rotation strategy"""

    @pytest.fixture
    def rotation_strategy(self):
        from src.strategies.sector_rotation_strategy import SectorRotationStrategy

        return SectorRotationStrategy()

    def test_sector_detection(self, rotation_strategy):
        """Test symbol to sector mapping"""
        from src.strategies.sector_rotation_strategy import get_sector_for_symbol

        assert get_sector_for_symbol("VCB") == "BANKING"
        assert get_sector_for_symbol("VNM") == "CONSUMER"
        assert get_sector_for_symbol("HPG") == "INDUSTRIAL"
        assert get_sector_for_symbol("FPT") == "TECHNOLOGY"

    def test_sector_phase_detection(self, rotation_strategy):
        """Test sector phase detection"""
        from src.strategies.sector_rotation_strategy import SectorPhase

        # Strong momentum should be MARKUP
        phase = rotation_strategy.detect_sector_phase(
            momentum_short=0.15,  # 15% 1-month return
            momentum_long=0.10,  # 10% 3-month return
            relative_strength=1.2,  # Outperforming
            foreign_flow=0.5,  # Net buying
        )
        assert phase == SectorPhase.MARKUP

        # Weak momentum should be DISTRIBUTION or MARKDOWN (depends on severity)
        phase = rotation_strategy.detect_sector_phase(
            momentum_short=-0.10, momentum_long=-0.05, relative_strength=0.8, foreign_flow=-0.5
        )
        assert phase in [SectorPhase.DISTRIBUTION, SectorPhase.MARKDOWN]

    def test_signal_generation(self, rotation_strategy):
        """Test sector signal generation"""
        from src.strategies.sector_rotation_strategy import SectorSignal, SectorPhase

        signal, confidence = rotation_strategy.generate_sector_signal(
            phase=SectorPhase.MARKUP,
            momentum=0.12,
            relative_strength=1.15,
            foreign_flow=0.6,
            economic_cycle=None,
            sector="TECHNOLOGY",
        )

        assert signal in [SectorSignal.STRONG_BUY, SectorSignal.BUY]
        assert confidence > 0.5

    def test_rotation_recommendations(self, rotation_strategy):
        """Test rotation recommendations generation"""
        # Create sample sector data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        sector_data = {}
        for sector in ["BANKING", "TECHNOLOGY"]:
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
            sector_data[sector] = pd.DataFrame({"close": prices}, index=dates)

        benchmark_df = pd.DataFrame(
            {"close": 1200 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 100))}, index=dates
        )

        # Analyze sectors
        rotation_strategy.analyze_all_sectors(sector_data, benchmark_df)

        # Get recommendations
        recs = rotation_strategy.get_rotation_recommendations(
            current_allocations={"BANKING": 0.50, "TECHNOLOGY": 0.50}
        )

        assert "target_allocations" in recs
        assert "recommended_changes" in recs


# =============================================================================
# DIVIDEND CAPTURE TESTS
# =============================================================================


class TestDividendCapture:
    """Tests for dividend capture strategy"""

    @pytest.fixture
    def dividend_strategy(self):
        from src.strategies.dividend_capture_strategy import DividendCaptureStrategy

        return DividendCaptureStrategy()

    def test_add_dividend_event(self, dividend_strategy):
        """Test adding dividend event"""
        from src.strategies.dividend_capture_strategy import DividendType

        event = dividend_strategy.add_dividend_event(
            symbol="VNM",
            ex_date=date.today() + timedelta(days=10),
            dividend_amount=1500,
            current_price=75000,
            dividend_type=DividendType.CASH,
        )

        assert event.symbol == "VNM"
        assert event.cash_amount == 1500
        assert event.yield_percent == 1500 / 75000

    def test_capture_opportunity_analysis(self, dividend_strategy):
        """Test dividend capture opportunity analysis"""
        from src.strategies.dividend_capture_strategy import DividendSignal

        # Add high-yield dividend
        dividend_strategy.add_dividend_event(
            symbol="TEST",
            ex_date=date.today() + timedelta(days=7),
            dividend_amount=5000,  # High dividend
            current_price=100000,  # 5% yield
        )

        rec = dividend_strategy.analyze_capture_opportunity("TEST", 100000)

        assert rec is not None
        assert rec.dividend_yield > 0.03  # Above min threshold
        assert rec.signal in [
            DividendSignal.STRONG_BUY,
            DividendSignal.BUY,
            DividendSignal.HOLD,
            DividendSignal.AVOID,
        ]

    def test_capture_too_late(self, dividend_strategy):
        """Test that capture is rejected if too close to ex-date"""
        from src.strategies.dividend_capture_strategy import DividendSignal

        # Ex-date is tomorrow (too late for T+2)
        dividend_strategy.add_dividend_event(
            symbol="LATE",
            ex_date=date.today() + timedelta(days=1),
            dividend_amount=2000,
            current_price=50000,
        )

        rec = dividend_strategy.analyze_capture_opportunity("LATE", 50000)

        assert rec is not None
        assert rec.signal == DividendSignal.AVOID

    def test_active_capture_management(self, dividend_strategy):
        """Test managing active capture position"""
        # Add dividend event
        dividend_strategy.add_dividend_event(
            symbol="VNM",
            ex_date=date.today() + timedelta(days=5),
            dividend_amount=1500,
            current_price=75000,
        )

        # Record entry
        dividend_strategy.record_capture_entry("VNM", 74500, 1000)

        # Check position status
        status = dividend_strategy.manage_active_capture("VNM", 75000)

        assert status["action"] == "HOLD"
        assert "days_held" in status


# =============================================================================
# ORDER BOOK INTEGRATION TESTS
# =============================================================================


class TestOrderBookIntegration:
    """Tests for order book integration"""

    @pytest.fixture
    def order_book(self):
        from src.strategies.order_book_integration import OrderBookIntegration

        return OrderBookIntegration()

    def test_order_book_update(self, order_book):
        """Test order book data update"""
        bids = [(75000, 50000, 25), (74900, 30000, 15), (74800, 20000, 10)]
        asks = [(75100, 20000, 12), (75200, 25000, 14), (75300, 35000, 18)]

        ob = order_book.update_order_book(
            symbol="VNM", bids=bids, asks=asks, last_price=75050, last_volume=1000
        )

        assert ob.best_bid == 75000
        assert ob.best_ask == 75100
        assert ob.spread == 100
        assert ob.mid_price == 75050

    def test_order_book_analysis(self, order_book):
        """Test order book analysis"""
        from src.strategies.order_book_integration import OrderBookImbalance

        # Update with bid-heavy order book
        bids = [(75000, 100000, 50), (74900, 80000, 40), (74800, 60000, 30)]
        asks = [(75100, 20000, 10), (75200, 25000, 12), (75300, 30000, 15)]

        order_book.update_order_book("VNM", bids, asks, 75050, 1000)

        analysis = order_book.analyze_order_book("VNM", 5000, "BUY")

        assert analysis is not None
        assert analysis.imbalance in [OrderBookImbalance.HEAVY_BID, OrderBookImbalance.SLIGHT_BID]
        assert analysis.confidence > 0

    def test_slippage_estimation(self, order_book):
        """Test slippage estimation"""
        bids = [(75000, 10000, 10), (74900, 10000, 10), (74800, 10000, 10)]
        asks = [(75100, 10000, 10), (75200, 10000, 10), (75300, 10000, 10)]

        order_book.update_order_book("VNM", bids, asks, 75050, 1000)

        # Small order should have low slippage
        analysis_small = order_book.analyze_order_book("VNM", 1000, "BUY")

        # Large order should have higher slippage
        analysis_large = order_book.analyze_order_book("VNM", 50000, "BUY")

        assert analysis_small.estimated_slippage_pct < analysis_large.estimated_slippage_pct

    def test_entry_adjustment(self, order_book):
        """Test entry confidence adjustment"""
        # Tight spread, bid-heavy
        bids = [(75000, 80000, 40), (74950, 60000, 30), (74900, 40000, 20)]
        asks = [(75050, 20000, 10), (75100, 25000, 12), (75150, 30000, 15)]

        order_book.update_order_book("VNM", bids, asks, 75025, 1000)

        adjustment, reason = order_book.get_entry_adjustment("VNM", 60, 5000)

        # With tight spread and buying pressure, should get positive adjustment
        assert adjustment >= 0
        assert len(reason) > 0

    def test_optimal_entry_price(self, order_book):
        """Test optimal entry price calculation"""
        bids = [(75000, 50000, 25), (74900, 30000, 15)]
        asks = [(75100, 20000, 12), (75200, 25000, 14)]

        order_book.update_order_book("VNM", bids, asks, 75050, 1000)

        # Passive entry should be close to bid
        price_passive = order_book.get_optimal_entry_price("VNM", 5000, "BUY", 0.0)

        # Aggressive entry should be close to ask
        price_aggressive = order_book.get_optimal_entry_price("VNM", 5000, "BUY", 1.0)

        assert price_passive <= price_aggressive


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestStrategyConfigUpdates:
    """Tests for updated strategy configuration"""

    def test_liquidity_tiers_updated(self):
        """Test that liquidity tiers are properly updated"""
        from src.config.strategy_config import LiquidityTiers

        tiers = LiquidityTiers()

        # Large cap should now be 1.5B (reduced from 3B)
        assert tiers.large_cap["min_value"] == 1_500_000_000

        # Mid cap should now be 500M (reduced from 1B)
        assert tiers.mid_cap["min_value"] == 500_000_000

        # Small cap should now be 200M (reduced from 500M)
        assert tiers.small_cap["min_value"] == 200_000_000

    def test_stop_loss_config_updated(self):
        """Test that stop loss config is properly updated"""
        from src.config.strategy_config import ExitConfig

        exit_config = ExitConfig()

        # Default stop loss should be 5% (was 6%)
        assert exit_config.default_stop_loss_pct == -5.0

        # High beta stop loss should be 10% (was 8%)
        assert exit_config.high_beta_stop_loss_pct == -10.0

        # ATR multiplier should be 2.5 (was 2.0)
        assert exit_config.stop_loss_atr_multiplier == 2.5

        # Should have very high beta threshold
        assert hasattr(exit_config, "very_high_beta_stop_loss_pct")
        assert exit_config.very_high_beta_stop_loss_pct == -12.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
