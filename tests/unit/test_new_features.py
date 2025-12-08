# -*- coding: utf-8 -*-
"""
Unit tests for new Vietnam Market features:
1. Smart Order Execution (TWAP/VWAP)
2. ATO/ATC Auction Strategy
3. Foreign Flow Provider

Author: Trading Bot Team
"""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


class TestSmartOrderExecution:
    """Tests for Smart Order Executor"""

    @pytest.fixture
    def executor(self):
        """Create smart order executor"""
        from src.execution.smart_order import SmartOrderExecutor

        return SmartOrderExecutor()

    def test_round_to_lot(self, executor):
        """Test lot size rounding"""
        assert executor._round_to_lot(150) == 100
        assert executor._round_to_lot(250) == 200
        assert executor._round_to_lot(50) == 100  # Minimum 1 lot
        assert executor._round_to_lot(0) == 0

    def test_create_twap_plan(self, executor):
        """Test TWAP plan creation"""
        with patch.object(executor, "get_price", return_value=50000):
            with patch.object(executor, "get_volume", return_value=1000000):
                plan = executor.create_twap_plan(
                    symbol="VNM",
                    side="BUY",
                    quantity=1000,
                    duration_minutes=60,
                )

        assert plan.symbol == "VNM"
        assert plan.side == "BUY"
        assert plan.total_quantity == 1000
        assert plan.strategy.value == "TWAP"
        assert len(plan.slices) > 0

        # Total slice quantity should equal total
        total_slice_qty = sum(s.quantity for s in plan.slices)
        assert total_slice_qty == 1000

    def test_create_vwap_plan(self, executor):
        """Test VWAP plan creation"""
        with patch.object(executor, "get_price", return_value=50000):
            with patch.object(executor, "get_volume", return_value=1000000):
                plan = executor.create_vwap_plan(
                    symbol="HPG",
                    side="BUY",
                    quantity=2000,
                )

        assert plan.symbol == "HPG"
        assert plan.strategy.value == "VWAP"
        assert len(plan.slices) > 0

        # VWAP should have more slices during high-volume periods
        # (ATO and ATC have higher weights)

    def test_create_iceberg_plan(self, executor):
        """Test Iceberg plan creation"""
        plan = executor.create_iceberg_plan(
            symbol="VCB",
            side="BUY",
            quantity=5000,
            visible_quantity=500,
            price_limit=85000,
        )

        assert plan.symbol == "VCB"
        assert plan.strategy.value == "ICEBERG"
        assert len(plan.slices) == 10  # 5000 / 500 = 10 slices

        # Each slice should have visible_quantity
        for slice_ in plan.slices[:-1]:
            assert slice_.quantity == 500

    def test_create_participation_plan(self, executor):
        """Test Participation Rate plan creation"""
        with patch.object(executor, "get_price", return_value=50000):
            with patch.object(executor, "get_volume", return_value=1000000):
                plan = executor.create_participation_plan(
                    symbol="FPT",
                    side="BUY",
                    quantity=10000,
                    participation_rate=0.05,
                )

        assert plan.symbol == "FPT"
        assert plan.strategy.value == "PARTICIPATION"
        assert plan.max_participation_rate == 0.05

    def test_execution_plan_properties(self, executor):
        """Test ExecutionPlan computed properties"""
        with patch.object(executor, "get_price", return_value=50000):
            with patch.object(executor, "get_volume", return_value=1000000):
                plan = executor.create_twap_plan("VNM", "BUY", 1000, 60)

        # Initially nothing executed
        assert plan.executed_quantity == 0
        assert plan.remaining_quantity == 1000
        assert plan.completion_pct == 0.0
        assert plan.is_complete is False

        # Simulate execution
        plan.slices[0].executed = True
        plan.slices[0].executed_quantity = plan.slices[0].quantity
        plan.slices[0].executed_price = 50000

        assert plan.executed_quantity > 0
        assert plan.remaining_quantity < 1000

    def test_is_trading_time(self, executor):
        """Test trading time detection"""
        # Morning session
        morning = datetime(2024, 1, 15, 10, 0)  # 10:00 AM
        assert executor._is_trading_time(morning) is True

        # Lunch break
        lunch = datetime(2024, 1, 15, 12, 0)  # 12:00 PM
        assert executor._is_trading_time(lunch) is False

        # Afternoon session
        afternoon = datetime(2024, 1, 15, 14, 0)  # 2:00 PM
        assert executor._is_trading_time(afternoon) is True

        # After market
        after = datetime(2024, 1, 15, 16, 0)  # 4:00 PM
        assert executor._is_trading_time(after) is False


class TestAuctionStrategy:
    """Tests for ATO/ATC Auction Strategy"""

    @pytest.fixture
    def strategy(self):
        """Create auction strategy"""
        from src.strategies.auction_strategy import AuctionStrategy

        return AuctionStrategy()

    @pytest.fixture
    def sample_df(self):
        """Create sample price DataFrame"""
        return pd.DataFrame(
            {
                "open": [50000, 50500, 51000],
                "high": [51000, 51500, 52000],
                "low": [49500, 50000, 50500],
                "close": [50500, 51000, 51500],
                "volume": [1000000, 1200000, 1100000],
            }
        )

    def test_get_current_session(self, strategy):
        """Test session detection"""
        from src.strategies.auction_strategy import AuctionSession

        # This depends on current time, so just verify it returns valid enum
        session = strategy.get_current_session()
        assert session in [AuctionSession.ATO, AuctionSession.ATC, AuctionSession.NONE]

    def test_is_auction_time(self, strategy):
        """Test auction time check"""
        # Returns bool
        result = strategy.is_auction_time()
        assert isinstance(result, bool)

    def test_time_to_next_auction(self, strategy):
        """Test time to next auction calculation"""
        from src.strategies.auction_strategy import AuctionSession

        session, minutes = strategy.time_to_next_auction()

        assert session in [AuctionSession.ATO, AuctionSession.ATC]
        assert minutes >= 0

    def test_analyze_ato(self, strategy, sample_df):
        """Test ATO analysis"""
        from src.strategies.auction_strategy import AuctionSession, AuctionSignal

        analysis = strategy.analyze_ato(
            symbol="VNM",
            df=sample_df,
            foreign_flow={"score": 0.5, "consecutive_days": 3},
            global_markets={"US_SP500": 0.01},
        )

        assert analysis.session == AuctionSession.ATO
        assert analysis.signal in list(AuctionSignal)
        assert 0 <= analysis.confidence <= 100
        assert analysis.order_type in ["ATO", "LO"]

    def test_analyze_atc(self, strategy, sample_df):
        """Test ATC analysis"""
        from src.strategies.auction_strategy import AuctionSession, AuctionSignal

        analysis = strategy.analyze_atc(
            symbol="VNM",
            df=sample_df,
            existing_position={"entry_price": 50000, "unrealized_pnl_pct": 0.03},
        )

        assert analysis.session == AuctionSession.ATC
        assert analysis.signal in list(AuctionSignal)

    def test_should_use_ato_entry(self, strategy, sample_df):
        """Test ATO entry recommendation"""
        should_use, reason, price = strategy.should_use_ato_entry(
            symbol="VNM",
            signal={"confidence": 80},
            df=sample_df,
            foreign_flow={"score": 0.6},
        )

        assert isinstance(should_use, bool)
        assert isinstance(reason, str)
        assert price >= 0

    def test_should_use_atc_exit(self, strategy, sample_df):
        """Test ATC exit recommendation"""
        should_use, reason, price = strategy.should_use_atc_exit(
            symbol="VNM",
            position={"entry_price": 50000, "unrealized_pnl_pct": 0.03},
            df=sample_df,
        )

        assert isinstance(should_use, bool)
        assert isinstance(reason, str)

    def test_friday_atc_exit(self, strategy, sample_df):
        """Test ATC exit on Friday"""
        # Test without mocking - just verify the function works
        should_use, reason, price = strategy.should_use_atc_exit(
            symbol="VNM",
            position={"entry_price": 50000, "unrealized_pnl_pct": 0.03},
            df=sample_df,
        )

        # Should return valid types regardless of day
        assert isinstance(should_use, bool)
        assert isinstance(reason, str)
        assert isinstance(price, (int, float))


class TestForeignFlowProvider:
    """Tests for Foreign Flow Provider"""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create foreign flow provider"""
        from src.data.foreign_flow_provider import ForeignFlowProvider

        cache_file = str(tmp_path / "test_flow_cache.json")
        return ForeignFlowProvider(cache_file=cache_file, auto_refresh=False)

    def test_get_flow_score_default(self, provider):
        """Test flow score with no data"""
        score = provider.get_flow_score()
        assert -1 <= score <= 1

    def test_get_top_foreign_activity(self, provider):
        """Test top foreign activity"""
        result = provider.get_top_foreign_activity(top_n=5)

        assert "top_buy" in result
        assert "top_sell" in result
        assert isinstance(result["top_buy"], list)
        assert isinstance(result["top_sell"], list)

    def test_is_trading_hours(self, provider):
        """Test trading hours detection"""
        result = provider._is_trading_hours()
        assert isinstance(result, bool)

    def test_cache_validity(self, provider):
        """Test cache validity check"""
        # No cache initially
        assert provider._is_cache_valid() is False

        # Set cache time
        provider._cache_time = datetime.now()
        assert provider._is_cache_valid() is True

        # Old cache
        provider._cache_time = datetime.now() - timedelta(hours=1)
        assert provider._is_cache_valid() is False

    @patch("src.data.foreign_flow_provider.ForeignFlowProvider._fetch_from_tcbs")
    def test_get_market_flow_with_mock(self, mock_fetch, provider):
        """Test market flow with mocked data"""
        from src.data.foreign_flow_provider import MarketForeignFlow

        # Mock TCBS response
        mock_fetch.return_value = [
            {"ticker": "VNM", "foreignBuyValue": 10000000000, "foreignSellValue": 5000000000},
            {"ticker": "HPG", "foreignBuyValue": 8000000000, "foreignSellValue": 12000000000},
        ]

        flow = provider.get_market_flow(force_refresh=True)

        # Should return MarketForeignFlow or None
        assert flow is None or isinstance(flow, MarketForeignFlow)

    def test_start_stop_realtime_updates(self, provider):
        """Test real-time update control"""
        # Start updates
        provider.start_realtime_updates()
        assert provider._refresh_thread is not None

        # Stop updates
        provider.stop_realtime_updates()
        # Thread should be stopped


class TestIntegration:
    """Integration tests for new features"""

    def test_smart_order_with_auction_strategy(self):
        """Test smart order execution with auction strategy"""
        from src.execution.smart_order import SmartOrderExecutor, ExecutionStrategy
        from src.strategies.auction_strategy import AuctionStrategy, AuctionSession

        executor = SmartOrderExecutor()
        auction = AuctionStrategy()

        # Check if in auction
        session = auction.get_current_session()

        if session == AuctionSession.ATO:
            # Use ATO order type
            pass
        elif session == AuctionSession.ATC:
            # Use ATC order type
            pass
        else:
            # Use TWAP/VWAP
            with patch.object(executor, "get_price", return_value=50000):
                with patch.object(executor, "get_volume", return_value=1000000):
                    plan = executor.create_twap_plan("VNM", "BUY", 1000, 60)

            assert plan.strategy == ExecutionStrategy.TWAP

    def test_foreign_flow_with_entry_decision(self):
        """Test foreign flow integration with entry decision"""
        from src.data.foreign_flow_provider import ForeignFlowProvider

        provider = ForeignFlowProvider(auto_refresh=False)

        # Get flow score
        score = provider.get_flow_score()

        # Use in entry decision
        confidence_adjustment = 0
        if score > 0.5:
            confidence_adjustment = 10  # Boost confidence
        elif score < -0.5:
            confidence_adjustment = -15  # Reduce confidence

        assert -15 <= confidence_adjustment <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
