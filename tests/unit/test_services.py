"""
Unit tests for services
Tests for RiskManagementService, EntrySignalService, ExitManagementService, NotificationService
"""

import pytest
import asyncio
from typing import Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import pandas as pd

# Import services
from services.risk_service import RiskManagementService
from services.entry_service import EntrySignalService
from services.exit_service import ExitManagementService
from services.notification_service import NotificationService


class TestRiskManagementService:
    """Test RiskManagementService"""

    @pytest.fixture
    def risk_service(self):
        """Create risk service with mocked dependencies"""
        with patch("services.risk_service.get_circuit_breaker"), patch(
            "services.risk_service.get_emergency_stop"
        ), patch("services.risk_service.get_portfolio_manager"):
            service = RiskManagementService()

            # Mock circuit breaker
            service.circuit_breaker = Mock()
            service.circuit_breaker.can_trade.return_value = (True, "OK")
            service.circuit_breaker.check_and_update.return_value = False
            service.circuit_breaker.tripped = False
            service.circuit_breaker.tripped_reason = ""

            # Mock emergency stop
            service.emergency_stop = Mock()
            service.emergency_stop.can_trade.return_value = (True, "OK")
            service.emergency_stop.is_emergency_active.return_value = False

            # Mock portfolio manager
            service.portfolio_manager = Mock()
            service.portfolio_manager.get_daily_pnl_pct.return_value = 0.0

            return service

    @pytest.mark.asyncio
    async def test_can_trade_success(self, risk_service):
        """Test can_trade when all checks pass"""
        can_trade, reason = await risk_service.can_trade()

        assert can_trade == True
        assert reason == "✅ OK to trade"

    @pytest.mark.asyncio
    async def test_can_trade_emergency_stop(self, risk_service):
        """Test can_trade when emergency stop active"""
        risk_service.emergency_stop.can_trade.return_value = (False, "Emergency")

        can_trade, reason = await risk_service.can_trade()

        assert can_trade == False
        assert "Emergency" in reason

    @pytest.mark.asyncio
    async def test_can_trade_circuit_breaker(self, risk_service):
        """Test can_trade when circuit breaker tripped"""
        risk_service.circuit_breaker.can_trade.return_value = (False, "Circuit breaker")

        can_trade, reason = await risk_service.can_trade()

        assert can_trade == False
        assert "Circuit breaker" in reason

    @pytest.mark.asyncio
    async def test_check_and_update_circuit_breaker(self, risk_service):
        """Test circuit breaker check and update"""
        tripped = await risk_service.check_and_update_circuit_breaker(
            portfolio_pnl_pct=-0.02, vnindex_change_pct=-0.01
        )

        assert tripped == False
        risk_service.circuit_breaker.check_and_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_tripped(self, risk_service):
        """Test when circuit breaker trips"""
        risk_service.circuit_breaker.check_and_update.return_value = True
        risk_service.circuit_breaker.tripped = True
        risk_service.circuit_breaker.tripped_reason = "Max loss reached"

        tripped = await risk_service.check_and_update_circuit_breaker(
            portfolio_pnl_pct=-0.06, vnindex_change_pct=-0.03
        )

        assert tripped == True

    def test_record_trade(self, risk_service):
        """Test recording a trade"""
        risk_service.record_trade(1_000_000)

        risk_service.circuit_breaker.record_trade.assert_called_once_with(1_000_000)

    def test_get_risk_status(self, risk_service):
        """Test getting risk status"""
        risk_service.circuit_breaker.get_daily_stats.return_value = {
            "trades_count": 5,
            "total_loss": 0,
            "total_profit": 5_000_000,
        }

        status = risk_service.get_risk_status()

        assert "circuit_breaker" in status
        assert "emergency_stop" in status
        assert "portfolio" in status


class TestEntrySignalService:
    """Test EntrySignalService"""

    @pytest.fixture
    def entry_service(self):
        """Create entry service with mocked dependencies"""
        with patch("services.entry_service.EnhancedMLSignalGenerator"), patch(
            "services.entry_service.ImprovedEntryLogic"
        ), patch("services.entry_service.EnhancedPositionSizer"), patch(
            "services.entry_service.get_portfolio_lock"
        ):
            service = EntrySignalService()

            # Mock ML generator
            service.ml_generator = Mock()
            service.ml_generator.analyze.return_value = {
                "signal": "BUY",
                "confidence": 75,
            }

            # Mock entry logic
            service.entry_logic = Mock()
            mock_signal = Mock()
            mock_signal.should_enter = True
            mock_signal.entry_price = 100_000
            mock_signal.stop_loss = 93_000
            mock_signal.take_profit_targets = [110_000, 120_000]
            mock_signal.confidence = 75
            mock_signal.strength = Mock(name="STRONG")
            service.entry_logic.analyze_entry.return_value = mock_signal

            # Mock position sizer
            service.position_sizer = Mock()
            mock_position = Mock()
            mock_position.shares = 100
            service.position_sizer.calculate_position_size.return_value = mock_position

            # Mock portfolio lock
            service.portfolio_lock = Mock()
            service.portfolio_lock.is_pending.return_value = False

            return service

    @pytest.mark.asyncio
    async def test_scan_single_ticker_success(self, entry_service):
        """Test scanning a single ticker successfully"""
        with patch("services.entry_service.load_data") as mock_load, patch(
            "services.entry_service.DataValidator"
        ):

            # Mock data
            import pandas as pd

            mock_df = pd.DataFrame(
                {
                    "open": [100] * 60,
                    "high": [105] * 60,
                    "low": [99] * 60,
                    "close": [103] * 60,
                    "volume": [1000] * 60,
                }
            )
            mock_load.return_value = mock_df

            result = await entry_service._scan_single_ticker(
                symbol="VNM",
                existing_symbols=set(),
                market_regime={"regime": "BULL"},
                vnindex_df=None,
            )

            assert result is not None
            assert result["symbol"] == "VNM"
            assert "signal" in result
            assert "position_size" in result

    @pytest.mark.asyncio
    async def test_scan_skip_existing_symbol(self, entry_service):
        """Test skipping symbols already in portfolio"""
        result = await entry_service._scan_single_ticker(
            symbol="VNM",
            existing_symbols={"VNM"},
            market_regime={"regime": "BULL"},
            vnindex_df=None,
        )

        assert result is None

    def test_filter_and_rank_signals(self, entry_service):
        """Test filtering and ranking signals"""
        # Create mock signals
        signals = []
        for i in range(10):
            signal = {
                "symbol": f"TEST{i}",
                "signal": Mock(confidence=50 + i * 5, strength=Mock(value=3)),
            }
            signals.append(signal)

        top_signals = entry_service.filter_and_rank_signals(signals, max_signals=5)

        assert len(top_signals) == 5
        # Should be sorted by score (confidence * strength)
        assert (
            top_signals[0]["signal"].confidence >= top_signals[-1]["signal"].confidence
        )


class TestExitManagementService:
    """Test ExitManagementService"""

    @pytest.fixture
    def exit_service(self):
        """Create exit service with mocked dependencies"""
        with patch("services.exit_service.ImprovedExitStrategy"), patch(
            "services.exit_service.EnhancedMLSignalGenerator"
        ), patch("services.exit_service.get_portfolio_manager"), patch(
            "services.exit_service.get_paper_account"
        ):
            service = ExitManagementService()

            # Mock exit strategy
            service.exit_strategy = Mock()
            mock_decision = Mock()
            mock_decision.should_exit = True
            mock_decision.exit_type = "FULL"
            mock_decision.exit_reason = Mock(value="Stop Loss")
            service.exit_strategy.check_exit.return_value = mock_decision

            # Mock ML generator
            service.ml_generator = Mock()
            service.ml_generator.analyze.return_value = {"signal": "SELL"}

            # Mock portfolio manager
            service.portfolio_manager = Mock()
            service.portfolio_manager.get_positions.return_value = {
                "VNM": {
                    "avg_price": 80_000,
                    "shares": 100,
                    "entry_date": datetime.now().isoformat(),
                }
            }

            # Mock paper account
            service.paper_account = Mock()
            service.paper_account.execute_sell.return_value = (True, "Success", {})

            return service

    @pytest.mark.asyncio
    async def test_check_single_position_exit(self, exit_service):
        """Test checking a single position for exit"""
        with patch("services.exit_service.load_data") as mock_load, patch(
            "services.exit_service.DataValidator"
        ):

            # Mock data
            import pandas as pd

            mock_df = pd.DataFrame(
                {
                    "open": [100] * 60,
                    "high": [105] * 60,
                    "low": [99] * 60,
                    "close": [75_000] * 60,  # Below entry
                    "volume": [1000] * 60,
                }
            )
            mock_load.return_value = mock_df

            result = await exit_service._check_single_position(
                symbol="VNM",
                pos_data={
                    "avg_price": 80_000,
                    "shares": 100,
                    "entry_date": datetime.now().isoformat(),
                    "stop_loss": 76_000,
                },
                market_regime={"regime": "BULL"},
                vnindex_df=None,
            )

            assert result is not None
            assert result["symbol"] == "VNM"
            assert "decision" in result

    @pytest.mark.asyncio
    async def test_execute_exit_success(self, exit_service):
        """Test executing an exit successfully"""
        exit_decision = {
            "symbol": "VNM",
            "decision": Mock(exit_type="FULL", exit_reason=Mock(value="Stop Loss")),
            "position": {"avg_price": 80_000, "shares": 100},
            "current_price": 75_000,
        }

        success = await exit_service.execute_exit(
            symbol="VNM", exit_decision=exit_decision, current_price=75_000
        )

        assert success == True
        exit_service.paper_account.execute_sell.assert_called_once()


class TestNotificationService:
    """Test NotificationService"""

    @pytest.fixture
    def notification_service(self):
        """Create notification service with mocked bot"""
        mock_bot = AsyncMock()
        service = NotificationService(mock_bot, "123456")
        return service

    @pytest.mark.asyncio
    async def test_send_scan_start(self, notification_service):
        """Test sending scan start notification"""
        await notification_service.send_scan_start(
            ticker_count=40, market_regime={"regime": "BULL", "confidence": 75}
        )

        notification_service.bot.send_message.assert_called_once()
        call_args = notification_service.bot.send_message.call_args
        assert "BULL" in call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_send_entry_signal(self, notification_service):
        """Test sending entry signal notification"""
        signal_data = {
            "symbol": "VNM",
            "signal": Mock(
                entry_price=100_000,
                stop_loss=93_000,
                take_profit_targets=[110_000, 120_000],
                confidence=75,
                strength=Mock(name="STRONG"),
                reasons=["Trend up", "Volume confirm"],
            ),
            "position_size": Mock(
                shares=100, value=10_000_000, risk_amount=700_000, risk_percent=0.7
            ),
        }

        await notification_service.send_entry_signal(signal_data)

        notification_service.bot.send_message.assert_called_once()
        call_args = notification_service.bot.send_message.call_args
        assert "VNM" in call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_send_exit_signal(self, notification_service):
        """Test sending exit signal notification"""
        exit_data = {
            "symbol": "VNM",
            "decision": Mock(
                exit_type="FULL",
                exit_reason=Mock(value="Stop Loss"),
                exit_price=75_000,
                expected_pnl_percent=-6.25,
                urgency=5,
                message="Stop loss hit",
            ),
        }

        await notification_service.send_exit_signal(exit_data)

        notification_service.bot.send_message.assert_called_once()
        call_args = notification_service.bot.send_message.call_args
        assert "VNM" in call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_send_risk_alert(self, notification_service):
        """Test sending risk alert"""
        await notification_service.send_risk_alert(
            alert_type="CIRCUIT_BREAKER", message="Max loss per day reached"
        )

        notification_service.bot.send_message.assert_called_once()
        call_args = notification_service.bot.send_message.call_args
        assert "CIRCUIT_BREAKER" in call_args[1]["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
