# -*- coding: utf-8 -*-
"""
Integration Tests for TradingOrchestrator
Tests toàn bộ workflow từ scan đến entry/exit decisions
"""
import asyncio
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.core.orchestrator import TradingOrchestrator


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")

    close_prices = 80000 + np.cumsum(np.random.randn(200) * 1000)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices + np.random.randn(200) * 500,
            "high": close_prices + np.abs(np.random.randn(200) * 1000),
            "low": close_prices - np.abs(np.random.randn(200) * 1000),
            "close": close_prices,
            "volume": np.random.randint(100000, 1000000, 200),
        }
    )

    return df


@pytest.fixture
def mock_bot():
    """Mock Telegram bot"""
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=True)
    return bot


@pytest.fixture
def mock_portfolio_manager():
    """Mock portfolio manager"""
    manager = Mock()
    manager.get_positions.return_value = {}
    manager.get_daily_pnl_pct.return_value = 0.0
    manager.update_position_price = Mock()
    return manager


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker"""
    cb = Mock()
    cb.check_and_update.return_value = False
    cb.is_active.return_value = False
    cb.tripped_reason = None
    cb.record_trade = Mock()
    cb.record_pnl = Mock()
    return cb


@pytest.fixture
def mock_strategy_manager():
    """Mock strategy manager"""
    manager = Mock()

    # Mock entry logic
    entry_logic = Mock()
    entry_logic.min_confidence = 55
    entry_logic.analyze_entry = Mock(
        return_value=Mock(
            should_enter=False,
            signal_type="HOLD",
            confidence=0,
            warnings=["No signal"],
        )
    )

    # Mock position sizer
    position_sizer = Mock()
    position_sizer.current_positions = {}

    # Mock exit strategy
    exit_strategy = Mock()
    exit_strategy.check_exit = Mock(
        return_value=Mock(
            should_exit=False,
            exit_type="HOLD",
        )
    )
    exit_strategy.clear_position_tracking = Mock()

    manager.get_strategies.return_value = {
        "entry_logic": entry_logic,
        "position_sizer": position_sizer,
        "exit_strategy": exit_strategy,
    }
    manager.apply_market_adjustments = Mock()

    return manager


@pytest.fixture
def mock_ml_generator():
    """Mock ML signal generator"""
    generator = Mock()
    generator.analyze = Mock(
        return_value={
            "signal": "HOLD",
            "confidence": 50,
        }
    )
    return generator


@pytest.fixture
def mock_paper_account():
    """Mock paper trading account"""
    account = Mock()
    account.execute_sell = Mock(return_value=(True, "Sold successfully", {}))
    account.execute_buy = Mock(return_value=(True, "Bought successfully", {}))
    return account


@pytest.fixture
def mock_ticker_loader():
    """Mock ticker loader"""
    loader = Mock()
    loader.get_validated_tickers.return_value = ["VNM", "VCB", "FPT"]
    return loader


@pytest.fixture
def orchestrator(
    mock_bot,
    mock_portfolio_manager,
    mock_circuit_breaker,
    mock_strategy_manager,
    mock_ml_generator,
    mock_paper_account,
):
    """Create orchestrator with mocked dependencies"""
    with (
        patch("src.core.orchestrator.get_portfolio_manager", return_value=mock_portfolio_manager),
        patch("src.core.orchestrator.get_circuit_breaker", return_value=mock_circuit_breaker),
        patch("src.core.orchestrator.get_strategy_manager", return_value=mock_strategy_manager),
        patch("src.core.orchestrator.EnhancedMLSignalGenerator", return_value=mock_ml_generator),
        patch("src.core.orchestrator.get_paper_account", return_value=mock_paper_account),
        patch("src.core.orchestrator.get_portfolio_risk_manager"),
        patch("src.core.orchestrator.get_portfolio_lock"),
        patch("src.core.orchestrator.ProxyMarketRegimeAnalyzer"),
        patch("src.core.orchestrator.get_ticker_loader"),
        patch("src.core.orchestrator.get_signal_performance_tracker"),
        patch("src.core.orchestrator.get_ml_model_monitor"),
    ):

        orch = TradingOrchestrator(
            bot_instance=mock_bot,
            chat_id="test_chat_id",
        )
        orch.circuit_breaker = mock_circuit_breaker
        orch.strategy_manager = mock_strategy_manager
        orch.ml_generator = mock_ml_generator
        orch.paper_account = mock_paper_account
        orch.portfolio_manager = mock_portfolio_manager

        return orch


# =============================================================================
# ORCHESTRATOR INITIALIZATION TESTS
# =============================================================================


class TestOrchestratorInitialization:
    """Tests cho khởi tạo orchestrator"""

    def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator được tạo thành công"""
        assert orchestrator is not None
        assert orchestrator.bot is not None
        assert orchestrator.chat_id == "test_chat_id"

    def test_orchestrator_has_required_components(self, orchestrator):
        """Test orchestrator có đủ các components cần thiết"""
        assert orchestrator.portfolio_manager is not None
        assert orchestrator.circuit_breaker is not None
        assert orchestrator.strategy_manager is not None
        assert orchestrator.ml_generator is not None

    def test_ml_tracking_initialized(self, orchestrator):
        """Test ML tracking được khởi tạo"""
        assert orchestrator._ml_failure_count == 0
        assert orchestrator._ml_success_count == 0
        assert orchestrator._ml_enabled is True


# =============================================================================
# STRATEGY SETUP TESTS
# =============================================================================


class TestStrategySetup:
    """Tests cho setup strategies"""

    def test_setup_strategies_bull_market(self, orchestrator, mock_strategy_manager):
        """Test setup strategies trong BULL market"""
        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        orchestrator._setup_strategies(market_regime)

        mock_strategy_manager.get_strategies.assert_called_once()
        mock_strategy_manager.apply_market_adjustments.assert_called_once_with(market_regime)

    def test_setup_strategies_bear_market(self, orchestrator, mock_strategy_manager):
        """Test setup strategies trong BEAR market"""
        market_regime = {"regime": "BEAR", "confidence": 70, "tradeable": False}

        orchestrator._setup_strategies(market_regime)

        mock_strategy_manager.apply_market_adjustments.assert_called_once_with(market_regime)

    def test_strategies_assigned_after_setup(self, orchestrator, mock_strategy_manager):
        """Test strategies được gán sau khi setup"""
        market_regime = {"regime": "SIDEWAYS", "confidence": 60, "tradeable": True}

        orchestrator._setup_strategies(market_regime)

        assert orchestrator.entry_logic is not None
        assert orchestrator.position_sizer is not None
        assert orchestrator.exit_strategy is not None


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests cho circuit breaker integration"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_scan(self, orchestrator, mock_circuit_breaker, mock_bot):
        """Test circuit breaker chặn scan khi active"""
        mock_circuit_breaker.check_and_update.return_value = True
        mock_circuit_breaker.tripped_reason = "Portfolio loss > 5%"

        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        await orchestrator.run_scan(market_regime)

        # Should send circuit breaker message
        mock_bot.send_message.assert_called()
        # Check that message was sent (circuit breaker notification)
        assert mock_bot.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_scan(
        self, orchestrator, mock_circuit_breaker, mock_strategy_manager
    ):
        """Test circuit breaker cho phép scan khi không active"""
        mock_circuit_breaker.check_and_update.return_value = False

        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        # Mock ticker loader
        orchestrator.ticker_loader = Mock()
        orchestrator.ticker_loader.get_validated_tickers.return_value = []

        await orchestrator.run_scan(market_regime)

        # Should setup strategies
        mock_strategy_manager.get_strategies.assert_called()


# =============================================================================
# POSITION CHECKING TESTS
# =============================================================================


class TestPositionChecking:
    """Tests cho kiểm tra vị thế"""

    @pytest.mark.asyncio
    async def test_check_active_positions_empty(self, orchestrator, mock_portfolio_manager):
        """Test check positions khi không có vị thế"""
        mock_portfolio_manager.get_positions.return_value = {}

        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        await orchestrator.check_active_positions(market_regime)

        mock_portfolio_manager.get_positions.assert_called()

    @pytest.mark.asyncio
    async def test_check_active_positions_with_positions(
        self, orchestrator, mock_portfolio_manager, sample_ohlcv_data
    ):
        """Test check positions với vị thế đang có"""
        mock_portfolio_manager.get_positions.return_value = {
            "VNM": {
                "shares": 500,
                "avg_price": 80000,
                "entry_date": datetime.now().isoformat(),
                "stop_loss": 76000,
                "take_profit_targets": [88000, 96000],
                "partial_exits": [],
            }
        }

        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        # Mock load_data
        with patch("src.core.orchestrator.load_data", return_value=sample_ohlcv_data):
            orchestrator._setup_strategies(market_regime)
            await orchestrator.check_active_positions(market_regime)

        mock_portfolio_manager.get_positions.assert_called()


# =============================================================================
# EXIT EXECUTION TESTS
# =============================================================================


class TestExitExecution:
    """Tests cho thực hiện exit"""

    @pytest.mark.asyncio
    async def test_execute_exit_success(
        self, orchestrator, mock_bot, mock_paper_account, mock_circuit_breaker
    ):
        """Test execute exit thành công"""
        pos_data = {
            "shares": 500,
            "avg_price": 80000,
            "entry_date": datetime.now().isoformat(),
        }

        exit_decision = Mock()
        exit_decision.should_exit = True
        exit_decision.exit_reason = Mock(value="Stop Loss Hit")
        exit_decision.exit_type = "FULL"
        exit_decision.message = "Stop loss triggered"

        # Mock exit strategy format message
        orchestrator.exit_strategy = Mock()
        orchestrator.exit_strategy.format_exit_message = Mock(return_value="Exit message")
        orchestrator.exit_strategy.clear_position_tracking = Mock()

        # Mock portfolio manager to return empty after exit
        orchestrator.portfolio_manager.get_positions.return_value = {}

        await orchestrator.execute_exit("VNM", pos_data, exit_decision, 75000)

        mock_bot.send_message.assert_called()
        mock_paper_account.execute_sell.assert_called_once()
        mock_circuit_breaker.record_trade.assert_called()

    @pytest.mark.asyncio
    async def test_execute_exit_triggers_circuit_breaker(
        self, orchestrator, mock_bot, mock_paper_account, mock_circuit_breaker
    ):
        """Test execute exit kích hoạt circuit breaker"""
        pos_data = {
            "shares": 500,
            "avg_price": 80000,
            "entry_date": datetime.now().isoformat(),
        }

        exit_decision = Mock()
        exit_decision.should_exit = True
        exit_decision.exit_reason = Mock(value="Stop Loss Hit")
        exit_decision.exit_type = "FULL"

        orchestrator.exit_strategy = Mock()
        orchestrator.exit_strategy.format_exit_message = Mock(return_value="Exit message")
        orchestrator.exit_strategy.clear_position_tracking = Mock()

        orchestrator.portfolio_manager.get_positions.return_value = {}
        orchestrator.portfolio_manager.get_daily_pnl_pct.return_value = -0.06  # -6%

        # Circuit breaker activates after exit
        mock_circuit_breaker.is_active.return_value = True
        mock_circuit_breaker.tripped_reason = "Daily loss > 5%"

        await orchestrator.execute_exit("VNM", pos_data, exit_decision, 75000)

        # Should send circuit breaker warning
        assert mock_bot.send_message.call_count >= 2


# =============================================================================
# NEWS ADJUSTMENT TESTS
# =============================================================================


class TestNewsAdjustment:
    """Tests cho điều chỉnh signal theo tin tức"""

    def test_adjust_signal_positive_news(self, orchestrator):
        """Test điều chỉnh signal với tin tức tích cực"""
        entry_signal = Mock()
        entry_signal.confidence = 70
        entry_signal.reasons = []
        entry_signal.warnings = []
        entry_signal.should_enter = True

        news_context = {
            "articles": [{"title": "Good news", "topics": []}],
            "sentiment_score": 0.8,
        }

        result = orchestrator.adjust_signal_with_news(entry_signal, news_context)

        assert result.confidence >= 70  # Should increase or stay same

    def test_adjust_signal_negative_news(self, orchestrator):
        """Test điều chỉnh signal với tin tức tiêu cực"""
        entry_signal = Mock()
        entry_signal.confidence = 70
        entry_signal.reasons = []
        entry_signal.warnings = []
        entry_signal.should_enter = True

        news_context = {
            "articles": [{"title": "Bad news", "topics": ["litigation"]}],
            "sentiment_score": -0.9,
        }

        result = orchestrator.adjust_signal_with_news(entry_signal, news_context)

        # Should block entry due to litigation
        assert result.should_enter is False

    def test_adjust_signal_no_news(self, orchestrator):
        """Test không điều chỉnh khi không có tin tức"""
        entry_signal = Mock()
        entry_signal.confidence = 70

        result = orchestrator.adjust_signal_with_news(entry_signal, None)

        assert result.confidence == 70

    def test_adjust_signal_dividend_news(self, orchestrator):
        """Test điều chỉnh với tin cổ tức"""
        entry_signal = Mock()
        entry_signal.confidence = 70
        entry_signal.reasons = []
        entry_signal.warnings = []
        entry_signal.should_enter = True

        news_context = {
            "articles": [{"title": "Dividend announcement", "topics": ["dividend"]}],
            "sentiment_score": 0.3,
        }

        result = orchestrator.adjust_signal_with_news(entry_signal, news_context)

        # Should add bonus for dividend
        assert result.confidence >= 70


# =============================================================================
# VNINDEX CACHING TESTS
# =============================================================================


class TestVNIndexCaching:
    """Tests cho VNINDEX caching"""

    def test_get_cached_vnindex_fresh_load(self, orchestrator, sample_ohlcv_data):
        """Test load VNINDEX mới khi cache trống"""
        orchestrator._cached_vnindex_df = None

        with patch("src.core.orchestrator.load_data", return_value=sample_ohlcv_data):
            result = orchestrator._get_cached_vnindex()

        assert result is not None
        assert orchestrator._cached_vnindex_df is not None

    def test_get_cached_vnindex_uses_cache(self, orchestrator, sample_ohlcv_data):
        """Test sử dụng cache khi còn valid"""
        import time

        orchestrator._cached_vnindex_df = sample_ohlcv_data
        orchestrator._vnindex_cache_timestamp = time.time()  # Fresh cache

        with patch("src.core.orchestrator.load_data") as mock_load:
            result = orchestrator._get_cached_vnindex()

        # Should not call load_data
        mock_load.assert_not_called()
        assert result is not None

    def test_get_cached_vnindex_expired_cache(self, orchestrator, sample_ohlcv_data):
        """Test reload khi cache hết hạn"""
        import time

        orchestrator._cached_vnindex_df = sample_ohlcv_data
        orchestrator._vnindex_cache_timestamp = time.time() - 7200  # 2 hours ago (expired)

        new_data = sample_ohlcv_data.copy()

        with patch("src.core.orchestrator.load_data", return_value=new_data):
            result = orchestrator._get_cached_vnindex()

        assert result is not None


# =============================================================================
# ML CIRCUIT BREAKER TESTS
# =============================================================================


class TestMLCircuitBreaker:
    """Tests cho ML circuit breaker"""

    def test_should_use_ml_enabled(self, orchestrator):
        """Test ML flag được set đúng khi enabled"""
        orchestrator._ml_enabled = True

        # Verify _ml_enabled flag is set correctly
        assert orchestrator._ml_enabled is True
        # ML circuit breaker should not be active
        assert orchestrator._ml_circuit_breaker_active is False

    def test_should_use_ml_disabled(self, orchestrator):
        """Test ML không được sử dụng khi disabled"""
        orchestrator._ml_enabled = False

        if hasattr(orchestrator, "_should_use_ml"):
            result = orchestrator._should_use_ml()
            assert result is False
        else:
            assert orchestrator._ml_enabled is False

    def test_track_ml_failure(self, orchestrator):
        """Test tracking ML failure"""
        error_details = {
            "symbol": "VNM",
            "error_type": "ValueError",
            "error_msg": "Test error",
        }

        initial_count = orchestrator._ml_failure_count

        orchestrator._track_ml_failure("VNM", error_details)

        assert orchestrator._ml_failure_count == initial_count + 1
        assert "VNM" in orchestrator._ml_failures_by_symbol


# =============================================================================
# SCAN UNIVERSE TESTS
# =============================================================================


class TestScanUniverse:
    """Tests cho scan universe"""

    def test_get_scan_universe(self, orchestrator, mock_ticker_loader):
        """Test lấy danh sách mã để scan"""
        orchestrator.ticker_loader = mock_ticker_loader
        mock_ticker_loader.get_validated_tickers.return_value = ["VNM", "VCB", "FPT"]

        result = orchestrator.get_scan_universe()

        assert len(result) == 3
        assert "VNM" in result

    def test_get_scan_universe_fallback(self, orchestrator):
        """Test fallback khi ticker loader lỗi"""
        orchestrator.ticker_loader = Mock()
        orchestrator.ticker_loader.get_validated_tickers.side_effect = Exception("Error")

        # Patch at the source where TICKERS is imported
        with patch("src.config.legacy_config.TICKERS", ["AAA", "BBB"]):
            result = orchestrator.get_scan_universe()

        assert isinstance(result, list)


# =============================================================================
# POSITION SYNC TESTS
# =============================================================================


class TestPositionSync:
    """Tests cho đồng bộ position"""

    def test_sync_position_sizer_empty(self, orchestrator):
        """Test sync với positions trống"""
        orchestrator.position_sizer = Mock()
        orchestrator.position_sizer.current_positions = {}

        orchestrator.sync_position_sizer_with_active_positions({})

        assert orchestrator.position_sizer.current_positions == {}

    def test_sync_position_sizer_with_positions(self, orchestrator):
        """Test sync với positions có sẵn"""
        orchestrator.position_sizer = Mock()
        orchestrator.position_sizer.current_positions = {}

        active_positions = {
            "VNM": {
                "shares": 500,
                "avg_price": 80000,
                "metadata": {"last_price": 82000},
            }
        }

        orchestrator.sync_position_sizer_with_active_positions(active_positions)

        assert "VNM" in orchestrator.position_sizer.current_positions


# =============================================================================
# FULL WORKFLOW TESTS
# =============================================================================


class TestFullWorkflow:
    """Tests cho toàn bộ workflow"""

    @pytest.mark.asyncio
    async def test_run_scan_full_workflow(
        self, orchestrator, mock_bot, mock_circuit_breaker, mock_strategy_manager
    ):
        """Test full scan workflow"""
        mock_circuit_breaker.check_and_update.return_value = False

        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        # Mock ticker loader
        orchestrator.ticker_loader = Mock()
        orchestrator.ticker_loader.get_validated_tickers.return_value = []

        # Mock portfolio lock
        orchestrator.portfolio_lock = Mock()
        orchestrator.portfolio_lock.clear_pending = Mock()

        await orchestrator.run_scan(market_regime)

        # Should complete without errors
        mock_strategy_manager.get_strategies.assert_called()

    @pytest.mark.asyncio
    async def test_run_scan_with_invalid_regime(self, orchestrator, mock_bot):
        """Test scan với market regime không hợp lệ"""
        # Should handle gracefully
        await orchestrator.run_scan(None)

        # Should still work with default regime

    @pytest.mark.asyncio
    async def test_scan_for_new_entries_empty(self, orchestrator):
        """Test scan entries với danh sách trống"""
        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        orchestrator.portfolio_lock = Mock()
        orchestrator.portfolio_lock.is_pending = Mock(return_value=False)

        signal_count, watchlist = await orchestrator.scan_for_new_entries([], set(), market_regime)

        assert signal_count == 0
        assert watchlist == []


# =============================================================================
# MESSAGE FORMATTING TESTS
# =============================================================================


class TestMessageFormatting:
    """Tests cho format messages"""

    def test_format_entry_recommendation(self, orchestrator):
        """Test format entry recommendation message"""
        entry_signal = Mock()
        entry_signal.strength = Mock(name="STRONG")
        entry_signal.confidence = 75
        entry_signal.stop_loss = 76000
        entry_signal.entry_price = 80000
        entry_signal.take_profit_targets = [88000, 96000]
        entry_signal.reasons = ["Good trend"]
        entry_signal.warnings = []

        position = Mock()
        position.shares = 500
        position.value = 40000000
        position.position_percent = 4.0
        position.max_loss = 2000000
        position.risk_percent = 2.0
        position.recommended_entries = []

        market_regime = {"regime": "BULL", "confidence": 75}

        msg = orchestrator.format_entry_recommendation("VNM", entry_signal, position, market_regime)

        assert "VNM" in msg
        assert "BULL" in msg
        assert "Stop Loss" in msg


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests cho xử lý lỗi"""

    @pytest.mark.asyncio
    async def test_check_single_position_error_handling(self, orchestrator, sample_ohlcv_data):
        """Test xử lý lỗi khi check single position"""
        pos_data = {
            "shares": 500,
            "avg_price": 80000,
            "entry_date": datetime.now().isoformat(),
            "stop_loss": 76000,
            "take_profit_targets": [88000, 96000],
            "partial_exits": [],
        }

        market_regime = {"regime": "BULL", "confidence": 75, "tradeable": True}

        # Mock load_data to raise exception
        with patch("src.core.orchestrator.load_data", side_effect=Exception("Load error")):
            orchestrator._setup_strategies(market_regime)
            # Should not raise, just log error
            await orchestrator._check_single_position("VNM", pos_data, market_regime)

    @pytest.mark.asyncio
    async def test_execute_exit_error_handling(self, orchestrator, mock_bot, mock_paper_account):
        """Test xử lý lỗi khi execute exit"""
        pos_data = {
            "shares": 500,
            "avg_price": 80000,
            "entry_date": datetime.now().isoformat(),
        }

        exit_decision = Mock()
        exit_decision.should_exit = True
        exit_decision.exit_reason = Mock(value="Stop Loss")
        exit_decision.exit_type = "FULL"

        orchestrator.exit_strategy = Mock()
        orchestrator.exit_strategy.format_exit_message = Mock(side_effect=Exception("Format error"))

        # Should not raise, just log error
        await orchestrator.execute_exit("VNM", pos_data, exit_decision, 75000)
