# -*- coding: utf-8 -*-
"""
Unit Tests for EntrySignalService
Tests cho service xử lý entry signals
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Mock all problematic modules before importing services
mock_modules = {
    "circuit_breaker": MagicMock(),
    "emergency_stop": MagicMock(),
    "portfolio_manager": MagicMock(),
}

# Apply mocks to sys.modules
for mod_name, mock_mod in mock_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock_mod

# Now import entry_service directly without going through services.__init__
import importlib.util

spec = importlib.util.spec_from_file_location(
    "entry_service",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
        "services",
        "entry_service.py",
    ),
)
entry_service_module = importlib.util.module_from_spec(spec)

# Mock dependencies before loading
with patch.dict(
    "sys.modules",
    {
        "src.data.loader": MagicMock(),
        "src.config.exceptions": MagicMock(DataQualityError=Exception),
        "src.strategies.entry_logic": MagicMock(),
        "src.ml.signals.enhanced": MagicMock(),
        "src.portfolio.lock": MagicMock(),
        "src.strategies.position_sizing": MagicMock(),
        "src.utils.validation": MagicMock(),
    },
):
    spec.loader.exec_module(entry_service_module)

EntrySignalService = entry_service_module.EntrySignalService
get_entry_service = entry_service_module.get_entry_service


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_df():
    """Sample OHLCV DataFrame"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    close_prices = 80000 + np.cumsum(np.random.randn(200) * 1000)

    return pd.DataFrame(
        {
            "time": dates,
            "open": close_prices + np.random.randn(200) * 500,
            "high": close_prices + np.abs(np.random.randn(200) * 1000),
            "low": close_prices - np.abs(np.random.randn(200) * 1000),
            "close": close_prices,
            "volume": np.random.randint(100000, 1000000, 200),
        }
    )


@pytest.fixture
def mock_entry_signal():
    """Mock entry signal"""
    signal = Mock()
    signal.should_enter = True
    signal.confidence = 75
    signal.strength = Mock(name="STRONG", value=4)
    signal.entry_price = 80000
    signal.stop_loss = 76000
    signal.take_profit_targets = [88000, 96000]
    signal.warnings = []
    signal.reasons = []
    return signal


@pytest.fixture
def mock_position_size():
    """Mock position size"""
    position = Mock()
    position.shares = 500
    position.value = 40000000
    position.warnings = []
    return position


@pytest.fixture
def bull_market_regime():
    """Bull market regime"""
    return {"regime": "BULL", "confidence": 75, "tradeable": True}


@pytest.fixture
def entry_service():
    """Create EntrySignalService with mocked dependencies"""
    # Disable optional features to simplify testing
    original_per_symbol_cb = entry_service_module.PER_SYMBOL_CB_AVAILABLE
    original_vn_validator = entry_service_module.VN_MARKET_VALIDATOR_AVAILABLE
    original_t2_settlement = entry_service_module.T2_SETTLEMENT_AVAILABLE

    entry_service_module.PER_SYMBOL_CB_AVAILABLE = False
    entry_service_module.VN_MARKET_VALIDATOR_AVAILABLE = False
    entry_service_module.T2_SETTLEMENT_AVAILABLE = False

    try:
        with (
            patch.object(entry_service_module, "EnhancedMLSignalGenerator") as mock_ml,
            patch.object(entry_service_module, "ImprovedEntryLogic") as mock_entry,
            patch.object(entry_service_module, "EnhancedPositionSizer") as mock_sizer,
            patch.object(entry_service_module, "get_portfolio_lock") as mock_lock,
            patch.object(entry_service_module, "get_config") as mock_config,
        ):
            # Mock config
            mock_cfg = Mock()
            mock_cfg.trading.min_confidence = 55
            mock_cfg.trading.min_risk_reward = 1.8
            mock_config.return_value = mock_cfg

            mock_lock.return_value = Mock()
            mock_lock.return_value.is_pending.return_value = False
            mock_lock.return_value.add_pending = Mock()

            service = EntrySignalService()
            service.ml_generator = mock_ml.return_value
            service.entry_logic = mock_entry.return_value
            service.position_sizer = mock_sizer.return_value
            service.portfolio_lock = mock_lock.return_value

            yield service
    finally:
        # Restore original values
        entry_service_module.PER_SYMBOL_CB_AVAILABLE = original_per_symbol_cb
        entry_service_module.VN_MARKET_VALIDATOR_AVAILABLE = original_vn_validator
        entry_service_module.T2_SETTLEMENT_AVAILABLE = original_t2_settlement


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestEntryServiceInit:
    """Tests cho khởi tạo EntrySignalService"""

    def test_init_creates_components(self):
        """Test khởi tạo tạo đủ các components"""
        with (
            patch("src.services.entry_service.EnhancedMLSignalGenerator"),
            patch("src.services.entry_service.ImprovedEntryLogic"),
            patch("src.services.entry_service.EnhancedPositionSizer"),
            patch("src.services.entry_service.get_portfolio_lock"),
        ):

            service = EntrySignalService()

            assert service.ml_generator is not None
            assert service.entry_logic is not None
            assert service.position_sizer is not None
            assert service.portfolio_lock is not None

    def test_get_entry_service_singleton(self):
        """Test singleton pattern"""
        # Reset singleton
        entry_service_module._entry_service = None

        with (
            patch.object(entry_service_module, "EnhancedMLSignalGenerator"),
            patch.object(entry_service_module, "ImprovedEntryLogic"),
            patch.object(entry_service_module, "EnhancedPositionSizer"),
            patch.object(entry_service_module, "get_portfolio_lock"),
        ):

            service1 = get_entry_service()
            service2 = get_entry_service()

            assert service1 is service2


# =============================================================================
# SCAN FOR ENTRIES TESTS
# =============================================================================


class TestScanForEntries:
    """Tests cho scan_for_entries"""

    @pytest.mark.asyncio
    async def test_scan_empty_tickers(self, entry_service, bull_market_regime):
        """Test scan với danh sách tickers trống"""
        result = await entry_service.scan_for_entries(
            tickers=[],
            existing_symbols=set(),
            market_regime=bull_market_regime,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_skips_existing_symbols(self, entry_service, sample_df, bull_market_regime):
        """Test scan bỏ qua symbols đã có trong portfolio"""
        with patch("src.services.entry_service.load_data", return_value=sample_df):
            result = await entry_service.scan_for_entries(
                tickers=["VNM", "VCB"],
                existing_symbols={"VNM", "VCB"},  # Cả 2 đều đã có
                market_regime=bull_market_regime,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_skips_pending_symbols(self, entry_service, sample_df, bull_market_regime):
        """Test scan bỏ qua symbols đang pending"""
        entry_service.portfolio_lock.is_pending.return_value = True

        with patch("src.services.entry_service.load_data", return_value=sample_df):
            result = await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_returns_valid_signals(
        self, entry_service, sample_df, mock_entry_signal, mock_position_size, bull_market_regime
    ):
        """Test scan trả về signals hợp lệ"""
        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = mock_position_size

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            result = await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert len(result) == 1
        assert result[0]["symbol"] == "VNM"
        assert result[0]["signal"] == mock_entry_signal

    @pytest.mark.asyncio
    async def test_scan_handles_data_validation_error(self, entry_service, bull_market_regime):
        """Test scan xử lý lỗi data validation"""
        # Use the DataQualityError from the module's namespace
        DataQualityError = entry_service_module.DataQualityError

        entry_service.portfolio_lock.is_pending.return_value = False

        with (
            patch.object(entry_service_module, "load_data", return_value=pd.DataFrame()),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.side_effect = DataQualityError("Invalid data")

            result = await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_skips_no_entry_signals(self, entry_service, sample_df, bull_market_regime):
        """Test scan bỏ qua signals không nên enter"""
        no_entry_signal = Mock()
        no_entry_signal.should_enter = False

        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "HOLD", "confidence": 50}
        entry_service.entry_logic.analyze_entry.return_value = no_entry_signal

        with (
            patch("src.services.entry_service.load_data", return_value=sample_df),
            patch("src.services.entry_service.DataValidator.validate_dataframe"),
        ):

            result = await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_skips_zero_position_size(
        self, entry_service, sample_df, mock_entry_signal, bull_market_regime
    ):
        """Test scan bỏ qua khi position size = 0"""
        zero_position = Mock()
        zero_position.shares = 0

        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = zero_position

        with (
            patch("src.services.entry_service.load_data", return_value=sample_df),
            patch("src.services.entry_service.DataValidator.validate_dataframe"),
        ):

            result = await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_adds_pending_on_valid_signal(
        self, entry_service, sample_df, mock_entry_signal, mock_position_size, bull_market_regime
    ):
        """Test scan thêm symbol vào pending khi có signal hợp lệ"""
        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = mock_position_size

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        # Check that add_pending was called with VNM and any position value
        entry_service.portfolio_lock.add_pending.assert_called()
        call_args = entry_service.portfolio_lock.add_pending.call_args
        assert call_args[0][0] == "VNM"

    @pytest.mark.asyncio
    async def test_scan_handles_exceptions(self, entry_service, bull_market_regime):
        """Test scan xử lý exceptions gracefully"""
        entry_service.portfolio_lock.is_pending.return_value = False

        with patch.object(entry_service_module, "load_data", side_effect=Exception("Load error")):
            result = await entry_service.scan_for_entries(
                tickers=["VNM"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_multiple_tickers(
        self, entry_service, sample_df, mock_entry_signal, mock_position_size, bull_market_regime
    ):
        """Test scan nhiều tickers"""
        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = mock_position_size

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            result = await entry_service.scan_for_entries(
                tickers=["VNM", "VCB", "FPT"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        assert len(result) == 3


# =============================================================================
# FILTER AND RANK SIGNALS TESTS
# =============================================================================


class TestFilterAndRankSignals:
    """Tests cho filter_and_rank_signals"""

    def test_filter_empty_signals(self, entry_service):
        """Test filter với signals trống"""
        result = entry_service.filter_and_rank_signals([])

        assert result == []

    def test_filter_single_signal(self, entry_service, mock_entry_signal):
        """Test filter với 1 signal"""
        signals = [{"symbol": "VNM", "signal": mock_entry_signal}]

        result = entry_service.filter_and_rank_signals(signals)

        assert len(result) == 1
        assert result[0]["symbol"] == "VNM"

    def test_filter_ranks_by_score(self, entry_service):
        """Test filter sắp xếp theo score (confidence * strength)"""
        # Signal 1: confidence=80, strength=4 -> score=320
        signal1 = Mock()
        signal1.confidence = 80
        signal1.strength = Mock(value=4)

        # Signal 2: confidence=90, strength=5 -> score=450 (highest)
        signal2 = Mock()
        signal2.confidence = 90
        signal2.strength = Mock(value=5)

        # Signal 3: confidence=70, strength=3 -> score=210
        signal3 = Mock()
        signal3.confidence = 70
        signal3.strength = Mock(value=3)

        signals = [
            {"symbol": "VNM", "signal": signal1},
            {"symbol": "VCB", "signal": signal2},
            {"symbol": "FPT", "signal": signal3},
        ]

        result = entry_service.filter_and_rank_signals(signals)

        # Should be sorted by score descending
        assert result[0]["symbol"] == "VCB"  # Highest score
        assert result[1]["symbol"] == "VNM"
        assert result[2]["symbol"] == "FPT"  # Lowest score

    def test_filter_limits_max_signals(self, entry_service):
        """Test filter giới hạn số lượng signals"""
        signals = []
        for i in range(10):
            signal = Mock()
            signal.confidence = 70 + i
            signal.strength = Mock(value=4)
            signals.append({"symbol": f"SYM{i}", "signal": signal})

        result = entry_service.filter_and_rank_signals(signals, max_signals=3)

        assert len(result) == 3

    def test_filter_default_max_signals(self, entry_service):
        """Test filter với default max_signals=5"""
        signals = []
        for i in range(10):
            signal = Mock()
            signal.confidence = 70 + i
            signal.strength = Mock(value=4)
            signals.append({"symbol": f"SYM{i}", "signal": signal})

        result = entry_service.filter_and_rank_signals(signals)

        assert len(result) == 5  # Default max

    def test_filter_returns_top_signals(self, entry_service):
        """Test filter trả về top signals theo score"""
        signals = []
        for i in range(10):
            signal = Mock()
            signal.confidence = 50 + i * 5  # 50, 55, 60, ..., 95
            signal.strength = Mock(value=4)
            signals.append({"symbol": f"SYM{i}", "signal": signal})

        result = entry_service.filter_and_rank_signals(signals, max_signals=3)

        # Should return top 3 by confidence (since strength is same)
        assert result[0]["symbol"] == "SYM9"  # confidence=95
        assert result[1]["symbol"] == "SYM8"  # confidence=90
        assert result[2]["symbol"] == "SYM7"  # confidence=85


# =============================================================================
# SCAN SINGLE TICKER TESTS
# =============================================================================


class TestScanSingleTicker:
    """Tests cho _scan_single_ticker"""

    @pytest.mark.asyncio
    async def test_scan_single_returns_none_for_existing(self, entry_service, bull_market_regime):
        """Test scan single trả về None cho symbol đã có"""
        result = await entry_service._scan_single_ticker(
            symbol="VNM",
            existing_symbols={"VNM"},
            market_regime=bull_market_regime,
            vnindex_df=None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_scan_single_returns_none_for_pending(self, entry_service, bull_market_regime):
        """Test scan single trả về None cho symbol đang pending"""
        entry_service.portfolio_lock.is_pending.return_value = True

        result = await entry_service._scan_single_ticker(
            symbol="VNM",
            existing_symbols=set(),
            market_regime=bull_market_regime,
            vnindex_df=None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_scan_single_returns_signal_dict(
        self, entry_service, sample_df, mock_entry_signal, mock_position_size, bull_market_regime
    ):
        """Test scan single trả về dict với signal hợp lệ"""
        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = mock_position_size

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            result = await entry_service._scan_single_ticker(
                symbol="VNM",
                existing_symbols=set(),
                market_regime=bull_market_regime,
                vnindex_df=None,
            )

        assert result is not None
        assert result["symbol"] == "VNM"
        assert result["signal"] == mock_entry_signal
        assert result["position_size"] == mock_position_size
        assert "ml_signal" in result

    @pytest.mark.asyncio
    async def test_scan_single_passes_vnindex_to_ml(
        self, entry_service, sample_df, mock_entry_signal, mock_position_size, bull_market_regime
    ):
        """Test scan single truyền vnindex_df cho ML generator"""
        vnindex_df = pd.DataFrame({"close": [1200] * 100})

        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = mock_position_size

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            await entry_service._scan_single_ticker(
                symbol="VNM",
                existing_symbols=set(),
                market_regime=bull_market_regime,
                vnindex_df=vnindex_df,
            )

        # Verify ML generator was called with vnindex_df
        entry_service.ml_generator.analyze.assert_called_once()
        call_args = entry_service.ml_generator.analyze.call_args
        assert call_args[0][1] is vnindex_df  # Second positional arg


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEntryServiceIntegration:
    """Integration tests cho EntrySignalService"""

    @pytest.mark.asyncio
    async def test_full_scan_workflow(
        self, entry_service, sample_df, mock_entry_signal, mock_position_size, bull_market_regime
    ):
        """Test full workflow: scan -> filter -> rank"""
        # Setup mocks
        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.entry_logic.analyze_entry.return_value = mock_entry_signal
        entry_service.position_sizer.calculate_position_size.return_value = mock_position_size

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            # Scan
            signals = await entry_service.scan_for_entries(
                tickers=["VNM", "VCB", "FPT"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

            # Filter and rank
            top_signals = entry_service.filter_and_rank_signals(signals, max_signals=2)

        assert len(signals) == 3
        assert len(top_signals) == 2

    @pytest.mark.asyncio
    async def test_scan_with_mixed_results(self, entry_service, sample_df, bull_market_regime):
        """Test scan với kết quả hỗn hợp (có signal và không có)"""
        # Signal for VNM
        vnm_signal = Mock()
        vnm_signal.should_enter = True
        vnm_signal.confidence = 75
        vnm_signal.strength = Mock(value=4)
        vnm_signal.entry_price = 80000
        vnm_signal.stop_loss = 76000
        vnm_signal.take_profit_targets = [88000]

        # No signal for VCB
        vcb_signal = Mock()
        vcb_signal.should_enter = False

        position = Mock()
        position.shares = 500

        entry_service.portfolio_lock.is_pending.return_value = False
        entry_service.ml_generator.analyze.return_value = {"signal": "BUY", "confidence": 70}
        entry_service.position_sizer.calculate_position_size.return_value = position

        # Return different signals for different symbols
        def mock_analyze_entry(df, ml_signal, market_regime, symbol=None):
            # Alternate between entry and no entry
            if not hasattr(mock_analyze_entry, "call_count"):
                mock_analyze_entry.call_count = 0
            mock_analyze_entry.call_count += 1

            if mock_analyze_entry.call_count % 2 == 1:
                return vnm_signal
            return vcb_signal

        entry_service.entry_logic.analyze_entry.side_effect = mock_analyze_entry

        with (
            patch.object(entry_service_module, "load_data", return_value=sample_df),
            patch.object(entry_service_module, "DataValidator") as mock_validator,
        ):
            mock_validator.validate_dataframe.return_value = None

            signals = await entry_service.scan_for_entries(
                tickers=["VNM", "VCB", "FPT", "HPG"],
                existing_symbols=set(),
                market_regime=bull_market_regime,
            )

        # Should have 2 signals (VNM and FPT - odd calls)
        assert len(signals) == 2
