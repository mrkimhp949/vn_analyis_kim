# -*- coding: utf-8 -*-
"""
Unit tests for the Bot Runner (src/core/bot_runner.py)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import unittest

import pandas as pd
import pytest

# Module to be tested
from src.core.bot_runner import run_bot_with_context, run_bot_sync
from src.config.exceptions import ConfigurationError


@pytest.fixture
def mock_bot():
    """Fixture for a mocked Telegram Bot instance."""
    bot = MagicMock()
    bot.send_message = AsyncMock()
    return bot


@pytest.fixture
def mock_orchestrator():
    """Fixture for a mocked TradingOrchestrator."""
    orchestrator = MagicMock()
    orchestrator.run_scan = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_market_analyzer():
    """Fixture for a mocked ProxyMarketRegimeAnalyzer."""
    analyzer = MagicMock()
    analyzer.analyze_market_regime.return_value = {
        "regime": "BULL",
        "confidence": 80,
        "tradeable": True,
    }
    return analyzer


@pytest.fixture
def mock_vnindex_df():
    """Fixture for a sample VNINDEX DataFrame with enough data and all required columns."""
    data = {
        "open": [i for i in range(1000, 1100)],
        "high": [i * 1.02 for i in range(1000, 1100)],
        "low": [i * 0.98 for i in range(1000, 1100)],
        "close": [i + 1 for i in range(1000, 1100)],
        "volume": [100000 + i * 100 for i in range(100)],
    }
    return pd.DataFrame(data)


@pytest.mark.asyncio
@patch("src.core.bot_runner.TradingOrchestrator")
@patch("src.core.bot_runner.market_analyzer")  # Patch the instance directly
@patch("src.core.bot_runner.load_data")
async def test_run_bot_with_context_success(
    mock_load_data,
    mock_market_analyzer_instance,
    mock_orchestrator_cls,
    mock_bot,
    mock_orchestrator,
    mock_vnindex_df,
):
    """
    Test the successful execution of run_bot_with_context.
    """
    # --- Arrange ---
    chat_id = "test_chat_id"
    mock_load_data.return_value = mock_vnindex_df

    # Configure the mocked instance
    mock_market_analyzer_instance.analyze_market_regime.return_value = {
        "regime": "BULL",
        "confidence": 80,
        "tradeable": True,
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    # --- Act ---
    await run_bot_with_context(mock_bot, chat_id)

    # --- Assert ---
    # 1. Data loading was called
    mock_load_data.assert_called_once()
    args, kwargs = mock_load_data.call_args
    assert args[0] == "VNINDEX"
    assert kwargs["data_type"] == "index"

    # 2. Market analyzer was used
    mock_market_analyzer_instance.analyze_market_regime.assert_called_once_with(
        vnindex_df=mock_vnindex_df
    )

    # 3. Orchestrator was initialized with correct context
    mock_orchestrator_cls.assert_called_once_with(
        bot_instance=mock_bot, chat_id=chat_id, vnindex_df=mock_vnindex_df
    )

    # 4. Orchestrator's scan was run with the market regime
    mock_orchestrator.run_scan.assert_called_once_with(
        market_regime=mock_market_analyzer_instance.analyze_market_regime.return_value
    )

    # 5. No error messages were sent
    mock_bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_run_bot_with_context_no_bot_instance(caplog):
    """
    Test that the function exits gracefully if no bot instance is provided.
    """
    # --- Act ---
    await run_bot_with_context(None, "test_chat_id")

    # --- Assert ---
    assert "Bot instance không khả dụng, không thể chạy." in caplog.text


@pytest.mark.asyncio
@patch("src.core.bot_runner.load_data", side_effect=Exception("Data load failed"))
async def test_run_bot_with_context_data_load_failure(mock_load_data, mock_bot, caplog):
    """
    Test that the process continues even if VNINDEX data loading fails.
    """
    # --- Arrange ---
    chat_id = "test_chat_id"
    # Mock other dependencies to avoid breaking the test
    with (
        patch("src.core.bot_runner.TradingOrchestrator"),
        patch("src.core.bot_runner.ProxyMarketRegimeAnalyzer"),
    ):
        # --- Act ---
        await run_bot_with_context(mock_bot, chat_id)

        # --- Assert ---
        assert "Lỗi khi tải dữ liệu VNINDEX" in caplog.text
        # Check that the orchestrator is still called, but with vnindex_df=None
        from src.core.bot_runner import TradingOrchestrator

        TradingOrchestrator.assert_called_with(
            bot_instance=mock_bot, chat_id=chat_id, vnindex_df=None
        )


@pytest.mark.asyncio
@patch("src.core.bot_runner.TradingOrchestrator", side_effect=Exception("Orchestrator init failed"))
async def test_run_bot_with_context_orchestrator_init_failure(
    mock_orchestrator_cls, mock_bot, caplog
):
    """
    Test that a fatal error is logged and a message is sent if the orchestrator fails to initialize.
    """
    # --- Arrange ---
    chat_id = "test_chat_id"
    with patch("src.core.bot_runner.load_data"):  # Mock load_data to avoid other errors
        # --- Act ---
        await run_bot_with_context(mock_bot, chat_id)

        # --- Assert ---
        assert "Lỗi khởi tạo TradingOrchestrator" in caplog.text
        mock_bot.send_message.assert_called_once_with(
            chat_id, "FATAL: Không thể khởi tạo Orchestrator"
        )


@pytest.mark.asyncio
@patch("src.core.bot_runner.TradingOrchestrator")
@patch("src.core.bot_runner.market_analyzer")  # Patch the instance
@patch("src.core.bot_runner.load_data")
async def test_run_bot_with_context_market_analysis_failure(
    mock_load_data,
    mock_market_analyzer_instance,
    mock_orchestrator_cls,
    mock_bot,
    mock_orchestrator,
    mock_vnindex_df,
    caplog,
):
    """
    Test that the scan continues with an empty market_regime if market analysis fails.
    """
    # --- Arrange ---
    chat_id = "test_chat_id"
    mock_load_data.return_value = mock_vnindex_df
    mock_market_analyzer_instance.analyze_market_regime.side_effect = Exception("Analysis failed")
    mock_orchestrator_cls.return_value = mock_orchestrator

    # --- Act ---
    await run_bot_with_context(mock_bot, chat_id)

    # --- Assert ---
    assert "Lỗi khi phân tích thị trường" in caplog.text
    mock_bot.send_message.assert_called_once_with(chat_id, "Lỗi phân tích thị trường")
    # Orchestrator's scan should be called with an empty market_regime
    mock_orchestrator.run_scan.assert_called_once_with(market_regime={})


@pytest.mark.asyncio
@patch("src.core.bot_runner.TradingOrchestrator")
@patch("src.core.bot_runner.market_analyzer")
@patch("src.core.bot_runner.load_data")
async def test_run_bot_with_context_scan_failure(
    mock_load_data,
    mock_market_analyzer_instance,
    mock_orchestrator_cls,
    mock_bot,
    mock_orchestrator,
    mock_vnindex_df,
    caplog,
):
    """
    Test that a critical error is logged and a message is sent if orchestrator.run_scan fails.
    """
    # --- Arrange ---
    chat_id = "test_chat_id"
    mock_load_data.return_value = mock_vnindex_df
    mock_market_analyzer_instance.analyze_market_regime.return_value = {"regime": "BULL"}
    mock_orchestrator.run_scan.side_effect = Exception("Scan failed miserably")
    mock_orchestrator_cls.return_value = mock_orchestrator

    # --- Act ---
    await run_bot_with_context(mock_bot, chat_id)

    # --- Assert ---
    assert "Lỗi nghiêm trọng trong quá trình quét của Orchestrator" in caplog.text
    mock_bot.send_message.assert_called_once_with(chat_id, "Lỗi nghiêm trọng khi đang quét")


@patch("src.core.bot_runner.asyncio.run")
@patch(
    "src.core.bot_runner.run_bot_with_context", side_effect=ConfigurationError("Config Test Error")
)
@patch("src.core.bot_runner.bot", new_callable=MagicMock)
@patch("src.core.bot_runner.CHAT_ID", "test_chat_id")
def test_run_bot_sync_configuration_error(
    mock_bot, mock_run_bot_with_context, mock_asyncio_run, caplog
):
    """
    Test that `run_bot_sync` catches and logs ConfigurationError.
    """
    # --- Act ---
    run_bot_sync()

    # --- Assert ---
    mock_asyncio_run.assert_called_once()
    assert "CRITICAL CONFIG ERROR: Config Test Error" in caplog.text


@patch("src.core.bot_runner.bot", None)
def test_run_bot_sync_no_bot(caplog):
    """
    Test that `run_bot_sync` logs an error and returns if the bot is not configured.
    """
    # --- Act ---
    run_bot_sync()

    # --- Assert ---
    assert "Không thể chạy bot: Thiếu TELEGRAM_TOKEN hoặc CHAT_ID" in caplog.text


def test_bot_initialization_failure(caplog):
    """
    Test that a critical error is logged if the Telegram Bot fails to initialize.
    This test reloads the module to check module-level exception handling.
    """
    import importlib

    # We need to patch both 'telegram.Bot' and 'TELEGRAM_TOKEN' before the module is reloaded
    with patch("telegram.Bot", side_effect=Exception("Bot init failed")), \
         patch("src.core.bot_runner.TELEGRAM_TOKEN", "test_token"):
        # Reload the bot_runner module to trigger the initialization code
        import src.core.bot_runner as bot_runner_reloaded

        importlib.reload(bot_runner_reloaded)

        # --- Assert ---
        # 1. The error was logged during module import
        assert "Lỗi khởi tạo Telegram bot" in caplog.text
        # 2. The global 'bot' variable in the reloaded module is None
        assert bot_runner_reloaded.bot is None

        # 3. Calling run_bot_sync in this state should fail gracefully
        bot_runner_reloaded.run_bot_sync()
        assert "Không thể chạy bot: Thiếu TELEGRAM_TOKEN hoặc CHAT_ID" in caplog.text


def test_market_analyzer_initialization_failure(caplog):
    """
    Test that a warning is logged if the ProxyMarketRegimeAnalyzer fails to initialize.
    """
    import importlib

    with patch(
        "src.market.regime_proxy.ProxyMarketRegimeAnalyzer",
        side_effect=ImportError("Analyzer init failed"),
    ):
        import src.core.bot_runner as bot_runner_reloaded

        importlib.reload(bot_runner_reloaded)

        assert "Không có market analyzer" in caplog.text
        assert bot_runner_reloaded.market_analyzer is None


@patch("src.core.bot_runner.run_bot_sync")
@patch("src.core.bot_runner.TELEGRAM_TOKEN", "fake_token")
@patch("src.core.bot_runner.CHAT_ID", "fake_chat_id")
def test_main_block(mock_run_bot_sync):
    """
    Test the __main__ block execution path.
    """
    # The __main__ block is special. We can't just import it.
    # We execute the file as a script and check the side effects.
    import subprocess
    import os

    # Run as module to avoid import path issues
    result = subprocess.run(
        ["python", "-m", "src.core.bot_runner"],
        capture_output=True,
        text=True,
        check=False,
        cwd=os.getcwd()  # Ensure we're in the project root
    )
    assert "TESTING BOT RUNNER" in result.stdout
    assert "Chạy thử bot trong 5 giây..." in result.stdout
    # Check that our mocked run_bot_sync was called inside the subprocess
    # This is tricky. A better way is to test the function called by main.
    # Here, we just check that the script runs.
    # To properly test this, we'd need to refactor the main guard.
    # For now, we confirm the script runs and calls the sync function.
    # The mock won't apply to the subprocess, but we can check the output.
    assert "Test run hoàn tất" in result.stdout
