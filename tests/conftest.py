# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Generate realistic price data
    np.random.seed(42)
    close_prices = 80000 + np.cumsum(np.random.randn(100) * 1000)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close_prices + np.random.randn(100) * 500,
            "high": close_prices + np.abs(np.random.randn(100) * 1000),
            "low": close_prices - np.abs(np.random.randn(100) * 1000),
            "close": close_prices,
            "volume": np.random.randint(100000, 1000000, 100),
        }
    )

    return df


@pytest.fixture
def sample_ml_signal():
    """Sample ML signal for testing"""
    return {
        "signal": "BUY",
        "confidence": 75,
        "ml_score": 0.75,
        "technical_score": {"trend": 0.5, "momentum": 0.3},
        "reason": "Test signal",
        "price": 80000,
        "rsi": 55,
        "ema_trend": "UP",
    }


@pytest.fixture
def sample_market_regime():
    """Sample market regime for testing"""
    return {
        "regime": "BULL",
        "confidence": 70,
        "tradeable": True,
        "vnindex_change": 0.02,
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    from trading_config import DataConfig, TradingConfig

    trading = TradingConfig(
        max_scan_universe=10,
        min_confidence=60,
        min_risk_reward=2.0,
        total_capital=100_000_000,
        max_position_size=0.10,
        max_positions=10,
    )

    data = DataConfig(lookback=100, min_volume=100000)

    return {"trading": trading, "data": data}


@pytest.fixture
def temp_database(tmp_path):
    """Create temporary database for testing"""
    db_path = tmp_path / "test_trading.db"

    # Set environment variable
    os.environ["DATABASE_PATH"] = str(db_path)

    yield str(db_path)

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_portfolio_positions():
    """Mock portfolio positions"""
    return {
        "VNM": {
            "shares": 500,
            "avg_price": 80000,
            "entry_price": 80000,
            "current_price": 82000,
            "entry_date": "2024-01-01T09:00:00",
            "stop_loss": 76000,
            "take_profit_targets": [84000, 88000, 92000],
            "partial_exits": [],
        },
        "VCB": {
            "shares": 300,
            "avg_price": 90000,
            "entry_price": 90000,
            "current_price": 91000,
            "entry_date": "2024-01-02T09:00:00",
            "stop_loss": 85500,
            "take_profit_targets": [94500, 99000, 103500],
            "partial_exits": [],
        },
    }
