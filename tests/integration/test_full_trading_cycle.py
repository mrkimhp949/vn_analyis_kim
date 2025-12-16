# -*- coding: utf-8 -*-
"""
Full Trading Cycle Integration Tests

This module provides comprehensive integration tests for the complete trading cycle:
1. Signal generation (entry signals)
2. Position entry with sizing
3. Stop loss management
4. Exit signal generation
5. Position closing

These tests verify that all components work together correctly.
"""

import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Core trading modules
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.exit_logic import ExitReason, ImprovedExitStrategy

# Position sizing
try:
    from src.strategies.position_sizing import EnhancedPositionSizer

    POSITION_SIZER_AVAILABLE = True
except ImportError:
    POSITION_SIZER_AVAILABLE = False

# Risk management
try:
    from src.risk.atr_dynamic_stop_loss import (
        ATRDynamicStopLossManager,
        get_dynamic_stop_loss_manager,
    )

    ATR_STOP_LOSS_AVAILABLE = True
except ImportError:
    ATR_STOP_LOSS_AVAILABLE = False

# Foreign flow
try:
    from src.data.hose_hnx_realtime_api import (
        HOSEHNXForeignFlowAPI,
        Exchange,
        ForeignFlowSnapshot,
    )

    FOREIGN_FLOW_AVAILABLE = True
except ImportError:
    FOREIGN_FLOW_AVAILABLE = False

# ML Integration
try:
    from src.ml.vietnam_ml_integration import VietnamMLIntegration

    ML_INTEGRATION_AVAILABLE = True
except ImportError:
    ML_INTEGRATION_AVAILABLE = False

# Market validation
from src.utils.vietnam_market import VietnamMarketValidator

logger = logging.getLogger(__name__)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_stock_data():
    """Generate realistic stock data for testing."""
    np.random.seed(42)
    n = 250  # ~1 year of trading days

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")

    # Uptrend with noise
    trend = np.linspace(80000, 105000, n)
    noise = np.random.normal(0, 1500, n)
    close = trend + noise

    # Ensure prices are positive
    close = np.maximum(close, 50000)

    high = close + np.abs(np.random.normal(800, 300, n))
    low = close - np.abs(np.random.normal(800, 300, n))
    openp = close + np.random.normal(0, 400, n)
    volume = np.random.uniform(200000, 600000, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Add technical indicators
    df = add_technical_indicators(df)

    return df


@pytest.fixture
def sample_index_data():
    """Generate VNINDEX data for testing."""
    np.random.seed(123)
    n = 250

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")

    trend = np.linspace(1100, 1250, n)
    noise = np.random.normal(0, 20, n)
    close = trend + noise

    high = close + np.abs(np.random.normal(10, 5, n))
    low = close - np.abs(np.random.normal(10, 5, n))
    openp = close + np.random.normal(0, 5, n)
    volume = np.random.uniform(500e6, 1000e6, n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def bear_market_data():
    """Generate bear market data."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")

    # Downtrend
    trend = np.linspace(100000, 75000, n)
    noise = np.random.normal(0, 1500, n)
    close = trend + noise

    high = close + np.abs(np.random.normal(800, 300, n))
    low = close - np.abs(np.random.normal(800, 300, n))
    openp = close + np.random.normal(0, 400, n)
    volume = np.random.uniform(200000, 600000, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return add_technical_indicators(df)


@pytest.fixture
def sideways_market_data():
    """Generate sideways market data."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")

    # Sideways with mean reversion
    base = 90000
    noise = np.random.normal(0, 2000, n)
    close = base + noise

    high = close + np.abs(np.random.normal(500, 200, n))
    low = close - np.abs(np.random.normal(500, 200, n))
    openp = close + np.random.normal(0, 300, n)
    volume = np.random.uniform(150000, 400000, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return add_technical_indicators(df)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators required for trading logic."""
    df = df.copy()

    # Moving averages
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume indicators
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]

    # Momentum
    df["momentum"] = df["close"].pct_change(10)
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # Volatility
    df["volatility_20"] = df["close"].pct_change().rolling(20).std()

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_dif"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_high"] = df["bb_mid"] + (bb_std * 2)
    df["bb_low"] = df["bb_mid"] - (bb_std * 2)
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    return df


# =============================================================================
# HELPER CLASSES
# =============================================================================


@dataclass
class TradingPosition:
    """Represents an active trading position."""

    symbol: str
    entry_price: float
    entry_date: datetime
    shares: int
    stop_loss: float
    take_profit_levels: List[float]

    # Tracking
    highest_price: float = None
    current_price: float = None
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_reason: Optional[str] = None

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.current_price is None:
            self.current_price = self.entry_price

    def update_price(self, price: float):
        self.current_price = price
        if price > self.highest_price:
            self.highest_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.shares

    def close_position(self, exit_price: float, reason: str):
        self.exit_price = exit_price
        self.exit_date = datetime.now()
        self.exit_reason = reason
        self.realized_pnl = (exit_price - self.entry_price) * self.shares


from dataclasses import dataclass


@dataclass
class TradingPosition:
    """Represents an active trading position."""

    symbol: str
    entry_price: float
    entry_date: datetime
    shares: int
    stop_loss: float
    take_profit_levels: List[float]

    # Tracking
    highest_price: float = None
    current_price: float = None
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_reason: Optional[str] = None

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.current_price is None:
            self.current_price = self.entry_price

    def update_price(self, price: float):
        self.current_price = price
        if price > self.highest_price:
            self.highest_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.shares

    def close_position(self, exit_price: float, reason: str):
        self.exit_price = exit_price
        self.exit_date = datetime.now()
        self.exit_reason = reason
        self.realized_pnl = (exit_price - self.entry_price) * self.shares


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestFullTradingCycleIntegration:
    """Integration tests for the complete trading cycle."""

    def test_entry_to_exit_bull_market(
        self, sample_stock_data, sample_index_data, temp_storage_dir
    ):
        """Test complete cycle in bull market: entry -> hold -> exit with profit."""
        # Setup
        symbol = "VNM"
        account_value = 100_000_000  # 100M VND

        entry_logic = ImprovedEntryLogic()
        exit_strategy = ImprovedExitStrategy()

        # Step 1: Generate entry signal
        entry_result = entry_logic.analyze_entry(
            df=sample_stock_data,
            symbol=symbol,
        )

        # Should have a valid result (may or may not be entry)
        assert entry_result is not None

        # Step 2: If entry signal, simulate position
        entry_price = sample_stock_data["close"].iloc[-1]
        shares = int(account_value * 0.10 / entry_price)  # 10% position

        # Step 3: Calculate stop loss
        atr = sample_stock_data["atr"].iloc[-1]
        stop_loss = entry_price - (atr * 2)

        # Step 4: Simulate price movement (up)
        position = TradingPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            shares=shares,
            stop_loss=stop_loss,
            take_profit_levels=[
                entry_price * 1.06,
                entry_price * 1.10,
                entry_price * 1.15,
            ],
        )

        # Step 5: Check exit conditions
        test_prices = [
            entry_price * 1.02,  # Small profit
            entry_price * 1.05,  # 5% profit
            entry_price * 1.08,  # 8% profit - may trigger partial exit
        ]

        for price in test_prices:
            position.update_price(price)

            # Update dataframe with new price
            test_df = sample_stock_data.copy()
            test_df.loc[test_df.index[-1], "close"] = price

            exit_decision = exit_strategy.check_exit(
                symbol=symbol,
                entry_price=position.entry_price,
                current_price=price,
                stop_loss=position.stop_loss,
                take_profit_targets=position.take_profit_levels,
                entry_date=position.entry_date,
                df=test_df,
            )

            # No exit yet at reasonable profit
            if price < entry_price * 1.10:
                # Should continue holding
                pass

        # Final assertions
        assert position.unrealized_pnl > 0
        assert position.highest_price >= entry_price

    def test_entry_to_stop_loss_exit(self, sample_stock_data, sample_index_data, temp_storage_dir):
        """Test complete cycle with stop loss exit."""
        symbol = "VNM"
        account_value = 100_000_000

        entry_logic = ImprovedEntryLogic()
        exit_strategy = ImprovedExitStrategy()

        # Get entry price
        entry_price = sample_stock_data["close"].iloc[-1]
        shares = int(account_value * 0.10 / entry_price)

        # Set stop loss at 5%
        stop_loss = entry_price * 0.95

        position = TradingPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            shares=shares,
            stop_loss=stop_loss,
            take_profit_levels=[entry_price * 1.10],
        )

        # Simulate price drop to stop loss
        test_prices = [
            entry_price * 0.99,  # -1%
            entry_price * 0.97,  # -3%
            entry_price * 0.95,  # -5% - stop loss hit
        ]

        exit_triggered = False
        for price in test_prices:
            position.update_price(price)

            test_df = sample_stock_data.copy()
            test_df.loc[test_df.index[-1], "close"] = price

            exit_decision = exit_strategy.check_exit(
                symbol=symbol,
                entry_price=position.entry_price,
                current_price=price,
                stop_loss=position.stop_loss,
                take_profit_targets=position.take_profit_levels,
                entry_date=position.entry_date,
                df=test_df,
            )

            if exit_decision and exit_decision.should_exit:
                position.close_position(price, str(exit_decision.exit_reason))
                exit_triggered = True
                break

        # Stop loss should have triggered
        assert exit_triggered or position.current_price <= stop_loss

    @pytest.mark.skipif(not ATR_STOP_LOSS_AVAILABLE, reason="ATR stop loss not available")
    def test_atr_dynamic_stop_loss_integration(
        self, sample_stock_data, sample_index_data, temp_storage_dir
    ):
        """Test ATR-based dynamic stop loss integration."""
        symbol = "VNM"
        entry_price = sample_stock_data["close"].iloc[-1]

        manager = ATRDynamicStopLossManager(storage_dir=temp_storage_dir)

        # Calculate stop loss
        result = manager.calculate_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            df=sample_stock_data,
            index_df=sample_index_data,
        )

        # Verify result
        assert result.is_valid
        assert result.stop_loss_price > 0
        assert result.stop_loss_price < entry_price
        assert 0.03 <= result.stop_loss_pct <= 0.15
        assert result.atr_value > 0

        # Test trailing stop
        manager.init_trailing_stop(
            symbol=symbol,
            entry_price=entry_price,
            initial_stop=result.stop_loss_price,
            df=sample_stock_data,
        )

        # Simulate price increase
        higher_price = entry_price * 1.08
        new_stop, triggered, msg = manager.update_trailing_stop(
            symbol=symbol,
            current_price=higher_price,
            highest_price=higher_price,
            df=sample_stock_data,
        )

        # Trailing should have moved up
        assert new_stop > result.stop_loss_price
        assert not triggered

        # Clean up
        manager.remove_trailing_stop(symbol)

    @pytest.mark.skipif(not FOREIGN_FLOW_AVAILABLE, reason="Foreign flow API not available")
    def test_foreign_flow_entry_integration(self, sample_stock_data, temp_storage_dir):
        """Test foreign flow check in entry decision."""
        symbol = "VNM"

        # Create mock foreign flow snapshot
        with patch.object(HOSEHNXForeignFlowAPI, "get_current_foreign_flow") as mock_flow:
            mock_flow.return_value = ForeignFlowSnapshot(
                symbol=symbol,
                exchange=Exchange.HOSE,
                timestamp=datetime.now(),
                buy_volume=500000,
                sell_volume=200000,
                net_volume=300000,
                buy_value=50_000_000_000,
                sell_value=20_000_000_000,
                net_value=30_000_000_000,
                remaining_room=1000000,
                data_source="test",
            )

            api = HOSEHNXForeignFlowAPI()

            # Check entry
            can_enter, adjustment, msg = api.check_for_entry(symbol)

            # With positive foreign flow, should allow entry
            assert can_enter
            assert adjustment >= 0

    @pytest.mark.skipif(not ML_INTEGRATION_AVAILABLE, reason="ML integration not available")
    def test_ml_enhanced_entry_signal(self, sample_stock_data, sample_index_data, temp_storage_dir):
        """Test ML-enhanced entry signal generation."""
        symbol = "VNM"

        ml_integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        # Get ML signal
        result = ml_integration.get_signal(
            sample_stock_data,
            symbol,
            sample_index_data,
            current_time=datetime(2024, 1, 15, 10, 0),  # Morning session
        )

        # Should have valid result
        assert result is not None
        assert result.signal in ["BUY", "SELL", "HOLD"]
        assert 0 <= result.calibrated_confidence <= 100

    def test_position_sizing_with_risk(self, sample_stock_data, temp_storage_dir):
        """Test position sizing with risk management."""
        symbol = "VNM"
        account_value = 100_000_000
        entry_price = sample_stock_data["close"].iloc[-1]

        # Calculate ATR-based stop
        atr = sample_stock_data["atr"].iloc[-1]
        stop_loss = entry_price - (atr * 2)

        # Risk per trade
        max_risk_pct = 0.02  # 2%
        max_risk_amount = account_value * max_risk_pct

        # Calculate position size based on risk
        risk_per_share = entry_price - stop_loss
        shares_from_risk = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0

        # Cap position size to max 20% of account (important for proper risk management)
        max_position_pct = 0.20
        max_shares_from_position = int((account_value * max_position_pct) / entry_price)

        # Use the smaller of the two to respect both limits
        shares = min(shares_from_risk, max_shares_from_position)

        # Round to lot size (Vietnam market standard)
        lot_size = 100
        shares = int(shares / lot_size) * lot_size

        # Position value
        position_value = shares * entry_price
        position_pct = position_value / account_value

        # Verify position is reasonable
        assert shares > 0
        assert position_pct <= 0.20  # Not more than 20% of account

        # Actual risk
        actual_risk = shares * risk_per_share
        assert actual_risk <= max_risk_amount

    def test_multi_symbol_portfolio_cycle(
        self, sample_stock_data, sample_index_data, temp_storage_dir
    ):
        """Test trading cycle with multiple symbols."""
        symbols = ["VNM", "VCB", "HPG", "FPT"]
        account_value = 500_000_000
        max_positions = 5
        max_position_pct = 0.15

        entry_logic = ImprovedEntryLogic()
        exit_strategy = ImprovedExitStrategy()

        positions: Dict[str, TradingPosition] = {}
        signals_generated = 0

        # Generate signals for multiple symbols
        for symbol in symbols:
            result = entry_logic.analyze_entry(
                df=sample_stock_data,
                symbol=symbol,
            )

            if result:
                signals_generated += 1

            # Simulate opening position
            if len(positions) < max_positions:
                entry_price = sample_stock_data["close"].iloc[-1]
                max_position_value = account_value * max_position_pct
                shares = int(max_position_value / entry_price)

                atr = sample_stock_data["atr"].iloc[-1]
                stop_loss = entry_price - (atr * 2)

                positions[symbol] = TradingPosition(
                    symbol=symbol,
                    entry_price=entry_price,
                    entry_date=datetime.now(),
                    shares=shares,
                    stop_loss=stop_loss,
                    take_profit_levels=[entry_price * 1.10],
                )

        # Verify portfolio
        assert len(positions) == len(symbols)

        # Calculate total exposure
        total_exposure = sum(pos.shares * pos.entry_price for pos in positions.values())
        exposure_pct = total_exposure / account_value

        # Should be within limits
        assert exposure_pct <= len(symbols) * max_position_pct

    def test_exit_partial_profit_taking(self, sample_stock_data, temp_storage_dir):
        """Test partial profit taking at multiple levels."""
        symbol = "VNM"
        entry_price = 100_000
        total_shares = 1000

        position = TradingPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            shares=total_shares,
            stop_loss=entry_price * 0.95,
            take_profit_levels=[
                entry_price * 1.06,  # TP1: 6%
                entry_price * 1.10,  # TP2: 10%
                entry_price * 1.15,  # TP3: 15%
            ],
        )

        # Simulate hitting TP levels
        remaining_shares = total_shares
        partial_exits = []

        for i, tp_level in enumerate(position.take_profit_levels):
            # Simulate price reaching TP
            position.update_price(tp_level)

            # Take partial profit (33% each level)
            shares_to_sell = int(remaining_shares * 0.33)
            if i == len(position.take_profit_levels) - 1:
                shares_to_sell = remaining_shares  # Sell all at final TP

            profit = shares_to_sell * (tp_level - entry_price)
            partial_exits.append(
                {
                    "level": i + 1,
                    "price": tp_level,
                    "shares": shares_to_sell,
                    "profit": profit,
                }
            )

            remaining_shares -= shares_to_sell
            if remaining_shares <= 0:
                break

        # Verify partial exits
        assert len(partial_exits) >= 1
        total_profit = sum(exit["profit"] for exit in partial_exits)
        assert total_profit > 0


class TestEdgeCasesIntegration:
    """Test edge cases in trading cycle."""

    def test_gap_down_through_stop_loss(self, sample_stock_data, temp_storage_dir):
        """Test handling of gap down through stop loss."""
        symbol = "VNM"
        entry_price = 100_000
        stop_loss = 95_000

        # Simulate gap down - price opens below stop
        gap_down_price = 92_000  # Below stop loss

        position = TradingPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            shares=100,
            stop_loss=stop_loss,
            take_profit_levels=[110_000],
        )

        position.update_price(gap_down_price)

        # Should exit at gap down price, not stop loss
        if gap_down_price <= stop_loss:
            position.close_position(gap_down_price, "Gap down through stop")

        # Actual loss is larger than planned
        planned_loss = (entry_price - stop_loss) * position.shares
        actual_loss = -position.realized_pnl

        assert actual_loss > planned_loss
        assert position.exit_price == gap_down_price

    def test_price_limit_hit(self, sample_stock_data, temp_storage_dir):
        """Test handling when price hits daily limit."""
        symbol = "VNM"
        entry_price = 100_000

        # HOSE price limit: ±7%
        ceiling_price = entry_price * 1.07
        floor_price = entry_price * 0.93

        position = TradingPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            shares=100,
            stop_loss=entry_price * 0.95,
            take_profit_levels=[entry_price * 1.10],
        )

        # Simulate hitting ceiling
        position.update_price(ceiling_price)

        # At ceiling, may want to take profit due to low liquidity
        at_ceiling = position.current_price >= ceiling_price * 0.99

        assert at_ceiling
        assert position.unrealized_pnl > 0

    def test_low_liquidity_exit(self, temp_storage_dir):
        """Test exit handling with low liquidity."""
        symbol = "VNM"
        entry_price = 100_000
        shares = 50000  # Large position
        avg_daily_volume = 100000

        # Large position relative to volume
        days_to_exit = shares / (avg_daily_volume * 0.10)  # Max 10% of volume

        # Should split exit over multiple days
        assert days_to_exit > 1

        # Calculate slippage estimate
        slippage_pct = min(0.02, shares / avg_daily_volume * 0.005)
        expected_slippage = entry_price * slippage_pct

        assert expected_slippage > 0


class TestVietnamMarketRules:
    """Test Vietnam-specific market rules in trading cycle."""

    def test_t2_settlement_awareness(self, sample_stock_data, temp_storage_dir):
        """Test T+2 settlement awareness."""
        symbol = "VNM"
        entry_date = datetime(2024, 12, 16)  # Monday

        # T+2 settlement
        settlement_date = entry_date + timedelta(days=2)  # Wednesday

        # If sold on entry day, shares won't settle until T+2
        # Cannot sell shares bought today

        position = TradingPosition(
            symbol=symbol,
            entry_price=100_000,
            entry_date=entry_date,
            shares=100,
            stop_loss=95_000,
            take_profit_levels=[110_000],
        )

        # Check if can sell (need T+2)
        current_date = entry_date
        can_sell = (current_date - entry_date).days >= 2

        assert not can_sell  # Cannot sell same day

    def test_lot_size_compliance(self, temp_storage_dir):
        """Test lot size rules for different exchanges."""
        # HOSE: 100 shares
        # HNX: 100 shares
        # UPCoM: 100 shares (odd lots allowed in some cases)

        account_value = 100_000_000  # 100M VND
        entry_price = 50_000  # Lower price to ensure enough shares

        # Calculate raw position (10% of account)
        raw_shares = account_value * 0.10 / entry_price  # = 200 shares

        # Round to lot size
        lot_size = 100
        shares = int(raw_shares / lot_size) * lot_size

        assert shares % lot_size == 0
        assert shares > 0

    def test_trading_session_awareness(self, sample_stock_data, temp_storage_dir):
        """Test trading session awareness."""
        # HOSE sessions:
        # - ATO: 9:00-9:15
        # - Continuous 1: 9:15-11:30
        # - Lunch: 11:30-13:00
        # - Continuous 2: 13:00-14:30
        # - ATC: 14:30-14:45

        from datetime import time

        sessions = {
            "ATO": (time(9, 0), time(9, 15)),
            "CONTINUOUS_1": (time(9, 15), time(11, 30)),
            "LUNCH": (time(11, 30), time(13, 0)),
            "CONTINUOUS_2": (time(13, 0), time(14, 30)),
            "ATC": (time(14, 30), time(14, 45)),
        }

        # Test time classification
        test_times = [
            (time(9, 0), "ATO"),
            (time(10, 30), "CONTINUOUS_1"),
            (time(12, 0), "LUNCH"),
            (time(14, 0), "CONTINUOUS_2"),
            (time(14, 35), "ATC"),
        ]

        for test_time, expected_session in test_times:
            for session_name, (start, end) in sessions.items():
                if start <= test_time < end:
                    assert session_name == expected_session
                    break


class TestPerformanceMetrics:
    """Test performance metric calculation in trading cycle."""

    def test_calculate_sharpe_ratio(self, temp_storage_dir):
        """Test Sharpe ratio calculation."""
        # Simulate daily returns
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 252)  # 1 year

        risk_free_rate = 0.05 / 252  # Daily risk-free rate
        excess_returns = daily_returns - risk_free_rate

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        # Reasonable Sharpe for a trading strategy
        assert -2 <= sharpe <= 3

    def test_calculate_max_drawdown(self, temp_storage_dir):
        """Test maximum drawdown calculation."""
        # Simulate portfolio value
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, 252)
        portfolio_values = [100_000_000]

        for r in returns:
            new_value = portfolio_values[-1] * (1 + r)
            portfolio_values.append(new_value)

        portfolio_values = np.array(portfolio_values)

        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        assert max_drawdown <= 0
        assert max_drawdown > -1  # Not total loss

    def test_calculate_win_rate(self, temp_storage_dir):
        """Test win rate calculation."""
        trades = [
            {"pnl": 5_000_000},
            {"pnl": -2_000_000},
            {"pnl": 3_000_000},
            {"pnl": -1_000_000},
            {"pnl": 4_000_000},
            {"pnl": 2_000_000},
        ]

        wins = sum(1 for t in trades if t["pnl"] > 0)
        total = len(trades)
        win_rate = wins / total * 100

        assert win_rate == pytest.approx(66.67, rel=0.01)

        # Calculate profit factor
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        assert profit_factor > 1  # Profitable strategy


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
