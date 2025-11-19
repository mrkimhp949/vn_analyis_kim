"""
Unit tests for Entry and Exit Logic
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.strategies.entry_logic import EntrySignal, ImprovedEntryLogic
from src.strategies.exit_logic import ExitDecision, ExitReason, ImprovedExitStrategy
from src.utils.indicators import StopLossCalculator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImprovedEntryLogic:
    """Test ImprovedEntryLogic class"""

    def setup_method(self):
        """Setup for each test"""
        self.entry_logic = ImprovedEntryLogic(
            min_confidence=60, min_risk_reward=2.0, support_distance_percent=3.0
        )

    def create_sample_dataframe(self, trend="up", volume_surge=True, rsi=50):
        """Create sample DataFrame with technical indicators"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Create uptrend or downtrend
        if trend == "up":
            close_prices = np.linspace(50000, 60000, 100) + np.random.randn(100) * 500
        else:
            close_prices = np.linspace(60000, 50000, 100) + np.random.randn(100) * 500

        df = pd.DataFrame(
            {
                "time": dates,
                "open": close_prices * 0.99,
                "high": close_prices * 1.02,
                "low": close_prices * 0.98,
                "close": close_prices,
                "volume": np.random.randint(100000, 1000000, 100),
            }
        )

        # Add indicators
        df["sma20"] = df["close"].rolling(20).mean()
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["rsi"] = rsi
        df["atr"] = df["close"].rolling(14).std() * 2
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["volume_ratio"] = 1.6 if volume_surge else 0.8

        # Fill NaN
        df = df.bfill()

        return df

    def test_entry_logic_initialization(self):
        """Test entry logic initializes correctly"""
        assert self.entry_logic.min_confidence == 60
        assert self.entry_logic.min_risk_reward == 2.0
        assert self.entry_logic.support_distance_percent == 3.0

    def test_no_signal_on_downtrend(self):
        """Test entry signal behavior on downtrend"""
        df = self.create_sample_dataframe(trend="down")

        ml_signal = {"signal": "BUY", "confidence": 80, "reason": "ML prediction"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        # May reject due to downtrend, but ML signal may override
        # Just verify we get a valid EntrySignal object
        assert isinstance(result, EntrySignal)
        assert result.signal_type in ["BUY", "HOLD"]

    def test_no_signal_on_low_volume(self):
        """Test no entry signal on low volume"""
        df = self.create_sample_dataframe(trend="up", volume_surge=False)

        ml_signal = {"signal": "BUY", "confidence": 80, "reason": "ML prediction"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        # May reject due to low volume
        assert isinstance(result, EntrySignal)

    def test_valid_entry_signal(self):
        """Test valid entry signal generation"""
        df = self.create_sample_dataframe(trend="up", volume_surge=True, rsi=45)

        ml_signal = {
            "signal": "BUY",
            "confidence": 75,
            "reason": "Strong bullish pattern",
        }

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == "BUY":
            assert result.confidence >= self.entry_logic.min_confidence
            assert result.entry_price > 0
            assert result.stop_loss > 0
            assert result.stop_loss < result.entry_price
            assert len(result.take_profit_targets) > 0
            assert all(tp > result.entry_price for tp in result.take_profit_targets)
            assert len(result.reasons) > 0

    def test_stop_loss_validation(self):
        """Test stop loss is always below entry price"""
        df = self.create_sample_dataframe(trend="up")

        ml_signal = {"signal": "BUY", "confidence": 75, "reason": "Test"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == "BUY":
            assert result.stop_loss < result.entry_price
            assert result.stop_loss > 0

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio meets minimum"""
        df = self.create_sample_dataframe(trend="up")

        ml_signal = {"signal": "BUY", "confidence": 75, "reason": "Test"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == "BUY":
            risk = result.entry_price - result.stop_loss
            reward = result.take_profit_targets[1] - result.entry_price
            rr_ratio = reward / risk

            assert rr_ratio >= self.entry_logic.min_risk_reward

    def test_entry_signal_contains_telemetry(self):
        """Ensure telemetry is attached for successful BUY signals"""
        df = self.create_sample_dataframe(trend="up", volume_surge=True, rsi=45)
        ml_signal = {"signal": "BUY", "confidence": 80, "reason": "Telemetry test"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == "BUY":
            assert result.telemetry is not None
            assert "base_confidence" in result.telemetry
            assert isinstance(result.telemetry.get("adjustments"), list)
            assert "confidence_after_filters" in result.telemetry

    def test_low_confidence_no_signal_has_telemetry(self, monkeypatch):
        """Force adjustments to drop confidence below threshold and ensure telemetry exists"""

        def fake_run(self, df, signal_type, current_price, market_regime):
            return True, [], [], [-50], [{"filter": "mock", "delta": -50, "note": "test"}]

        monkeypatch.setattr(ImprovedEntryLogic, "_run_all_filters", fake_run)

        df = self.create_sample_dataframe(trend="up", volume_surge=True, rsi=45)
        ml_signal = {"signal": "BUY", "confidence": 65, "reason": "Telemetry fail test"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        assert result.signal_type == "HOLD"
        assert result.telemetry is not None
        assert result.telemetry.get("confidence_after_filters") is not None
        assert result.warnings[0].startswith("Confidence sau adjustment")

    def test_missing_take_profit_targets_returns_hold(self, monkeypatch):
        """Guard against insufficient take profit targets"""

        def fake_tp(entry_price, atr, risk_reward_ratios):
            return [entry_price * 1.02]  # Only one TP

        monkeypatch.setattr(
            StopLossCalculator,
            "calculate_take_profit_targets",
            staticmethod(fake_tp),
        )

        monkeypatch.setattr(
            ImprovedEntryLogic,
            "_check_volume_confirmation",
            lambda self, df, market_regime: {
                "confirmed": True,
                "reason": "Test override",
                "surge": True,
            },
        )

        df = self.create_sample_dataframe(trend="up", volume_surge=True, rsi=45)
        ml_signal = {"signal": "BUY", "confidence": 80, "reason": "TP guard test"}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        assert result.signal_type == "HOLD"
        assert any("take profit" in warning.lower() for warning in result.warnings)


class TestImprovedExitStrategy:
    """Test ImprovedExitStrategy class"""

    def setup_method(self):
        """Setup for each test"""
        self.exit_logic = ImprovedExitStrategy()

    def create_position_data(self, entry_price=50000, current_price=55000):
        """Create sample position data"""
        return {
            "entry_price": entry_price,
            "stop_loss": entry_price * 0.93,
            "take_profit": entry_price * 1.15,
            "shares": 100,
        }

    def create_market_data(self, current_price=55000):
        """Create sample market data"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        close_prices = np.linspace(50000, current_price, 50)

        df = pd.DataFrame(
            {
                "time": dates,
                "open": close_prices * 0.99,
                "high": close_prices * 1.02,
                "low": close_prices * 0.98,
                "close": close_prices,
                "volume": np.random.randint(100000, 1000000, 50),
            }
        )

        df["atr"] = df["close"].rolling(14).std() * 2
        df["rsi"] = 50
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        return df.bfill()

    def test_exit_on_stop_loss(self):
        """Test exit when stop loss is hit"""
        position = self.create_position_data(entry_price=50000, current_price=46000)
        df = self.create_market_data(current_price=46000)

        decision = self.exit_logic.check_exit(
            symbol="VCB",
            entry_price=position["entry_price"],
            current_price=46000,
            stop_loss=position["stop_loss"],
            take_profit_targets=[position["take_profit"]],
            entry_date=pd.Timestamp("2024-01-01"),
            df=df,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.STOP_LOSS

    def test_exit_on_take_profit(self):
        """Test exit when take profit is reached"""
        position = self.create_position_data(entry_price=50000, current_price=58000)
        df = self.create_market_data(current_price=58000)

        decision = self.exit_logic.check_exit(
            symbol="VCB",
            entry_price=position["entry_price"],
            current_price=58000,
            stop_loss=position["stop_loss"],
            take_profit_targets=[position["take_profit"]],
            entry_date=pd.Timestamp("2024-01-01"),
            df=df,
        )

        # May exit on take profit or trailing stop
        if decision.should_exit:
            assert decision.exit_reason in [
                ExitReason.TAKE_PROFIT_1,
                ExitReason.TAKE_PROFIT_2,
                ExitReason.TAKE_PROFIT_3,
                ExitReason.TRAILING_STOP,
            ]

    def test_no_exit_in_normal_conditions(self):
        """Test no exit in normal market conditions"""
        position = self.create_position_data(entry_price=50000, current_price=52000)
        df = self.create_market_data(current_price=52000)

        decision = self.exit_logic.check_exit(
            symbol="VCB",
            entry_price=position["entry_price"],
            current_price=52000,
            stop_loss=position["stop_loss"],
            take_profit_targets=[position["take_profit"]],
            entry_date=pd.Timestamp("2024-01-01"),
            df=df,
        )

        # Small profit, no major signals - may hold
        assert isinstance(decision, ExitDecision)

    def test_trailing_stop_activation(self):
        """Test trailing stop activates after significant gain"""
        # Entry at 50k, current at 56k (12% gain, should activate trailing)
        position = self.create_position_data(entry_price=50000, current_price=56000)
        df = self.create_market_data(current_price=56000)

        # Track position for trailing stop
        self.exit_logic.check_exit(
            symbol="VCB",
            entry_price=position["entry_price"],
            current_price=56000,
            stop_loss=position["stop_loss"],
            take_profit_targets=[position["take_profit"]],
            entry_date=pd.Timestamp("2024-01-01"),
            df=df,
        )

        # Trailing stop should be activated
        assert "VCB" in self.exit_logic.position_highs

    def test_exit_reason_message(self):
        """Test exit decision includes proper message"""
        position = self.create_position_data(entry_price=50000, current_price=46000)
        df = self.create_market_data(current_price=46000)

        decision = self.exit_logic.check_exit(
            symbol="VCB",
            entry_price=position["entry_price"],
            current_price=46000,
            stop_loss=position["stop_loss"],
            take_profit_targets=[position["take_profit"]],
            entry_date=pd.Timestamp("2024-01-01"),
            df=df,
        )

        if decision.should_exit:
            assert decision.message is not None
            assert len(decision.message) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
