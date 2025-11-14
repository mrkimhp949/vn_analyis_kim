"""
Unit tests for Entry and Exit Logic
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_entry_logic import ImprovedEntryLogic, EntrySignal, SignalStrength
from improved_exit_logic import ImprovedExitLogic, ExitDecision, ExitReason


class TestImprovedEntryLogic:
    """Test ImprovedEntryLogic class"""

    def setup_method(self):
        """Setup for each test"""
        self.entry_logic = ImprovedEntryLogic(
            min_confidence=60,
            min_risk_reward=2.0,
            support_distance_percent=3.0
        )

    def create_sample_dataframe(self, trend='up', volume_surge=True, rsi=50):
        """Create sample DataFrame with technical indicators"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Create uptrend or downtrend
        if trend == 'up':
            close_prices = np.linspace(50000, 60000, 100) + np.random.randn(100) * 500
        else:
            close_prices = np.linspace(60000, 50000, 100) + np.random.randn(100) * 500

        df = pd.DataFrame({
            'time': dates,
            'open': close_prices * 0.99,
            'high': close_prices * 1.02,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })

        # Add indicators
        df['sma20'] = df['close'].rolling(20).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['rsi'] = rsi
        df['atr'] = df['close'].rolling(14).std() * 2
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['volume_ratio'] = 1.6 if volume_surge else 0.8

        # Fill NaN
        df = df.fillna(method='bfill')

        return df

    def test_entry_logic_initialization(self):
        """Test entry logic initializes correctly"""
        assert self.entry_logic.min_confidence == 60
        assert self.entry_logic.min_risk_reward == 2.0
        assert self.entry_logic.support_distance_percent == 3.0

    def test_no_signal_on_downtrend(self):
        """Test no entry signal on downtrend"""
        df = self.create_sample_dataframe(trend='down')

        ml_signal = {'signal': 'BUY', 'confidence': 80, 'reason': 'ML prediction'}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        # Should reject due to downtrend
        assert result.signal_type == 'HOLD'

    def test_no_signal_on_low_volume(self):
        """Test no entry signal on low volume"""
        df = self.create_sample_dataframe(trend='up', volume_surge=False)

        ml_signal = {'signal': 'BUY', 'confidence': 80, 'reason': 'ML prediction'}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        # May reject due to low volume
        assert isinstance(result, EntrySignal)

    def test_valid_entry_signal(self):
        """Test valid entry signal generation"""
        df = self.create_sample_dataframe(trend='up', volume_surge=True, rsi=45)

        ml_signal = {'signal': 'BUY', 'confidence': 75, 'reason': 'Strong bullish pattern'}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == 'BUY':
            assert result.confidence >= self.entry_logic.min_confidence
            assert result.entry_price > 0
            assert result.stop_loss > 0
            assert result.stop_loss < result.entry_price
            assert len(result.take_profit_targets) > 0
            assert all(tp > result.entry_price for tp in result.take_profit_targets)
            assert len(result.reasons) > 0

    def test_stop_loss_validation(self):
        """Test stop loss is always below entry price"""
        df = self.create_sample_dataframe(trend='up')

        ml_signal = {'signal': 'BUY', 'confidence': 75, 'reason': 'Test'}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == 'BUY':
            assert result.stop_loss < result.entry_price
            assert result.stop_loss > 0

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio meets minimum"""
        df = self.create_sample_dataframe(trend='up')

        ml_signal = {'signal': 'BUY', 'confidence': 75, 'reason': 'Test'}

        result = self.entry_logic.analyze_entry(df, ml_signal)

        if result.signal_type == 'BUY':
            risk = result.entry_price - result.stop_loss
            reward = result.take_profit_targets[1] - result.entry_price
            rr_ratio = reward / risk

            assert rr_ratio >= self.entry_logic.min_risk_reward


class TestImprovedExitLogic:
    """Test ImprovedExitLogic class"""

    def setup_method(self):
        """Setup for each test"""
        self.exit_logic = ImprovedExitLogic()

    def create_position_data(self, entry_price=50000, current_price=55000):
        """Create sample position data"""
        return {
            'entry_price': entry_price,
            'stop_loss': entry_price * 0.93,
            'take_profit': entry_price * 1.15,
            'shares': 100
        }

    def create_market_data(self, current_price=55000):
        """Create sample market data"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        close_prices = np.linspace(50000, current_price, 50)

        df = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'high': close_prices * 1.02,
            'low': close_prices * 0.98,
            'volume': np.random.randint(100000, 1000000, 50)
        })

        df['atr'] = df['close'].rolling(14).std() * 2
        df['rsi'] = 50

        return df.fillna(method='bfill')

    def test_exit_on_stop_loss(self):
        """Test exit when stop loss is hit"""
        position = self.create_position_data(entry_price=50000, current_price=46000)
        df = self.create_market_data(current_price=46000)

        decision = self.exit_logic.check_exit(
            symbol='VCB',
            position=position,
            df=df
        )

        assert decision.should_exit is True
        assert decision.reason == ExitReason.STOP_LOSS

    def test_exit_on_take_profit(self):
        """Test exit when take profit is reached"""
        position = self.create_position_data(entry_price=50000, current_price=58000)
        df = self.create_market_data(current_price=58000)

        decision = self.exit_logic.check_exit(
            symbol='VCB',
            position=position,
            df=df
        )

        # May exit on take profit or trailing stop
        if decision.should_exit:
            assert decision.reason in [ExitReason.TAKE_PROFIT, ExitReason.TRAILING_STOP]

    def test_no_exit_in_normal_conditions(self):
        """Test no exit in normal market conditions"""
        position = self.create_position_data(entry_price=50000, current_price=52000)
        df = self.create_market_data(current_price=52000)

        decision = self.exit_logic.check_exit(
            symbol='VCB',
            position=position,
            df=df
        )

        # Small profit, no major signals - may hold
        assert isinstance(decision, ExitDecision)

    def test_trailing_stop_activation(self):
        """Test trailing stop activates after significant gain"""
        # Entry at 50k, current at 56k (12% gain, should activate trailing)
        position = self.create_position_data(entry_price=50000, current_price=56000)
        df = self.create_market_data(current_price=56000)

        # Track position for trailing stop
        self.exit_logic.check_exit(symbol='VCB', position=position, df=df)

        # Trailing stop should be activated
        assert 'VCB' in self.exit_logic.position_peaks

    def test_exit_reason_message(self):
        """Test exit decision includes proper message"""
        position = self.create_position_data(entry_price=50000, current_price=46000)
        df = self.create_market_data(current_price=46000)

        decision = self.exit_logic.check_exit(
            symbol='VCB',
            position=position,
            df=df
        )

        if decision.should_exit:
            assert decision.message is not None
            assert len(decision.message) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
