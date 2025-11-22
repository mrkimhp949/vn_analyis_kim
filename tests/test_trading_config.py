"""
Unit tests for Trading Configuration
"""

import os
import sys

import pytest

from src.config.exceptions import ConfigurationError
from src.config.trading_config import Config, TradingConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTradingConfig:
    """Test TradingConfig validation"""

    def test_valid_config(self):
        """Test valid configuration"""
        config = TradingConfig(
            min_confidence=70,
            min_risk_reward=2.5,
            max_position_size=0.10,
            max_positions=5,
        )

        # Should not raise exception
        config.validate()

    def test_invalid_confidence(self):
        """Test invalid confidence values"""
        # Too low
        config = TradingConfig(min_confidence=-10)
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "min_confidence must be between 0 and 100" in str(exc_info.value)

        # Too high
        config = TradingConfig(min_confidence=150)
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "min_confidence must be between 0 and 100" in str(exc_info.value)

    def test_invalid_risk_reward(self):
        """Test invalid risk/reward ratio"""
        config = TradingConfig(min_risk_reward=0.5)
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "min_risk_reward must be >= 1.0" in str(exc_info.value)

    def test_invalid_position_size(self):
        """Test invalid position sizes"""
        # Max position too high
        config = TradingConfig(max_position_size=1.5)
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "max_position_size must be between 0 and 1.0" in str(exc_info.value)

        # Min > Max
        config = TradingConfig(min_position_size=0.20, max_position_size=0.10)
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "min_position_size" in str(exc_info.value)
        assert "max_position_size" in str(exc_info.value)

    def test_invalid_max_positions(self):
        """Test invalid max positions"""
        config = TradingConfig(max_positions=0)
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "max_positions must be >= 1" in str(exc_info.value)

    def test_edge_cases(self):
        """Test edge case values"""
        # Minimum valid values (respecting cross-field constraints)
        config = TradingConfig(
            total_capital=100_000_000,  # Need enough capital for min practical size
            min_confidence=0,
            min_risk_reward=1.0,
            min_position_size=0.01,
            max_position_size=0.02,
            max_positions=1,
            max_positions_per_sector=1,  # Must be <= max_positions
            max_portfolio_risk=0.01,
            max_sector_exposure=0.02,  # Must be >= max_position_size
            stop_loss_percent=-2.0,  # Need to keep within portfolio risk
        )
        config.validate()  # Should not raise

        # Maximum valid values (respecting cross-field constraints)
        config = TradingConfig(
            min_confidence=100,
            min_risk_reward=10.0,
            max_position_size=1.0,  # 100% in one position
            max_positions=1,  # Can only have 1 position if size is 100%
            max_positions_per_sector=1,  # Must be <= max_positions
            max_portfolio_risk=1.0,
            max_sector_exposure=1.0,
            stop_loss_percent=-100.0,  # Extreme but mathematically valid
        )
        config.validate()  # Should not raise


class TestConfig:
    """Test full Config class"""

    def test_config_load(self):
        """Test loading configuration"""
        # Set some env vars for testing
        os.environ["MIN_CONFIDENCE"] = "75"
        os.environ["MAX_POSITIONS"] = "8"

        config = Config.load()

        assert config.trading.min_confidence == 75
        assert config.trading.max_positions == 8

    def test_config_summary(self):
        """Test configuration summary"""
        config = Config.load()
        summary = config.summary()

        assert "CONFIGURATION" in summary
        assert "Data:" in summary
        assert "Trading:" in summary
        assert "API:" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
