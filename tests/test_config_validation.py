"""
Quick test to verify cross-field validation is working
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.trading_config import TradingConfig
from src.config.exceptions import ConfigurationError


def test_max_exposure_validation():
    """Test that max_position_size * max_positions > 1.0 raises error"""
    print("Testing max exposure validation...")

    try:
        # This should FAIL: 0.15 * 10 = 1.5 > 1.0
        config = TradingConfig(
            total_capital=100_000_000,
            max_position_size=0.15,  # 15%
            max_positions=10,  # 10 positions
            min_position_size=0.02,
            max_sector_exposure=0.40,
            stop_loss_percent=-3.0,
            take_profit_percent=6.0,
            max_portfolio_risk=0.10,
        )
        config.validate()
        print("❌ FAIL: Should have raised ConfigurationError!")
        return False

    except ConfigurationError as e:
        print(f"✅ PASS: Correctly caught error:\n   {str(e)[:100]}...")
        return True


def test_valid_config():
    """Test that valid config passes"""
    print("\nTesting valid configuration...")

    try:
        # This should PASS: 0.10 * 8 = 0.8 < 1.0
        config = TradingConfig(
            total_capital=100_000_000,
            max_position_size=0.10,  # 10%
            max_positions=8,  # 8 positions
            min_position_size=0.02,
            max_sector_exposure=0.40,
            stop_loss_percent=-3.0,
            take_profit_percent=6.0,
            max_portfolio_risk=0.10,
        )
        config.validate()
        print("✅ PASS: Valid config accepted")
        return True

    except ConfigurationError as e:
        print(f"❌ FAIL: Should not have raised error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CROSS-FIELD VALIDATION")
    print("=" * 60)

    test1 = test_max_exposure_validation()
    test2 = test_valid_config()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED - Cross-field validation is working!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
