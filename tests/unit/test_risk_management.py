# -*- coding: utf-8 -*-
"""
Unit Tests for Enhanced Risk Management
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch

from src.strategies.risk_management import EnhancedRiskManager


class TestEnhancedRiskManager:
    """Test EnhancedRiskManager class"""

    @pytest.fixture
    def risk_manager(self):
        """Create a risk manager instance for testing"""
        return EnhancedRiskManager(
            total_capital=100_000_000, max_position_pct=0.2, risk_per_trade_pct=0.02
        )

    def test_initialization(self, risk_manager):
        """Test risk manager initialization"""
        assert risk_manager.total_capital == 100_000_000
        assert risk_manager.max_position_pct == 0.2
        assert risk_manager.risk_per_trade_pct == 0.02
        assert risk_manager.volatility_adjustment is True
        assert risk_manager.correlation_penalty is True
        assert risk_manager.market_regime_adjustment is True

    def test_volatility_factor_low(self, risk_manager):
        """Test volatility factor calculation for low volatility"""
        # Low volatility (< 0.015) should increase position
        factor = risk_manager._calculate_volatility_factor(0.01)
        assert factor == 1.2

    def test_volatility_factor_normal(self, risk_manager):
        """Test volatility factor calculation for normal volatility"""
        # Normal volatility (0.015-0.03) should keep position unchanged
        factor = risk_manager._calculate_volatility_factor(0.02)
        assert factor == 1.0

    def test_volatility_factor_high(self, risk_manager):
        """Test volatility factor calculation for high volatility"""
        # High volatility (0.03-0.05) should decrease position
        factor = risk_manager._calculate_volatility_factor(0.04)
        assert factor == 0.7

    def test_volatility_factor_very_high(self, risk_manager):
        """Test volatility factor calculation for very high volatility"""
        # Very high volatility (> 0.05) should decrease position significantly
        factor = risk_manager._calculate_volatility_factor(0.06)
        assert factor == 0.5

    def test_confidence_factor_high(self, risk_manager):
        """Test confidence factor for high confidence"""
        # High confidence (>= 80) should increase position
        factor = risk_manager._calculate_confidence_factor(85)
        assert factor == 1.2

    def test_confidence_factor_medium(self, risk_manager):
        """Test confidence factor for medium confidence"""
        # Medium confidence (60-80) should keep position unchanged
        factor = risk_manager._calculate_confidence_factor(70)
        assert factor == 1.0

    def test_confidence_factor_low(self, risk_manager):
        """Test confidence factor for low confidence"""
        # Low confidence (40-60) should decrease position
        factor = risk_manager._calculate_confidence_factor(50)
        assert factor == 0.7

    def test_confidence_factor_very_low(self, risk_manager):
        """Test confidence factor for very low confidence"""
        # Very low confidence (< 40) should decrease position significantly
        factor = risk_manager._calculate_confidence_factor(30)
        assert factor == 0.3

    @patch("src.data.vnindex_cache.get_cached_vnindex")
    @patch("utils.dataframe_utils.safe_get_latest")
    def test_market_regime_factor_bull(self, mock_safe_get_latest, mock_get_vnindex, risk_manager):
        """Test market regime factor in bull market"""
        # IMPROVEMENT #4: Test with market_regime dict from regime_detector
        # Bull market with high confidence
        market_regime = {
            "regime": "BULL",
            "confidence": 75,
            "tradeable": True,
            "components": {"trend": 0.5, "momentum": 0.3, "volatility": 0.3},
        }

        factor = risk_manager._calculate_market_regime_factor(market_regime)
        assert factor == 1.15  # Strong bull -> 1.15x (TIGHTENED from 1.2)

    def test_market_regime_factor_bear(self, risk_manager):
        """Test market regime factor in bear market"""
        # IMPROVEMENT #4: Test with market_regime dict from regime_detector
        # Bear market with high confidence
        market_regime = {
            "regime": "BEAR",
            "confidence": 75,
            "tradeable": True,
            "components": {"trend": -0.5, "momentum": -0.3, "volatility": 0.4},
        }

        factor = risk_manager._calculate_market_regime_factor(market_regime)
        assert factor == 0.35  # Strong bear -> 0.35x (TIGHTENED from 0.4)

    def test_market_regime_factor_sideway(self, risk_manager):
        """Test market regime factor in sideway market"""
        # IMPROVEMENT #4: Test with market_regime dict from regime_detector
        # Sideways market with low volatility
        market_regime = {
            "regime": "SIDEWAYS",
            "confidence": 60,
            "tradeable": True,
            "components": {"trend": 0.1, "momentum": 0.0, "volatility": 0.3},
        }

        factor = risk_manager._calculate_market_regime_factor(market_regime)
        assert factor == 0.80  # Sideways + low vol -> 0.80x (TIGHTENED from 0.85)

    @patch("src.data.vnindex_cache.get_cached_vnindex")
    def test_market_regime_factor_error_handling(self, mock_get_vnindex, risk_manager):
        """Test market regime factor returns default on error"""
        # Mock get_cached_vnindex to raise exception
        mock_get_vnindex.side_effect = Exception("Data not available")

        factor = risk_manager._calculate_market_regime_factor()
        assert factor == 1.0  # Default value

    def test_enhanced_limit_orders_buy_high_confidence(self, risk_manager):
        """Test enhanced limit orders for BUY with high confidence"""
        current_price = 50000
        atr = 1000
        confidence = 80

        orders = risk_manager.suggest_enhanced_limit_orders(
            current_price, atr, signal="BUY", confidence=confidence
        )

        assert "aggressive" in orders
        assert "moderate" in orders
        assert "conservative" in orders
        assert "note" in orders

        # Check that prices are below current price (for BUY)
        assert orders["aggressive"] < current_price
        assert orders["moderate"] < current_price
        assert orders["conservative"] < current_price

        # Check ordering (conservative < moderate < aggressive)
        assert orders["conservative"] < orders["moderate"] < orders["aggressive"]

    def test_enhanced_limit_orders_buy_low_confidence(self, risk_manager):
        """Test enhanced limit orders for BUY with low confidence"""
        current_price = 50000
        atr = 1000
        confidence = 30

        orders = risk_manager.suggest_enhanced_limit_orders(
            current_price, atr, signal="BUY", confidence=confidence
        )

        # With low confidence, suggested prices should be more conservative
        # (further from current price)
        high_conf_orders = risk_manager.suggest_enhanced_limit_orders(
            current_price, atr, signal="BUY", confidence=80
        )

        assert orders["moderate"] < high_conf_orders["moderate"]

    def test_enhanced_limit_orders_sell(self, risk_manager):
        """Test enhanced limit orders for SELL"""
        current_price = 50000
        atr = 1000
        confidence = 70

        orders = risk_manager.suggest_enhanced_limit_orders(
            current_price, atr, signal="SELL", confidence=confidence
        )

        assert "aggressive" in orders
        assert "moderate" in orders
        assert "conservative" in orders

        # Check that prices are above current price (for SELL)
        assert orders["aggressive"] > current_price
        assert orders["moderate"] > current_price
        assert orders["conservative"] > current_price

        # Check ordering (aggressive < moderate < conservative)
        assert orders["aggressive"] < orders["moderate"] < orders["conservative"]

    @patch("src.strategies.risk_management.RiskManager.calculate_position_size")
    def test_calculate_enhanced_position_size_basic(self, mock_parent_calc, risk_manager):
        """Test basic enhanced position size calculation"""
        # Mock parent class method
        mock_parent_calc.return_value = {
            "shares": 1000,
            "value": 50_000_000,
            "risk_per_share": 500,
            "max_loss": 500_000,
            "stop_loss": 49_500,
        }

        result = risk_manager.calculate_enhanced_position_size(
            symbol="VCB",
            price=50000,
            atr=1000,
            confidence=70,
            signal="BUY",
            market_volatility=0.02,
        )

        assert "shares" in result
        assert "value" in result
        assert result["shares"] > 0

    @patch("src.strategies.risk_management.RiskManager.calculate_position_size")
    def test_calculate_enhanced_position_size_zero_shares(self, mock_parent_calc, risk_manager):
        """Test enhanced position size when base calculation returns zero"""
        # Mock parent class to return zero shares
        mock_parent_calc.return_value = {
            "shares": 0,
            "value": 0,
            "risk_per_share": 0,
            "max_loss": 0,
            "stop_loss": 0,
        }

        result = risk_manager.calculate_enhanced_position_size(
            symbol="VCB",
            price=50000,
            atr=1000,
            confidence=70,
            signal="BUY",
            market_volatility=0.02,
        )

        assert result["shares"] == 0

    @patch("src.strategies.risk_management.RiskManager.calculate_position_size")
    def test_calculate_enhanced_position_size_rounded_to_lot(self, mock_parent_calc, risk_manager):
        """Test that position size is rounded to lot of 100"""
        # Mock parent class method
        mock_parent_calc.return_value = {
            "shares": 1234,  # Not a multiple of 100
            "value": 61_700_000,
            "risk_per_share": 500,
            "max_loss": 617_000,
            "stop_loss": 49_500,
        }

        result = risk_manager.calculate_enhanced_position_size(
            symbol="VCB",
            price=50000,
            atr=1000,
            confidence=70,
            signal="BUY",
            market_volatility=0.02,
        )

        # Should be rounded to nearest 100
        assert result["shares"] % 100 == 0

    @patch("src.strategies.risk_management.RiskManager.calculate_position_size")
    def test_calculate_enhanced_position_size_respects_max_capital(
        self, mock_parent_calc, risk_manager
    ):
        """Test that position size respects maximum capital limit"""
        # Mock parent class to return very large position
        mock_parent_calc.return_value = {
            "shares": 100000,  # Very large
            "value": 5_000_000_000,
            "risk_per_share": 500,
            "max_loss": 50_000_000,
            "stop_loss": 49_500,
        }

        result = risk_manager.calculate_enhanced_position_size(
            symbol="VCB",
            price=50000,
            atr=1000,
            confidence=70,
            signal="BUY",
            market_volatility=0.02,
        )

        # Maximum allowed shares based on 20% of 100M capital at 50k price
        max_allowed_shares = int((100_000_000 * 0.2) / 50000)
        assert result["shares"] <= max_allowed_shares


class TestEnhancedRiskManagerEdgeCases:
    """Test edge cases for EnhancedRiskManager"""

    def test_extreme_volatility_values(self):
        """Test with extreme volatility values"""
        rm = EnhancedRiskManager()

        # Test with zero volatility
        factor = rm._calculate_volatility_factor(0.0)
        assert factor == 1.2

        # Test with very high volatility
        factor = rm._calculate_volatility_factor(1.0)
        assert factor == 0.5

    def test_confidence_boundary_values(self):
        """Test confidence factor at boundary values"""
        rm = EnhancedRiskManager()

        # Test at boundaries
        assert rm._calculate_confidence_factor(80) == 1.2
        assert rm._calculate_confidence_factor(60) == 1.0
        assert rm._calculate_confidence_factor(40) == 0.7
        assert rm._calculate_confidence_factor(0) == 0.3
        assert rm._calculate_confidence_factor(100) == 1.2

    def test_initialization_with_custom_values(self):
        """Test initialization with custom parameter values"""
        rm = EnhancedRiskManager(
            total_capital=500_000_000, max_position_pct=0.15, risk_per_trade_pct=0.01
        )

        assert rm.total_capital == 500_000_000
        assert rm.max_position_pct == 0.15
        assert rm.risk_per_trade_pct == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
