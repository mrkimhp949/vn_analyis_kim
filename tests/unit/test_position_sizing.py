"""
Unit tests for EnhancedPositionSizer
Comprehensive test coverage for position sizing logic
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.strategies.position_sizing import (
    EnhancedPositionSizer,
    EnhancedPositionSize,
    PositionSizingConstants,
    MarketRegimeInfo,
    CorrelationCache,
)
from src.config.exceptions import RiskManagementError
from src.config.constants import VIETNAM_LOT_SIZE


class TestPositionSizingConstants:
    """Test constants are properly defined."""

    def test_risk_thresholds(self):
        assert 0 < PositionSizingConstants.MIN_RISK_PERCENT < 0.05
        assert 0 < PositionSizingConstants.DEFAULT_RISK_PERCENT < 0.10

    def test_kelly_constants(self):
        assert 0 < PositionSizingConstants.MAX_KELLY_PERCENT <= 0.30
        assert 0 < PositionSizingConstants.DEFAULT_KELLY_FRACTION <= 1.0

    def test_correlation_thresholds(self):
        assert (
            PositionSizingConstants.MEDIUM_CORRELATION_THRESHOLD
            < PositionSizingConstants.HIGH_CORRELATION_THRESHOLD
        )


class TestMarketRegimeInfo:
    """Test MarketRegimeInfo dataclass."""

    def test_default_values(self):
        regime = MarketRegimeInfo()
        assert regime.regime == "SIDEWAYS"
        assert regime.confidence == 50.0
        assert regime.tradeable is True

    def test_from_dict(self):
        data = {
            "regime": "BULL",
            "confidence": 75.0,
            "tradeable": True,
            "description": "Strong uptrend",
        }
        regime = MarketRegimeInfo.from_dict(data)
        assert regime.regime == "BULL"
        assert regime.confidence == 75.0

    def test_from_dict_none(self):
        regime = MarketRegimeInfo.from_dict(None)
        assert regime.regime == "SIDEWAYS"


class TestCorrelationCache:
    """Test thread-safe correlation cache."""

    def test_cache_miss(self):
        cache = CorrelationCache(ttl=3600, maxsize=100)
        result = cache.get("VNM", "FPT")
        assert result is None

    def test_cache_hit(self):
        cache = CorrelationCache(ttl=3600, maxsize=100)
        cache.set("VNM", "FPT", 0.75)
        result = cache.get("VNM", "FPT")
        assert result == 0.75

    def test_cache_key_order_independent(self):
        cache = CorrelationCache(ttl=3600, maxsize=100)
        cache.set("VNM", "FPT", 0.75)
        # Should work regardless of order
        assert cache.get("FPT", "VNM") == 0.75

    def test_cache_expiry(self):
        cache = CorrelationCache(ttl=0, maxsize=100)  # Immediate expiry
        cache.set("VNM", "FPT", 0.75)
        result = cache.get("VNM", "FPT")
        assert result is None  # Should be expired

    def test_hit_rate(self):
        cache = CorrelationCache(ttl=3600, maxsize=100)
        cache.set("VNM", "FPT", 0.75)
        cache.get("VNM", "FPT")  # Hit
        cache.get("VNM", "FPT")  # Hit
        cache.get("ABC", "XYZ")  # Miss
        assert cache.hit_rate == 2 / 3

    def test_clear(self):
        cache = CorrelationCache(ttl=3600, maxsize=100)
        cache.set("VNM", "FPT", 0.75)
        cache.clear()
        assert cache.get("VNM", "FPT") is None
        assert cache.hit_rate == 0.0


class TestEnhancedPositionSize:
    """Test EnhancedPositionSize dataclass."""

    def test_is_valid_true(self):
        pos = EnhancedPositionSize(
            shares=100,
            value=8_000_000,
            risk_amount=400_000,
            risk_percent=0.4,
            max_loss=400_000,
            position_percent=8.0,
            kelly_percent=5.0,
        )
        assert pos.is_valid() is True

    def test_is_valid_false_zero_shares(self):
        pos = EnhancedPositionSize(
            shares=0,
            value=0,
            risk_amount=0,
            risk_percent=0,
            max_loss=0,
            position_percent=0,
            kelly_percent=0,
        )
        assert pos.is_valid() is False

    def test_is_valid_false_below_lot_size(self):
        pos = EnhancedPositionSize(
            shares=50,  # Below VIETNAM_LOT_SIZE (100)
            value=4_000_000,
            risk_amount=200_000,
            risk_percent=0.2,
            max_loss=200_000,
            position_percent=4.0,
            kelly_percent=2.5,
        )
        assert pos.is_valid() is False


class TestEnhancedPositionSizer:
    """Test main position sizer class."""

    @pytest.fixture
    def sizer(self):
        """Create a basic sizer for testing."""
        return EnhancedPositionSizer(
            total_capital=100_000_000,
            max_risk_per_trade=0.015,
            max_position_size=0.12,
            min_position_size=0.03,
        )

    # =========================================================================
    # Basic Calculation Tests
    # =========================================================================

    def test_basic_calculation(self, sizer):
        """Test basic position size calculation."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert result.shares > 0
        assert result.shares % VIETNAM_LOT_SIZE == 0
        assert result.value > 0
        assert result.risk_percent <= sizer.max_risk_per_trade * 100 * 1.1

    def test_lot_size_rounding(self, sizer):
        """Test that shares are rounded to lot size."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert result.shares % VIETNAM_LOT_SIZE == 0

    def test_minimum_position_enforced(self, sizer):
        """Test minimum position size is enforced."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        min_value = sizer.total_capital * sizer.min_position_size
        # Either zero or at least minimum
        assert result.value == 0 or result.value >= min_value * 0.9

    # =========================================================================
    # Risk Validation Tests
    # =========================================================================

    def test_invalid_stop_loss_returns_zero(self, sizer):
        """Test that invalid stop loss returns zero position."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=80_000,  # Same as entry = 0 risk
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert result.shares == 0
        assert "Invalid stop loss" in result.warnings

    def test_tight_stop_loss_adjusted(self, sizer):
        """Test that very tight stop loss is adjusted."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=79_900,  # Only 0.125% risk - too tight
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        # Should have warning about adjustment
        assert any("tight" in w.lower() or "adjusted" in w.lower() for w in result.warnings)

    def test_portfolio_risk_limit_exceeded(self, sizer):
        """Test that exceeding portfolio risk raises error."""
        with pytest.raises(RiskManagementError) as exc_info:
            sizer.calculate_position_size(
                symbol="VNM",
                entry_price=80_000,
                stop_loss=76_000,
                take_profit=92_000,
                confidence=70,
                portfolio_risk=0.20,  # Exceeds 15% limit
                auto_detect_regime=False,
            )

        assert "Portfolio risk" in str(exc_info.value)

    def test_sector_exposure_limit_exceeded(self, sizer):
        """Test that exceeding sector exposure raises error."""
        # Add existing positions in same sector
        sizer.current_positions = {
            "FPT": {"shares": 1000, "current_price": 100_000, "sector": "Technology"},
            "CMG": {"shares": 500, "current_price": 50_000, "sector": "Technology"},
        }

        with pytest.raises(RiskManagementError) as exc_info:
            sizer.calculate_position_size(
                symbol="VNG",
                entry_price=80_000,
                stop_loss=76_000,
                take_profit=92_000,
                confidence=70,
                sector="Technology",  # Same sector
                auto_detect_regime=False,
            )

        assert "Sector" in str(exc_info.value)

    # =========================================================================
    # Kelly Criterion Tests
    # =========================================================================

    def test_kelly_calculation_valid(self, sizer):
        """Test Kelly calculation with valid inputs."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            win_rate=0.55,
            avg_win_loss_ratio=1.5,
            auto_detect_regime=False,
        )

        assert result.kelly_percent > 0
        assert "kelly" in result.adjustments

    def test_kelly_negative_returns_minimum(self, sizer):
        """Test that negative Kelly returns minimum fallback."""
        kelly = sizer._calculate_kelly(
            win_rate=0.30,  # Low win rate
            avg_win_loss_ratio=0.8,  # Poor ratio
        )

        # Should return minimum fallback, not negative
        assert kelly >= 0
        assert kelly == PositionSizingConstants.MIN_KELLY_FALLBACK

    def test_kelly_invalid_win_rate(self, sizer):
        """Test Kelly with invalid win rate."""
        kelly = sizer._calculate_kelly(
            win_rate=1.5,  # Invalid > 1
            avg_win_loss_ratio=1.5,
        )
        assert kelly == 0.0

    def test_kelly_invalid_ratio(self, sizer):
        """Test Kelly with invalid win/loss ratio."""
        kelly = sizer._calculate_kelly(
            win_rate=0.55,
            avg_win_loss_ratio=-1.0,  # Invalid negative
        )
        assert kelly == 0.0

    def test_kelly_clamped_to_max(self, sizer):
        """Test Kelly is clamped to maximum."""
        kelly = sizer._calculate_kelly(
            win_rate=0.80,  # Very high
            avg_win_loss_ratio=3.0,  # Very good
        )

        assert kelly <= PositionSizingConstants.MAX_KELLY_PERCENT

    # =========================================================================
    # Risk Multiplier Tests
    # =========================================================================

    def test_risk_multiplier_high_confidence(self, sizer):
        """Test risk multiplier with high confidence."""
        regime = MarketRegimeInfo(regime="BULL", confidence=70)
        mult = sizer._calculate_risk_multiplier(
            confidence=85,
            signal_strength="STRONG",
            regime_info=regime,
        )

        assert mult >= 1.0

    def test_risk_multiplier_low_confidence(self, sizer):
        """Test risk multiplier with low confidence."""
        regime = MarketRegimeInfo(regime="SIDEWAYS", confidence=50)
        mult = sizer._calculate_risk_multiplier(
            confidence=50,
            signal_strength="WEAK",
            regime_info=regime,
        )

        assert mult < 1.0

    def test_risk_multiplier_bear_market(self, sizer):
        """Test risk multiplier in bear market."""
        regime = MarketRegimeInfo(regime="BEAR", confidence=80)
        mult = sizer._calculate_risk_multiplier(
            confidence=70,
            signal_strength="MODERATE",
            regime_info=regime,
        )

        # Should be reduced in bear market
        assert mult < 0.8

    def test_risk_multiplier_bounds(self, sizer):
        """Test risk multiplier stays within bounds."""
        for confidence in [30, 50, 70, 90]:
            for strength in ["VERY_WEAK", "MODERATE", "VERY_STRONG"]:
                for regime in ["BULL", "BEAR", "SIDEWAYS"]:
                    regime_info = MarketRegimeInfo(regime=regime, confidence=60)
                    mult = sizer._calculate_risk_multiplier(
                        confidence=confidence,
                        signal_strength=strength,
                        regime_info=regime_info,
                    )

                    assert PositionSizingConstants.MIN_RISK_MULTIPLIER <= mult
                    assert mult <= PositionSizingConstants.MAX_RISK_MULTIPLIER

    # =========================================================================
    # Correlation Adjustment Tests
    # =========================================================================

    def test_correlation_adjustment_no_positions(self, sizer):
        """Test correlation adjustment with no existing positions."""
        adj = sizer._calculate_correlation_adjustment("VNM", "Consumer")
        assert adj == 1.0

    def test_sector_based_correlation_high(self, sizer):
        """Test sector-based correlation with many same-sector positions."""
        sizer.current_positions = {
            "FPT": {"shares": 100, "current_price": 100_000, "sector": "Tech"},
            "CMG": {"shares": 100, "current_price": 50_000, "sector": "Tech"},
            "VNG": {"shares": 100, "current_price": 80_000, "sector": "Tech"},
        }

        adj = sizer._sector_based_correlation_adjustment("Tech")
        assert adj == PositionSizingConstants.SECTOR_HIGH_ADJUSTMENT

    def test_sector_based_correlation_medium(self, sizer):
        """Test sector-based correlation with some same-sector positions."""
        sizer.current_positions = {
            "FPT": {"shares": 100, "current_price": 100_000, "sector": "Tech"},
            "CMG": {"shares": 100, "current_price": 50_000, "sector": "Tech"},
        }

        adj = sizer._sector_based_correlation_adjustment("Tech")
        assert adj == PositionSizingConstants.SECTOR_MEDIUM_ADJUSTMENT

    # =========================================================================
    # DCA Entry Tests
    # =========================================================================

    def test_dca_entries_structure(self, sizer):
        """Test DCA entries have correct structure."""
        entries = sizer._calculate_dca_entries(80_000, 1000)

        assert len(entries) == 3
        for entry in entries:
            assert "level" in entry
            assert "price" in entry
            assert "shares" in entry
            assert "percent" in entry

    def test_dca_entries_prices_descending(self, sizer):
        """Test DCA entry prices are descending."""
        entries = sizer._calculate_dca_entries(80_000, 1000)

        prices = [e["price"] for e in entries]
        assert prices[0] > prices[1] > prices[2]

    def test_dca_entries_shares_sum(self, sizer):
        """Test DCA entry shares approximately sum to total."""
        total_shares = 1000
        entries = sizer._calculate_dca_entries(80_000, total_shares)

        total_dca = sum(e["shares"] for e in entries)
        # Allow some rounding difference
        assert abs(total_dca - total_shares) <= VIETNAM_LOT_SIZE * 3

    # =========================================================================
    # Position Management Tests
    # =========================================================================

    def test_add_position(self, sizer):
        """Test adding a position."""
        sizer.add_position(
            symbol="VNM",
            shares=100,
            entry_price=80_000,
            current_price=82_000,
            sector="Consumer",
        )

        assert "VNM" in sizer.current_positions
        assert sizer.current_positions["VNM"]["shares"] == 100

    def test_remove_position(self, sizer):
        """Test removing a position."""
        sizer.add_position("VNM", 100, 80_000, 82_000, "Consumer")
        sizer.remove_position("VNM")

        assert "VNM" not in sizer.current_positions

    def test_update_position_price(self, sizer):
        """Test updating position price."""
        sizer.add_position("VNM", 100, 80_000, 82_000, "Consumer")
        sizer.update_position_price("VNM", 85_000)

        assert sizer.current_positions["VNM"]["current_price"] == 85_000

    def test_clear_positions(self, sizer):
        """Test clearing all positions."""
        sizer.add_position("VNM", 100, 80_000, 82_000, "Consumer")
        sizer.add_position("FPT", 200, 100_000, 105_000, "Tech")
        sizer.clear_positions()

        assert len(sizer.current_positions) == 0

    def test_portfolio_summary(self, sizer):
        """Test portfolio summary."""
        sizer.add_position("VNM", 100, 80_000, 82_000, "Consumer")

        summary = sizer.get_portfolio_summary()

        assert "total_capital" in summary
        assert "total_exposure" in summary
        assert "exposure_percent" in summary
        assert "position_count" in summary
        assert summary["position_count"] == 1

    # =========================================================================
    # Circuit Breaker Integration Tests
    # =========================================================================

    def test_circuit_breaker_caution_mode(self, sizer):
        """Test position reduction in circuit breaker caution mode."""
        mock_cb = Mock()
        mock_cb.is_caution_mode.return_value = True
        sizer._circuit_breaker = mock_cb

        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert "caution_mode_adj" in result.adjustments
        assert any("caution" in w.lower() for w in result.warnings)

    def test_circuit_breaker_normal_mode(self, sizer):
        """Test no reduction when circuit breaker is normal."""
        mock_cb = Mock()
        mock_cb.is_caution_mode.return_value = False
        sizer._circuit_breaker = mock_cb

        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert "caution_mode_adj" not in result.adjustments

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_zero_capital(self):
        """Test with zero capital."""
        sizer = EnhancedPositionSizer(total_capital=0)

        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        assert result.shares == 0

    def test_exposure_limit_reached(self, sizer):
        """Test when exposure limit is reached."""
        # Fill up to max exposure
        sizer.current_positions = {
            "FPT": {"shares": 10000, "current_price": 10_000, "sector": "Tech"},
        }
        # This creates 100M exposure, which is 100% of capital

        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        # Should return zero or very small position
        assert result.shares == 0 or result.value < sizer.total_capital * 0.1

    def test_very_high_price_stock(self, sizer):
        """Test with very high price stock."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=500_000,  # 500k per share
            stop_loss=475_000,
            take_profit=575_000,
            confidence=70,
            auto_detect_regime=False,
        )

        # Should still work and respect lot size
        assert result.shares % VIETNAM_LOT_SIZE == 0

    def test_very_low_price_stock(self, sizer):
        """Test with very low price stock."""
        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=5_000,  # 5k per share
            stop_loss=4_750,
            take_profit=5_750,
            confidence=70,
            auto_detect_regime=False,
        )

        # Should still work and respect lot size
        assert result.shares % VIETNAM_LOT_SIZE == 0


class TestDependencyInjection:
    """Test dependency injection functionality."""

    def test_custom_data_loader(self):
        """Test with custom data loader."""
        mock_loader = Mock(return_value=None)

        sizer = EnhancedPositionSizer(
            total_capital=100_000_000,
            data_loader=mock_loader,
        )

        # Trigger correlation calculation
        sizer.current_positions = {"FPT": {"shares": 100, "current_price": 100_000}}
        sizer._calculate_correlation("VNM", "FPT")

        # Verify custom loader was used
        assert mock_loader.called

    def test_custom_circuit_breaker(self):
        """Test with custom circuit breaker."""
        mock_cb = Mock()
        mock_cb.is_caution_mode.return_value = False

        sizer = EnhancedPositionSizer(
            total_capital=100_000_000,
            circuit_breaker=mock_cb,
        )

        result = sizer.calculate_position_size(
            symbol="VNM",
            entry_price=80_000,
            stop_loss=76_000,
            take_profit=92_000,
            confidence=70,
            auto_detect_regime=False,
        )

        # Verify custom circuit breaker was used
        assert mock_cb.is_caution_mode.called
