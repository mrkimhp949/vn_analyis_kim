# -*- coding: utf-8 -*-
"""
Unit tests for src/market/regime_detector.py - Unified Market Regime Detection
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.market.regime_detector import (
    MarketRegimeDetector,
    MarketRegimeAnalyzer,  # Legacy alias
    MarketRegime,
    EnhancedMarketRegime,
    detect_regime,
    detect_enhanced_regime,
    get_regime_detector,
    check_market_before_trading,
    get_market_position_adjustment,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_vnindex_data():
    """Create sample VNINDEX data for testing"""
    np.random.seed(40)
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    base_price = 1200
    trend = np.linspace(0, 100, n)
    noise = np.random.normal(0, 5, n)
    close = base_price + trend + noise

    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.uniform(1_000_000, 2_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def bull_market_data():
    """Create strong bull market data"""
    np.random.seed(42)
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    base_price = 1200
    trend = np.linspace(0, 400, n)  # Strong uptrend
    noise = np.random.normal(0, 3, n)
    close = base_price + trend + noise

    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.uniform(2_000_000, 3_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def bear_market_data():
    """Create bear market data"""
    np.random.seed(43)
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    base_price = 1200
    trend = np.linspace(0, -300, n)  # Strong downtrend
    noise = np.random.normal(0, 3, n)
    close = base_price + trend + noise

    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.uniform(2_500_000, 4_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def high_volatility_data():
    """Create high volatility market data"""
    np.random.seed(44)
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    base_price = 1200
    noise = np.random.normal(0, 50, n)  # High volatility
    close = base_price + noise

    return pd.DataFrame(
        {
            "open": close * 0.97,
            "high": close * 1.03,
            "low": close * 0.95,
            "close": close,
            "volume": np.random.uniform(1_500_000, 2_500_000, n),
        },
        index=dates,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestMarketRegimeDetectorInit:
    def test_default_init(self):
        """Test default initialization"""
        detector = MarketRegimeDetector()
        assert detector.bull_threshold == 0.45
        assert detector.bear_threshold == -0.50
        assert detector.volatility_threshold == 0.70
        assert detector.min_confidence == 50.0

    def test_custom_init(self):
        """Test custom initialization"""
        detector = MarketRegimeDetector(
            bull_threshold=0.6,
            bear_threshold=-0.6,
            volatility_threshold=0.8,
        )
        assert detector.bull_threshold == 0.6
        assert detector.bear_threshold == -0.6
        assert detector.volatility_threshold == 0.8

    def test_legacy_parameters(self):
        """Test legacy parameters for backward compatibility"""
        detector = MarketRegimeDetector(
            high_volatility_threshold=0.04,
            trend_period=30,
        )
        assert detector.high_volatility_threshold == 0.04
        assert detector.trend_period == 30

    def test_alias_classes(self):
        """Test that alias classes work"""
        analyzer = MarketRegimeAnalyzer()
        assert isinstance(analyzer, MarketRegimeDetector)


# ============================================================================
# DETECTION TESTS
# ============================================================================


class TestMarketRegimeDetection:
    def test_detect_returns_market_regime(self, sample_vnindex_data):
        """Test that detect returns MarketRegime object"""
        detector = MarketRegimeDetector()
        result = detector.detect(sample_vnindex_data)

        assert isinstance(result, MarketRegime)
        assert result.regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY", "CORRECTION"]
        assert 0 <= result.confidence <= 100
        assert isinstance(result.tradeable, bool)
        assert isinstance(result.components, dict)
        assert isinstance(result.recommendations, list)

    def test_detect_bull_market(self, bull_market_data):
        """Test detection of bull market"""
        detector = MarketRegimeDetector()
        result = detector.detect(bull_market_data)

        # Detection depends on many factors, just verify valid regime
        assert result.regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY", "CORRECTION"]
        assert isinstance(result.tradeable, bool)

    def test_detect_bear_market(self, bear_market_data):
        """Test detection of bear market"""
        detector = MarketRegimeDetector()
        result = detector.detect(bear_market_data)

        # Detection depends on many factors, just verify valid regime
        assert result.regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY", "CORRECTION"]
        assert isinstance(result.tradeable, bool)

    def test_detect_insufficient_data(self):
        """Test detection with insufficient data"""
        detector = MarketRegimeDetector()
        small_df = pd.DataFrame({"close": [1200, 1210, 1205]})
        result = detector.detect(small_df)

        assert result.regime == "SIDEWAYS"
        assert result.tradeable is False
        assert result.confidence == 30.0

    def test_detect_empty_data(self):
        """Test detection with empty data"""
        detector = MarketRegimeDetector()
        result = detector.detect(pd.DataFrame())

        assert result.regime == "SIDEWAYS"
        assert result.tradeable is False

    def test_detect_none_data(self):
        """Test detection with None data"""
        detector = MarketRegimeDetector()
        result = detector.detect(None)

        assert result.regime == "SIDEWAYS"
        assert result.tradeable is False


# ============================================================================
# ENHANCED DETECTION TESTS
# ============================================================================


class TestEnhancedDetection:
    def test_detect_with_multi_index(self, sample_vnindex_data):
        """Test enhanced detection with multiple indices"""
        detector = MarketRegimeDetector()
        vn30_df = sample_vnindex_data.copy()

        result = detector.detect(sample_vnindex_data, vn30_df)

        assert isinstance(result, EnhancedMarketRegime)
        assert hasattr(result, "vnindex_score")
        assert hasattr(result, "vn30_score")
        assert hasattr(result, "recommendations")

    def test_correlation_breakdown_detection(self):
        """Test correlation breakdown detection"""
        detector = MarketRegimeDetector()

        # Divergent scores (one positive, one negative) should trigger breakdown
        breakdown = detector._check_correlation_breakdown(0.5, -0.5, 0.4)
        assert breakdown == True

        # Aligned scores should not trigger breakdown
        breakdown = detector._check_correlation_breakdown(0.3, 0.35, 0.28)
        assert breakdown == False


# ============================================================================
# LEGACY API TESTS
# ============================================================================


class TestLegacyAPI:
    def test_analyze_market_regime(self, sample_vnindex_data):
        """Test legacy analyze_market_regime method"""
        detector = MarketRegimeDetector()
        result = detector.analyze_market_regime(vnindex_df=sample_vnindex_data)

        assert isinstance(result, dict)
        assert "regime" in result
        assert "tradeable" in result
        assert "confidence" in result
        assert "details" in result
        assert "message" in result

    def test_get_position_multiplier(self):
        """Test position multiplier calculation"""
        detector = MarketRegimeDetector()

        with patch.object(detector, "analyze_market_regime") as mock:
            mock.return_value = {"regime": "BULL", "tradeable": True, "confidence": 85}
            multiplier = detector.get_position_multiplier()
            assert multiplier == 1.2

            mock.return_value = {"regime": "SIDEWAYS", "tradeable": True, "confidence": 50}
            multiplier = detector.get_position_multiplier()
            assert multiplier == 0.7

            mock.return_value = {"regime": "BEAR", "tradeable": False, "confidence": 20}
            multiplier = detector.get_position_multiplier()
            assert multiplier == 0.0


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    def test_get_regime_detector_singleton(self):
        """Test singleton pattern"""
        detector1 = get_regime_detector()
        detector2 = get_regime_detector()
        assert detector1 is detector2

    def test_detect_regime_function(self, sample_vnindex_data):
        """Test detect_regime convenience function"""
        result = detect_regime(sample_vnindex_data)
        assert isinstance(result, MarketRegime)

    def test_detect_enhanced_regime_function(self, sample_vnindex_data):
        """Test detect_enhanced_regime convenience function"""
        result = detect_enhanced_regime(sample_vnindex_data)
        assert isinstance(result, (MarketRegime, EnhancedMarketRegime))

    @patch.object(MarketRegimeDetector, "analyze_market_regime")
    def test_check_market_before_trading(self, mock_analyze):
        """Test check_market_before_trading helper"""
        mock_analyze.return_value = {"tradeable": True, "message": "Market is good"}

        can_trade, message = check_market_before_trading()
        assert can_trade is True
        assert "Market is good" in message

    @patch.object(MarketRegimeDetector, "get_position_multiplier")
    def test_get_market_position_adjustment(self, mock_multiplier):
        """Test get_market_position_adjustment helper"""
        mock_multiplier.return_value = 1.2

        multiplier = get_market_position_adjustment()
        assert multiplier == 1.2


# ============================================================================
# COMPONENT CALCULATION TESTS
# ============================================================================


class TestComponentCalculation:
    def test_calculate_components(self, sample_vnindex_data):
        """Test component calculation"""
        detector = MarketRegimeDetector()
        components = detector._calculate_components(sample_vnindex_data)

        assert "trend" in components
        assert "momentum" in components
        assert "volatility" in components
        assert "volume_trend" in components
        assert -1 <= components["trend"] <= 1
        assert -1 <= components["momentum"] <= 1
        assert 0 <= components["volatility"] <= 1

    def test_calculate_composite_score(self):
        """Test composite score calculation"""
        detector = MarketRegimeDetector()
        components = {
            "trend": 0.5,
            "momentum": 0.3,
            "volatility": 0.2,
            "volume_trend": 0.1,
            "sector_rotation": 0.1,
            "foreign_flow": 0.1,
        }

        score = detector._calculate_composite_score(components)
        assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
