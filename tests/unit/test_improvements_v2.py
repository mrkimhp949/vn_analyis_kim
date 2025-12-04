# -*- coding: utf-8 -*-
"""
Test suite for v2.0 improvements
Verifies all enhancements work correctly
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestCircuitBreakerImprovements:
    """Test circuit breaker gradual response and configurable thresholds"""

    def test_gradual_response_levels(self, tmp_path):
        """Test warning -> caution -> trip progression (v3.0 - tightened thresholds)"""
        from src.risk.circuit_breaker import CircuitBreaker

        # Use temp stats file to avoid interference from existing state
        temp_stats = tmp_path / "test_cb_stats.json"
        breaker = CircuitBreaker(
            vnindex_drop_threshold=-2.5,
            use_conservative_threshold=False,
            warning_threshold_pct=-1.0,  # v3.0: Warning at -1.0%
            caution_threshold_pct=-1.5,  # v3.0: Caution at -1.5%
            stats_file=str(temp_stats),
        )
        # Ensure clean state
        breaker.reset()

        # Level 1: Warning at -1.0% (v3.0 tightened)
        result = breaker.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=-0.010,  # -1.0%
        )
        assert not result, "Should not trip at warning level"
        assert not breaker.caution_mode, "Should not be in caution mode yet"

        # Level 2: Caution at -1.5% (v3.0 tightened)
        result = breaker.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=-0.016,  # -1.6%
        )
        assert not result, "Should not trip at caution level"
        assert breaker.caution_mode, "Should be in caution mode"

        # Level 3: Trip at -2.5%
        result = breaker.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=-0.026,  # -2.6%
        )
        assert result, "Should trip at threshold"
        assert breaker.tripped, "Should be tripped"

    def test_position_size_multiplier(self):
        """Test position size multiplier based on market conditions"""
        from src.risk.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        # Normal conditions
        assert breaker.get_position_size_multiplier() == 1.0

        # Caution mode
        breaker.caution_mode = True
        assert breaker.get_position_size_multiplier() == 0.5

        # Tripped
        breaker.tripped = True
        assert breaker.get_position_size_multiplier() == 0.0

    def test_conservative_threshold_option(self):
        """Test conservative threshold configuration"""
        from src.risk.circuit_breaker import CircuitBreaker

        # Normal mode
        breaker_normal = CircuitBreaker(
            vnindex_drop_threshold=-2.5,
            use_conservative_threshold=False,
        )
        assert breaker_normal.vnindex_drop_threshold == -0.025

        # Conservative mode
        breaker_conservative = CircuitBreaker(
            vnindex_drop_threshold=-2.5,
            vnindex_drop_threshold_conservative=-2.0,
            use_conservative_threshold=True,
        )
        assert breaker_conservative.vnindex_drop_threshold == -0.020


class TestPositionSizingImprovements:
    """Test Kelly criterion improvements"""

    def test_negative_kelly_fallback(self):
        """Test that negative Kelly returns minimum size instead of exception"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        # Negative Kelly scenario (win_rate=30%, W/L ratio=0.5)
        # Kelly = 0.3 - (0.7/0.5) = 0.3 - 1.4 = -1.1 (negative)
        kelly = sizer._calculate_kelly(win_rate=0.30, avg_win_loss_ratio=0.5)

        # Should return minimum (1%) instead of raising exception
        assert kelly == 0.01, f"Expected 0.01, got {kelly}"

    def test_positive_kelly_calculation(self):
        """Test normal Kelly calculation"""
        from src.strategies.position_sizing import EnhancedPositionSizer

        sizer = EnhancedPositionSizer()

        # Positive Kelly scenario (win_rate=60%, W/L ratio=2.0)
        # Kelly = 0.6 - (0.4/2.0) = 0.6 - 0.2 = 0.4
        # Half-Kelly = 0.2
        kelly = sizer._calculate_kelly(win_rate=0.60, avg_win_loss_ratio=2.0)

        assert kelly > 0, "Kelly should be positive"
        assert kelly <= 0.25, "Kelly should be capped at 25%"


class TestEntryLogicImprovements:
    """Test simplified entry logic with soft filters"""

    def test_reduced_min_confidence(self):
        """Test that min_confidence is set correctly (v3.0 - constructor defaults)"""
        from src.strategies.entry_logic import ImprovedEntryLogic

        logic = ImprovedEntryLogic()

        # v3.0: These are constructor defaults
        assert logic.min_confidence == 45, f"Expected 45, got {logic.min_confidence}"
        assert logic.min_risk_reward == 1.0, f"Expected 1.0, got {logic.min_risk_reward}"

    def test_soft_filter_mode_enabled(self):
        """Test that soft filter mode is enabled by default (v2.1 further relaxed)"""
        from src.strategies.entry_logic import ImprovedEntryLogic

        logic = ImprovedEntryLogic()

        assert logic.soft_filter_mode is True
        assert logic.max_warnings_allowed == 5  # Further relaxed from 3
        assert logic.require_volume_confirmation is False  # Now soft filter


class TestExitLogicImprovements:
    """Test simplified exit logic"""

    def test_simplified_tp_levels(self):
        """Test that TP levels are configured correctly (v4.0 optimized)"""
        from src.strategies.exit_logic import ImprovedExitStrategy

        strategy = ImprovedExitStrategy()

        # v4.0: 3 TP levels optimized for Vietnam market with transaction costs
        assert len(strategy.tp_levels) == 3, f"Expected 3 TP levels, got {len(strategy.tp_levels)}"
        assert strategy.tp_levels == [
            0.04,
            0.08,
            0.15,
        ], f"Expected [0.04, 0.08, 0.15], got {strategy.tp_levels}"

    def test_partial_exit_tracker(self):
        """Test PartialExitTracker functionality"""
        from src.strategies.exit_logic import PartialExitTracker

        tracker = PartialExitTracker()

        # Initial state
        assert tracker.get_state("VCB") == 0
        assert not tracker.has_partial_exit("VCB")

        # Record partial exit
        tracker.record_partial_exit("VCB", "PARTIAL_50%", 60000, 500)
        assert tracker.get_state("VCB") == 1
        assert tracker.has_partial_exit("VCB")
        assert not tracker.is_fully_exited("VCB")

        # Record full exit
        tracker.record_partial_exit("VCB", "FULL", 65000, 500)
        assert tracker.get_state("VCB") == 2
        assert tracker.is_fully_exited("VCB")

        # Check history
        history = tracker.get_exit_history("VCB")
        assert len(history) == 2


class TestMarketRegimeImprovements:
    """Test market regime detector with sector rotation and foreign flow"""

    def test_sector_rotation_integration(self):
        """Test sector rotation is integrated"""
        from src.market.regime_detector import MarketRegimeDetector

        detector = MarketRegimeDetector(
            enable_sector_rotation=True,
            enable_foreign_flow=True,
        )

        assert detector.enable_sector_rotation is True
        assert detector.enable_foreign_flow is True

    def test_composite_score_weights(self):
        """Test updated composite score weights"""
        from src.market.regime_detector import MarketRegimeDetector

        detector = MarketRegimeDetector()

        # Create test components
        components = {
            "trend": 0.5,
            "momentum": 0.3,
            "volume_trend": 0.2,
            "volatility": 0.1,
            "sector_rotation": 0.5,
            "foreign_flow": 0.5,
        }

        score = detector._calculate_composite_score(components)

        # Expected: 0.5*0.35 + 0.3*0.25 + 0.2*0.15 - 0.1*0.10 + 0.5*0.075 + 0.5*0.075
        # = 0.175 + 0.075 + 0.03 - 0.01 + 0.0375 + 0.0375 = 0.345
        expected = 0.175 + 0.075 + 0.03 - 0.01 + 0.0375 + 0.0375
        assert abs(score - expected) < 0.01, f"Expected ~{expected}, got {score}"


class TestMLPredictorImprovements:
    """Test ML predictor online learning capabilities"""

    def test_incremental_learning_buffer(self):
        """Test incremental learning buffer initialization"""
        from src.ml.models.predictor import MLPredictor

        predictor = MLPredictor()

        assert hasattr(predictor, "incremental_samples")
        assert hasattr(predictor, "incremental_buffer_size")
        assert predictor.incremental_buffer_size == 100
        assert predictor.retrain_interval_days == 7

    def test_add_training_sample(self):
        """Test adding training samples"""
        from src.ml.models.predictor import MLPredictor

        predictor = MLPredictor()

        # Add sample
        features = np.random.randn(predictor.expected_features)
        predictor.add_training_sample(features, label=1, trade_result={"pnl": 5.0})

        assert len(predictor.incremental_samples) == 1
        assert predictor.incremental_samples[0]["label"] == 1

    def test_model_health_status(self):
        """Test model health status reporting"""
        from src.ml.models.predictor import MLPredictor

        predictor = MLPredictor()

        status = predictor.get_model_health_status()

        assert "ml_enabled" in status
        assert "models_loaded" in status
        assert "incremental_buffer_size" in status
        assert "ensemble_weights" in status


class TestSectorRotationModule:
    """Test new sector rotation module"""

    def test_sector_definitions(self):
        """Test Vietnam sector definitions"""
        from src.market.sector_rotation import VIETNAM_SECTORS

        assert "banking" in VIETNAM_SECTORS
        assert "real_estate" in VIETNAM_SECTORS
        assert "technology" in VIETNAM_SECTORS

        # Check banking sector
        banking = VIETNAM_SECTORS["banking"]
        assert "VCB" in banking["symbols"]
        assert banking["cycle_phase"] == "EARLY"
        assert banking["defensive"] is False

    def test_sector_analyzer_singleton(self):
        """Test sector analyzer singleton"""
        from src.market.sector_rotation import get_sector_analyzer

        analyzer1 = get_sector_analyzer()
        analyzer2 = get_sector_analyzer()

        assert analyzer1 is analyzer2


class TestForeignFlowModule:
    """Test new foreign flow module"""

    def test_foreign_flow_analyzer_singleton(self):
        """Test foreign flow analyzer singleton"""
        from src.market.foreign_flow import get_foreign_flow_analyzer

        analyzer1 = get_foreign_flow_analyzer()
        analyzer2 = get_foreign_flow_analyzer()

        assert analyzer1 is analyzer2

    def test_manual_data_addition(self):
        """Test manual data addition for testing"""
        from src.market.foreign_flow import ForeignFlowAnalyzer

        analyzer = ForeignFlowAnalyzer()

        # Add manual data
        analyzer.add_manual_data(
            date="2024-01-15",
            buy_value=500_000_000_000,  # 500B VND
            sell_value=300_000_000_000,  # 300B VND
        )

        assert len(analyzer._historical_data) == 1
        assert analyzer._historical_data[0]["net_value"] == 200_000_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
