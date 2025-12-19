# -*- coding: utf-8 -*-
"""
Tests for Enhanced ML Integration v5.0

Tests:
1. Walk-Forward Validation
2. Performance Decay Detection
3. Advanced Confidence Calibration
4. Risk-Adjusted Metrics
5. Regime-Specific Model Selection
6. Integration with Base ML
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.ml.enhanced_ml_integration import (
    EnhancedMLIntegration,
    WalkForwardValidator,
    PerformanceDecayDetector,
    AdvancedConfidenceCalibrator,
    RiskAdjustedMetrics,
    RegimeModelSelector,
    ModelHealth,
    PerformanceMetrics,
    WalkForwardResult,
    DecayDetection,
    RegimeModelConfig,
    get_enhanced_ml_integration,
    reset_enhanced_ml_integration,
    ACCURACY_TARGET_MINIMUM,
    ACCURACY_TARGET_ACCEPTABLE,
    ACCURACY_TARGET_GOOD,
    ACCURACY_TARGET_EXCELLENT,
)

# Try importing base integration for mock
try:
    from src.ml.vietnam_ml_integration import (
        MLPredictionRecord,
        SignalQuality,
        VietnamMarketSession,
    )

    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.random.rand(100) * 2
    low = close - np.random.rand(100) * 2
    open_price = close + np.random.randn(100) * 0.5
    volume = np.random.randint(100000, 1000000, 100)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_predictions():
    """Create sample prediction records."""
    if not BASE_AVAILABLE:
        pytest.skip("Base ML integration not available")

    predictions = []
    base_date = datetime.now() - timedelta(days=90)

    for i in range(100):
        is_correct = np.random.random() > 0.45  # ~55% accuracy

        pred = MLPredictionRecord(
            prediction_id=f"pred_{i}",
            symbol="VCB",
            timestamp=base_date + timedelta(days=i),
            signal="BUY" if np.random.random() > 0.3 else "SELL",
            raw_confidence=50 + np.random.random() * 40,
            calibrated_confidence=50 + np.random.random() * 40,
            signal_quality=SignalQuality.MEDIUM,
            session=VietnamMarketSession.MORNING_CONTINUOUS,
            model_version="4.0.0",
            feature_snapshot={},
            actual_outcome="WIN" if is_correct else "LOSS",
            outcome_timestamp=base_date + timedelta(days=i + 1),
            pnl_percent=np.random.random() * 10 if is_correct else -np.random.random() * 5,
            is_correct=is_correct,
        )
        predictions.append(pred)

    return predictions


@pytest.fixture
def walk_forward_validator():
    """Create WalkForwardValidator instance."""
    return WalkForwardValidator(
        train_window_days=30,
        test_window_days=7,
        step_days=7,
    )


@pytest.fixture
def decay_detector():
    """Create PerformanceDecayDetector instance."""
    return PerformanceDecayDetector(
        baseline_accuracy=0.55,
        warning_threshold=0.05,
        critical_threshold=0.10,
    )


@pytest.fixture
def calibrator():
    """Create AdvancedConfidenceCalibrator instance."""
    return AdvancedConfidenceCalibrator(
        method="bucket",
        num_buckets=10,
        min_samples_per_bucket=5,
    )


# =============================================================================
# WALK-FORWARD VALIDATION TESTS
# =============================================================================


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    def test_init(self, walk_forward_validator):
        """Test initialization."""
        assert walk_forward_validator.train_window_days == 30
        assert walk_forward_validator.test_window_days == 7
        assert walk_forward_validator.step_days == 7
        assert walk_forward_validator.results == []

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_validate_with_predictions(self, walk_forward_validator, sample_predictions):
        """Test validation with sample predictions."""
        results = walk_forward_validator.validate(
            sample_predictions,
            min_train_samples=10,
            min_test_samples=5,
        )

        # Should have some results
        assert isinstance(results, list)

        # Each result should be WalkForwardResult
        for result in results:
            assert isinstance(result, WalkForwardResult)
            assert 0 <= result.accuracy <= 1
            assert result.train_samples > 0
            assert result.test_samples > 0

    def test_validate_empty_predictions(self, walk_forward_validator):
        """Test validation with empty predictions."""
        results = walk_forward_validator.validate([])
        assert results == []

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_get_summary(self, walk_forward_validator, sample_predictions):
        """Test summary generation."""
        walk_forward_validator.validate(
            sample_predictions,
            min_train_samples=10,
            min_test_samples=5,
        )

        summary = walk_forward_validator.get_summary()

        assert isinstance(summary, dict)
        if walk_forward_validator.results:
            assert "mean_accuracy" in summary
            assert "std_accuracy" in summary
            assert "stability_score" in summary


# =============================================================================
# PERFORMANCE DECAY DETECTION TESTS
# =============================================================================


class TestPerformanceDecayDetector:
    """Tests for PerformanceDecayDetector."""

    def test_init(self, decay_detector):
        """Test initialization."""
        assert decay_detector.baseline_accuracy == 0.55
        assert decay_detector.warning_threshold == 0.05
        assert decay_detector.critical_threshold == 0.10

    def test_record_prediction(self, decay_detector):
        """Test recording predictions."""
        decay_detector.record_prediction(True)
        decay_detector.record_prediction(False)
        decay_detector.record_prediction(True)

        assert len(decay_detector._accuracy_7d) == 3
        assert len(decay_detector._accuracy_30d) == 3

    def test_detect_decay_no_data(self, decay_detector):
        """Test decay detection with no data."""
        result = decay_detector.detect_decay()

        assert isinstance(result, DecayDetection)
        assert result.decay_severity == "none"  # No data = use baseline

    def test_detect_decay_with_good_performance(self, decay_detector):
        """Test decay detection with good performance."""
        # Record 40 correct predictions (100% accuracy)
        for _ in range(40):
            decay_detector.record_prediction(True)

        result = decay_detector.detect_decay()

        assert result.decay_severity == "none"
        assert not result.requires_retraining
        assert result.current_accuracy >= decay_detector.baseline_accuracy

    def test_detect_decay_with_bad_performance(self, decay_detector):
        """Test decay detection with degraded performance."""
        # Record 40 incorrect predictions (0% accuracy)
        for _ in range(40):
            decay_detector.record_prediction(False)

        result = decay_detector.detect_decay()

        assert result.decay_severity in ["warning", "critical"]
        assert result.accuracy_change < 0

    def test_update_baseline(self, decay_detector):
        """Test baseline update."""
        decay_detector.update_baseline(0.60)
        assert decay_detector.baseline_accuracy == 0.60


# =============================================================================
# CONFIDENCE CALIBRATION TESTS
# =============================================================================


class TestAdvancedConfidenceCalibrator:
    """Tests for AdvancedConfidenceCalibrator."""

    def test_init(self, calibrator):
        """Test initialization."""
        assert calibrator.method == "bucket"
        assert calibrator.num_buckets == 10
        assert not calibrator._is_fitted

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_fit_bucket(self, calibrator, sample_predictions):
        """Test bucket calibration fitting."""
        calibrator.fit(sample_predictions)

        assert calibrator._is_fitted
        assert len(calibrator._bucket_calibration) > 0

    def test_calibrate_not_fitted(self, calibrator):
        """Test calibration without fitting."""
        raw = 65.0
        calibrated = calibrator.calibrate(raw)

        # Should return raw confidence when not fitted
        assert calibrated == raw

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_calibrate_after_fit(self, calibrator, sample_predictions):
        """Test calibration after fitting."""
        calibrator.fit(sample_predictions)

        raw = 65.0
        calibrated = calibrator.calibrate(raw)

        # Should return calibrated value
        assert isinstance(calibrated, float)
        assert 0 <= calibrated <= 100

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_calibration_metrics(self, calibrator, sample_predictions):
        """Test calibration metrics calculation."""
        calibrator.fit(sample_predictions)

        metrics = calibrator.calculate_calibration_metrics(sample_predictions)

        assert isinstance(metrics, dict)
        if "brier_score" in metrics:
            assert 0 <= metrics["brier_score"] <= 1


class TestAdvancedCalibratorMethods:
    """Test different calibration methods."""

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_platt_calibrator(self, sample_predictions):
        """Test Platt scaling calibrator."""
        calibrator = AdvancedConfidenceCalibrator(method="platt")
        calibrator.fit(sample_predictions)

        raw = 65.0
        calibrated = calibrator.calibrate(raw)

        assert isinstance(calibrated, float)

    @pytest.mark.skipif(not BASE_AVAILABLE, reason="Base ML not available")
    def test_temperature_calibrator(self, sample_predictions):
        """Test temperature scaling calibrator."""
        calibrator = AdvancedConfidenceCalibrator(method="temperature")
        calibrator.fit(sample_predictions)

        raw = 65.0
        calibrated = calibrator.calibrate(raw)

        assert isinstance(calibrated, float)


# =============================================================================
# RISK-ADJUSTED METRICS TESTS
# =============================================================================


class TestRiskAdjustedMetrics:
    """Tests for RiskAdjustedMetrics."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio with positive returns."""
        returns = [0.01, 0.02, -0.005, 0.015, 0.01, -0.002, 0.008]

        sharpe = RiskAdjustedMetrics.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)

    def test_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty returns."""
        sharpe = RiskAdjustedMetrics.calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = [0.01, 0.02, -0.005, 0.015, 0.01, -0.002, 0.008]

        sortino = RiskAdjustedMetrics.calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_information_ratio(self):
        """Test Information ratio calculation."""
        returns = [0.01, 0.02, -0.005, 0.015]
        benchmark = [0.005, 0.008, -0.002, 0.006]

        ir = RiskAdjustedMetrics.calculate_information_ratio(returns, benchmark)

        assert isinstance(ir, float)

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        equity_curve = [100, 105, 110, 95, 100, 90, 95, 100]

        max_dd = RiskAdjustedMetrics.calculate_max_drawdown(equity_curve)

        assert 0 <= max_dd <= 1
        # Max drawdown should be from 110 to 90 = 18.18%
        assert abs(max_dd - 0.1818) < 0.01

    def test_profit_factor(self):
        """Test profit factor calculation."""
        wins = [0.05, 0.03, 0.02, 0.04]  # Total: 0.14
        losses = [-0.02, -0.01, -0.015]  # Total: -0.045

        pf = RiskAdjustedMetrics.calculate_profit_factor(wins, losses)

        # PF = 0.14 / 0.045 = 3.11
        assert pf > 3.0


# =============================================================================
# REGIME MODEL SELECTOR TESTS
# =============================================================================


class TestRegimeModelSelector:
    """Tests for RegimeModelSelector."""

    def test_init(self):
        """Test initialization."""
        selector = RegimeModelSelector()

        assert "BULL" in selector.configs
        assert "BEAR" in selector.configs
        assert "SIDEWAYS" in selector.configs
        assert "HIGH_VOLATILITY" in selector.configs

    def test_get_config_bull(self):
        """Test getting BULL config."""
        selector = RegimeModelSelector()

        config = selector.get_config("BULL")

        assert isinstance(config, RegimeModelConfig)
        assert config.regime == "BULL"
        assert config.position_multiplier > 1.0  # Higher in bull

    def test_get_config_bear(self):
        """Test getting BEAR config."""
        selector = RegimeModelSelector()

        config = selector.get_config("BEAR")

        assert config.position_multiplier < 1.0  # Lower in bear
        assert config.min_confidence > 60  # Higher threshold in bear

    def test_select_models(self):
        """Test model selection."""
        selector = RegimeModelSelector()
        available = ["xgboost", "lightgbm", "random_forest"]

        models, weights = selector.select_models("BULL", available)

        assert len(models) > 0
        assert all(m in available for m in models)
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Weights sum to 1


# =============================================================================
# ENHANCED ML INTEGRATION TESTS
# =============================================================================


class TestEnhancedMLIntegration:
    """Tests for EnhancedMLIntegration."""

    def test_init(self):
        """Test initialization."""
        # Reset singleton first
        reset_enhanced_ml_integration()

        integration = EnhancedMLIntegration(
            base_integration=None,
            min_confidence=55.0,
            enable_walk_forward=True,
            enable_decay_detection=True,
        )

        assert integration.min_confidence == 55.0
        assert integration.walk_forward is not None
        assert integration.decay_detector is not None
        assert integration.calibrator is not None
        assert integration.regime_selector is not None

    def test_get_signal(self, sample_df):
        """Test getting ML signal."""
        reset_enhanced_ml_integration()

        integration = EnhancedMLIntegration(
            base_integration=None,
            min_confidence=55.0,
        )

        result = integration.get_signal(
            df=sample_df,
            symbol="VCB",
            market_regime={"regime": "BULL"},
        )

        assert isinstance(result, dict)
        assert "signal" in result
        assert "calibrated_confidence" in result
        assert "recommendation" in result
        assert "regime" in result
        assert result["regime"] == "BULL"

    def test_record_outcome(self):
        """Test recording trade outcome."""
        reset_enhanced_ml_integration()

        integration = EnhancedMLIntegration(
            base_integration=None,
            min_confidence=55.0,
        )

        # Record some outcomes
        integration.record_outcome("pred_1", 0.05, 0.02)
        integration.record_outcome("pred_2", -0.02, 0.01)

        assert len(integration._prediction_returns) == 2
        assert len(integration._equity_curve) == 3  # Initial + 2

    def test_get_comprehensive_report(self):
        """Test comprehensive report generation."""
        reset_enhanced_ml_integration()

        integration = EnhancedMLIntegration(
            base_integration=None,
            min_confidence=55.0,
        )

        # Record some outcomes first
        for i in range(15):
            is_win = i % 2 == 0
            integration.record_outcome(
                f"pred_{i}",
                0.03 if is_win else -0.02,
                0.01,
            )

        report = integration.get_comprehensive_report(days=30)

        assert isinstance(report, dict)
        assert "model_version" in report
        assert "recommendations" in report
        assert isinstance(report["recommendations"], list)

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        reset_enhanced_ml_integration()

        instance1 = get_enhanced_ml_integration()
        instance2 = get_enhanced_ml_integration()

        assert instance1 is instance2

        # Reset should create new instance
        reset_enhanced_ml_integration()
        instance3 = get_enhanced_ml_integration()

        assert instance3 is not instance1


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Test accuracy target constants."""

    def test_accuracy_targets_order(self):
        """Test accuracy targets are in correct order."""
        assert ACCURACY_TARGET_MINIMUM < ACCURACY_TARGET_ACCEPTABLE
        assert ACCURACY_TARGET_ACCEPTABLE < ACCURACY_TARGET_GOOD
        assert ACCURACY_TARGET_GOOD < ACCURACY_TARGET_EXCELLENT

    def test_accuracy_targets_realistic(self):
        """Test accuracy targets are realistic."""
        # Minimum should be above random (50%)
        assert ACCURACY_TARGET_MINIMUM > 0.50

        # Excellent should be achievable but challenging
        assert ACCURACY_TARGET_EXCELLENT < 0.70


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
