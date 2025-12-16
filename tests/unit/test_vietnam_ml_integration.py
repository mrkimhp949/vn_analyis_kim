# -*- coding: utf-8 -*-
"""
Tests for Vietnam ML Integration

Comprehensive tests for:
- VietnamMLIntegration
- ConfidenceCalibrator
- VietnamFeatureGenerator
- MLPredictionTracker
- MLIntegrationBridge
"""

import pytest
from datetime import datetime, timedelta, time
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import numpy as np
import pandas as pd


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    base_price = 50000
    close_prices = base_price + np.cumsum(np.random.randn(100) * 500)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close_prices - np.random.rand(100) * 200,
            "high": close_prices + np.random.rand(100) * 500,
            "low": close_prices - np.random.rand(100) * 500,
            "close": close_prices,
            "volume": np.random.randint(100000, 1000000, 100),
        }
    )

    return df


@pytest.fixture
def sample_index_df():
    """Create sample VNINDEX DataFrame."""
    np.random.seed(123)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    base_index = 1200
    close_prices = base_index + np.cumsum(np.random.randn(100) * 10)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close_prices - np.random.rand(100) * 5,
            "high": close_prices + np.random.rand(100) * 10,
            "low": close_prices - np.random.rand(100) * 10,
            "close": close_prices,
            "volume": np.random.randint(100000000, 500000000, 100),
        }
    )

    return df


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# TEST VIETNAM MARKET SESSION
# =============================================================================


class TestVietnamMarketSession:
    """Test session detection."""

    def test_session_detection_morning(self):
        """Test morning session detection."""
        from src.ml.vietnam_ml_integration import (
            VietnamFeatureGenerator,
            VietnamMarketSession,
        )

        generator = VietnamFeatureGenerator()

        # 10:00 AM should be morning continuous
        morning_time = datetime(2024, 1, 15, 10, 0)
        features = generator.generate_features(
            pd.DataFrame(
                {
                    "close": [50000],
                    "open": [49500],
                    "high": [50500],
                    "low": [49000],
                    "volume": [100000],
                }
            ),
            "VNM",
            current_time=morning_time,
        )

        assert features.current_session == VietnamMarketSession.MORNING_CONTINUOUS

    def test_session_detection_lunch(self):
        """Test lunch break detection."""
        from src.ml.vietnam_ml_integration import (
            VietnamFeatureGenerator,
            VietnamMarketSession,
        )

        generator = VietnamFeatureGenerator()

        # 12:00 should be lunch break
        lunch_time = datetime(2024, 1, 15, 12, 0)
        features = generator.generate_features(
            pd.DataFrame(
                {
                    "close": [50000],
                    "open": [49500],
                    "high": [50500],
                    "low": [49000],
                    "volume": [100000],
                }
            ),
            "VNM",
            current_time=lunch_time,
        )

        assert features.current_session == VietnamMarketSession.LUNCH_BREAK

    def test_session_detection_atc(self):
        """Test ATC session detection."""
        from src.ml.vietnam_ml_integration import (
            VietnamFeatureGenerator,
            VietnamMarketSession,
        )

        generator = VietnamFeatureGenerator()

        # 14:35 should be ATC
        atc_time = datetime(2024, 1, 15, 14, 35)
        features = generator.generate_features(
            pd.DataFrame(
                {
                    "close": [50000],
                    "open": [49500],
                    "high": [50500],
                    "low": [49000],
                    "volume": [100000],
                }
            ),
            "VNM",
            current_time=atc_time,
        )

        assert features.current_session == VietnamMarketSession.ATC


# =============================================================================
# TEST VIETNAM FEATURE GENERATOR
# =============================================================================


class TestVietnamFeatureGenerator:
    """Test Vietnam-specific feature generation."""

    def test_price_limit_features(self, sample_ohlcv_df):
        """Test price limit feature calculation."""
        from src.ml.vietnam_ml_integration import VietnamFeatureGenerator

        generator = VietnamFeatureGenerator()
        features = generator.generate_features(sample_ohlcv_df, "VNM")

        # Should have distance to ceiling and floor
        assert features.distance_to_ceiling_pct > 0
        assert features.distance_to_floor_pct > 0

    def test_near_ceiling_detection(self):
        """Test near ceiling detection."""
        from src.ml.vietnam_ml_integration import VietnamFeatureGenerator

        generator = VietnamFeatureGenerator()

        # Create data where price is near ceiling (>5% up from previous close)
        df = pd.DataFrame(
            {
                "close": [50000, 53000],  # 6% increase, near 7% ceiling
                "open": [50000, 52000],
                "high": [50500, 53500],
                "low": [49500, 52500],
                "volume": [100000, 150000],
            }
        )

        features = generator.generate_features(df, "VNM")

        assert features.distance_to_ceiling_pct < 2  # Less than 2% to ceiling
        assert features.near_ceiling == True

    def test_liquidity_score_high(self, sample_ohlcv_df):
        """Test liquidity score for high liquidity stock."""
        from src.ml.vietnam_ml_integration import VietnamFeatureGenerator

        generator = VietnamFeatureGenerator()

        # High volume stock
        df = sample_ohlcv_df.copy()
        df["volume"] = 5000000  # 5M shares
        df["close"] = 50000  # 50k VND = 250B VND/day

        features = generator.generate_features(df, "VNM")

        assert features.liquidity_score >= 0.8

    def test_foreign_flow_estimation(self, sample_ohlcv_df):
        """Test foreign flow estimation."""
        from src.ml.vietnam_ml_integration import VietnamFeatureGenerator

        generator = VietnamFeatureGenerator()
        features = generator.generate_features(sample_ohlcv_df, "VNM")

        # Should have a foreign flow signal
        assert features.foreign_flow_signal in ["BUYING", "SELLING", "NEUTRAL"]


# =============================================================================
# TEST CONFIDENCE CALIBRATOR
# =============================================================================


class TestConfidenceCalibrator:
    """Test confidence calibration."""

    def test_calibration_no_history(self):
        """Test calibration with no historical data."""
        from src.ml.vietnam_ml_integration import (
            ConfidenceCalibrator,
            VietnamMarketSession,
        )

        calibrator = ConfidenceCalibrator()

        calibrated, quality, reasons = calibrator.calibrate(
            raw_confidence=65.0,
            symbol="VNM",
            session=VietnamMarketSession.MORNING_CONTINUOUS,
            historical_predictions=[],
        )

        # Should return raw confidence when no history
        assert calibrated == 65.0

    def test_session_penalty_ato(self):
        """Test ATO session applies penalty."""
        from src.ml.vietnam_ml_integration import (
            ConfidenceCalibrator,
            VietnamMarketSession,
        )

        calibrator = ConfidenceCalibrator()

        calibrated, quality, reasons = calibrator.calibrate(
            raw_confidence=70.0,
            symbol="VNM",
            session=VietnamMarketSession.ATO,
            historical_predictions=[],
        )

        # ATO should have -10 penalty
        assert calibrated == 60.0
        assert any("ATO" in r for r in reasons)

    def test_session_penalty_atc(self):
        """Test ATC session applies penalty."""
        from src.ml.vietnam_ml_integration import (
            ConfidenceCalibrator,
            VietnamMarketSession,
        )

        calibrator = ConfidenceCalibrator()

        calibrated, quality, reasons = calibrator.calibrate(
            raw_confidence=70.0,
            symbol="VNM",
            session=VietnamMarketSession.ATC,
            historical_predictions=[],
        )

        # ATC should have -15 penalty
        assert calibrated == 55.0

    def test_quality_levels(self):
        """Test signal quality level determination."""
        from src.ml.vietnam_ml_integration import (
            ConfidenceCalibrator,
            VietnamMarketSession,
            SignalQuality,
        )

        calibrator = ConfidenceCalibrator()

        # Premium quality (75%+)
        _, quality, _ = calibrator.calibrate(
            80.0, "VNM", VietnamMarketSession.MORNING_CONTINUOUS, []
        )
        assert quality == SignalQuality.PREMIUM

        # High quality (65-75%)
        _, quality, _ = calibrator.calibrate(
            70.0, "VNM", VietnamMarketSession.MORNING_CONTINUOUS, []
        )
        assert quality == SignalQuality.HIGH

        # Medium quality (55-65%)
        _, quality, _ = calibrator.calibrate(
            60.0, "VNM", VietnamMarketSession.MORNING_CONTINUOUS, []
        )
        assert quality == SignalQuality.MEDIUM

        # Low quality (<55%)
        _, quality, _ = calibrator.calibrate(
            50.0, "VNM", VietnamMarketSession.MORNING_CONTINUOUS, []
        )
        assert quality == SignalQuality.LOW


# =============================================================================
# TEST ML PREDICTION TRACKER
# =============================================================================


class TestMLPredictionTracker:
    """Test prediction tracking."""

    def test_log_prediction(self, temp_storage_dir):
        """Test logging a prediction."""
        from src.ml.vietnam_ml_integration import (
            MLPredictionTracker,
            VietnamMarketSession,
            SignalQuality,
        )

        tracker = MLPredictionTracker(storage_dir=temp_storage_dir)

        prediction_id = tracker.log_prediction(
            symbol="VNM",
            signal="BUY",
            raw_confidence=65.0,
            calibrated_confidence=62.0,
            signal_quality=SignalQuality.MEDIUM,
            session=VietnamMarketSession.MORNING_CONTINUOUS,
            model_version="4.0.0",
        )

        assert prediction_id is not None
        assert len(prediction_id) == 12  # MD5 hash prefix

    def test_update_outcome(self, temp_storage_dir):
        """Test updating prediction outcome."""
        from src.ml.vietnam_ml_integration import (
            MLPredictionTracker,
            VietnamMarketSession,
            SignalQuality,
        )

        tracker = MLPredictionTracker(storage_dir=temp_storage_dir)

        # Log prediction
        prediction_id = tracker.log_prediction(
            symbol="VNM",
            signal="BUY",
            raw_confidence=65.0,
            calibrated_confidence=62.0,
            signal_quality=SignalQuality.HIGH,
            session=VietnamMarketSession.MORNING_CONTINUOUS,
            model_version="4.0.0",
        )

        # Update outcome
        tracker.update_outcome(
            prediction_id=prediction_id,
            actual_outcome="CLOSED",
            pnl_percent=5.0,
        )

        # Check outcome was recorded
        predictions = tracker.get_recent_predictions(days=1)
        assert len(predictions) == 1
        assert predictions[0].is_correct == True
        assert predictions[0].pnl_percent == 5.0

    def test_accuracy_calculation(self, temp_storage_dir):
        """Test accuracy calculation."""
        from src.ml.vietnam_ml_integration import (
            MLPredictionTracker,
            VietnamMarketSession,
            SignalQuality,
        )

        tracker = MLPredictionTracker(storage_dir=temp_storage_dir)

        # Log multiple predictions with outcomes
        for i in range(20):
            prediction_id = tracker.log_prediction(
                symbol="VNM",
                signal="BUY",
                raw_confidence=65.0,
                calibrated_confidence=62.0,
                signal_quality=SignalQuality.HIGH,
                session=VietnamMarketSession.MORNING_CONTINUOUS,
                model_version="4.0.0",
            )

            # 60% win rate
            pnl = 5.0 if i < 12 else -3.0
            tracker.update_outcome(prediction_id, "CLOSED", pnl)

        # Calculate accuracy
        metrics = tracker.calculate_accuracy(days=30)

        assert metrics["total_predictions"] == 20
        assert metrics["accuracy"] == 0.6  # 12/20


# =============================================================================
# TEST VIETNAM ML INTEGRATION
# =============================================================================


class TestVietnamMLIntegration:
    """Test main integration class."""

    def test_integration_initialization(self, temp_storage_dir):
        """Test integration initializes correctly."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(
            min_confidence=55.0,
            storage_dir=temp_storage_dir,
        )

        assert integration is not None
        assert integration.min_confidence == 55.0

    def test_get_signal_insufficient_data(self, temp_storage_dir):
        """Test signal with insufficient data."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        # Only 10 rows - insufficient
        df = pd.DataFrame(
            {
                "close": range(10),
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "volume": [1000] * 10,
            }
        )

        result = integration.get_signal(df, "VNM")

        assert result.is_valid == False
        assert "Insufficient data" in result.validation_warnings[0]

    def test_get_signal_lunch_break(self, sample_ohlcv_df, temp_storage_dir):
        """Test signal during lunch break is blocked."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        # Simulate lunch break time
        lunch_time = datetime(2024, 1, 15, 12, 30)

        result = integration.get_signal(sample_ohlcv_df, "VNM", current_time=lunch_time)

        assert result.is_valid == False
        assert "LUNCH" in result.validation_warnings[0]

    def test_get_signal_valid(self, sample_ohlcv_df, sample_index_df, temp_storage_dir):
        """Test valid signal generation."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        # Simulate morning session
        morning_time = datetime(2024, 1, 15, 10, 0)

        result = integration.get_signal(
            sample_ohlcv_df, "VNM", sample_index_df, current_time=morning_time
        )

        assert result.is_valid == True
        assert result.signal in ["BUY", "SELL", "HOLD"]
        assert 0 <= result.calibrated_confidence <= 100
        assert result.model_version == "4.0.0"

    def test_near_ceiling_penalty(self, temp_storage_dir):
        """Test near ceiling applies penalty."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        # Create data near ceiling
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "open": [50000] * 60 + [53000] * 10,
                "high": [50500] * 60 + [53500] * 10,
                "low": [49500] * 60 + [52500] * 10,
                "close": [50000] * 60 + [53400] * 10,  # Near 7% ceiling
                "volume": [100000] * 70,
            }
        )

        morning_time = datetime(2024, 1, 15, 10, 0)
        result = integration.get_signal(df, "VNM", current_time=morning_time)

        # Should have near ceiling warning
        if result.vietnam_features.near_ceiling:
            assert any("ceiling" in w.lower() for w in result.validation_warnings)


# =============================================================================
# TEST ML INTEGRATION BRIDGE
# =============================================================================


class TestMLIntegrationBridge:
    """Test integration bridge."""

    def test_bridge_initialization(self):
        """Test bridge initializes correctly."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge(min_confidence=55.0)

        assert bridge is not None
        assert bridge.min_confidence == 55.0

    def test_should_trade_invalid_signal(self):
        """Test should_trade with invalid signal."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge()

        result = {
            "is_valid": False,
            "warnings": ["Test warning"],
        }

        should_trade, reason = bridge.should_trade(result)

        assert should_trade == False
        assert "Invalid signal" in reason

    def test_should_trade_hold_signal(self):
        """Test should_trade with HOLD signal."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge()

        result = {
            "is_valid": True,
            "signal": "HOLD",
            "confidence": 70.0,
        }

        should_trade, reason = bridge.should_trade(result)

        assert should_trade == False
        assert "HOLD" in reason

    def test_should_trade_low_confidence(self):
        """Test should_trade with low confidence."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge(min_confidence=60.0)

        result = {
            "is_valid": True,
            "signal": "BUY",
            "confidence": 55.0,
            "signal_quality": "MEDIUM",
        }

        should_trade, reason = bridge.should_trade(result)

        assert should_trade == False
        assert "below threshold" in reason

    def test_should_trade_valid(self):
        """Test should_trade with valid signal."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge(min_confidence=55.0)

        result = {
            "is_valid": True,
            "signal": "BUY",
            "confidence": 65.0,
            "signal_quality": "HIGH",
            "near_ceiling": False,
        }

        should_trade, reason = bridge.should_trade(result)

        assert should_trade == True
        assert "Valid BUY" in reason

    def test_position_size_adjustment(self):
        """Test position size adjustment calculation."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge()

        # Premium quality, high agreement, high liquidity
        result = {
            "position_multiplier": 1.2,
            "signal_quality": "PREMIUM",
            "ensemble_agreement": 0.9,
            "liquidity_score": 0.9,
        }

        adjustment = bridge.get_position_size_adjustment(result)

        # Should be higher than 1.0 for premium signal
        assert adjustment > 1.0
        assert adjustment <= 1.5

    def test_position_size_adjustment_low_quality(self):
        """Test position size adjustment for low quality signal."""
        from src.ml.integration_bridge import MLIntegrationBridge

        bridge = MLIntegrationBridge()

        # Low quality, low agreement, low liquidity
        result = {
            "position_multiplier": 0.8,
            "signal_quality": "LOW",
            "ensemble_agreement": 0.5,
            "liquidity_score": 0.3,
        }

        adjustment = bridge.get_position_size_adjustment(result)

        # Should be lower for low quality
        assert adjustment < 1.0
        assert adjustment >= 0.5


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, temp_storage_dir):
        """Test with empty DataFrame."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        result = integration.get_signal(pd.DataFrame(), "VNM")

        assert result.is_valid == False

    def test_none_dataframe(self, temp_storage_dir):
        """Test with None DataFrame."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        result = integration.get_signal(None, "VNM")

        assert result.is_valid == False

    def test_missing_columns(self, temp_storage_dir):
        """Test with missing required columns."""
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)

        # Missing 'close' column
        df = pd.DataFrame(
            {
                "open": range(100),
                "volume": [1000] * 100,
            }
        )

        # Should handle gracefully
        result = integration.get_signal(df, "VNM")

        # May be invalid or have warnings
        assert result is not None

    def test_concurrent_predictions(self, temp_storage_dir):
        """Test concurrent prediction logging."""
        from src.ml.vietnam_ml_integration import (
            MLPredictionTracker,
            VietnamMarketSession,
            SignalQuality,
        )
        import threading

        tracker = MLPredictionTracker(storage_dir=temp_storage_dir)

        prediction_ids = []

        def log_prediction():
            pid = tracker.log_prediction(
                symbol="VNM",
                signal="BUY",
                raw_confidence=65.0,
                calibrated_confidence=62.0,
                signal_quality=SignalQuality.HIGH,
                session=VietnamMarketSession.MORNING_CONTINUOUS,
                model_version="4.0.0",
            )
            prediction_ids.append(pid)

        # Create multiple threads
        threads = [threading.Thread(target=log_prediction) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All predictions should be logged
        assert len(prediction_ids) == 10
        assert len(set(prediction_ids)) == 10  # All unique IDs


# =============================================================================
# TEST PERFORMANCE
# =============================================================================


class TestPerformance:
    """Test performance and efficiency."""

    def test_signal_generation_time(self, sample_ohlcv_df, sample_index_df, temp_storage_dir):
        """Test signal generation is fast enough."""
        import time
        from src.ml.vietnam_ml_integration import VietnamMLIntegration

        integration = VietnamMLIntegration(storage_dir=temp_storage_dir)
        morning_time = datetime(2024, 1, 15, 10, 0)

        # Warm up
        integration.get_signal(sample_ohlcv_df, "VNM", sample_index_df, morning_time)

        # Time 10 signals
        start = time.time()
        for _ in range(10):
            integration.get_signal(sample_ohlcv_df, "VNM", sample_index_df, morning_time)
        elapsed = time.time() - start

        # Should be less than 1 second for 10 signals
        avg_time = elapsed / 10
        assert avg_time < 0.1, f"Signal generation too slow: {avg_time:.3f}s per signal"

    def test_tracker_persistence(self, temp_storage_dir):
        """Test tracker saves and loads correctly."""
        from src.ml.vietnam_ml_integration import (
            MLPredictionTracker,
            VietnamMarketSession,
            SignalQuality,
        )

        # Create tracker and log predictions
        tracker1 = MLPredictionTracker(storage_dir=temp_storage_dir)

        for i in range(5):
            tracker1.log_prediction(
                symbol=f"VNM{i}",
                signal="BUY",
                raw_confidence=60.0 + i,
                calibrated_confidence=58.0 + i,
                signal_quality=SignalQuality.HIGH,
                session=VietnamMarketSession.MORNING_CONTINUOUS,
                model_version="4.0.0",
            )

        # Create new tracker instance (simulates restart)
        tracker2 = MLPredictionTracker(storage_dir=temp_storage_dir)

        # Should have loaded all predictions
        predictions = tracker2.get_recent_predictions(days=1)
        assert len(predictions) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
