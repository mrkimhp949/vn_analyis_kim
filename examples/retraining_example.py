#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Automated Retraining Pipeline Usage

Demonstrates how to use the automated retraining pipeline and scheduler
"""

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_1_manual_retraining():
    """Example 1: Manual retraining trigger"""

    from src.ml.retraining_pipeline import get_retraining_pipeline, RetrainingTrigger

    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Manual Retraining")
    logger.info("=" * 80)

    pipeline = get_retraining_pipeline()

    # Manual trigger
    result = pipeline.run_retraining(
        trigger=RetrainingTrigger.MANUAL, trigger_reason="Testing automated retraining pipeline"
    )

    if result.training_successful:
        logger.info(f"✅ Retraining successful!")
        logger.info(f"   New model: {result.new_model_id} (v{result.new_version})")
        logger.info(f"   Train accuracy: {result.train_accuracy:.1%}")
        logger.info(f"   Val accuracy: {result.val_accuracy:.1%}")
        logger.info(f"   Test accuracy: {result.test_accuracy:.1%}")

        if result.current_model_id:
            logger.info(f"   Improvement: {result.improvement_pct:+.1f}%")

        if result.deployed:
            logger.info(f"   🚀 Model deployed automatically!")
        else:
            logger.info(f"   ℹ️ Model registered but not deployed")
    else:
        logger.error(f"❌ Retraining failed!")
        for error in result.errors:
            logger.error(f"   - {error}")


def example_2_check_triggers():
    """Example 2: Check if retraining triggers are active"""

    import pandas as pd
    import numpy as np
    from src.ml.retraining_pipeline import get_retraining_pipeline

    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Check Retraining Triggers")
    logger.info("=" * 80)

    pipeline = get_retraining_pipeline()

    # Simulate current performance
    current_performance = {
        "win_rate": 0.55,  # Below 60% threshold
        "accuracy": 0.58,
    }

    # Simulate current features (for drift detection)
    # In production, these would be real features from recent predictions
    feature_names = ["rsi_14", "macd", "volume_ratio", "ema_20_50"]
    current_features = pd.DataFrame(np.random.randn(100, len(feature_names)), columns=feature_names)

    # Check triggers
    should_retrain, trigger, reason = pipeline.check_triggers(
        current_performance=current_performance, current_features=current_features
    )

    if should_retrain:
        logger.info(f"✅ Retraining trigger detected!")
        logger.info(f"   Trigger: {trigger.value}")
        logger.info(f"   Reason: {reason}")

        # You can auto-retrain here
        # result = pipeline.run_retraining(trigger, reason)
    else:
        logger.info(f"ℹ️ No retraining needed")


def example_3_background_scheduler():
    """Example 3: Start background scheduler"""

    from src.ml.retraining_scheduler import get_retraining_scheduler
    import time

    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Background Scheduler")
    logger.info("=" * 80)

    scheduler = get_retraining_scheduler()

    # Start scheduler
    logger.info("Starting background scheduler...")
    scheduler.start()

    # Let it run for a bit (in production, this runs forever)
    logger.info("Scheduler running in background...")
    logger.info("(In production, this would run continuously)")

    # Get status
    time.sleep(2)
    status = scheduler.get_status()
    logger.info(f"Scheduler status:")
    logger.info(f"   Running: {status['is_running']}")
    logger.info(f"   Last check: {status['last_check_time']}")
    logger.info(f"   Total retrainings: {status['total_retrainings']}")
    logger.info(f"   Next check: {status['next_check']}")

    # Stop scheduler
    logger.info("Stopping scheduler...")
    scheduler.stop()
    logger.info("✅ Scheduler stopped")


def example_4_manual_trigger_scheduler():
    """Example 4: Manual trigger via scheduler"""

    from src.ml.retraining_scheduler import get_retraining_scheduler

    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Manual Trigger via Scheduler")
    logger.info("=" * 80)

    scheduler = get_retraining_scheduler()

    # Manual trigger
    result = scheduler.manual_trigger(reason="Testing manual trigger from scheduler")

    if result.training_successful:
        logger.info(f"✅ Manual retraining successful!")
        logger.info(f"   New model: {result.new_model_id}")
        logger.info(f"   Val accuracy: {result.val_accuracy:.1%}")
    else:
        logger.error(f"❌ Manual retraining failed")


def example_5_model_versioning():
    """Example 5: Model versioning and comparison"""

    from src.ml.model_version_manager import get_model_version_manager

    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Model Versioning")
    logger.info("=" * 80)

    manager = get_model_version_manager()

    # Get active model
    active_model_id = manager.get_active_model_id()

    if active_model_id:
        logger.info(f"Active model: {active_model_id}")

        # Load model
        model = manager.load_model()
        logger.info(f"✅ Model loaded: {type(model).__name__}")

        # Get performance history
        perf_history = manager.get_performance_history(model_id=active_model_id, days=7)

        if not perf_history.empty:
            logger.info(f"\nPerformance history (last 7 days):")
            logger.info(f"   Total signals: {perf_history['total_signals'].sum()}")
            logger.info(f"   Avg accuracy: {perf_history['accuracy'].mean():.1%}")
            logger.info(f"   Avg win rate: {perf_history['win_rate'].mean():.1%}")

    # Compare all models
    comparison = manager.compare_models()

    if not comparison.empty:
        logger.info(f"\nModel comparison:")
        logger.info(comparison.to_string())


def example_6_drift_detection():
    """Example 6: Feature drift detection"""

    import pandas as pd
    import numpy as np
    from src.ml.feature_drift_detector import get_drift_detector

    logger.info("=" * 80)
    logger.info("EXAMPLE 6: Feature Drift Detection")
    logger.info("=" * 80)

    detector = get_drift_detector()

    # Simulate baseline features (training data)
    feature_names = ["rsi_14", "macd", "volume_ratio", "ema_20_50"]
    baseline_features = pd.DataFrame(
        np.random.randn(1000, len(feature_names)), columns=feature_names
    )

    # Set baseline
    logger.info("Setting baseline from training data...")
    detector.set_baseline(baseline_features, feature_names)

    # Simulate current features (with drift)
    # Add shift to simulate drift
    current_features = pd.DataFrame(
        np.random.randn(500, len(feature_names)) + 0.5, columns=feature_names  # +0.5 shift
    )

    # Detect drift
    logger.info("Detecting drift in current features...")
    drift_report = detector.detect_drift(current_features)

    logger.info(f"\nDrift report:")
    logger.info(f"   Overall PSI: {drift_report.overall_drift_score:.3f}")
    logger.info(f"   Severity: {drift_report.drift_severity}")
    logger.info(
        f"   Features with drift: {drift_report.features_with_drift}/{drift_report.total_features}"
    )
    logger.info(f"   Requires retraining: {drift_report.requires_retraining}")

    if drift_report.features_with_drift > 0:
        logger.info(f"\nDrifted features:")
        for feature_drift in drift_report.feature_drifts:
            if feature_drift.has_drift:
                logger.info(
                    f"   - {feature_drift.feature_name}: PSI={feature_drift.psi_score:.3f} ({feature_drift.drift_severity})"
                )


def example_7_full_workflow():
    """Example 7: Complete automated workflow"""

    from src.ml.retraining_scheduler import get_retraining_scheduler
    from src.ml.model_version_manager import get_model_version_manager
    import time

    logger.info("=" * 80)
    logger.info("EXAMPLE 7: Complete Automated Workflow")
    logger.info("=" * 80)

    # 1. Start scheduler
    logger.info("Step 1: Starting background scheduler...")
    scheduler = get_retraining_scheduler()
    scheduler.start()

    # 2. Simulate some time passing
    logger.info("Step 2: Simulating production usage...")
    time.sleep(2)

    # 3. Check status
    status = scheduler.get_status()
    logger.info(f"Step 3: Scheduler status")
    logger.info(f"   Running: {status['is_running']}")
    logger.info(f"   Total retrainings: {status['total_retrainings']}")

    # 4. Model versioning
    logger.info(f"\nStep 4: Model versioning")
    manager = get_model_version_manager()
    active_id = manager.get_active_model_id()

    if active_id:
        logger.info(f"   Active model: {active_id}")

        registry = manager.registry
        if active_id in registry:
            metadata = registry[active_id]
            logger.info(f"   Version: {metadata['version']}")
            logger.info(f"   Val accuracy: {metadata['val_accuracy']:.1%}")
            logger.info(f"   Created: {metadata['created_at']}")

    # 5. Stop scheduler
    logger.info(f"\nStep 5: Stopping scheduler...")
    scheduler.stop()

    logger.info(f"\n✅ Complete workflow demonstrated!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AUTOMATED RETRAINING PIPELINE - EXAMPLES")
    print("=" * 80 + "\n")

    # Run examples
    try:
        # Example 1: Manual retraining
        # example_1_manual_retraining()

        # Example 2: Check triggers
        example_2_check_triggers()

        # Example 3: Background scheduler
        example_3_background_scheduler()

        # Example 4: Manual trigger via scheduler
        # example_4_manual_trigger_scheduler()

        # Example 5: Model versioning
        example_5_model_versioning()

        # Example 6: Drift detection
        example_6_drift_detection()

        # Example 7: Full workflow
        example_7_full_workflow()

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")
