#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate Trading Improvements

Script to validate all trading improvements:
1. Execution cost tracking
2. ML performance validation
3. Odd-lot handling
4. Backtest-live consistency

Run: python scripts/validate_trading_improvements.py

Author: Trading Bot Team
Version: 1.0.0
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_execution_tracker():
    """Test ExecutionCostTracker."""
    print("\n" + "=" * 60)
    print("TEST 1: Execution Cost Tracker")
    print("=" * 60)

    try:
        from src.monitoring.execution_tracker import get_execution_tracker

        tracker = get_execution_tracker()

        # Simulate some executions
        test_executions = [
            {
                "symbol": "VNM",
                "order_type": "LIMIT",
                "side": "BUY",
                "expected_price": 80000,
                "executed_price": 80100,
                "shares": 500,
                "expected_slippage_pct": 0.002,
                "expected_commission": 100000,
                "actual_commission": 100500,
                "execution_time_ms": 1500,
                "session": "CONTINUOUS",
                "avg_daily_volume": 1000000,
            },
            {
                "symbol": "VNM",
                "order_type": "MARKET",
                "side": "SELL",
                "expected_price": 82000,
                "executed_price": 81800,
                "shares": 500,
                "expected_slippage_pct": 0.004,
                "expected_commission": 102500,
                "actual_commission": 102000,
                "execution_time_ms": 500,
                "session": "CONTINUOUS",
                "avg_daily_volume": 1000000,
            },
            {
                "symbol": "HPG",
                "order_type": "LIMIT",
                "side": "BUY",
                "expected_price": 25000,
                "executed_price": 25050,
                "shares": 1000,
                "expected_slippage_pct": 0.003,
                "expected_commission": 62500,
                "actual_commission": 62625,
                "execution_time_ms": 2000,
                "session": "CONTINUOUS",
                "avg_daily_volume": 5000000,
            },
        ]

        for exec_data in test_executions:
            record = tracker.record_execution(**exec_data)
            print(
                f"✅ Recorded: {record.symbol} {record.side} - "
                f"slippage: {record.actual_slippage_pct:.2%}"
            )

        # Get calibrated slippage
        vnm_slippage = tracker.get_calibrated_slippage("VNM")
        print(f"\n📊 Calibrated slippage for VNM: {vnm_slippage:.2%}")

        # Get execution quality
        score, desc = tracker.get_execution_quality_score("VNM")
        print(f"📊 VNM execution quality: {score:.1f}/100 - {desc}")

        # Get recommendations
        recommendations = tracker.get_calibration_recommendations()
        print(f"\n📋 Calibration recommendations: {len(recommendations)} symbols")

        # Get daily report
        report = tracker.get_daily_report()
        print(f"📊 Daily report: {report.get('total_executions', 0)} executions")

        print("\n✅ ExecutionCostTracker: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ ExecutionCostTracker: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ml_validator():
    """Test MLPerformanceValidator."""
    print("\n" + "=" * 60)
    print("TEST 2: ML Performance Validator")
    print("=" * 60)

    try:
        from src.monitoring.ml_performance_validator import get_ml_validator

        validator = get_ml_validator()

        # Simulate predictions and outcomes
        test_predictions = [
            {"symbol": "VNM", "signal": "BUY", "confidence": 75, "entry_price": 80000},
            {"symbol": "HPG", "signal": "BUY", "confidence": 65, "entry_price": 25000},
            {"symbol": "FPT", "signal": "BUY", "confidence": 80, "entry_price": 120000},
            {"symbol": "VCB", "signal": "BUY", "confidence": 55, "entry_price": 90000},
            {"symbol": "MWG", "signal": "BUY", "confidence": 70, "entry_price": 50000},
        ]

        prediction_ids = []
        for pred in test_predictions:
            pred_id = validator.record_prediction(**pred)
            prediction_ids.append(pred_id)
            print(f"✅ Recorded prediction: {pred['symbol']} conf={pred['confidence']}%")

        # Simulate outcomes (3 wins, 2 losses)
        outcomes = [
            (prediction_ids[0], 84000, "TP1"),  # Win +5%
            (prediction_ids[1], 26500, "TP1"),  # Win +6%
            (prediction_ids[2], 115000, "STOP_LOSS"),  # Loss -4%
            (prediction_ids[3], 94500, "TP1"),  # Win +5%
            (prediction_ids[4], 47000, "STOP_LOSS"),  # Loss -6%
        ]

        for pred_id, exit_price, reason in outcomes:
            result = validator.record_outcome(pred_id, exit_price, reason)
            if result:
                print(
                    f"✅ Outcome: {result.symbol} {result.outcome} "
                    f"return={result.actual_return_pct:+.2%}"
                )

        # Get model health
        health = validator.get_model_health()
        print(f"\n📊 Model Health:")
        print(f"   - Healthy: {health.is_healthy}")
        print(f"   - Accuracy: {health.accuracy_status}")
        print(f"   - EV: {health.ev_status}")
        print(f"   - Calibration: {health.calibration_status}")
        print(f"   - Recommendation: {health.recommendation}")

        # Check if should retrain
        should_retrain, reason = validator.should_retrain()
        print(f"\n🔄 Should retrain: {should_retrain} - {reason}")

        # Get confidence adjustment
        raw_conf = 70
        adjusted_conf = validator.get_confidence_adjustment(raw_conf)
        print(f"\n🎯 Confidence adjustment: {raw_conf}% -> {adjusted_conf:.1f}%")

        # Get stats summary
        summary = validator.get_stats_summary()
        print(f"\n📊 Stats Summary:")
        print(f"   - Accuracy: {summary.get('accuracy_pct', 0):.1f}%")
        print(f"   - Expected Value: {summary.get('expected_value_pct', 0):.2f}%")
        print(f"   - Profit Factor: {summary.get('profit_factor', 0):.2f}")

        print("\n✅ MLPerformanceValidator: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ MLPerformanceValidator: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_odd_lot_handler():
    """Test OddLotHandler."""
    print("\n" + "=" * 60)
    print("TEST 3: Odd-Lot Handler")
    print("=" * 60)

    try:
        from src.utils.odd_lot_handler import get_odd_lot_handler

        handler = get_odd_lot_handler()

        # Test positions with odd-lots
        test_positions = {
            "VNM": {
                "shares": 550,  # 500 full + 50 odd
                "avg_price": 80000,
                "metadata": {"last_price": 82000},
            },
            "HPG": {
                "shares": 1000,  # No odd-lot
                "avg_price": 25000,
                "metadata": {"last_price": 26000},
            },
            "FPT": {
                "shares": 75,  # Only odd-lot
                "avg_price": 120000,
                "metadata": {"last_price": 118000},
            },
        }

        # Detect odd-lots
        odd_lots = handler.detect_odd_lots(test_positions)
        print(f"\n📊 Detected {len(odd_lots)} odd-lot positions:")

        for ol in odd_lots:
            print(
                f"   - {ol.symbol}: {ol.odd_lot_shares} odd-lot shares "
                f"(value: {ol.odd_lot_value:,.0f} VND)"
            )

        # Get recommendations
        print("\n📋 Exit Recommendations:")
        for ol in odd_lots:
            rec = handler.get_exit_recommendation(
                ol,
                market_regime="SIDEWAYS",
                days_held=3,
            )
            print(f"   - {rec.symbol}: {rec.action} (priority: {rec.priority}/5)")
            print(f"     Reason: {rec.reason}")
            print(f"     Fill probability: {rec.expected_fill_probability:.0%}")

        # Calculate cleanup cost
        cleanup_cost = handler.calculate_cleanup_cost(odd_lots, "SIDEWAYS")
        print(f"\n💰 Cleanup Cost Estimate:")
        print(f"   - Total odd-lot value: {cleanup_cost['total_odd_lot_value']:,.0f} VND")
        print(
            f"   - Estimated cost: {cleanup_cost['total_estimated_cost']:,.0f} VND "
            f"({cleanup_cost['cost_as_pct_of_value']:.1f}%)"
        )

        # Test odd-lot avoidance
        would_create, recommended = handler.should_avoid_creating_odd_lot(
            current_shares=1000,
            shares_to_sell=350,
        )
        print(f"\n🔍 Odd-lot avoidance check:")
        print(f"   - Selling 350 from 1000 would create odd-lot: {would_create}")
        print(f"   - Recommended shares to sell: {recommended}")

        # Test lot rounding
        rounded_down = handler.round_to_lot_size(550, round_up=False)
        rounded_up = handler.round_to_lot_size(550, round_up=True)
        print(f"\n📐 Lot rounding: 550 -> {rounded_down} (down) / {rounded_up} (up)")

        print("\n✅ OddLotHandler: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ OddLotHandler: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_consistency_checker():
    """Test LiveConsistencyChecker."""
    print("\n" + "=" * 60)
    print("TEST 4: Backtest-Live Consistency Checker")
    print("=" * 60)

    try:
        from src.backtesting.live_consistency_checker import LiveConsistencyChecker
        import pandas as pd
        import numpy as np

        checker = LiveConsistencyChecker()

        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="D")

        df = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(200) * 0.5),
                "high": 0,
                "low": 0,
                "close": 0,
                "volume": np.random.randint(100000, 1000000, 200),
            },
            index=dates,
        )

        df["close"] = df["open"] + np.random.randn(200) * 0.3
        df["high"] = df[["open", "close"]].max(axis=1) + np.random.rand(200) * 0.5
        df["low"] = df[["open", "close"]].min(axis=1) - np.random.rand(200) * 0.5

        # Test indicator consistency (using same function = should pass)
        def calc_sma(data, period=20):
            return data["close"].rolling(period).mean()

        result = checker.check_indicator_consistency(
            df,
            "SMA_20",
            calc_sma,
            calc_sma,
            {"period": 20},
        )
        print(f"✅ SMA indicator consistency: {'PASSED' if result else 'FAILED'}")

        # Test transaction cost consistency
        result = checker.check_transaction_cost_consistency(0.01, 0.01)
        print(f"✅ Transaction cost consistency: {'PASSED' if result else 'FAILED'}")

        # Test with different costs (should warn)
        result = checker.check_transaction_cost_consistency(0.008, 0.012)
        print(f"⚠️ Different transaction costs: {'PASSED' if result else 'FAILED'}")

        # Test look-ahead bias
        result = checker.check_look_ahead_bias(df, calc_sma, "SMA_20")
        print(f"✅ Look-ahead bias check: {'PASSED' if result else 'FAILED'}")

        # Generate report
        checker.print_report()

        report = checker.generate_report()
        print(f"\n📊 Consistency Report:")
        print(f"   - Consistent: {report.is_consistent}")
        print(f"   - Checks: {report.passed_checks}/{report.total_checks} passed")
        print(f"   - Issues: {len(report.issues)}")

        print("\n✅ LiveConsistencyChecker: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ LiveConsistencyChecker: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("TRADING IMPROVEMENTS VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    results = {
        "Execution Tracker": test_execution_tracker(),
        "ML Validator": test_ml_validator(),
        "Odd-Lot Handler": test_odd_lot_handler(),
        "Consistency Checker": test_consistency_checker(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
