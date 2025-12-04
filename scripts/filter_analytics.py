#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter Analytics Script
Generates filter effectiveness dashboard and config consistency report

Usage:
    python scripts/filter_analytics.py [--dashboard] [--config-check] [--export]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.filter_performance import get_filter_performance_tracker
from src.config.trading_config import get_config, TradingConfig
from src.config import constants

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_config_consistency() -> dict:
    """
    Check for configuration inconsistencies between trading_config and constants.

    Returns:
        Dict with consistency check results
    """
    results = {
        "status": "OK",
        "issues": [],
        "warnings": [],
        "checked_items": [],
    }

    try:
        config = get_config(validate=False)
        trading = config.trading
    except Exception as e:
        results["status"] = "ERROR"
        results["issues"].append(f"Failed to load config: {e}")
        return results

    # Check 1: min_confidence consistency
    results["checked_items"].append("min_confidence")
    if hasattr(constants, "TECH_ONLY_MIN_CONFIDENCE"):
        if trading.min_confidence < constants.TECH_ONLY_MIN_CONFIDENCE:
            results["warnings"].append(
                f"trading_config.min_confidence ({trading.min_confidence}) < "
                f"constants.TECH_ONLY_MIN_CONFIDENCE ({constants.TECH_ONLY_MIN_CONFIDENCE})"
            )

    # Check 2: Risk/Reward consistency
    results["checked_items"].append("min_risk_reward")
    if hasattr(constants, "DEFAULT_MIN_RISK_REWARD"):
        if abs(trading.min_risk_reward - constants.DEFAULT_MIN_RISK_REWARD) > 0.5:
            results["warnings"].append(
                f"trading_config.min_risk_reward ({trading.min_risk_reward}) differs from "
                f"constants.DEFAULT_MIN_RISK_REWARD ({constants.DEFAULT_MIN_RISK_REWARD})"
            )

    # Check 3: Stop loss consistency
    results["checked_items"].append("stop_loss")
    if hasattr(constants, "VN_STOP_LOSS_BASE"):
        config_sl = abs(trading.stop_loss_percent) / 100
        if abs(config_sl - constants.VN_STOP_LOSS_BASE) > 0.02:
            results["warnings"].append(
                f"trading_config.stop_loss_percent ({trading.stop_loss_percent}%) differs from "
                f"constants.VN_STOP_LOSS_BASE ({constants.VN_STOP_LOSS_BASE * 100}%)"
            )

    # Check 4: Liquidity thresholds
    results["checked_items"].append("liquidity_thresholds")
    if hasattr(constants, "VN_MIN_LIQUIDITY_VALUE"):
        if trading.vn_min_daily_value != constants.VN_MIN_LIQUIDITY_VALUE:
            results["warnings"].append(
                f"trading_config.vn_min_daily_value ({trading.vn_min_daily_value:,.0f}) differs from "
                f"constants.VN_MIN_LIQUIDITY_VALUE ({constants.VN_MIN_LIQUIDITY_VALUE:,.0f})"
            )

    # Check 5: Position sizing consistency
    results["checked_items"].append("position_sizing")
    if hasattr(constants, "DEFAULT_MAX_POSITION_SIZE"):
        if abs(trading.max_position_size - constants.DEFAULT_MAX_POSITION_SIZE) > 0.03:
            results["warnings"].append(
                f"trading_config.max_position_size ({trading.max_position_size:.1%}) differs from "
                f"constants.DEFAULT_MAX_POSITION_SIZE ({constants.DEFAULT_MAX_POSITION_SIZE:.1%})"
            )

    # Check 6: Market regime penalty scales
    results["checked_items"].append("regime_penalty_scales")
    expected_scales = {
        "bull": 0.7,
        "bear": 1.2,
        "high_volatility": 1.3,
    }
    if trading.bull_market_penalty_scale != expected_scales["bull"]:
        results["warnings"].append(
            f"bull_market_penalty_scale ({trading.bull_market_penalty_scale}) "
            f"differs from expected ({expected_scales['bull']})"
        )

    # Determine overall status
    if results["issues"]:
        results["status"] = "ERROR"
    elif results["warnings"]:
        results["status"] = "WARNING"

    return results


def print_config_report(results: dict) -> None:
    """Print configuration consistency report."""
    print("\n" + "=" * 80)
    print("🔧 CONFIGURATION CONSISTENCY REPORT")
    print("=" * 80)

    status_icon = {
        "OK": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
    }.get(results["status"], "❓")

    print(f"\nStatus: {status_icon} {results['status']}")
    print(f"Checked Items: {len(results['checked_items'])}")

    if results["issues"]:
        print(f"\n❌ ISSUES ({len(results['issues'])}):")
        for issue in results["issues"]:
            print(f"   • {issue}")

    if results["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"   • {warning}")

    if not results["issues"] and not results["warnings"]:
        print("\n✅ All configuration values are consistent!")

    print("\n" + "-" * 80)
    print("💡 RECOMMENDATIONS:")
    print("   1. Keep threshold values in constants.py for centralized management")
    print("   2. Use trading_config.py for runtime-configurable values")
    print("   3. entry_logic.py should import from constants.py, not hardcode values")
    print("=" * 80 + "\n")


def list_magic_numbers() -> dict:
    """
    Scan for potential magic numbers that should be in constants.

    Returns:
        Dict with magic number analysis
    """
    # Known thresholds that should be in constants
    expected_constants = {
        "SR_BOUNCE_THRESHOLD": 0.02,
        "SR_RESISTANCE_CLOSE_THRESHOLD": 2.0,
        "ENTRY_PULLBACK_MAX_PCT": 5.0,
        "ENTRY_PULLBACK_MIN_PCT": 1.0,
        "ENTRY_BREAKOUT_VOLUME_MULT": 1.2,
        "VN_FLOOR_PENALTY": -20,
        "VN_CEILING_DISTANCE_THRESHOLD": 0.5,
        "TECH_ONLY_MIN_CONFIDENCE": 55,
        "RSI_OVERSOLD": 30,
        "RSI_OVERBOUGHT": 70,
    }

    results = {
        "defined_in_constants": [],
        "missing_from_constants": [],
    }

    for name, expected_value in expected_constants.items():
        if hasattr(constants, name):
            actual_value = getattr(constants, name)
            results["defined_in_constants"].append(
                {
                    "name": name,
                    "value": actual_value,
                    "expected": expected_value,
                    "match": actual_value == expected_value,
                }
            )
        else:
            results["missing_from_constants"].append(
                {
                    "name": name,
                    "expected_value": expected_value,
                }
            )

    return results


def print_magic_numbers_report(results: dict) -> None:
    """Print magic numbers analysis report."""
    print("\n" + "=" * 80)
    print("🔢 MAGIC NUMBERS ANALYSIS")
    print("=" * 80)

    print(f"\n✅ Defined in constants.py ({len(results['defined_in_constants'])}):")
    for item in results["defined_in_constants"]:
        match_icon = "✓" if item["match"] else "≠"
        print(f"   {match_icon} {item['name']} = {item['value']}")

    if results["missing_from_constants"]:
        print(f"\n❌ Missing from constants.py ({len(results['missing_from_constants'])}):")
        for item in results["missing_from_constants"]:
            print(f"   • {item['name']} (suggested: {item['expected_value']})")
    else:
        print("\n✅ All expected constants are defined!")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter Analytics and Config Checker")
    parser.add_argument(
        "--dashboard", action="store_true", help="Show filter effectiveness dashboard"
    )
    parser.add_argument(
        "--config-check", action="store_true", help="Check configuration consistency"
    )
    parser.add_argument("--magic-numbers", action="store_true", help="Analyze magic numbers")
    parser.add_argument("--export", type=str, help="Export dashboard to JSON file")
    parser.add_argument("--all", action="store_true", help="Run all checks")

    args = parser.parse_args()

    # Default to --all if no args
    if not any([args.dashboard, args.config_check, args.magic_numbers, args.export, args.all]):
        args.all = True

    if args.all or args.config_check:
        results = check_config_consistency()
        print_config_report(results)

    if args.all or args.magic_numbers:
        results = list_magic_numbers()
        print_magic_numbers_report(results)

    if args.all or args.dashboard:
        tracker = get_filter_performance_tracker()
        tracker.print_dashboard()

    if args.export:
        tracker = get_filter_performance_tracker()
        filepath = tracker.export_dashboard_json(args.export)
        print(f"✅ Dashboard exported to: {filepath}")


if __name__ == "__main__":
    main()
