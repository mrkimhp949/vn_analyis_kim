#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate Trading Improvements v12.0

This script validates that all improvements are working correctly:
1. Balanced filter configuration
2. Regime-aware thresholds
3. Improved circuit breaker
4. Error handling

Run: python scripts/validate_improvements_v12.py

Author: Trading Bot Team
Version: 12.0.0
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | {test_name}")
    if details:
        print(f"         {details}")


def validate_balanced_config() -> bool:
    """Validate balanced entry configuration."""
    print_header("1. Balanced Entry Configuration")
    
    all_passed = True
    
    try:
        from src.config.trading_improvements_v12 import (
            BalancedEntryConfig,
            get_balanced_entry_config,
        )
        
        config = get_balanced_entry_config()
        
        # Test 1: Confidence thresholds are balanced
        test_passed = (
            40 <= config.min_confidence_bull <= 50 and
            50 <= config.min_confidence_sideways <= 60 and
            60 <= config.min_confidence_bear <= 70
        )
        print_result(
            "Confidence thresholds balanced",
            test_passed,
            f"BULL={config.min_confidence_bull}, SIDEWAYS={config.min_confidence_sideways}, BEAR={config.min_confidence_bear}"
        )
        all_passed &= test_passed
        
        # Test 2: R:R thresholds are balanced
        test_passed = (
            1.3 <= config.min_rr_bull <= 1.8 and
            1.5 <= config.min_rr_sideways <= 2.0 and
            2.0 <= config.min_rr_bear <= 2.5
        )
        print_result(
            "R:R thresholds balanced",
            test_passed,
            f"BULL={config.min_rr_bull}, SIDEWAYS={config.min_rr_sideways}, BEAR={config.min_rr_bear}"
        )
        all_passed &= test_passed
        
        # Test 3: Important filters re-enabled
        test_passed = (
            config.use_sector_strength_filter and
            config.use_market_breadth_filter and
            config.use_foreign_flow_filter
        )
        print_result(
            "Important filters re-enabled",
            test_passed,
            f"sector={config.use_sector_strength_filter}, breadth={config.use_market_breadth_filter}, foreign={config.use_foreign_flow_filter}"
        )
        all_passed &= test_passed
        
        # Test 4: Regime config works
        bull_cfg = config.get_regime_config("BULL")
        bear_cfg = config.get_regime_config("BEAR")
        test_passed = (
            bull_cfg["position_multiplier"] > bear_cfg["position_multiplier"] and
            bull_cfg["filter_strictness"] < bear_cfg["filter_strictness"]
        )
        print_result(
            "Regime-aware configuration",
            test_passed,
            f"BULL pos_mult={bull_cfg['position_multiplier']}, BEAR pos_mult={bear_cfg['position_multiplier']}"
        )
        all_passed &= test_passed
        
    except Exception as e:
        print_result("Configuration import", False, str(e))
        all_passed = False
    
    return all_passed


def validate_balanced_filters() -> bool:
    """Validate balanced filter definitions."""
    print_header("2. Balanced Filter Definitions")
    
    all_passed = True
    
    try:
        from src.config.trading_improvements_v12 import (
            BALANCED_FILTERS,
            FilterPriority,
            get_filter_config,
            get_enabled_filters,
            get_critical_filters,
        )
        
        # Test 1: Critical filters exist and are enabled
        critical = get_critical_filters()
        required_critical = ["price_limit", "liquidity", "stop_loss_valid", "consecutive_loss"]
        test_passed = all(f in critical for f in required_critical)
        print_result(
            "Critical filters defined",
            test_passed,
            f"Found: {critical}"
        )
        all_passed &= test_passed
        
        # Test 2: Critical filters can block
        test_passed = all(
            BALANCED_FILTERS[f].can_block for f in required_critical
        )
        print_result(
            "Critical filters can block",
            test_passed,
        )
        all_passed &= test_passed
        
        # Test 3: Important filters re-enabled
        important_filters = ["sector_strength", "market_breadth", "foreign_flow"]
        test_passed = all(
            BALANCED_FILTERS[f].enabled for f in important_filters
        )
        print_result(
            "Important filters enabled",
            test_passed,
            f"Checked: {important_filters}"
        )
        all_passed &= test_passed
        
        # Test 4: Regime overrides work
        cfg_bull = get_filter_config("sector_strength", "BULL")
        cfg_bear = get_filter_config("sector_strength", "BEAR")
        test_passed = (
            cfg_bull["can_block"] is False and
            cfg_bear["can_block"] is True
        )
        print_result(
            "Regime overrides work",
            test_passed,
            f"sector_strength: BULL can_block={cfg_bull['can_block']}, BEAR can_block={cfg_bear['can_block']}"
        )
        all_passed &= test_passed
        
        # Test 5: Enabled filters count
        enabled = get_enabled_filters("SIDEWAYS")
        test_passed = len(enabled) >= 10
        print_result(
            "Sufficient filters enabled",
            test_passed,
            f"Enabled: {len(enabled)} filters"
        )
        all_passed &= test_passed
        
    except Exception as e:
        print_result("Filter definitions import", False, str(e))
        all_passed = False
    
    return all_passed


def validate_circuit_breaker() -> bool:
    """Validate improved circuit breaker."""
    print_header("3. Improved Circuit Breaker")
    
    all_passed = True
    
    try:
        from src.risk.circuit_breaker_improved import (
            ImprovedCircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerState,
        )
        
        # Create test instance
        config = CircuitBreakerConfig(
            max_trades_per_day=8,
            max_loss_per_day_pct=0.03,
            max_consecutive_losses_sideways=3,
        )
        cb = ImprovedCircuitBreaker(
            config=config,
            total_capital=100_000_000,
            stats_file="test_validation_cb.json",
        )
        
        # Test 1: Initial state is NORMAL
        test_passed = cb.state == CircuitBreakerState.NORMAL
        print_result(
            "Initial state is NORMAL",
            test_passed,
            f"State: {cb.state.value}"
        )
        all_passed &= test_passed
        
        # Test 2: Position multiplier is 1.0 in NORMAL
        test_passed = cb.get_position_multiplier() == 1.0
        print_result(
            "Position multiplier 1.0 in NORMAL",
            test_passed,
            f"Multiplier: {cb.get_position_multiplier()}"
        )
        all_passed &= test_passed
        
        # Test 3: Warning state on VNINDEX drop
        cb2 = ImprovedCircuitBreaker(
            config=config,
            stats_file="test_validation_cb2.json",
        )
        cb2.check_and_update(portfolio_pnl_pct=0.0, vnindex_change_pct=-0.018)
        test_passed = cb2.state == CircuitBreakerState.WARNING
        print_result(
            "WARNING state on VNINDEX -1.8%",
            test_passed,
            f"State: {cb2.state.value}"
        )
        all_passed &= test_passed
        
        # Test 4: Caution state on deeper VNINDEX drop
        cb3 = ImprovedCircuitBreaker(
            config=config,
            stats_file="test_validation_cb3.json",
        )
        cb3.check_and_update(portfolio_pnl_pct=0.0, vnindex_change_pct=-0.022)
        test_passed = cb3.state == CircuitBreakerState.CAUTION
        print_result(
            "CAUTION state on VNINDEX -2.2%",
            test_passed,
            f"State: {cb3.state.value}, Multiplier: {cb3.get_position_multiplier()}"
        )
        all_passed &= test_passed
        
        # Test 5: Trip on daily loss
        cb4 = ImprovedCircuitBreaker(
            config=config,
            stats_file="test_validation_cb4.json",
        )
        is_tripped, _ = cb4.check_and_update(portfolio_pnl_pct=-0.04, vnindex_change_pct=0.0)
        test_passed = is_tripped and cb4.is_tripped
        print_result(
            "TRIPPED on -4% daily loss",
            test_passed,
            f"Tripped: {is_tripped}"
        )
        all_passed &= test_passed
        
        # Test 6: Regime-aware consecutive loss limits
        bull_limit = config.get_max_consecutive_losses("BULL")
        bear_limit = config.get_max_consecutive_losses("BEAR")
        test_passed = bull_limit > bear_limit
        print_result(
            "Regime-aware loss limits",
            test_passed,
            f"BULL={bull_limit}, BEAR={bear_limit}"
        )
        all_passed &= test_passed
        
        # Test 7: Stats tracking
        cb5 = ImprovedCircuitBreaker(
            config=config,
            stats_file="test_validation_cb5.json",
        )
        cb5.record_trade(pnl=1_000_000)
        cb5.record_trade(pnl=-500_000)
        stats = cb5.get_stats()
        test_passed = (
            stats["trades_today"] == 2 and
            stats["consecutive_losses"] == 1
        )
        print_result(
            "Stats tracking works",
            test_passed,
            f"Trades: {stats['trades_today']}, Losses: {stats['consecutive_losses']}"
        )
        all_passed &= test_passed
        
        # Cleanup test files
        for f in ["test_validation_cb.json", "test_validation_cb2.json", 
                  "test_validation_cb3.json", "test_validation_cb4.json",
                  "test_validation_cb5.json"]:
            if os.path.exists(f):
                os.remove(f)
        
    except Exception as e:
        print_result("Circuit breaker import", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def validate_thresholds() -> bool:
    """Validate balanced thresholds."""
    print_header("4. Balanced Thresholds")
    
    all_passed = True
    
    try:
        from src.config.trading_improvements_v12 import (
            BALANCED_THRESHOLDS,
            get_threshold,
        )
        
        # Test 1: Liquidity threshold is balanced
        liquidity = get_threshold("min_liquidity_value")
        test_passed = 500_000_000 <= liquidity <= 1_000_000_000
        print_result(
            "Liquidity threshold balanced",
            test_passed,
            f"Value: {liquidity:,.0f} VND"
        )
        all_passed &= test_passed
        
        # Test 2: Gap thresholds are balanced
        gap_block = get_threshold("gap_block_threshold")
        gap_warn = get_threshold("gap_warn_threshold")
        test_passed = (
            0.04 <= gap_block <= 0.07 and
            0.02 <= gap_warn <= 0.05 and
            gap_warn < gap_block
        )
        print_result(
            "Gap thresholds balanced",
            test_passed,
            f"Block: {gap_block:.1%}, Warn: {gap_warn:.1%}"
        )
        all_passed &= test_passed
        
        # Test 3: Regime-specific confidence
        conf_bull = get_threshold("min_confidence", "BULL")
        conf_bear = get_threshold("min_confidence", "BEAR")
        test_passed = conf_bull < conf_bear
        print_result(
            "Regime confidence thresholds",
            test_passed,
            f"BULL: {conf_bull}, BEAR: {conf_bear}"
        )
        all_passed &= test_passed
        
        # Test 4: Consecutive loss limit is reasonable
        loss_limit = get_threshold("consecutive_loss_limit")
        test_passed = 3 <= loss_limit <= 5
        print_result(
            "Consecutive loss limit reasonable",
            test_passed,
            f"Limit: {loss_limit}"
        )
        all_passed &= test_passed
        
    except Exception as e:
        print_result("Thresholds import", False, str(e))
        all_passed = False
    
    return all_passed


def validate_error_handling() -> bool:
    """Validate error handling."""
    print_header("5. Error Handling")
    
    all_passed = True
    
    try:
        from src.config.trading_improvements_v12 import (
            get_filter_config,
            get_threshold,
        )
        
        # Test 1: Unknown filter returns safe default
        cfg = get_filter_config("unknown_filter_xyz", "SIDEWAYS")
        test_passed = cfg["enabled"] is False and cfg["can_block"] is False
        print_result(
            "Unknown filter returns safe default",
            test_passed,
            f"enabled={cfg['enabled']}, can_block={cfg['can_block']}"
        )
        all_passed &= test_passed
        
        # Test 2: Unknown threshold returns zero
        value = get_threshold("unknown_threshold_xyz")
        test_passed = value == 0.0
        print_result(
            "Unknown threshold returns zero",
            test_passed,
            f"Value: {value}"
        )
        all_passed &= test_passed
        
        # Test 3: Invalid regime defaults to SIDEWAYS
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        config = BalancedEntryConfig()
        cfg_unknown = config.get_regime_config("INVALID_REGIME")
        cfg_sideways = config.get_regime_config("SIDEWAYS")
        test_passed = cfg_unknown["min_confidence"] == cfg_sideways["min_confidence"]
        print_result(
            "Invalid regime defaults to SIDEWAYS",
            test_passed,
        )
        all_passed &= test_passed
        
    except Exception as e:
        print_result("Error handling", False, str(e))
        all_passed = False
    
    return all_passed


def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("  TRADING IMPROVEMENTS v12.0 VALIDATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    
    # Run all validations
    results.append(("Balanced Config", validate_balanced_config()))
    results.append(("Balanced Filters", validate_balanced_filters()))
    results.append(("Circuit Breaker", validate_circuit_breaker()))
    results.append(("Thresholds", validate_thresholds()))
    results.append(("Error Handling", validate_error_handling()))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} | {name}")
    
    print("\n" + "-" * 60)
    print(f"  Total: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n  🎉 ALL VALIDATIONS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
