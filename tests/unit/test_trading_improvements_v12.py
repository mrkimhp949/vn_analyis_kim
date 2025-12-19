# -*- coding: utf-8 -*-
"""
Tests for Trading Improvements v12.0

Tests cover:
1. Balanced filter configuration
2. Regime-aware thresholds
3. Improved circuit breaker
4. Error handling and logging

Author: Trading Bot Team
Version: 12.0.0
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date
import pandas as pd
import numpy as np


# =============================================================================
# TEST: Balanced Entry Configuration
# =============================================================================

class TestBalancedEntryConfig:
    """Tests for BalancedEntryConfig."""
    
    def test_default_config_values(self):
        """Test default configuration values are balanced."""
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        
        config = BalancedEntryConfig()
        
        # Check confidence thresholds are balanced (not too strict, not too loose)
        assert 40 <= config.min_confidence_bull <= 50
        assert 50 <= config.min_confidence_sideways <= 60
        assert 60 <= config.min_confidence_bear <= 70
        assert 65 <= config.min_confidence_high_vol <= 75
        
        # Check R:R thresholds
        assert 1.3 <= config.min_rr_bull <= 1.8
        assert 1.5 <= config.min_rr_sideways <= 2.0
        assert 2.0 <= config.min_rr_bear <= 2.5
        
        # Check filters are re-enabled
        assert config.use_sector_strength_filter is True
        assert config.use_market_breadth_filter is True
        assert config.use_foreign_flow_filter is True
        assert config.use_session_timing_filter is True
        
        # Check some filters remain disabled (data availability issues)
        assert config.use_vn30_correlation_filter is False
        assert config.use_order_book_timing is False
    
    def test_regime_config_bull(self):
        """Test BULL regime configuration."""
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        
        config = BalancedEntryConfig()
        regime_cfg = config.get_regime_config("BULL")
        
        assert regime_cfg["min_confidence"] == config.min_confidence_bull
        assert regime_cfg["min_risk_reward"] == config.min_rr_bull
        assert regime_cfg["position_multiplier"] > 1.0  # Larger positions in bull
        assert regime_cfg["filter_strictness"] < 1.0  # More lenient
    
    def test_regime_config_bear(self):
        """Test BEAR regime configuration."""
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        
        config = BalancedEntryConfig()
        regime_cfg = config.get_regime_config("BEAR")
        
        assert regime_cfg["min_confidence"] == config.min_confidence_bear
        assert regime_cfg["min_risk_reward"] == config.min_rr_bear
        assert regime_cfg["position_multiplier"] < 1.0  # Smaller positions in bear
        assert regime_cfg["filter_strictness"] > 1.0  # More strict
    
    def test_regime_config_high_volatility(self):
        """Test HIGH_VOLATILITY regime configuration."""
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        
        config = BalancedEntryConfig()
        regime_cfg = config.get_regime_config("HIGH_VOLATILITY")
        
        assert regime_cfg["min_confidence"] == config.min_confidence_high_vol
        assert regime_cfg["position_multiplier"] <= 0.5  # Very small positions
        assert regime_cfg["filter_strictness"] >= 1.2  # Most strict
    
    def test_regime_config_case_insensitive(self):
        """Test regime config handles case insensitivity."""
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        
        config = BalancedEntryConfig()
        
        # Should handle lowercase
        cfg1 = config.get_regime_config("bull")
        cfg2 = config.get_regime_config("BULL")
        
        assert cfg1["min_confidence"] == cfg2["min_confidence"]
    
    def test_regime_config_unknown_defaults_to_sideways(self):
        """Test unknown regime defaults to SIDEWAYS."""
        from src.config.trading_improvements_v12 import BalancedEntryConfig
        
        config = BalancedEntryConfig()
        
        cfg_unknown = config.get_regime_config("UNKNOWN_REGIME")
        cfg_sideways = config.get_regime_config("SIDEWAYS")
        
        assert cfg_unknown["min_confidence"] == cfg_sideways["min_confidence"]


class TestBalancedFilters:
    """Tests for balanced filter definitions."""
    
    def test_critical_filters_always_enabled(self):
        """Test critical filters are always enabled."""
        from src.config.trading_improvements_v12 import (
            BALANCED_FILTERS, FilterPriority, get_critical_filters
        )
        
        critical = get_critical_filters()
        
        # Must have critical filters
        assert "price_limit" in critical
        assert "liquidity" in critical
        assert "stop_loss_valid" in critical
        assert "consecutive_loss" in critical
        
        # All critical filters must be enabled
        for name in critical:
            assert BALANCED_FILTERS[name].enabled is True
            assert BALANCED_FILTERS[name].can_block is True
    
    def test_important_filters_re_enabled(self):
        """Test important filters are re-enabled."""
        from src.config.trading_improvements_v12 import BALANCED_FILTERS, FilterPriority
        
        important_filters = [
            "sector_strength",
            "market_breadth",
            "foreign_flow",
            "trend_alignment",
            "volatility",
        ]
        
        for name in important_filters:
            assert name in BALANCED_FILTERS
            assert BALANCED_FILTERS[name].enabled is True
            assert BALANCED_FILTERS[name].priority == FilterPriority.IMPORTANT
    
    def test_filter_regime_overrides(self):
        """Test filter behavior changes by regime."""
        from src.config.trading_improvements_v12 import get_filter_config
        
        # sector_strength should block in BEAR but not in BULL
        cfg_bull = get_filter_config("sector_strength", "BULL")
        cfg_bear = get_filter_config("sector_strength", "BEAR")
        
        assert cfg_bull["can_block"] is False  # Warn only in bull
        assert cfg_bear["can_block"] is True   # Block in bear
    
    def test_get_enabled_filters(self):
        """Test getting list of enabled filters."""
        from src.config.trading_improvements_v12 import get_enabled_filters
        
        enabled = get_enabled_filters("SIDEWAYS")
        
        # Should have multiple enabled filters
        assert len(enabled) >= 10
        
        # Critical filters must be included
        assert "price_limit" in enabled
        assert "liquidity" in enabled
        
        # Disabled filters should not be included
        assert "vn30_correlation" not in enabled
        assert "order_book" not in enabled


class TestBalancedThresholds:
    """Tests for balanced threshold values."""
    
    def test_get_threshold_simple(self):
        """Test getting simple threshold values."""
        from src.config.trading_improvements_v12 import get_threshold
        
        liquidity = get_threshold("min_liquidity_value")
        assert liquidity == 800_000_000  # 800M VND
        
        volume = get_threshold("min_avg_volume")
        assert volume == 40_000  # 40k shares
    
    def test_get_threshold_regime_specific(self):
        """Test getting regime-specific thresholds."""
        from src.config.trading_improvements_v12 import get_threshold
        
        conf_bull = get_threshold("min_confidence", "BULL")
        conf_bear = get_threshold("min_confidence", "BEAR")
        
        assert conf_bull < conf_bear  # Bull should be more lenient
    
    def test_threshold_values_are_balanced(self):
        """Test threshold values are balanced (not extreme)."""
        from src.config.trading_improvements_v12 import BALANCED_THRESHOLDS
        
        # Gap thresholds should be reasonable
        gap_block = BALANCED_THRESHOLDS["gap_block_threshold"]
        assert 0.04 <= gap_block <= 0.07  # 4-7%
        
        # Consecutive loss should be reasonable
        loss_limit = BALANCED_THRESHOLDS["consecutive_loss_limit"]
        assert 3 <= loss_limit <= 5


# =============================================================================
# TEST: Improved Circuit Breaker
# =============================================================================

class TestImprovedCircuitBreaker:
    """Tests for ImprovedCircuitBreaker."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        from src.risk.circuit_breaker_improved import (
            ImprovedCircuitBreaker, CircuitBreakerConfig
        )
        
        config = CircuitBreakerConfig(
            max_trades_per_day=8,
            max_loss_per_day_pct=0.03,
            max_consecutive_losses_sideways=3,
        )
        
        return ImprovedCircuitBreaker(
            config=config,
            total_capital=100_000_000,
            stats_file="test_cb_stats.json",
        )
    
    def test_initial_state_is_normal(self, circuit_breaker):
        """Test circuit breaker starts in NORMAL state."""
        from src.risk.circuit_breaker_improved import CircuitBreakerState
        
        assert circuit_breaker.state == CircuitBreakerState.NORMAL
        assert circuit_breaker.is_tripped is False
        assert circuit_breaker.get_position_multiplier() == 1.0
    
    def test_daily_loss_trips_breaker(self, circuit_breaker):
        """Test daily loss exceeding limit trips breaker."""
        is_tripped, message = circuit_breaker.check_and_update(
            portfolio_pnl_pct=-0.04,  # -4% loss (exceeds 3% limit)
            vnindex_change_pct=0.0,
        )
        
        assert is_tripped is True
        assert "daily loss" in message.lower() or "max_daily_loss" in message.lower()
        assert circuit_breaker.is_tripped is True
        assert circuit_breaker.get_position_multiplier() == 0.0
    
    def test_vnindex_drop_trips_breaker(self, circuit_breaker):
        """Test VNINDEX drop trips breaker."""
        is_tripped, message = circuit_breaker.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=-0.03,  # -3% drop (exceeds -2.5% threshold)
        )
        
        assert is_tripped is True
        assert "vnindex" in message.lower()
    
    def test_vnindex_warning_sets_warning_state(self, circuit_breaker):
        """Test VNINDEX warning level sets WARNING state."""
        from src.risk.circuit_breaker_improved import CircuitBreakerState
        
        is_tripped, _ = circuit_breaker.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=-0.018,  # -1.8% (between warning and caution)
        )
        
        assert is_tripped is False
        assert circuit_breaker.state == CircuitBreakerState.WARNING
        assert circuit_breaker.get_position_multiplier() == 0.75
    
    def test_vnindex_caution_sets_caution_state(self, circuit_breaker):
        """Test VNINDEX caution level sets CAUTION state."""
        from src.risk.circuit_breaker_improved import CircuitBreakerState
        
        is_tripped, _ = circuit_breaker.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=-0.022,  # -2.2% (between caution and trip)
        )
        
        assert is_tripped is False
        assert circuit_breaker.state == CircuitBreakerState.CAUTION
        assert circuit_breaker.get_position_multiplier() == 0.5
    
    def test_record_trade_updates_stats(self, circuit_breaker):
        """Test recording trades updates statistics."""
        # Record a winning trade
        circuit_breaker.record_trade(pnl=1_000_000)
        
        stats = circuit_breaker.get_stats()
        assert stats["trades_today"] == 1
        assert stats["consecutive_wins"] == 1
        assert stats["consecutive_losses"] == 0
        
        # Record a losing trade
        circuit_breaker.record_trade(pnl=-500_000)
        
        stats = circuit_breaker.get_stats()
        assert stats["trades_today"] == 2
        assert stats["consecutive_wins"] == 0
        assert stats["consecutive_losses"] == 1
    
    def test_consecutive_losses_trips_breaker(self, circuit_breaker):
        """Test consecutive losses trip breaker."""
        # Record 3 consecutive losses
        for _ in range(3):
            circuit_breaker.record_trade(pnl=-100_000)
        
        is_tripped, message = circuit_breaker.check_and_update(
            portfolio_pnl_pct=-0.01,
            vnindex_change_pct=0.0,
        )
        
        assert is_tripped is True
        assert "consecutive" in message.lower()
    
    def test_regime_aware_consecutive_loss_limit(self, circuit_breaker):
        """Test consecutive loss limit changes by regime."""
        from src.risk.circuit_breaker_improved import CircuitBreakerConfig
        
        config = CircuitBreakerConfig()
        
        # BULL should allow more losses
        bull_limit = config.get_max_consecutive_losses("BULL")
        bear_limit = config.get_max_consecutive_losses("BEAR")
        
        assert bull_limit > bear_limit
    
    def test_trade_count_trips_breaker(self, circuit_breaker):
        """Test max trade count trips breaker."""
        # Record 8 trades (max limit)
        for _ in range(8):
            circuit_breaker.record_trade(pnl=100_000)
        
        is_tripped, message = circuit_breaker.check_and_update(
            portfolio_pnl_pct=0.01,
            vnindex_change_pct=0.0,
        )
        
        assert is_tripped is True
        assert "trade" in message.lower()
    
    def test_manual_reset(self, circuit_breaker):
        """Test manual reset of circuit breaker."""
        from src.risk.circuit_breaker_improved import CircuitBreakerState
        
        # Trip the breaker
        circuit_breaker.check_and_update(
            portfolio_pnl_pct=-0.05,
            vnindex_change_pct=0.0,
        )
        assert circuit_breaker.is_tripped is True
        
        # Manual reset
        circuit_breaker.manual_reset()
        
        assert circuit_breaker.state == CircuitBreakerState.NORMAL
        assert circuit_breaker.is_tripped is False
    
    def test_get_stats_returns_complete_info(self, circuit_breaker):
        """Test get_stats returns complete information."""
        stats = circuit_breaker.get_stats()
        
        required_keys = [
            "state",
            "regime",
            "position_multiplier",
            "trades_today",
            "max_trades",
            "net_pnl",
            "consecutive_losses",
            "consecutive_wins",
            "max_consecutive_losses",
            "current_drawdown",
        ]
        
        for key in required_keys:
            assert key in stats


# =============================================================================
# TEST: Logging Helpers
# =============================================================================

class TestLoggingHelpers:
    """Tests for logging helper functions."""
    
    def test_log_filter_decision(self):
        """Test filter decision logging."""
        from src.config.trading_improvements_v12 import log_filter_decision
        
        # Should not raise any errors
        log_filter_decision(
            filter_name="test_filter",
            symbol="VNM",
            passed=True,
            reason="Test passed",
            regime="BULL",
            level="DEBUG",
        )
        
        log_filter_decision(
            filter_name="test_filter",
            symbol="VNM",
            passed=False,
            reason="Test blocked",
            regime="BEAR",
            level="WARNING",
        )
    
    def test_log_entry_summary(self):
        """Test entry summary logging."""
        from src.config.trading_improvements_v12 import log_entry_summary
        
        # Should not raise any errors
        log_entry_summary(
            symbol="VNM",
            should_enter=True,
            confidence=75,
            filters_passed=10,
            filters_total=12,
            warnings=["Warning 1", "Warning 2"],
            regime="BULL",
        )


# =============================================================================
# TEST: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for improved trading logic."""
    
    def test_singleton_instances(self):
        """Test singleton pattern works correctly."""
        from src.config.trading_improvements_v12 import get_balanced_entry_config
        from src.risk.circuit_breaker_improved import get_improved_circuit_breaker
        
        # Should return same instance
        config1 = get_balanced_entry_config()
        config2 = get_balanced_entry_config()
        assert config1 is config2
        
        # Force new should create new instance
        cb1 = get_improved_circuit_breaker()
        cb2 = get_improved_circuit_breaker(force_new=True)
        assert cb1 is not cb2
    
    def test_filter_config_consistency(self):
        """Test filter configuration is consistent across modules."""
        from src.config.trading_improvements_v12 import (
            BALANCED_FILTERS,
            get_filter_config,
            get_enabled_filters,
        )
        
        # All filters in BALANCED_FILTERS should be accessible via get_filter_config
        for name in BALANCED_FILTERS:
            cfg = get_filter_config(name, "SIDEWAYS")
            assert "enabled" in cfg
            assert "weight" in cfg
        
        # Enabled filters should match BALANCED_FILTERS
        enabled = get_enabled_filters("SIDEWAYS")
        for name in enabled:
            assert BALANCED_FILTERS[name].enabled is True


# =============================================================================
# TEST: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_unknown_filter_returns_default(self):
        """Test unknown filter returns safe default."""
        from src.config.trading_improvements_v12 import get_filter_config
        
        cfg = get_filter_config("unknown_filter_xyz", "SIDEWAYS")
        
        assert cfg["enabled"] is False
        assert cfg["can_block"] is False
    
    def test_unknown_threshold_returns_zero(self):
        """Test unknown threshold returns zero."""
        from src.config.trading_improvements_v12 import get_threshold
        
        value = get_threshold("unknown_threshold_xyz")
        assert value == 0.0
    
    def test_circuit_breaker_handles_invalid_inputs(self):
        """Test circuit breaker handles invalid inputs gracefully."""
        from src.risk.circuit_breaker_improved import ImprovedCircuitBreaker
        
        cb = ImprovedCircuitBreaker(stats_file="test_cb_invalid.json")
        
        # Should not crash with edge case inputs
        is_tripped, _ = cb.check_and_update(
            portfolio_pnl_pct=0.0,
            vnindex_change_pct=0.0,
            portfolio_heat=0.0,
        )
        
        assert is_tripped is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
