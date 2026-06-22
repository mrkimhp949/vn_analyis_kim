# -*- coding: utf-8 -*-
"""
Entry Logic Improved - Wrapper module for balanced filter configuration

This module provides improved entry logic that:
1. Uses balanced filter configuration from trading_improvements_v12
2. Provides better error handling with proper logging
3. Maintains backward compatibility with existing code

Usage:
    from src.strategies.entry_logic_improved import ImprovedEntryLogicV12
    
    entry_logic = ImprovedEntryLogicV12()
    signal = entry_logic.analyze_entry(df, ml_signal, ml_confidence, symbol, market_regime)

Author: Trading Bot Team
Version: 12.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# Import balanced configuration
from src.config.trading_improvements_v12 import (
    BalancedEntryConfig,
    get_balanced_entry_config,
    get_filter_config,
    get_enabled_filters,
    get_critical_filters,
    get_threshold,
    log_filter_decision,
    log_entry_summary,
    BALANCED_FILTERS,
    FilterPriority,
)

# Import original entry logic for delegation
from src.strategies.entry_logic import ImprovedEntryLogic, EntrySignal, create_no_signal

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of a single filter check."""
    filter_name: str
    passed: bool
    blocked: bool
    warning: Optional[str]
    positive: Optional[str]
    adjustment: int
    note: Optional[str]
    execution_time_ms: float = 0.0
    error: Optional[str] = None


class ImprovedEntryLogicV12(ImprovedEntryLogic):
    """
    Improved Entry Logic v12.0 with balanced filter configuration.
    
    Key improvements over v10.3:
    1. Re-enabled important filters with balanced thresholds
    2. Regime-aware filter behavior (stricter in BEAR, more lenient in BULL)
    3. Better error handling - filters don't crash on errors
    4. Comprehensive logging for debugging
    5. Filter performance tracking
    
    Backward compatible with ImprovedEntryLogic interface.
    """
    
    def __init__(
        self,
        config: Optional[BalancedEntryConfig] = None,
        **kwargs
    ) -> None:
        """
        Initialize improved entry logic.
        
        Args:
            config: Optional BalancedEntryConfig, uses default if not provided
            **kwargs: Additional arguments passed to parent class
        """
        # Get balanced config
        self._balanced_config = config or get_balanced_entry_config()
        
        # Override kwargs with balanced config values
        balanced_kwargs = self._get_balanced_kwargs()
        balanced_kwargs.update(kwargs)  # Allow explicit overrides
        
        # Initialize parent class
        super().__init__(**balanced_kwargs)
        
        # Filter tracking
        self._filter_results: Dict[str, FilterResult] = {}
        self._last_regime: str = "SIDEWAYS"
        
        logger.info("✅ ImprovedEntryLogicV12 initialized with balanced configuration")
    
    def _get_balanced_kwargs(self) -> Dict[str, Any]:
        """Get kwargs from balanced config."""
        cfg = self._balanced_config
        return {
            "min_confidence": cfg.min_confidence_sideways,  # Base value
            "min_risk_reward": cfg.min_rr_sideways,
            "min_liquidity_value": cfg.min_liquidity_value,
            "min_avg_volume": cfg.min_avg_volume,
            "max_warnings_allowed": cfg.max_warnings_sideways,
            # Re-enabled filters
            "use_sector_strength_filter": cfg.use_sector_strength_filter,
            "use_market_breadth_filter": cfg.use_market_breadth_filter,
            "use_foreign_flow_filter": cfg.use_foreign_flow_filter,
            "use_session_timing_filter": cfg.use_session_timing_filter,
            "use_gap_analysis_filter": cfg.use_gap_analysis_filter,
            "use_accumulation_filter": cfg.use_accumulation_filter,
            "use_pre_holiday_filter": cfg.use_pre_holiday_filter,
            "use_intraday_momentum_filter": cfg.use_intraday_momentum_filter,
            "use_margin_check": cfg.use_margin_check,
            "use_consecutive_loss_protection": cfg.use_consecutive_loss_protection,
            # Keep disabled
            "use_vn30_correlation_filter": cfg.use_vn30_correlation_filter,
            "use_order_book_timing": cfg.use_order_book_timing,
            "use_vn_news_sentiment_filter": cfg.use_vn_news_sentiment_filter,
            # Balanced thresholds
            "gap_block_threshold": get_threshold("gap_block_threshold"),
            "gap_warn_threshold": get_threshold("gap_warn_threshold"),
            "intraday_momentum_threshold": get_threshold("intraday_momentum_threshold"),
            "consecutive_loss_limit": int(get_threshold("consecutive_loss_limit")),
            "consecutive_loss_cooldown_days": int(get_threshold("consecutive_loss_cooldown_days")),
            # Soft filter mode with balanced warnings
            "soft_filter_mode": True,
        }
    
    def _adjust_thresholds_for_market(self, market_regime: Optional[Dict]) -> None:
        """
        Override parent method to use balanced regime-aware thresholds.
        
        Args:
            market_regime: Market regime dict with 'regime' key
        """
        if not market_regime:
            self._last_regime = "SIDEWAYS"
            regime_config = self._balanced_config.get_regime_config("SIDEWAYS")
        else:
            regime = market_regime.get("regime", "SIDEWAYS")
            self._last_regime = regime
            regime_config = self._balanced_config.get_regime_config(regime)
        
        # Apply regime-specific thresholds
        self.min_confidence = regime_config["min_confidence"]
        self.min_risk_reward = regime_config["min_risk_reward"]
        self.max_warnings_allowed = regime_config["max_warnings"]
        self._regime_position_multiplier = regime_config["position_multiplier"]
        
        logger.debug(
            f"Adjusted thresholds for {self._last_regime}: "
            f"min_conf={self.min_confidence}, min_rr={self.min_risk_reward}, "
            f"max_warn={self.max_warnings_allowed}, pos_mult={self._regime_position_multiplier}"
        )
    
    def _safe_run_filter_v12(
        self,
        filter_name: str,
        filter_func: callable,
        symbol: str,
        **kwargs,
    ) -> FilterResult:
        """
        Safely run a filter with comprehensive error handling and logging.
        
        Improvements over parent _safe_run_filter:
        1. Returns structured FilterResult
        2. Tracks execution time
        3. Logs all decisions consistently
        4. Never crashes - always returns valid result
        
        Args:
            filter_name: Name of the filter
            filter_func: Filter function to call
            symbol: Stock symbol
            **kwargs: Arguments for filter function
        
        Returns:
            FilterResult with all details
        """
        import time
        start_time = time.time()
        
        # Get filter config for current regime
        filter_cfg = get_filter_config(filter_name, self._last_regime)
        
        # Check if filter is enabled
        if not filter_cfg.get("enabled", True):
            return FilterResult(
                filter_name=filter_name,
                passed=True,
                blocked=False,
                warning=None,
                positive=None,
                adjustment=0,
                note=f"Filter disabled for regime {self._last_regime}",
            )
        
        try:
            # Run the filter
            result = filter_func(**kwargs)
            
            execution_time = (time.time() - start_time) * 1000
            
            if result is None:
                result = {}
            
            # Determine if blocked based on filter config
            raw_blocked = result.get("blocked", False)
            can_block = filter_cfg.get("can_block", True)
            actual_blocked = raw_blocked and can_block
            
            # Create filter result
            filter_result = FilterResult(
                filter_name=filter_name,
                passed=not actual_blocked,
                blocked=actual_blocked,
                warning=result.get("warning"),
                positive=result.get("positive"),
                adjustment=result.get("adjustment", 0),
                note=result.get("note"),
                execution_time_ms=execution_time,
            )
            
            # Log decision
            if actual_blocked:
                log_filter_decision(
                    filter_name, symbol, False,
                    result.get("note", "Blocked"),
                    self._last_regime, "WARNING"
                )
            elif filter_result.warning:
                log_filter_decision(
                    filter_name, symbol, True,
                    f"Warning: {filter_result.warning}",
                    self._last_regime, "DEBUG"
                )
            
            # Track filter
            self._track_filter(filter_name, not actual_blocked, symbol, bool(filter_result.warning))
            
            return filter_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error but don't crash
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"[{symbol}] Filter '{filter_name}' error: {error_msg}")
            
            # Return pass result on error (don't block due to errors)
            return FilterResult(
                filter_name=filter_name,
                passed=True,
                blocked=False,
                warning=None,
                positive=None,
                adjustment=0,
                note=f"Error: {error_msg}",
                execution_time_ms=execution_time,
                error=error_msg,
            )
    
    def analyze_entry(
        self,
        df: pd.DataFrame,
        ml_signal: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        symbol: Optional[str] = None,
        market_regime: Optional[Dict] = None,
    ) -> EntrySignal:
        """
        Analyze entry with improved logging and error handling.
        
        This method wraps the parent analyze_entry with:
        1. Pre-analysis logging
        2. Post-analysis summary
        3. Filter result tracking
        
        Args:
            df: DataFrame with OHLCV and indicators
            ml_signal: Optional ML model signal
            ml_confidence: Optional ML model confidence
            symbol: Stock symbol
            market_regime: Market regime analysis result
        
        Returns:
            EntrySignal with recommendation
        """
        # Clear previous filter results
        self._filter_results.clear()
        
        # Log analysis start
        regime_str = market_regime.get("regime", "UNKNOWN") if market_regime else "UNKNOWN"
        logger.debug(f"[{symbol}] Starting entry analysis (regime: {regime_str})")
        
        try:
            # Call parent analyze_entry
            signal = super().analyze_entry(
                df=df,
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                symbol=symbol,
                market_regime=market_regime,
            )
            
            # Log summary
            log_entry_summary(
                symbol=symbol or "UNKNOWN",
                should_enter=signal.should_enter,
                confidence=signal.confidence,
                filters_passed=len([r for r in self._filter_results.values() if r.passed]),
                filters_total=len(self._filter_results),
                warnings=signal.warnings if hasattr(signal, 'warnings') else [],
                regime=regime_str,
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"[{symbol}] Entry analysis failed: {e}", exc_info=True)
            return create_no_signal(f"Analysis error: {str(e)}", {"symbol": symbol})
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        Get summary of filter results from last analysis.
        
        Returns:
            Dict with filter statistics
        """
        if not self._filter_results:
            return {"message": "No filter results available"}
        
        passed = [r for r in self._filter_results.values() if r.passed]
        blocked = [r for r in self._filter_results.values() if r.blocked]
        warnings = [r for r in self._filter_results.values() if r.warning]
        errors = [r for r in self._filter_results.values() if r.error]
        
        total_time = sum(r.execution_time_ms for r in self._filter_results.values())
        
        return {
            "total_filters": len(self._filter_results),
            "passed": len(passed),
            "blocked": len(blocked),
            "warnings": len(warnings),
            "errors": len(errors),
            "total_execution_time_ms": total_time,
            "blocked_by": [r.filter_name for r in blocked],
            "warning_filters": [r.filter_name for r in warnings],
            "error_filters": [r.filter_name for r in errors],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_improved_entry_logic: Optional[ImprovedEntryLogicV12] = None


def get_improved_entry_logic(
    config: Optional[BalancedEntryConfig] = None,
    force_new: bool = False,
) -> ImprovedEntryLogicV12:
    """
    Get singleton instance of improved entry logic.
    
    Args:
        config: Optional custom config
        force_new: Force create new instance
    
    Returns:
        ImprovedEntryLogicV12 instance
    """
    global _improved_entry_logic
    
    if _improved_entry_logic is None or force_new:
        _improved_entry_logic = ImprovedEntryLogicV12(config=config)
    
    return _improved_entry_logic
