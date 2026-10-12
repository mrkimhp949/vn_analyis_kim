# -*- coding: utf-8 -*-
"""
Trading Improvements v12.0 - Balanced Filter Configuration

This module provides balanced filter configurations that:
1. Re-enable critical filters that were disabled in v10.3
2. Use adaptive thresholds based on market regime
3. Provide proper error handling and logging
4. Balance between signal quality and quantity

Author: Trading Bot Team
Version: 12.0.0
Date: 2025-01
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FilterPriority(Enum):
    """Filter priority levels for entry logic."""
    CRITICAL = "critical"      # Must pass, no exceptions
    IMPORTANT = "important"    # High weight, can add warnings
    OPTIONAL = "optional"      # Low weight, informational only


class MarketRegimeType(Enum):
    """Market regime types."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class FilterConfig:
    """Configuration for a single filter."""
    name: str
    enabled: bool
    priority: FilterPriority
    weight: float = 1.0
    can_block: bool = True
    can_warn: bool = True
    description: str = ""
    
    # Regime-specific overrides
    regime_overrides: Dict[str, Dict] = field(default_factory=dict)
    
    def get_config_for_regime(self, regime: str) -> Dict:
        """Get filter config adjusted for market regime."""
        base_config = {
            "enabled": self.enabled,
            "weight": self.weight,
            "can_block": self.can_block,
            "can_warn": self.can_warn,
        }
        
        if regime in self.regime_overrides:
            base_config.update(self.regime_overrides[regime])
        
        return base_config


@dataclass
class BalancedEntryConfig:
    """
    Balanced entry configuration v12.0
    
    Key changes from v10.3 (RELAXED):
    - Re-enable important filters with balanced thresholds
    - Use regime-aware settings instead of blanket disable
    - Maintain signal quality while allowing reasonable quantity
    """
    
    # Core confidence thresholds - BALANCED (not too strict, not too loose)
    min_confidence_bull: int = 45      # BALANCED: 45% in bull (was 40 in v10.3)
    min_confidence_sideways: int = 55  # BALANCED: 55% in sideways (was 50)
    min_confidence_bear: int = 65      # BALANCED: 65% in bear (was 60)
    min_confidence_high_vol: int = 70  # BALANCED: 70% in high vol (was 65)
    
    # Risk/Reward thresholds - BALANCED
    min_rr_bull: float = 1.5           # BALANCED: 1.5 in bull
    min_rr_sideways: float = 1.8       # BALANCED: 1.8 in sideways
    min_rr_bear: float = 2.2           # BALANCED: 2.2 in bear
    min_rr_high_vol: float = 2.5       # BALANCED: 2.5 in high vol
    
    # Warning limits - BALANCED
    max_warnings_bull: int = 6         # Allow more in bull
    max_warnings_sideways: int = 5     # Standard
    max_warnings_bear: int = 3         # Strict in bear
    max_warnings_high_vol: int = 3     # Strict in high vol
    
    # Liquidity thresholds - BALANCED (not too restrictive)
    min_liquidity_value: float = 800_000_000   # 800M VND (was 500M in v10.3)
    min_avg_volume: int = 40_000               # 40k shares (was 30k)
    
    # Filter enables - RE-ENABLED with balanced settings
    use_sector_strength_filter: bool = True    # RE-ENABLED (was False)
    use_market_breadth_filter: bool = True     # RE-ENABLED (was False)
    use_foreign_flow_filter: bool = True       # RE-ENABLED (was False)
    use_session_timing_filter: bool = True     # RE-ENABLED (was False)
    use_gap_analysis_filter: bool = True       # RE-ENABLED (was False)
    use_accumulation_filter: bool = True       # RE-ENABLED (was False)
    use_vn30_correlation_filter: bool = False  # Keep disabled (often unavailable)
    use_order_book_timing: bool = False        # Keep disabled (often unavailable)
    use_vn_news_sentiment_filter: bool = False # Keep disabled (data often delayed)
    use_pre_holiday_filter: bool = True        # RE-ENABLED (was False)
    use_intraday_momentum_filter: bool = True  # RE-ENABLED (was False)
    use_margin_check: bool = True              # RE-ENABLED (was False)
    use_consecutive_loss_protection: bool = True  # Keep enabled (critical safety)
    
    def get_regime_config(self, regime: str) -> Dict:
        """Get configuration adjusted for market regime."""
        regime = regime.upper() if regime else "SIDEWAYS"
        
        if regime == "BULL":
            return {
                "min_confidence": self.min_confidence_bull,
                "min_risk_reward": self.min_rr_bull,
                "max_warnings": self.max_warnings_bull,
                "position_multiplier": 1.1,
                "filter_strictness": 0.8,  # 80% strictness (more lenient)
            }
        elif regime == "BEAR":
            return {
                "min_confidence": self.min_confidence_bear,
                "min_risk_reward": self.min_rr_bear,
                "max_warnings": self.max_warnings_bear,
                "position_multiplier": 0.6,
                "filter_strictness": 1.2,  # 120% strictness (more strict)
            }
        elif regime == "HIGH_VOLATILITY":
            return {
                "min_confidence": self.min_confidence_high_vol,
                "min_risk_reward": self.min_rr_high_vol,
                "max_warnings": self.max_warnings_high_vol,
                "position_multiplier": 0.5,
                "filter_strictness": 1.3,  # 130% strictness (most strict)
            }
        else:  # SIDEWAYS
            return {
                "min_confidence": self.min_confidence_sideways,
                "min_risk_reward": self.min_rr_sideways,
                "max_warnings": self.max_warnings_sideways,
                "position_multiplier": 0.9,
                "filter_strictness": 1.0,  # 100% strictness (baseline)
            }


# =============================================================================
# BALANCED FILTER DEFINITIONS
# =============================================================================

BALANCED_FILTERS: Dict[str, FilterConfig] = {
    # CRITICAL FILTERS - Always enabled, must pass
    "price_limit": FilterConfig(
        name="price_limit",
        enabled=True,
        priority=FilterPriority.CRITICAL,
        weight=2.0,
        can_block=True,
        description="Vietnam ±7% price limit check - ALWAYS REQUIRED",
    ),
    "liquidity": FilterConfig(
        name="liquidity",
        enabled=True,
        priority=FilterPriority.CRITICAL,
        weight=2.0,
        can_block=True,
        description="Minimum liquidity check - ALWAYS REQUIRED",
    ),
    "stop_loss_valid": FilterConfig(
        name="stop_loss_valid",
        enabled=True,
        priority=FilterPriority.CRITICAL,
        weight=2.0,
        can_block=True,
        description="Valid stop loss check - ALWAYS REQUIRED",
    ),
    "consecutive_loss": FilterConfig(
        name="consecutive_loss",
        enabled=True,
        priority=FilterPriority.CRITICAL,
        weight=2.0,
        can_block=True,
        description="Consecutive loss protection - ALWAYS REQUIRED",
    ),
    
    # IMPORTANT FILTERS - Re-enabled with balanced thresholds
    "sector_strength": FilterConfig(
        name="sector_strength",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.IMPORTANT,
        weight=1.5,
        can_block=False,  # Warn only, don't block
        can_warn=True,
        description="Sector strength analysis - warns if sector weak",
        regime_overrides={
            "BEAR": {"can_block": True, "weight": 2.0},  # Block in bear
            "HIGH_VOLATILITY": {"can_block": True, "weight": 2.0},
        },
    ),
    "market_breadth": FilterConfig(
        name="market_breadth",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.IMPORTANT,
        weight=1.5,
        can_block=False,  # Warn only in normal conditions
        can_warn=True,
        description="Market breadth check - warns if breadth weak",
        regime_overrides={
            "BEAR": {"can_block": True},
            "HIGH_VOLATILITY": {"can_block": True},
        },
    ),
    "foreign_flow": FilterConfig(
        name="foreign_flow",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.IMPORTANT,
        weight=1.5,
        can_block=False,  # Warn only
        can_warn=True,
        description="Foreign investor flow - warns if heavy selling",
        regime_overrides={
            "BEAR": {"can_block": True, "weight": 2.0},  # Critical in bear
        },
    ),
    "trend_alignment": FilterConfig(
        name="trend_alignment",
        enabled=True,
        priority=FilterPriority.IMPORTANT,
        weight=1.5,
        can_block=False,
        can_warn=True,
        description="EMA trend alignment check",
    ),
    "volatility": FilterConfig(
        name="volatility",
        enabled=True,
        priority=FilterPriority.IMPORTANT,
        weight=1.5,
        can_block=True,  # Block if extreme volatility
        can_warn=True,
        description="Volatility filter - blocks extreme volatility",
    ),
    "rsi_check": FilterConfig(
        name="rsi_check",
        enabled=True,
        priority=FilterPriority.IMPORTANT,
        weight=1.3,
        can_block=False,  # Warn only
        can_warn=True,
        description="RSI overbought/oversold check",
    ),
    "correlation": FilterConfig(
        name="correlation",
        enabled=True,
        priority=FilterPriority.IMPORTANT,
        weight=1.3,
        can_block=False,
        can_warn=True,
        description="Portfolio correlation check",
    ),
    
    # OPTIONAL FILTERS - Re-enabled but with lower weight
    "session_timing": FilterConfig(
        name="session_timing",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.OPTIONAL,
        weight=1.0,
        can_block=False,  # Never block, only adjust confidence
        can_warn=True,
        description="Trading session timing optimization",
    ),
    "gap_analysis": FilterConfig(
        name="gap_analysis",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.OPTIONAL,
        weight=1.0,
        can_block=False,  # Warn only for gaps
        can_warn=True,
        description="Gap up/down analysis",
        regime_overrides={
            "HIGH_VOLATILITY": {"can_block": True},  # Block large gaps in high vol
        },
    ),
    "accumulation": FilterConfig(
        name="accumulation",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.OPTIONAL,
        weight=1.0,
        can_block=False,
        can_warn=True,
        description="Accumulation/Distribution analysis",
    ),
    "pre_holiday": FilterConfig(
        name="pre_holiday",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.OPTIONAL,
        weight=1.0,
        can_block=False,  # Warn only
        can_warn=True,
        description="Pre-holiday risk warning",
    ),
    "intraday_momentum": FilterConfig(
        name="intraday_momentum",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.OPTIONAL,
        weight=1.0,
        can_block=False,
        can_warn=True,
        description="Intraday momentum check",
        regime_overrides={
            "HIGH_VOLATILITY": {"can_block": True},
        },
    ),
    "margin_check": FilterConfig(
        name="margin_check",
        enabled=True,  # RE-ENABLED
        priority=FilterPriority.OPTIONAL,
        weight=1.0,
        can_block=True,  # Block if margin critical
        can_warn=True,
        description="Margin availability check",
    ),
    
    # DISABLED FILTERS - Keep disabled due to data availability issues
    "vn30_correlation": FilterConfig(
        name="vn30_correlation",
        enabled=False,  # Keep disabled
        priority=FilterPriority.OPTIONAL,
        weight=0.5,
        can_block=False,
        description="VN30 correlation - disabled (data often unavailable)",
    ),
    "order_book": FilterConfig(
        name="order_book",
        enabled=False,  # Keep disabled
        priority=FilterPriority.OPTIONAL,
        weight=0.5,
        can_block=False,
        description="Order book analysis - disabled (data often unavailable)",
    ),
    "news_sentiment": FilterConfig(
        name="news_sentiment",
        enabled=False,  # Keep disabled
        priority=FilterPriority.OPTIONAL,
        weight=0.5,
        can_block=False,
        description="News sentiment - disabled (data often delayed)",
    ),
}


def get_filter_config(filter_name: str, regime: str = "SIDEWAYS") -> Dict:
    """
    Get filter configuration adjusted for market regime.
    
    Args:
        filter_name: Name of the filter
        regime: Market regime (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY)
    
    Returns:
        Dict with filter configuration
    """
    if filter_name not in BALANCED_FILTERS:
        logger.warning(f"Unknown filter: {filter_name}, using default config")
        return {
            "enabled": False,
            "weight": 1.0,
            "can_block": False,
            "can_warn": True,
        }
    
    filter_cfg = BALANCED_FILTERS[filter_name]
    return filter_cfg.get_config_for_regime(regime)


def get_enabled_filters(regime: str = "SIDEWAYS") -> List[str]:
    """Get list of enabled filters for given regime."""
    enabled = []
    for name, cfg in BALANCED_FILTERS.items():
        regime_cfg = cfg.get_config_for_regime(regime)
        if regime_cfg.get("enabled", cfg.enabled):
            enabled.append(name)
    return enabled


def get_critical_filters() -> List[str]:
    """Get list of critical filters that must always pass."""
    return [
        name for name, cfg in BALANCED_FILTERS.items()
        if cfg.priority == FilterPriority.CRITICAL
    ]


# =============================================================================
# BALANCED THRESHOLD CONSTANTS
# =============================================================================

# These replace the "RELAXED v10.3" values with balanced alternatives

BALANCED_THRESHOLDS = {
    # Entry confidence - BALANCED
    "min_confidence": {
        "BULL": 45,
        "SIDEWAYS": 55,
        "BEAR": 65,
        "HIGH_VOLATILITY": 70,
    },
    
    # Risk/Reward - BALANCED
    "min_risk_reward": {
        "BULL": 1.5,
        "SIDEWAYS": 1.8,
        "BEAR": 2.2,
        "HIGH_VOLATILITY": 2.5,
    },
    
    # Max warnings - BALANCED
    "max_warnings": {
        "BULL": 6,
        "SIDEWAYS": 5,
        "BEAR": 3,
        "HIGH_VOLATILITY": 3,
    },
    
    # Liquidity - BALANCED (not too restrictive)
    "min_liquidity_value": 800_000_000,  # 800M VND
    "min_avg_volume": 40_000,            # 40k shares
    
    # Gap thresholds - BALANCED
    "gap_block_threshold": 0.055,        # 5.5% (was 7% in v10.3)
    "gap_warn_threshold": 0.035,         # 3.5% (was 5%)
    
    # Intraday momentum - BALANCED
    "intraday_momentum_threshold": 0.04, # 4% (was 5%)
    
    # Consecutive loss - BALANCED
    "consecutive_loss_limit": 4,         # 4 losses (was 5)
    "consecutive_loss_cooldown_days": 4, # 4 days (was 3)
    
    # Foreign flow - BALANCED
    "foreign_heavy_selling_block": -0.25,  # -25% (was -30%)
    "foreign_moderate_selling_warn": -0.12, # -12% (was -15%)
    
    # Sector breadth - BALANCED
    "sector_breadth_block": 0.30,        # 30% (was 25%)
    "sector_breadth_warn": 0.40,         # 40%
}


def get_threshold(name: str, regime: str = "SIDEWAYS") -> float:
    """
    Get threshold value, optionally adjusted for regime.
    
    Args:
        name: Threshold name
        regime: Market regime
    
    Returns:
        Threshold value
    """
    if name not in BALANCED_THRESHOLDS:
        logger.warning(f"Unknown threshold: {name}")
        return 0.0
    
    value = BALANCED_THRESHOLDS[name]
    
    # If dict, get regime-specific value
    if isinstance(value, dict):
        regime = regime.upper() if regime else "SIDEWAYS"
        return value.get(regime, value.get("SIDEWAYS", 0.0))
    
    return value


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_balanced_config: Optional[BalancedEntryConfig] = None


def get_balanced_entry_config() -> BalancedEntryConfig:
    """Get singleton instance of balanced entry config."""
    global _balanced_config
    if _balanced_config is None:
        _balanced_config = BalancedEntryConfig()
    return _balanced_config


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log_filter_decision(
    filter_name: str,
    symbol: str,
    passed: bool,
    reason: str,
    regime: str = "SIDEWAYS",
    level: str = "DEBUG",
) -> None:
    """
    Log filter decision with consistent format.
    
    Args:
        filter_name: Name of the filter
        symbol: Stock symbol
        passed: Whether filter passed
        reason: Reason for decision
        regime: Current market regime
        level: Log level (DEBUG, INFO, WARNING)
    """
    status = "✅ PASS" if passed else "❌ BLOCK"
    msg = f"[{symbol}] Filter '{filter_name}' {status}: {reason} (regime: {regime})"
    
    if level == "WARNING":
        logger.warning(msg)
    elif level == "INFO":
        logger.info(msg)
    else:
        logger.debug(msg)


def log_entry_summary(
    symbol: str,
    should_enter: bool,
    confidence: int,
    filters_passed: int,
    filters_total: int,
    warnings: List[str],
    regime: str,
) -> None:
    """
    Log entry decision summary.
    
    Args:
        symbol: Stock symbol
        should_enter: Whether entry is recommended
        confidence: Final confidence score
        filters_passed: Number of filters passed
        filters_total: Total number of filters checked
        warnings: List of warning messages
        regime: Current market regime
    """
    status = "🎯 ENTRY" if should_enter else "🚫 NO ENTRY"
    
    logger.info(
        f"{status} [{symbol}] "
        f"Confidence: {confidence}% | "
        f"Filters: {filters_passed}/{filters_total} | "
        f"Warnings: {len(warnings)} | "
        f"Regime: {regime}"
    )
    
    if warnings and len(warnings) <= 3:
        for w in warnings:
            logger.info(f"  ⚠️ {w}")
    elif warnings:
        logger.info(f"  ⚠️ {len(warnings)} warnings (see debug log for details)")
        for w in warnings:
            logger.debug(f"  ⚠️ {w}")
