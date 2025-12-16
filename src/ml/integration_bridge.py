# -*- coding: utf-8 -*-
"""
ML Integration Bridge - Connect VietnamMLIntegration with Entry Logic

This module bridges the new VietnamMLIntegration with the existing
entry logic system for seamless trading integration.

Author: Trading Bot Team  
Version: 1.0.0
"""

import logging
from typing import Dict, Optional, Any, Tuple
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# Import the new ML integration
try:
    from src.ml.vietnam_ml_integration import (
        VietnamMLIntegration,
        VietnamMarketSession,
        SignalQuality,
        MLIntegrationResult,
        get_vietnam_ml_integration,
    )

    VIETNAM_ML_AVAILABLE = True
except ImportError:
    VIETNAM_ML_AVAILABLE = False
    logger.warning("⚠️ Vietnam ML Integration not available")


class MLIntegrationBridge:
    """
    Bridge between VietnamMLIntegration and Entry Logic.

    Converts ML signals to entry logic format and provides
    confidence adjustments for position sizing.
    """

    # Quality to confidence bonus mapping
    QUALITY_BONUSES = {
        SignalQuality.PREMIUM: 10,
        SignalQuality.HIGH: 5,
        SignalQuality.MEDIUM: 0,
        SignalQuality.LOW: -5,
        SignalQuality.UNRELIABLE: -20,
    }

    # Recommendation to entry signal mapping
    RECOMMENDATION_MAP = {
        "STRONG_BUY": ("BUY", 1.2),  # Signal, position_multiplier
        "BUY": ("BUY", 1.0),
        "WEAK_BUY": ("BUY", 0.8),
        "HOLD": ("HOLD", 0.0),
        "WEAK_SELL": ("SELL", 0.8),
        "SELL": ("SELL", 1.0),
        "STRONG_SELL": ("SELL", 1.2),
        "AVOID": ("HOLD", 0.0),
    }

    def __init__(
        self,
        min_confidence: float = 55.0,
        require_quality: SignalQuality = SignalQuality.MEDIUM,
    ):
        self.min_confidence = min_confidence
        self.require_quality = require_quality

        if VIETNAM_ML_AVAILABLE:
            self._ml_integration = get_vietnam_ml_integration(
                min_confidence=min_confidence,
                require_quality=require_quality,
            )
        else:
            self._ml_integration = None

    def get_ml_signal_for_entry(
        self,
        df: pd.DataFrame,
        symbol: str,
        index_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Get ML signal in format compatible with entry logic.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            index_df: Index DataFrame (optional)

        Returns:
            Dict compatible with ImprovedEntryLogic.analyze_entry()
        """
        if not VIETNAM_ML_AVAILABLE or self._ml_integration is None:
            return self._fallback_response()

        try:
            # Get ML integration result
            result = self._ml_integration.get_signal(df, symbol, index_df)

            # Convert to entry logic format
            return self._convert_to_entry_format(result, symbol)

        except Exception as e:
            logger.error(f"ML bridge error for {symbol}: {e}")
            return self._fallback_response()

    def _convert_to_entry_format(
        self,
        result: MLIntegrationResult,
        symbol: str,
    ) -> Dict[str, Any]:
        """Convert MLIntegrationResult to entry logic format."""
        # Get signal and multiplier from recommendation
        recommendation = result.final_recommendation
        signal, position_mult = self.RECOMMENDATION_MAP.get(recommendation, ("HOLD", 0.0))

        # Calculate quality bonus
        quality_bonus = self.QUALITY_BONUSES.get(result.signal_quality, 0)

        # Build response
        response = {
            # Core signal info
            "signal": signal,
            "confidence": result.calibrated_confidence,
            "raw_confidence": result.raw_confidence,
            # Position sizing hints
            "position_multiplier": position_mult,
            "quality_bonus": quality_bonus,
            # Vietnam-specific info
            "session": result.vietnam_features.current_session.value,
            "near_ceiling": result.vietnam_features.near_ceiling,
            "near_floor": result.vietnam_features.near_floor,
            "foreign_flow": result.vietnam_features.foreign_flow_signal,
            "liquidity_score": result.vietnam_features.liquidity_score,
            # Model info
            "model_version": result.model_version,
            "ensemble_agreement": result.ensemble_agreement,
            "signal_quality": result.signal_quality.value,
            # Validation
            "is_valid": result.is_valid,
            "warnings": result.validation_warnings,
            "reasons": result.recommendation_reasons,
            # Performance context
            "model_accuracy_7d": result.model_accuracy_7d,
            "model_accuracy_30d": result.model_accuracy_30d,
            # Feature importance (top 5)
            "top_features": dict(list(result.top_features.items())[:5]),
            # Metadata
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        return response

    def _fallback_response(self) -> Dict[str, Any]:
        """Return fallback response when ML unavailable."""
        return {
            "signal": "HOLD",
            "confidence": 50.0,
            "raw_confidence": 50.0,
            "position_multiplier": 0.0,
            "quality_bonus": 0,
            "session": "UNKNOWN",
            "is_valid": False,
            "warnings": ["ML integration not available"],
            "reasons": [],
            "model_version": "fallback",
        }

    def should_trade(
        self,
        result: Dict[str, Any],
        min_confidence: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if trade should be executed based on ML result.

        Args:
            result: Result from get_ml_signal_for_entry()
            min_confidence: Override minimum confidence

        Returns:
            Tuple of (should_trade, reason)
        """
        min_conf = min_confidence or self.min_confidence

        # Check validity
        if not result.get("is_valid", False):
            warnings = result.get("warnings", [])
            return False, f"Invalid signal: {warnings[0] if warnings else 'unknown'}"

        # Check signal
        signal = result.get("signal", "HOLD")
        if signal == "HOLD":
            return False, "Signal is HOLD"

        # Check confidence
        confidence = result.get("confidence", 0)
        if confidence < min_conf:
            return False, f"Confidence {confidence:.1f}% below threshold {min_conf}%"

        # Check quality
        quality = result.get("signal_quality", "LOW")
        if quality == "UNRELIABLE":
            return False, "Model showing unreliable performance"

        # Check Vietnam-specific blocks
        if result.get("near_ceiling") and signal == "BUY":
            if confidence < 70:  # Higher threshold near ceiling
                return False, "Near ceiling - requires 70%+ confidence to buy"

        # All checks passed
        return True, f"Valid {signal} signal with {confidence:.1f}% confidence"

    def get_position_size_adjustment(
        self,
        result: Dict[str, Any],
    ) -> float:
        """
        Get position size adjustment factor based on ML signal.

        Args:
            result: Result from get_ml_signal_for_entry()

        Returns:
            Adjustment factor (0.5 - 1.5)
        """
        base_mult = result.get("position_multiplier", 1.0)

        # Quality adjustment
        quality = result.get("signal_quality", "MEDIUM")
        quality_mult = {
            "PREMIUM": 1.1,
            "HIGH": 1.0,
            "MEDIUM": 0.9,
            "LOW": 0.7,
            "UNRELIABLE": 0.5,
        }.get(quality, 1.0)

        # Ensemble agreement adjustment
        agreement = result.get("ensemble_agreement", 0.5)
        agreement_mult = 0.8 + (agreement * 0.4)  # 0.8 to 1.2

        # Liquidity adjustment
        liquidity = result.get("liquidity_score", 0.5)
        liquidity_mult = 0.8 + (liquidity * 0.4)  # 0.8 to 1.2

        # Combined adjustment
        adjustment = base_mult * quality_mult * agreement_mult * liquidity_mult

        # Clamp to reasonable range
        return max(0.5, min(1.5, adjustment))

    def update_trade_outcome(
        self,
        prediction_id: str,
        pnl_percent: float,
        outcome: str = "CLOSED",
    ):
        """
        Update ML prediction with trade outcome.

        This is CRITICAL for confidence calibration!
        Call this when a trade is closed.
        """
        if VIETNAM_ML_AVAILABLE and self._ml_integration:
            self._ml_integration.update_prediction_outcome(
                prediction_id=prediction_id,
                actual_outcome=outcome,
                pnl_percent=pnl_percent,
            )

    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Get ML performance report."""
        if VIETNAM_ML_AVAILABLE and self._ml_integration:
            return self._ml_integration.get_performance_report(days=days)
        return {"error": "ML integration not available"}


# =============================================================================
# INTEGRATION WITH ENTRY SERVICE
# =============================================================================


def enhance_entry_signal_with_ml(
    entry_signal: Any,  # EntrySignal from entry_logic
    ml_result: Dict[str, Any],
) -> Any:
    """
    Enhance an entry signal with ML insights.

    Args:
        entry_signal: EntrySignal from ImprovedEntryLogic
        ml_result: Result from MLIntegrationBridge

    Returns:
        Enhanced EntrySignal
    """
    if entry_signal is None:
        return None

    # Add ML confidence to entry signal
    ml_confidence = ml_result.get("confidence", 0)
    original_confidence = getattr(entry_signal, "confidence", 50)

    # Weighted average: 60% technical, 40% ML
    combined_confidence = (original_confidence * 0.6) + (ml_confidence * 0.4)

    # Apply quality bonus
    quality_bonus = ml_result.get("quality_bonus", 0)
    combined_confidence += quality_bonus

    # Update entry signal
    if hasattr(entry_signal, "confidence"):
        entry_signal.confidence = int(min(100, max(0, combined_confidence)))

    # Add ML metadata
    if hasattr(entry_signal, "metadata"):
        if entry_signal.metadata is None:
            entry_signal.metadata = {}
        entry_signal.metadata["ml_confidence"] = ml_confidence
        entry_signal.metadata["ml_quality"] = ml_result.get("signal_quality", "UNKNOWN")
        entry_signal.metadata["ml_version"] = ml_result.get("model_version", "unknown")
        entry_signal.metadata["ensemble_agreement"] = ml_result.get("ensemble_agreement", 0)

    # Add warnings
    if hasattr(entry_signal, "warnings"):
        ml_warnings = ml_result.get("warnings", [])
        if ml_warnings:
            entry_signal.warnings.extend(ml_warnings)

    # Add reasons
    if hasattr(entry_signal, "reasons"):
        ml_reasons = ml_result.get("reasons", [])
        if ml_reasons:
            entry_signal.reasons.extend(ml_reasons)

    return entry_signal


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================


_bridge_instance: Optional[MLIntegrationBridge] = None


def get_ml_integration_bridge(
    min_confidence: float = 55.0,
) -> MLIntegrationBridge:
    """Get singleton instance of MLIntegrationBridge."""
    global _bridge_instance

    if _bridge_instance is None:
        _bridge_instance = MLIntegrationBridge(min_confidence=min_confidence)

    return _bridge_instance


def reset_ml_integration_bridge():
    """Reset singleton instance."""
    global _bridge_instance
    _bridge_instance = None


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "MLIntegrationBridge",
    "enhance_entry_signal_with_ml",
    "get_ml_integration_bridge",
    "reset_ml_integration_bridge",
    "VIETNAM_ML_AVAILABLE",
]
