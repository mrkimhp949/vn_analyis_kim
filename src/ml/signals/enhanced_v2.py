# -*- coding: utf-8 -*-
"""
Enhanced ML Signal Generator V2
Drop-in replacement for EnhancedMLSignalGenerator with improved accuracy
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check if V2 models exist
MODELS_DIR = "models"
V2_AVAILABLE = os.path.exists(os.path.join(MODELS_DIR, "scaler_v2.pkl"))


class EnhancedMLSignalGeneratorV2:
    """
    Enhanced ML Signal Generator V2 - Improved accuracy (58-62%)

    Drop-in replacement for EnhancedMLSignalGenerator.
    Uses V2 features and models when available, falls back to V1.
    """

    def __init__(self, use_v2: bool = True):
        """
        Initialize signal generator.

        Args:
            use_v2: Use V2 models if available (default True)
        """
        self.use_v2 = use_v2 and V2_AVAILABLE
        self.model_loaded = False
        self.generator = None

        self._init_generator()

    def _init_generator(self) -> None:
        """Initialize the appropriate generator."""
        if self.use_v2:
            try:
                from src.ml.signals.generator_v2 import MLSignalGeneratorV2

                self.generator = MLSignalGeneratorV2(
                    model_name="rf",  # Best on unseen data
                    confidence_threshold=0.55,
                    use_ensemble=False,
                )
                self.model_loaded = True
                logger.info("✅ Using ML Signal Generator V2 (improved accuracy)")
            except Exception as e:
                logger.warning(f"⚠️ V2 init failed: {e}, falling back to V1")
                self._init_v1_fallback()
        else:
            self._init_v1_fallback()

    def _init_v1_fallback(self) -> None:
        """Initialize V1 generator as fallback."""
        try:
            from src.ml.signals.enhanced import EnhancedMLSignalGenerator

            self.generator = EnhancedMLSignalGenerator()
            self.model_loaded = self.generator.model_loaded
            self.use_v2 = False
            logger.info("📊 Using ML Signal Generator V1 (fallback)")
        except Exception as e:
            logger.error(f"❌ V1 init also failed: {e}")
            self.model_loaded = False

    def analyze(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        explain: bool = False,
        symbol: Optional[str] = None,
    ) -> Dict:
        """
        Analyze and generate ML signal.

        Compatible with EnhancedMLSignalGenerator interface.

        Args:
            df: DataFrame with OHLCV data
            index_df: VNINDEX DataFrame
            explain: Include explanation (V1 only)
            symbol: Stock symbol for logging

        Returns:
            Dict with signal, confidence, and metadata
        """
        if not self.model_loaded or self.generator is None:
            return self._fallback_response()

        try:
            if self.use_v2:
                # V2 generator
                result = self.generator.generate_signal(df, index_df, symbol)

                # Add compatibility fields for V1 interface
                result["raw_confidence"] = result["confidence"]
                result["ml_score"] = result["probabilities"]["buy"]
                result["technical_score"] = {}
                result["reason"] = f"ML V2 {result['model']}"

                # Add price info
                if not df.empty:
                    latest = df.iloc[-1]
                    result["price"] = float(latest.get("close", 0))
                    result["rsi"] = float(latest.get("rsi", 50)) if "rsi" in latest else 50

                    # EMA trend
                    ema20 = latest.get("ema20", latest.get("close", 0))
                    ema50 = latest.get("ema50", latest.get("close", 0))
                    result["ema_trend"] = "UP" if ema20 > ema50 else "DOWN"

                return result
            else:
                # V1 generator
                return self.generator.analyze(df, index_df, explain, symbol)

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._fallback_response()

    def _fallback_response(self) -> Dict:
        """Return fallback response when models unavailable."""
        return {
            "signal": "HOLD",
            "confidence": 0,
            "raw_confidence": 0,
            "ml_score": 0.5,
            "technical_score": {},
            "reason": "Models not available",
            "price": 0,
            "rsi": 50,
            "ema_trend": "NEUTRAL",
        }


# Factory function for easy switching
def get_enhanced_signal_generator(use_v2: bool = True) -> EnhancedMLSignalGeneratorV2:
    """
    Get enhanced signal generator.

    Args:
        use_v2: Use V2 models if available

    Returns:
        EnhancedMLSignalGeneratorV2 instance
    """
    return EnhancedMLSignalGeneratorV2(use_v2=use_v2)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING ENHANCED ML SIGNAL GENERATOR V2")
    print("=" * 60)

    from src.data.loader import load_data

    generator = EnhancedMLSignalGeneratorV2(use_v2=True)

    print(f"\n📊 Using V2: {generator.use_v2}")
    print(f"📊 Model loaded: {generator.model_loaded}")

    # Test with sample symbol
    symbol = "VNM"
    df = load_data(symbol, lookback=200)
    index_df = load_data("VNINDEX", lookback=200, is_index=True)

    result = generator.analyze(df, index_df, symbol=symbol)

    print(f"\n📊 Signal for {symbol}:")
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   ML Score: {result.get('ml_score', 'N/A')}")

    print("\n✅ Test complete!")
