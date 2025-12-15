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


def validate_ml_prediction(
    signal_result: dict,
    symbol: str,
    min_confidence: float = 50.0,
    require_technical_confirmation: bool = True,
) -> dict:
    """
    Validate ML prediction before using for trading decisions.

    This function adds an extra layer of validation to ensure ML signals
    meet minimum quality standards and are properly calibrated.

    Args:
        signal_result: Result from signal generator
        symbol: Stock symbol
        min_confidence: Minimum confidence threshold
        require_technical_confirmation: Require technical indicator alignment

    Returns:
        Validated signal with additional metadata
    """
    validated = signal_result.copy()
    validated["validation"] = {
        "passed": True,
        "warnings": [],
        "adjustments": [],
    }

    # 1. Confidence calibration using historical accuracy
    try:
        from src.ml.monitor import get_ml_model_monitor

        monitor = get_ml_model_monitor()
        original_confidence = validated.get("confidence", 0)

        # Calibrate confidence based on recent model performance
        calibrated_confidence = monitor.calibrate_confidence(
            confidence=original_confidence,
            model_version=validated.get("model", "unknown"),
        )

        if calibrated_confidence != original_confidence:
            validated["confidence"] = calibrated_confidence
            validated["raw_confidence"] = original_confidence
            validated["validation"]["adjustments"].append(
                f"Confidence calibrated: {original_confidence:.0f}% -> {calibrated_confidence:.0f}%"
            )

        # Log prediction for future validation
        if validated.get("signal") in ["BUY", "SELL"]:
            monitor.log_prediction(
                symbol=symbol,
                prediction=validated.get("ml_score", 0.5),
                predicted_class=1 if validated.get("signal") == "BUY" else 0,
                confidence=calibrated_confidence,
                model_version=validated.get("model", "unknown"),
            )

    except ImportError:
        validated["validation"]["warnings"].append("ML monitor not available for calibration")
    except Exception as e:
        validated["validation"]["warnings"].append(f"Calibration error: {str(e)}")

    # 2. Minimum confidence check
    if validated.get("confidence", 0) < min_confidence:
        validated["validation"]["passed"] = False
        validated["validation"]["warnings"].append(
            f"Confidence {validated.get('confidence', 0):.0f}% below minimum {min_confidence:.0f}%"
        )
        # Downgrade signal to HOLD if confidence too low
        if validated.get("signal") in ["BUY", "SELL"]:
            validated["original_signal"] = validated["signal"]
            validated["signal"] = "HOLD"
            validated["validation"]["adjustments"].append(
                "Signal downgraded to HOLD due to low confidence"
            )

    # 3. Technical confirmation check
    if require_technical_confirmation:
        signal = validated.get("signal", "HOLD")
        ema_trend = validated.get("ema_trend", "NEUTRAL")
        rsi = validated.get("rsi", 50)

        if signal == "BUY":
            # BUY signal should have bullish technicals
            if ema_trend == "DOWN":
                validated["validation"]["warnings"].append(
                    "BUY signal conflicts with EMA downtrend"
                )
                validated["confidence"] = max(0, validated.get("confidence", 0) - 10)
                validated["validation"]["adjustments"].append(
                    "Confidence reduced 10% due to EMA conflict"
                )

            if rsi > 70:
                validated["validation"]["warnings"].append(
                    f"BUY signal with overbought RSI ({rsi:.0f})"
                )
                validated["confidence"] = max(0, validated.get("confidence", 0) - 5)
                validated["validation"]["adjustments"].append(
                    "Confidence reduced 5% due to overbought RSI"
                )

        elif signal == "SELL":
            # SELL signal should have bearish technicals
            if ema_trend == "UP":
                validated["validation"]["warnings"].append("SELL signal conflicts with EMA uptrend")
                validated["confidence"] = max(0, validated.get("confidence", 0) - 10)
                validated["validation"]["adjustments"].append(
                    "Confidence reduced 10% due to EMA conflict"
                )

            if rsi < 30:
                validated["validation"]["warnings"].append(
                    f"SELL signal with oversold RSI ({rsi:.0f})"
                )
                validated["confidence"] = max(0, validated.get("confidence", 0) - 5)
                validated["validation"]["adjustments"].append(
                    "Confidence reduced 5% due to oversold RSI"
                )

    # 4. ML probability sanity check
    ml_score = validated.get("ml_score", 0.5)
    probabilities = validated.get("probabilities", {})

    if probabilities:
        buy_prob = probabilities.get("buy", 0.5)
        sell_prob = probabilities.get("sell", 0.5)

        # Check for extreme probabilities (model might be overfit)
        if buy_prob > 0.95 or sell_prob > 0.95:
            validated["validation"]["warnings"].append(
                "Extreme ML probability detected - possible overfit"
            )
            validated["confidence"] = max(0, validated.get("confidence", 0) - 15)
            validated["validation"]["adjustments"].append(
                "Confidence reduced 15% due to extreme probability"
            )

        # Check for probability mismatch with signal
        if validated.get("signal") == "BUY" and buy_prob < 0.5:
            validated["validation"]["warnings"].append("BUY signal with low buy probability")
            validated["validation"]["passed"] = False

    # 5. Model drift check
    try:
        from src.ml.monitor import get_ml_model_monitor

        monitor = get_ml_model_monitor()
        drift_status = monitor.check_drift()

        if drift_status.get("drift_detected"):
            validated["validation"]["warnings"].append(
                "Model drift detected - predictions may be unreliable"
            )
            validated["confidence"] = max(0, validated.get("confidence", 0) - 20)
            validated["validation"]["adjustments"].append(
                "Confidence reduced 20% due to model drift"
            )

    except Exception:
        pass

    # Update passed status based on warnings
    if len(validated["validation"]["warnings"]) >= 3:
        validated["validation"]["passed"] = False

    return validated


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
