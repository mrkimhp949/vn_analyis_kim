"""
Signal Quality Scoring System
Evaluates and scores trading signals based on multiple criteria
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Container for signal quality assessment"""

    total_score: float  # 0-100
    grade: str  # A+, A, B, C, D, F
    components: Dict[str, float]  # Individual component scores
    warnings: list  # Quality warnings
    recommendations: list  # Recommendations for improvement
    is_tradeable: bool  # Whether signal meets minimum quality


class SignalQualityScorer:
    """
    Comprehensive signal quality scoring system

    Evaluates signals based on:
    1. ML Confidence (25 points)
    2. Technical Confirmation (20 points)
    3. Volume Confirmation (15 points)
    4. Market Regime Alignment (15 points)
    5. Timing Quality (10 points)
    6. Risk/Reward Ratio (10 points)
    7. Historical Accuracy (5 points)
    """

    def __init__(self, min_tradeable_score: float = 60.0):
        """
        Initialize quality scorer

        Args:
            min_tradeable_score: Minimum score to consider signal tradeable (default: 60)
        """
        self.min_tradeable_score = min_tradeable_score

        # Component weights (must sum to 100)
        self.weights = {
            "ml_confidence": 25,
            "technical_confirmation": 20,
            "volume_confirmation": 15,
            "market_regime": 15,
            "timing": 10,
            "risk_reward": 10,
            "historical_accuracy": 5,
        }

    def score_signal(self, signal: dict) -> QualityScore:
        """
        Evaluate and score a trading signal

        Args:
            signal: Signal dictionary with all metadata

        Returns:
            QualityScore with detailed assessment
        """
        components = {}
        warnings = []
        recommendations = []

        # =================================================================
        # 1. ML CONFIDENCE (25 points)
        # =================================================================
        ml_score = signal.get("ml_score", 0.5)
        confidence = signal.get("confidence", 0) / 100.0  # Normalize to 0-1

        # Score based on confidence level
        if confidence >= 0.8:
            ml_points = 25
        elif confidence >= 0.7:
            ml_points = 20
        elif confidence >= 0.6:
            ml_points = 15
        elif confidence >= 0.5:
            ml_points = 10
            warnings.append("⚠️ Low ML confidence")
        else:
            ml_points = 5
            warnings.append("❌ Very low ML confidence")
            recommendations.append("Wait for higher confidence signal")

        components["ml_confidence"] = ml_points

        # =================================================================
        # 2. TECHNICAL CONFIRMATION (20 points)
        # =================================================================
        tech_score = signal.get("technical_score", {})

        tech_points = 0

        # Trend alignment
        if isinstance(tech_score, dict):
            trend = tech_score.get("trend", 0)
            if abs(trend) > 0.5:
                tech_points += 10
            elif abs(trend) > 0.2:
                tech_points += 6
            else:
                tech_points += 3
                warnings.append("⚠️ Weak trend")

            # Momentum
            momentum = tech_score.get("momentum", 0)
            if abs(momentum) > 0.5:
                tech_points += 7
            elif abs(momentum) > 0.2:
                tech_points += 4
            else:
                tech_points += 2

            # Volatility (optimal = positive score)
            volatility = tech_score.get("volatility", 0)
            if volatility > 0.3:
                tech_points += 3
            else:
                tech_points += 1
        else:
            tech_points = 10  # Default if tech_score not available

        components["technical_confirmation"] = tech_points

        # =================================================================
        # 3. VOLUME CONFIRMATION (15 points)
        # =================================================================
        volume_ratio = signal.get("volume_ratio", 1.0)

        if volume_ratio >= 1.5:
            volume_points = 15
        elif volume_ratio >= 1.2:
            volume_points = 12
        elif volume_ratio >= 1.0:
            volume_points = 8
        elif volume_ratio >= 0.8:
            volume_points = 5
            warnings.append("⚠️ Low volume")
        else:
            volume_points = 2
            warnings.append("❌ Very low volume")
            recommendations.append("Avoid low-volume signals")

        components["volume_confirmation"] = volume_points

        # =================================================================
        # 4. MARKET REGIME ALIGNMENT (15 points)
        # =================================================================
        # Check if market regime info is in reason
        reason = signal.get("reason", "")

        if "Bull" in reason:
            regime_points = 15 if signal.get("action") == "BUY" else 8
        elif "Bear" in reason:
            regime_points = 5 if signal.get("action") == "BUY" else 12
            if signal.get("action") == "BUY":
                warnings.append("⚠️ Buying in bear market")
        elif "Sideways" in reason:
            regime_points = 10
        elif "High Vol" in reason:
            regime_points = 5
            warnings.append("⚠️ High volatility market")
        else:
            regime_points = 8  # Default if no regime info

        components["market_regime"] = regime_points

        # =================================================================
        # 5. TIMING QUALITY (10 points)
        # =================================================================
        timing_info = signal.get("timing", {})

        if isinstance(timing_info, dict):
            timing_score = timing_info.get("timing_score", 0.5)

            if timing_score >= 0.9:
                timing_points = 10
            elif timing_score >= 0.7:
                timing_points = 7
            elif timing_score >= 0.5:
                timing_points = 5
            else:
                timing_points = 3
                warnings.append("⚠️ Suboptimal timing")
                time_until = timing_info.get("time_until_best")
                if time_until:
                    recommendations.append(f"Wait {time_until} for better timing")
        else:
            timing_points = 5  # Default

        components["timing"] = timing_points

        # =================================================================
        # 6. RISK/REWARD RATIO (10 points)
        # =================================================================
        entry_price = signal.get("entry_price", 0)
        stop_loss = signal.get("stop_loss", 0)
        take_profit = signal.get("take_profit", 0)

        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)

            if risk > 0:
                rr_ratio = reward / risk

                if rr_ratio >= 3.0:
                    rr_points = 10
                elif rr_ratio >= 2.0:
                    rr_points = 8
                elif rr_ratio >= 1.5:
                    rr_points = 6
                elif rr_ratio >= 1.0:
                    rr_points = 4
                    warnings.append(f"⚠️ Low R:R ratio ({rr_ratio:.1f}:1)")
                else:
                    rr_points = 2
                    warnings.append(f"❌ Poor R:R ratio ({rr_ratio:.1f}:1)")
                    recommendations.append("Adjust SL/TP for better risk/reward")
            else:
                rr_points = 5  # Default
        else:
            rr_points = 5  # Default if SL/TP not set

        components["risk_reward"] = rr_points

        # =================================================================
        # 7. HISTORICAL ACCURACY (5 points)
        # =================================================================
        # This would require tracking past signal performance
        # For now, use a placeholder
        raw_confidence = signal.get("raw_confidence", signal.get("confidence", 50)) / 100.0

        if raw_confidence >= 0.7:
            hist_points = 5
        elif raw_confidence >= 0.6:
            hist_points = 4
        else:
            hist_points = 3

        components["historical_accuracy"] = hist_points

        # =================================================================
        # CALCULATE TOTAL SCORE
        # =================================================================
        total_score = sum(components.values())

        # Grade assignment
        if total_score >= 90:
            grade = "A+"
        elif total_score >= 85:
            grade = "A"
        elif total_score >= 80:
            grade = "A-"
        elif total_score >= 75:
            grade = "B+"
        elif total_score >= 70:
            grade = "B"
        elif total_score >= 65:
            grade = "B-"
        elif total_score >= 60:
            grade = "C+"
        elif total_score >= 55:
            grade = "C"
        else:
            grade = "D"

        # Tradeable check
        is_tradeable = total_score >= self.min_tradeable_score

        if not is_tradeable:
            warnings.append(
                f"❌ Signal below tradeable threshold ({total_score:.0f} < {self.min_tradeable_score:.0f})"
            )
            recommendations.append("Skip this signal and wait for better opportunity")

        return QualityScore(
            total_score=total_score,
            grade=grade,
            components=components,
            warnings=warnings,
            recommendations=recommendations,
            is_tradeable=is_tradeable,
        )

    def add_quality_score_to_signal(self, signal: dict) -> dict:
        """
        Add quality score to signal dictionary

        Args:
            signal: Signal dict

        Returns:
            Updated signal with quality_score field
        """
        quality = self.score_signal(signal)

        signal["quality_score"] = {
            "total": quality.total_score,
            "grade": quality.grade,
            "components": quality.components,
            "warnings": quality.warnings,
            "recommendations": quality.recommendations,
            "is_tradeable": quality.is_tradeable,
        }

        logger.info(
            f"Quality score for {signal.get('symbol', 'UNKNOWN')}: "
            f"{quality.total_score:.0f}/100 (Grade: {quality.grade})"
        )

        return signal


# Singleton instance
_quality_scorer = None


def get_quality_scorer() -> SignalQualityScorer:
    """Get quality scorer singleton"""
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = SignalQualityScorer()
    return _quality_scorer


# Convenience function
def add_quality_score(signal: dict) -> dict:
    """Add quality score to a signal"""
    scorer = get_quality_scorer()
    return scorer.add_quality_score_to_signal(signal)


# Testing
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎯 TESTING SIGNAL QUALITY SCORER")
    print("=" * 70 + "\n")

    scorer = SignalQualityScorer()

    # Test signal 1: High quality
    high_quality_signal = {
        "symbol": "VNM",
        "action": "BUY",
        "confidence": 85,
        "ml_score": 0.75,
        "technical_score": {"trend": 0.6, "momentum": 0.5, "volatility": 0.4},
        "volume_ratio": 1.6,
        "reason": "Bull Market | High Volume",
        "entry_price": 100000,
        "stop_loss": 95000,
        "take_profit": 112000,
        "timing": {"timing_score": 0.95, "is_good_timing": True},
    }

    quality = scorer.score_signal(high_quality_signal)

    print(f"Signal: {high_quality_signal['symbol']}")
    print(f"Total Score: {quality.total_score:.0f}/100")
    print(f"Grade: {quality.grade}")
    print(f"Tradeable: {quality.is_tradeable}")
    print(f"\nComponent Scores:")
    for component, score in quality.components.items():
        print(f"  {component:25s}: {score:4.0f}")

    if quality.warnings:
        print(f"\nWarnings:")
        for warning in quality.warnings:
            print(f"  {warning}")

    if quality.recommendations:
        print(f"\nRecommendations:")
        for rec in quality.recommendations:
            print(f"  • {rec}")

    print("\n" + "=" * 70)
    print("✅ Quality scoring complete!")
    print("=" * 70 + "\n")
