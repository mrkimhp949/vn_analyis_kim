"""
Advanced Risk Management Module
Comprehensive risk management with VaR, drawdown limits, and dynamic adjustments

New Features:
1. Value at Risk (VaR) calculation
2. Conditional Value at Risk (CVaR)
3. Dynamic drawdown-based position reduction
4. Market regime-aware risk limits
5. Correlation-based risk assessment
6. Real-time portfolio heat monitoring
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics"""

    portfolio_var_1day: float  # 1-day Value at Risk (95% confidence)
    portfolio_cvar_1day: float  # 1-day Conditional VaR (Expected Shortfall)
    current_drawdown_pct: float  # Current drawdown from peak
    max_drawdown_pct: float  # Maximum historical drawdown
    portfolio_heat: float  # Portfolio heat index (0-100)
    correlation_risk: float  # Average portfolio correlation
    concentration_risk: float  # Concentration in top positions
    sector_concentration: Dict[str, float]  # Sector exposure breakdown
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    warnings: List[str]
    recommendations: List[str]


class AdvancedRiskManager:
    """
    Advanced risk management with dynamic adjustments

    Features:
    - VaR and CVaR calculation
    - Drawdown monitoring and limits
    - Position heat index
    - Correlation-based risk
    - Market regime integration
    """

    def __init__(
        self,
        total_capital: float = 100_000_000,
        max_var_percent: float = 0.03,  # Max 3% VaR per day
        max_drawdown_percent: float = 0.15,  # Max 15% drawdown
        critical_drawdown_percent: float = 0.10,  # Reduce positions at 10% DD
        max_portfolio_heat: float = 75.0,  # Max heat index
        max_correlation: float = 0.70,  # Max avg correlation
        max_concentration: float = 0.40,  # Max 40% in top 3 positions
    ):
        self.total_capital = total_capital
        self.max_var_percent = max_var_percent
        self.max_drawdown_percent = max_drawdown_percent
        self.critical_drawdown_percent = critical_drawdown_percent
        self.max_portfolio_heat = max_portfolio_heat
        self.max_correlation = max_correlation
        self.max_concentration = max_concentration

        # Tracking
        self.portfolio_peak_value = total_capital
        self.risk_history = []  # Historical risk metrics
        self._var_cache = {}  # Cache VaR calculations

    def calculate_portfolio_risk(
        self,
        positions: Dict,
        current_prices: Dict[str, float],
        historical_returns: Optional[Dict[str, pd.Series]] = None,
        market_regime: Optional[Dict] = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics

        Args:
            positions: Dict of positions {symbol: position_data}
            current_prices: Current prices for each symbol
            historical_returns: Historical return series for VaR calc
            market_regime: Current market regime

        Returns:
            RiskMetrics with full risk assessment
        """
        warnings = []
        recommendations = []

        if not positions:
            return self._zero_risk_metrics()

        # 1. Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value(positions, current_prices)

        # 2. Calculate VaR and CVaR
        var_1day, cvar_1day = self._calculate_var_cvar(
            positions, current_prices, historical_returns
        )

        # 3. Calculate drawdown
        current_dd, max_dd = self._calculate_drawdown(portfolio_value)

        # 4. Calculate portfolio heat
        heat = self._calculate_portfolio_heat(positions, current_prices, market_regime)

        # 5. Calculate correlation risk
        corr_risk = self._calculate_correlation_risk(positions)

        # 6. Calculate concentration risk
        concentration, sector_concentration = self._calculate_concentration(
            positions, current_prices
        )

        # 7. Determine risk level and generate warnings
        risk_level = self._determine_risk_level(
            var_1day, current_dd, heat, corr_risk, concentration
        )

        # Generate warnings based on risk metrics
        if var_1day > self.max_var_percent:
            warnings.append(
                f"⚠️ VaR ({var_1day:.1%}) exceeds limit ({self.max_var_percent:.1%})"
            )
            recommendations.append("Reduce position sizes or hedge portfolio")

        if current_dd > self.critical_drawdown_percent:
            warnings.append(
                f"🚨 Drawdown ({current_dd:.1%}) reached critical level ({self.critical_drawdown_percent:.1%})"
            )
            recommendations.append("CRITICAL: Reduce positions by 50% immediately")
        elif current_dd > self.max_drawdown_percent * 0.75:
            warnings.append(
                f"⚠️ Drawdown ({current_dd:.1%}) approaching limit ({self.max_drawdown_percent:.1%})"
            )
            recommendations.append("Consider reducing position sizes by 25%")

        if heat > self.max_portfolio_heat:
            warnings.append(
                f"🔥 Portfolio heat ({heat:.0f}) exceeds safe limit ({self.max_portfolio_heat:.0f})"
            )
            recommendations.append("Portfolio overheated - pause new entries")

        if corr_risk > self.max_correlation:
            warnings.append(
                f"⚠️ Portfolio correlation ({corr_risk:.1%}) too high (limit: {self.max_correlation:.1%})"
            )
            recommendations.append("Diversify - positions are too correlated")

        if concentration > self.max_concentration:
            warnings.append(
                f"⚠️ Position concentration ({concentration:.1%}) exceeds limit ({self.max_concentration:.1%})"
            )
            recommendations.append("Rebalance - reduce oversized positions")

        # Market regime adjustments
        if market_regime:
            regime_warnings, regime_recs = self._get_regime_risk_adjustments(
                market_regime, risk_level
            )
            warnings.extend(regime_warnings)
            recommendations.extend(regime_recs)

        return RiskMetrics(
            portfolio_var_1day=var_1day,
            portfolio_cvar_1day=cvar_1day,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=max_dd,
            portfolio_heat=heat,
            correlation_risk=corr_risk,
            concentration_risk=concentration,
            sector_concentration=sector_concentration,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_position_size_adjustment(
        self, risk_metrics: RiskMetrics, base_multiplier: float = 1.0
    ) -> Tuple[float, str]:
        """
        Get position size adjustment based on current risk levels

        Args:
            risk_metrics: Current portfolio risk metrics
            base_multiplier: Base position multiplier

        Returns:
            Tuple of (adjusted_multiplier, reason)
        """
        adjustments = []
        multiplier = base_multiplier

        # Adjust for drawdown
        if risk_metrics.current_drawdown_pct > self.critical_drawdown_percent:
            # Critical drawdown - reduce 70%
            multiplier *= 0.3
            adjustments.append(f"Critical DD ({risk_metrics.current_drawdown_pct:.1%})")
        elif risk_metrics.current_drawdown_pct > self.max_drawdown_percent * 0.75:
            # High drawdown - reduce 40%
            multiplier *= 0.6
            adjustments.append(f"High DD ({risk_metrics.current_drawdown_pct:.1%})")
        elif risk_metrics.current_drawdown_pct > self.max_drawdown_percent * 0.5:
            # Moderate drawdown - reduce 20%
            multiplier *= 0.8
            adjustments.append(f"Moderate DD ({risk_metrics.current_drawdown_pct:.1%})")

        # Adjust for portfolio heat
        if risk_metrics.portfolio_heat > self.max_portfolio_heat:
            # Portfolio too hot - reduce 50%
            multiplier *= 0.5
            adjustments.append(f"High heat ({risk_metrics.portfolio_heat:.0f})")
        elif risk_metrics.portfolio_heat > self.max_portfolio_heat * 0.8:
            # Portfolio warming up - reduce 25%
            multiplier *= 0.75
            adjustments.append(f"Moderate heat ({risk_metrics.portfolio_heat:.0f})")

        # Adjust for correlation risk
        if risk_metrics.correlation_risk > self.max_correlation:
            # Too correlated - reduce 30%
            multiplier *= 0.7
            adjustments.append(f"High correlation ({risk_metrics.correlation_risk:.1%})")

        # Adjust for VaR
        if risk_metrics.portfolio_var_1day > self.max_var_percent:
            # VaR too high - reduce 40%
            multiplier *= 0.6
            adjustments.append(f"High VaR ({risk_metrics.portfolio_var_1day:.1%})")

        # Overall risk level override
        if risk_metrics.risk_level == "CRITICAL":
            # Critical risk - max 20% of normal size
            multiplier = min(multiplier, 0.2)
            adjustments.append("CRITICAL RISK LEVEL")
        elif risk_metrics.risk_level == "HIGH":
            # High risk - max 50% of normal size
            multiplier = min(multiplier, 0.5)
            adjustments.append("HIGH RISK LEVEL")
        elif risk_metrics.risk_level == "MEDIUM":
            # Medium risk - max 75% of normal size
            multiplier = min(multiplier, 0.75)

        reason = " | ".join(adjustments) if adjustments else "Normal risk level"
        return multiplier, reason

    def should_close_position(
        self, symbol: str, position: Dict, risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be closed due to risk limits

        Args:
            symbol: Stock symbol
            position: Position data
            risk_metrics: Current risk metrics

        Returns:
            Tuple of (should_close, reason)
        """
        # Critical drawdown - close losing positions
        if risk_metrics.current_drawdown_pct > self.critical_drawdown_percent:
            pnl_pct = self._calculate_position_pnl_pct(position)
            if pnl_pct < -0.03:  # Losing more than 3%
                return (
                    True,
                    f"Critical drawdown ({risk_metrics.current_drawdown_pct:.1%}) - closing losing position",
                )

        # Critical risk level - close all positions gradually
        if risk_metrics.risk_level == "CRITICAL":
            return True, "CRITICAL risk level - emergency exit"

        # Very high heat - close weakest positions
        if risk_metrics.portfolio_heat > self.max_portfolio_heat * 1.2:
            pnl_pct = self._calculate_position_pnl_pct(position)
            if pnl_pct < 0:  # Any losing position
                return (
                    True,
                    f"Extreme heat ({risk_metrics.portfolio_heat:.0f}) - closing losing position",
                )

        return False, ""

    def _calculate_var_cvar(
        self,
        positions: Dict,
        current_prices: Dict[str, float],
        historical_returns: Optional[Dict[str, pd.Series]] = None,
        confidence: float = 0.95,
        lookback_days: int = 60,
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR

        Uses historical simulation method with actual returns

        Returns:
            Tuple of (VaR, CVaR) as percentage of portfolio value
        """
        if not historical_returns or len(positions) == 0:
            # Fallback: Use simple volatility-based estimation
            return self._estimate_var_simple(positions, current_prices)

        try:
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(positions, current_prices)

            if portfolio_value == 0:
                return 0.0, 0.0

            # Get portfolio weights
            weights = {}
            for symbol, pos in positions.items():
                if symbol in current_prices:
                    position_value = pos["shares"] * current_prices[symbol]
                    weights[symbol] = position_value / portfolio_value

            # Calculate portfolio returns for each historical period
            portfolio_returns = []
            max_length = min(
                [len(historical_returns.get(symbol, [])) for symbol in weights.keys()]
                + [lookback_days]
            )

            if max_length < 10:
                # Not enough historical data
                return self._estimate_var_simple(positions, current_prices)

            for i in range(max_length):
                period_return = 0
                for symbol, weight in weights.items():
                    if symbol in historical_returns and i < len(historical_returns[symbol]):
                        period_return += weight * historical_returns[symbol].iloc[i]
                portfolio_returns.append(period_return)

            # Sort returns
            portfolio_returns = sorted(portfolio_returns)

            # Calculate VaR (95% confidence = 5th percentile)
            var_index = int(len(portfolio_returns) * (1 - confidence))
            var_return = portfolio_returns[var_index] if var_index < len(portfolio_returns) else 0

            # Calculate CVaR (average of returns worse than VaR)
            cvar_returns = portfolio_returns[:var_index] if var_index > 0 else [var_return]
            cvar_return = np.mean(cvar_returns) if cvar_returns else 0

            # Convert to percentage (take absolute value for loss)
            var_pct = abs(var_return)
            cvar_pct = abs(cvar_return)

            logger.debug(
                f"📊 VaR/CVaR calculated: VaR={var_pct:.2%}, CVaR={cvar_pct:.2%} "
                f"(based on {len(portfolio_returns)} periods)"
            )

            return var_pct, cvar_pct

        except Exception as e:
            logger.warning(f"⚠️ VaR calculation failed: {e}, using simple estimate")
            return self._estimate_var_simple(positions, current_prices)

    def _estimate_var_simple(
        self, positions: Dict, current_prices: Dict[str, float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Simple VaR estimation using average position volatility

        Fallback when historical returns not available

        Assumes normal distribution: VaR = z-score * volatility
        For 95% confidence: z = 1.645
        For 99% confidence: z = 2.326

        Typical Vietnam stock volatility: 2-3% per day
        """
        # Assume typical daily volatility of 2.5%
        assumed_daily_vol = 0.025

        # Z-score for 95% confidence
        z_score = 1.645 if confidence == 0.95 else 2.326

        # VaR = volatility * z-score
        var_pct = assumed_daily_vol * z_score

        # CVaR approximately 1.25x VaR for normal distribution
        cvar_pct = var_pct * 1.25

        logger.debug(
            f"📊 Simple VaR estimate: VaR={var_pct:.2%}, CVaR={cvar_pct:.2%} "
            f"(assuming {assumed_daily_vol:.2%} daily volatility)"
        )

        return var_pct, cvar_pct

    def _calculate_drawdown(self, current_value: float) -> Tuple[float, float]:
        """Calculate current and maximum drawdown"""
        # Update peak if current value is higher
        if current_value > self.portfolio_peak_value:
            self.portfolio_peak_value = current_value

        # Current drawdown from peak
        current_dd = (
            (self.portfolio_peak_value - current_value) / self.portfolio_peak_value
            if self.portfolio_peak_value > 0
            else 0
        )

        # Track maximum drawdown (historical worst)
        max_dd = max([current_dd] + [m.current_drawdown_pct for m in self.risk_history])

        return current_dd, max_dd

    def _calculate_portfolio_heat(
        self,
        positions: Dict,
        current_prices: Dict[str, float],
        market_regime: Optional[Dict] = None,
    ) -> float:
        """
        Calculate portfolio heat index (0-100)

        Heat increases with:
        - Number of positions
        - Position size
        - Market volatility
        - Correlation
        """
        if not positions:
            return 0.0

        heat = 0.0

        # Base heat from number of positions
        # 1 position = 10 heat, 10 positions = 100 heat
        num_positions = len(positions)
        heat += num_positions * 10

        # Heat from position concentration
        portfolio_value = self._calculate_portfolio_value(positions, current_prices)
        if portfolio_value > 0:
            position_sizes = []
            for symbol, pos in positions.items():
                if symbol in current_prices:
                    position_value = pos["shares"] * current_prices[symbol]
                    position_pct = position_value / portfolio_value
                    position_sizes.append(position_pct)

                    # Large positions add heat
                    if position_pct > 0.15:  # >15% position
                        heat += 20
                    elif position_pct > 0.10:  # >10% position
                        heat += 10

        # Heat from market regime
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "HIGH_VOLATILITY":
                heat += 20
            elif regime == "BEAR":
                heat += 15
            elif regime == "BULL":
                heat -= 10  # Reduce heat in bull market

        # Clamp to 0-100
        return max(0, min(heat, 100))

    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """
        Calculate average portfolio correlation

        Higher correlation = higher risk (all positions move together)
        """
        if len(positions) < 2:
            return 0.0

        # In real implementation, calculate actual correlation matrix
        # For now, estimate based on sector diversity

        sectors = {}
        for symbol, pos in positions.items():
            sector = pos.get("metadata", {}).get("sector", "UNKNOWN")
            sectors[sector] = sectors.get(sector, 0) + 1

        # More sectors = lower correlation
        num_sectors = len(sectors)
        if num_sectors >= 5:
            return 0.3  # Low correlation
        elif num_sectors >= 3:
            return 0.5  # Medium correlation
        else:
            return 0.7  # High correlation

    def _calculate_concentration(
        self, positions: Dict, current_prices: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate position concentration risk

        Returns:
            Tuple of (top3_concentration, sector_breakdown)
        """
        if not positions:
            return 0.0, {}

        portfolio_value = self._calculate_portfolio_value(positions, current_prices)
        if portfolio_value == 0:
            return 0.0, {}

        # Calculate position values
        position_values = []
        sector_values = {}

        for symbol, pos in positions.items():
            if symbol in current_prices:
                value = pos["shares"] * current_prices[symbol]
                position_values.append(value)

                sector = pos.get("metadata", {}).get("sector", "UNKNOWN")
                sector_values[sector] = sector_values.get(sector, 0) + value

        # Top 3 concentration
        position_values.sort(reverse=True)
        top3_value = sum(position_values[:3])
        top3_concentration = top3_value / portfolio_value if portfolio_value > 0 else 0

        # Sector concentration
        sector_concentration = {
            sector: (value / portfolio_value)
            for sector, value in sector_values.items()
            if portfolio_value > 0
        }

        return top3_concentration, sector_concentration

    def _determine_risk_level(
        self,
        var: float,
        drawdown: float,
        heat: float,
        correlation: float,
        concentration: float,
    ) -> str:
        """Determine overall risk level"""
        critical_count = 0
        high_count = 0

        # Check each metric
        if var > self.max_var_percent * 1.2:
            critical_count += 1
        elif var > self.max_var_percent:
            high_count += 1

        if drawdown > self.critical_drawdown_percent:
            critical_count += 1
        elif drawdown > self.max_drawdown_percent * 0.75:
            high_count += 1

        if heat > self.max_portfolio_heat * 1.2:
            critical_count += 1
        elif heat > self.max_portfolio_heat:
            high_count += 1

        if correlation > self.max_correlation * 1.1:
            high_count += 1

        if concentration > self.max_concentration * 1.2:
            high_count += 1

        # Determine level
        if critical_count >= 2:
            return "CRITICAL"
        elif critical_count >= 1 or high_count >= 3:
            return "HIGH"
        elif high_count >= 1:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_regime_risk_adjustments(
        self, market_regime: Dict, current_risk_level: str
    ) -> Tuple[List[str], List[str]]:
        """Get risk warnings and recommendations based on market regime"""
        warnings = []
        recommendations = []

        regime = market_regime.get("regime", "SIDEWAYS")
        confidence = market_regime.get("confidence", 50)

        if regime == "BEAR" and confidence >= 70:
            warnings.append(f"🐻 Strong bear market detected ({confidence:.0f}% confidence)")
            recommendations.append("Reduce exposure - bear market requires defensive positioning")

        elif regime == "HIGH_VOLATILITY":
            warnings.append("⚡ High volatility detected")
            recommendations.append("Tighten stop losses and reduce position sizes")

        elif regime == "BULL" and current_risk_level in ["HIGH", "CRITICAL"]:
            warnings.append("⚠️ Bull market but portfolio risk is elevated")
            recommendations.append("Rebalance - lock in profits on winning positions")

        return warnings, recommendations

    def _calculate_portfolio_value(
        self, positions: Dict, current_prices: Dict[str, float]
    ) -> float:
        """Calculate total portfolio value"""
        total = 0.0
        for symbol, pos in positions.items():
            if symbol in current_prices:
                total += pos["shares"] * current_prices[symbol]
        return total

    def _calculate_position_pnl_pct(self, position: Dict) -> float:
        """Calculate position P&L percentage"""
        entry_value = position["entry_value"]
        metadata = position.get("metadata", {})
        current_price = metadata.get("last_price", position["avg_price"])
        shares = position["shares"]

        current_value = shares * current_price
        pnl = current_value - entry_value
        pnl_pct = pnl / entry_value if entry_value > 0 else 0

        return pnl_pct

    def _zero_risk_metrics(self) -> RiskMetrics:
        """Return zero risk metrics for empty portfolio"""
        return RiskMetrics(
            portfolio_var_1day=0.0,
            portfolio_cvar_1day=0.0,
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            portfolio_heat=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            sector_concentration={},
            risk_level="LOW",
            warnings=[],
            recommendations=[],
        )

    def format_risk_report(self, risk_metrics: RiskMetrics) -> str:
        """Format comprehensive risk report"""
        lines = []
        lines.append("🛡️ *ADVANCED RISK ANALYSIS*")
        lines.append("=" * 50)

        # Risk Level
        emoji = {
            "LOW": "✅",
            "MEDIUM": "⚠️",
            "HIGH": "🚨",
            "CRITICAL": "🚨🚨🚨",
        }
        lines.append(
            f"\n{emoji.get(risk_metrics.risk_level, '⚠️')} *Risk Level:* "
            f"{risk_metrics.risk_level}"
        )

        # Key Metrics
        lines.append("\n📊 *Risk Metrics:*")
        lines.append(
            f"• VaR (1-day, 95%): {risk_metrics.portfolio_var_1day:.2%} of portfolio"
        )
        lines.append(f"• CVaR (Expected Shortfall): {risk_metrics.portfolio_cvar_1day:.2%}")
        lines.append(f"• Current Drawdown: {risk_metrics.current_drawdown_pct:.2%}")
        lines.append(f"• Max Drawdown: {risk_metrics.max_drawdown_pct:.2%}")
        lines.append(f"• Portfolio Heat: {risk_metrics.portfolio_heat:.0f}/100")
        lines.append(f"• Correlation Risk: {risk_metrics.correlation_risk:.1%}")
        lines.append(f"• Concentration (Top 3): {risk_metrics.concentration_risk:.1%}")

        # Sector Breakdown
        if risk_metrics.sector_concentration:
            lines.append("\n🏭 *Sector Exposure:*")
            for sector, pct in sorted(
                risk_metrics.sector_concentration.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"• {sector}: {pct:.1%}")

        # Warnings
        if risk_metrics.warnings:
            lines.append("\n⚠️ *WARNINGS:*")
            for warning in risk_metrics.warnings:
                lines.append(f"• {warning}")

        # Recommendations
        if risk_metrics.recommendations:
            lines.append("\n💡 *RECOMMENDATIONS:*")
            for rec in risk_metrics.recommendations:
                lines.append(f"• {rec}")

        return "\n".join(lines)


# Factory function
def get_advanced_risk_manager() -> AdvancedRiskManager:
    """Get advanced risk manager singleton"""
    global _risk_manager
    if "_risk_manager" not in globals():
        _risk_manager = AdvancedRiskManager()
    return _risk_manager


# Test
if __name__ == "__main__":
    print("Testing Advanced Risk Manager...")

    manager = AdvancedRiskManager()

    # Simulate positions
    positions = {
        "VCB": {
            "shares": 100,
            "avg_price": 90000,
            "entry_value": 9000000,
            "metadata": {"last_price": 92000, "sector": "BANKING"},
        },
        "VNM": {
            "shares": 200,
            "avg_price": 80000,
            "entry_value": 16000000,
            "metadata": {"last_price": 82000, "sector": "CONSUMER"},
        },
    }

    current_prices = {"VCB": 92000, "VNM": 82000}

    risk_metrics = manager.calculate_portfolio_risk(positions, current_prices)

    print(manager.format_risk_report(risk_metrics))
    print("\n✅ Advanced Risk Manager test completed!")
