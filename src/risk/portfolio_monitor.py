"""
Real-time Portfolio Risk Monitor
Tracks portfolio risk in real-time and provides alerts
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics"""

    total_exposure: float  # Total position value
    total_exposure_pct: float  # As % of capital
    total_risk: float  # Total at-risk amount
    total_risk_pct: float  # As % of capital
    position_count: int
    largest_position_pct: float
    sector_exposures: Dict[str, float]  # Sector -> exposure %
    correlation_risk: float  # Correlation-adjusted risk
    max_drawdown: float  # Current drawdown %
    alerts: List[str]  # Active alerts
    timestamp: datetime


@dataclass
class PositionRisk:
    """Risk data for a single position"""

    symbol: str
    entry_price: float
    current_price: float
    shares: int
    stop_loss: float
    position_value: float
    position_pct: float
    risk_amount: float
    risk_pct: float
    pnl: float
    pnl_pct: float
    sector: Optional[str] = None


class PortfolioRiskMonitor:
    """
    Real-time Portfolio Risk Monitor

    Features:
    1. Track total exposure and risk
    2. Monitor position concentration
    3. Track sector exposure
    4. Calculate correlation-adjusted risk
    5. Generate real-time alerts
    6. Track max drawdown
    """

    def __init__(
        self,
        total_capital: float,
        max_total_exposure: float = 0.60,  # 60%
        max_portfolio_risk: float = 0.20,  # 20%
        max_position_size: float = 0.10,  # 10% (FIXED: was 0.15 which exceeded 100% with 10 positions)
        max_sector_exposure: float = 0.40,  # 40%
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            total_capital: Total capital
            max_total_exposure: Max total exposure (% of capital)
            max_portfolio_risk: Max portfolio risk (% of capital)
            max_position_size: Max single position (% of capital)
            max_sector_exposure: Max sector exposure (% of capital)
            alert_thresholds: Custom alert thresholds
        """
        self.total_capital = total_capital
        self.max_total_exposure = max_total_exposure
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "exposure_warning": 0.50,  # 50% exposure
            "exposure_critical": 0.55,  # 55% exposure
            "risk_warning": 0.15,  # 15% portfolio risk
            "risk_critical": 0.18,  # 18% portfolio risk
            "position_warning": 0.12,  # 12% single position
            "sector_warning": 0.35,  # 35% sector exposure
        }

        # State tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.peak_capital = total_capital
        self.metrics_history: List[RiskMetrics] = []

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        sector: Optional[str] = None,
    ):
        """Add a new position to monitoring"""
        position_value = shares * entry_price
        position_pct = (position_value / self.total_capital) * 100
        risk_amount = shares * abs(entry_price - stop_loss)
        risk_pct = (risk_amount / self.total_capital) * 100

        position = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            pnl=0.0,
            pnl_pct=0.0,
            sector=sector,
        )

        self.positions[symbol] = position
        logger.info(
            f"📊 Added position: {symbol} - {shares} shares @ {entry_price:,.0f} VND "
            f"(Value: {position_pct:.2f}%, Risk: {risk_pct:.2f}%)"
        )

        # Update metrics and check alerts
        self.calculate_metrics()

    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol not in self.positions:
            logger.warning(f"⚠️ Position {symbol} not found in monitor")
            return

        position = self.positions[symbol]
        position.current_price = current_price
        position.position_value = position.shares * current_price
        position.position_pct = (position.position_value / self.total_capital) * 100
        position.pnl = (current_price - position.entry_price) * position.shares
        position.pnl_pct = ((current_price / position.entry_price) - 1) * 100

        # Update metrics
        self.calculate_metrics()

    def remove_position(self, symbol: str, exit_price: float, reason: str = "CLOSED"):
        """Remove position from monitoring"""
        if symbol not in self.positions:
            logger.warning(f"⚠️ Position {symbol} not found in monitor")
            return

        position = self.positions[symbol]
        final_pnl = (exit_price - position.entry_price) * position.shares
        final_pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        logger.info(
            f"📊 Removed position: {symbol} - {reason} @ {exit_price:,.0f} VND "
            f"(PnL: {final_pnl:,.0f} VND, {final_pnl_pct:+.2f}%)"
        )

        del self.positions[symbol]

        # Update capital if profitable
        if final_pnl > 0:
            self.total_capital += final_pnl
            self.peak_capital = max(self.peak_capital, self.total_capital)

        # Update metrics
        self.calculate_metrics()

    def calculate_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        if not self.positions:
            # No positions - return zero metrics
            return RiskMetrics(
                total_exposure=0.0,
                total_exposure_pct=0.0,
                total_risk=0.0,
                total_risk_pct=0.0,
                position_count=0,
                largest_position_pct=0.0,
                sector_exposures={},
                correlation_risk=0.0,
                max_drawdown=0.0,
                alerts=[],
                timestamp=datetime.now(),
            )

        # Calculate aggregates
        total_exposure = sum(p.position_value for p in self.positions.values())
        total_exposure_pct = (total_exposure / self.total_capital) * 100

        total_risk = sum(p.risk_amount for p in self.positions.values())
        total_risk_pct = (total_risk / self.total_capital) * 100

        position_count = len(self.positions)

        largest_position_pct = max(p.position_pct for p in self.positions.values())

        # Sector exposures
        sector_exposures = {}
        for position in self.positions.values():
            if position.sector:
                sector_exposures[position.sector] = (
                    sector_exposures.get(position.sector, 0) + position.position_value
                )

        # Convert to percentages
        sector_exposures = {
            sector: (value / self.total_capital) * 100 for sector, value in sector_exposures.items()
        }

        # Correlation risk (simplified - assume 50% correlation)
        # TODO: Use real correlation matrix
        correlation_adjustment = 0.7  # Reduce by 30% due to correlation
        correlation_risk = total_risk * correlation_adjustment

        # Max drawdown
        current_equity = self.total_capital + sum(p.pnl for p in self.positions.values())
        max_drawdown = ((self.peak_capital - current_equity) / self.peak_capital) * 100

        # Generate alerts
        alerts = self._generate_alerts(
            total_exposure_pct, total_risk_pct, largest_position_pct, sector_exposures
        )

        metrics = RiskMetrics(
            total_exposure=total_exposure,
            total_exposure_pct=total_exposure_pct,
            total_risk=total_risk,
            total_risk_pct=total_risk_pct,
            position_count=position_count,
            largest_position_pct=largest_position_pct,
            sector_exposures=sector_exposures,
            correlation_risk=correlation_risk,
            max_drawdown=max_drawdown,
            alerts=alerts,
            timestamp=datetime.now(),
        )

        # Store in history
        self.metrics_history.append(metrics)

        # Log alerts if any
        if alerts:
            for alert in alerts:
                logger.warning(f"⚠️ RISK ALERT: {alert}")

        return metrics

    def _generate_alerts(
        self,
        exposure_pct: float,
        risk_pct: float,
        largest_position_pct: float,
        sector_exposures: Dict[str, float],
    ) -> List[str]:
        """Generate risk alerts based on thresholds"""
        alerts = []

        # Exposure alerts
        if exposure_pct >= self.max_total_exposure * 100:
            alerts.append(
                f"🚨 CRITICAL: Total exposure ({exposure_pct:.1f}%) "
                f"exceeds limit ({self.max_total_exposure*100:.0f}%)"
            )
        elif exposure_pct >= self.alert_thresholds["exposure_critical"] * 100:
            alerts.append(
                f"⚠️ WARNING: Total exposure ({exposure_pct:.1f}%) "
                f"near limit ({self.max_total_exposure*100:.0f}%)"
            )
        elif exposure_pct >= self.alert_thresholds["exposure_warning"] * 100:
            alerts.append(f"ℹ️ INFO: Moderate exposure ({exposure_pct:.1f}%)")

        # Risk alerts
        if risk_pct >= self.max_portfolio_risk * 100:
            alerts.append(
                f"🚨 CRITICAL: Portfolio risk ({risk_pct:.1f}%) "
                f"exceeds limit ({self.max_portfolio_risk*100:.0f}%)"
            )
        elif risk_pct >= self.alert_thresholds["risk_critical"] * 100:
            alerts.append(
                f"⚠️ WARNING: Portfolio risk ({risk_pct:.1f}%) "
                f"near limit ({self.max_portfolio_risk*100:.0f}%)"
            )
        elif risk_pct >= self.alert_thresholds["risk_warning"] * 100:
            alerts.append(f"ℹ️ INFO: Moderate portfolio risk ({risk_pct:.1f}%)")

        # Position size alerts
        if largest_position_pct >= self.max_position_size * 100:
            alerts.append(
                f"🚨 CRITICAL: Largest position ({largest_position_pct:.1f}%) "
                f"exceeds limit ({self.max_position_size*100:.0f}%)"
            )
        elif largest_position_pct >= self.alert_thresholds["position_warning"] * 100:
            alerts.append(f"⚠️ WARNING: Large position detected ({largest_position_pct:.1f}%)")

        # Sector exposure alerts
        for sector, exposure_pct in sector_exposures.items():
            if exposure_pct >= self.max_sector_exposure * 100:
                alerts.append(
                    f"🚨 CRITICAL: {sector} sector ({exposure_pct:.1f}%) "
                    f"exceeds limit ({self.max_sector_exposure*100:.0f}%)"
                )
            elif exposure_pct >= self.alert_thresholds["sector_warning"] * 100:
                alerts.append(f"⚠️ WARNING: High {sector} sector exposure ({exposure_pct:.1f}%)")

        return alerts

    def get_risk_summary(self) -> str:
        """Get formatted risk summary"""
        metrics = self.calculate_metrics()

        summary = []
        summary.append("\n" + "=" * 80)
        summary.append("📊 PORTFOLIO RISK SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Timestamp:          {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Total Capital:      {self.total_capital:,.0f} VND")
        summary.append("")
        summary.append(f"Position Count:     {metrics.position_count}")
        summary.append(
            f"Total Exposure:     {metrics.total_exposure:,.0f} VND  "
            f"({metrics.total_exposure_pct:.2f}%)"
        )
        summary.append(
            f"Total Risk:         {metrics.total_risk:,.0f} VND  ({metrics.total_risk_pct:.2f}%)"
        )
        summary.append(f"Largest Position:   {metrics.largest_position_pct:.2f}%")
        summary.append(f"Max Drawdown:       {metrics.max_drawdown:.2f}%")

        if metrics.sector_exposures:
            summary.append("")
            summary.append("Sector Exposures:")
            for sector, exposure_pct in sorted(
                metrics.sector_exposures.items(), key=lambda x: x[1], reverse=True
            ):
                summary.append(f"  - {sector:15s}: {exposure_pct:6.2f}%")

        if metrics.alerts:
            summary.append("")
            summary.append("⚠️ ACTIVE ALERTS:")
            for alert in metrics.alerts:
                summary.append(f"  {alert}")

        summary.append("=" * 80 + "\n")

        return "\n".join(summary)

    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard visualization"""
        metrics = self.calculate_metrics()

        # Position details
        positions_data = []
        for symbol, pos in self.positions.items():
            positions_data.append(
                {
                    "symbol": symbol,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "value": pos.position_value,
                    "value_pct": pos.position_pct,
                    "risk": pos.risk_amount,
                    "risk_pct": pos.risk_pct,
                    "pnl": pos.pnl,
                    "pnl_pct": pos.pnl_pct,
                    "sector": pos.sector,
                }
            )

        return {
            "timestamp": metrics.timestamp.isoformat(),
            "capital": self.total_capital,
            "peak_capital": self.peak_capital,
            "metrics": {
                "total_exposure": metrics.total_exposure,
                "total_exposure_pct": metrics.total_exposure_pct,
                "total_risk": metrics.total_risk,
                "total_risk_pct": metrics.total_risk_pct,
                "position_count": metrics.position_count,
                "largest_position_pct": metrics.largest_position_pct,
                "max_drawdown": metrics.max_drawdown,
                "correlation_risk": metrics.correlation_risk,
            },
            "sector_exposures": metrics.sector_exposures,
            "positions": positions_data,
            "alerts": metrics.alerts,
            "limits": {
                "max_exposure": self.max_total_exposure * 100,
                "max_risk": self.max_portfolio_risk * 100,
                "max_position": self.max_position_size * 100,
                "max_sector": self.max_sector_exposure * 100,
            },
        }


# Singleton instance
_monitor_instance: Optional[PortfolioRiskMonitor] = None


def get_portfolio_monitor(
    total_capital: float = 100_000_000, reset: bool = False
) -> PortfolioRiskMonitor:
    """Get singleton instance of portfolio monitor"""
    global _monitor_instance
    if _monitor_instance is None or reset:
        _monitor_instance = PortfolioRiskMonitor(total_capital=total_capital)
    return _monitor_instance
