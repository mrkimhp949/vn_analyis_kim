"""
Portfolio Risk Manager - Real-time risk calculation, circuit breakers, correlation limits
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from exceptions import RiskManagementError
from risk_metrics import calculate_portfolio_correlation_risk, get_sector_for_symbol


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics"""
    total_portfolio_value: float
    total_risk_amount: float
    portfolio_risk_percent: float
    max_position_risk: float
    sector_exposure: Dict[str, float]  # {sector: exposure_percent}
    correlation_risk: float
    max_correlation: float
    num_positions: int
    diversification_score: float  # 0-1, higher is better
    risk_status: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


class PortfolioRiskManager:
    """
    Real-time portfolio risk management:
    1. Calculate portfolio risk in real-time
    2. Enforce correlation limits
    3. Circuit breakers for portfolio drawdown
    4. Sector exposure limits
    """
    
    def __init__(self,
                 total_capital: float = 100_000_000,
                 max_portfolio_risk: float = 0.20,  # 20% max risk
                 max_sector_exposure: float = 0.40,  # 40% per sector
                 max_correlation: float = 0.70,  # Max correlation between positions
                 max_drawdown: float = 0.15,  # 15% max drawdown before circuit breaker
                 correlation_lookback: int = 60):
        self.total_capital = total_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.max_drawdown = max_drawdown
        self.correlation_lookback = correlation_lookback
        
        # Track initial portfolio value for drawdown calculation
        self.initial_portfolio_value = total_capital
        self.peak_portfolio_value = total_capital
    
    def calculate_portfolio_risk(self, positions: Dict) -> PortfolioRiskMetrics:
        """
        Calculate real-time portfolio risk metrics
        
        Args:
            positions: {symbol: position_data}
                position_data should have: shares, avg_price, stop_loss, current_price
        
        Returns:
            PortfolioRiskMetrics
        """
        if not positions:
            return PortfolioRiskMetrics(
                total_portfolio_value=0,
                total_risk_amount=0,
                portfolio_risk_percent=0,
                max_position_risk=0,
                sector_exposure={},
                correlation_risk=0,
                max_correlation=0,
                num_positions=0,
                diversification_score=1.0,
                risk_status='LOW'
            )
        
        # Calculate total portfolio value
        total_value = sum(
            pos.get('shares', 0) * pos.get('current_price', pos.get('avg_price', 0))
            for pos in positions.values()
        )
        
        # Calculate total risk (sum of all stop loss risks)
        total_risk = 0
        max_pos_risk = 0
        
        for symbol, pos in positions.items():
            shares = pos.get('shares', 0)
            entry_price = pos.get('avg_price', 0)
            stop_loss = pos.get('stop_loss', entry_price * 0.93)  # Default -7%
            current_price = pos.get('current_price', entry_price)
            
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss)
            position_risk = risk_per_share * shares
            total_risk += position_risk
            
            # Track max position risk
            max_pos_risk = max(max_pos_risk, position_risk)
        
        portfolio_risk_percent = (total_risk / self.total_capital) * 100 if self.total_capital > 0 else 0
        
        # Calculate sector exposure
        sector_exposure = self._calculate_sector_exposure(positions)
        
        # Calculate correlation risk
        symbols = list(positions.keys())
        correlation_metrics = calculate_portfolio_correlation_risk(
            symbols,
            lookback=self.correlation_lookback,
            max_avg_correlation=self.max_correlation
        )
        
        correlation_risk = correlation_metrics.get('avg_correlation', 0)
        max_correlation = correlation_metrics.get('max_correlation', 0)
        
        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(
            positions, sector_exposure, correlation_risk
        )
        
        # Determine risk status
        risk_status = self._determine_risk_status(
            portfolio_risk_percent,
            max(sector_exposure.values()) if sector_exposure else 0,
            correlation_risk
        )
        
        return PortfolioRiskMetrics(
            total_portfolio_value=total_value,
            total_risk_amount=total_risk,
            portfolio_risk_percent=portfolio_risk_percent,
            max_position_risk=max_pos_risk,
            sector_exposure=sector_exposure,
            correlation_risk=correlation_risk,
            max_correlation=max_correlation,
            num_positions=len(positions),
            diversification_score=diversification_score,
            risk_status=risk_status
        )
    
    def _calculate_sector_exposure(self, positions: Dict) -> Dict[str, float]:
        """Calculate sector exposure as % of portfolio"""
        sector_values = {}
        total_value = 0
        
        for symbol, pos in positions.items():
            shares = pos.get('shares', 0)
            current_price = pos.get('current_price', pos.get('avg_price', 0))
            position_value = shares * current_price
            
            sector = get_sector_for_symbol(symbol)
            sector_values[sector] = sector_values.get(sector, 0) + position_value
            total_value += position_value
        
        # Convert to percentages
        if total_value > 0:
            return {
                sector: (value / total_value) * 100
                for sector, value in sector_values.items()
            }
        
        return {}
    
    def _calculate_diversification_score(self,
                                        positions: Dict,
                                        sector_exposure: Dict[str, float],
                                        correlation_risk: float) -> float:
        """
        Calculate diversification score (0-1)
        Higher = more diversified
        """
        if not positions:
            return 1.0
        
        score = 1.0
        
        # Penalize high sector concentration
        if sector_exposure:
            max_sector_exp = max(sector_exposure.values())
            if max_sector_exp > self.max_sector_exposure * 100:
                score *= 0.5  # Heavy penalty
            elif max_sector_exp > self.max_sector_exposure * 100 * 0.8:
                score *= 0.7  # Moderate penalty
        
        # Penalize high correlation
        if correlation_risk > self.max_correlation:
            score *= 0.5
        elif correlation_risk > self.max_correlation * 0.8:
            score *= 0.7
        
        # Reward more positions (up to a point)
        num_positions = len(positions)
        if num_positions < 5:
            score *= 0.8  # Penalty for too few positions
        elif num_positions > 15:
            score *= 0.9  # Slight penalty for too many positions
        
        return max(0.0, min(score, 1.0))
    
    def _determine_risk_status(self,
                              portfolio_risk_percent: float,
                              max_sector_exposure: float,
                              correlation_risk: float) -> str:
        """Determine overall risk status"""
        critical_count = 0
        high_count = 0
        
        # Check portfolio risk
        if portfolio_risk_percent > self.max_portfolio_risk * 100 * 1.2:
            critical_count += 1
        elif portfolio_risk_percent > self.max_portfolio_risk * 100:
            high_count += 1
        
        # Check sector exposure
        if max_sector_exposure > self.max_sector_exposure * 100 * 1.2:
            critical_count += 1
        elif max_sector_exposure > self.max_sector_exposure * 100:
            high_count += 1
        
        # Check correlation
        if correlation_risk > self.max_correlation * 1.2:
            critical_count += 1
        elif correlation_risk > self.max_correlation:
            high_count += 1
        
        if critical_count > 0:
            return 'CRITICAL'
        elif high_count >= 2:
            return 'HIGH'
        elif high_count > 0 or portfolio_risk_percent > self.max_portfolio_risk * 100 * 0.8:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def check_circuit_breaker(self, current_portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if circuit breaker should trigger based on drawdown
        
        Args:
            current_portfolio_value: Current total portfolio value
        
        Returns:
            (should_trigger, reason)
        """
        # Update peak
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        # Calculate drawdown from peak
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        drawdown_percent = drawdown * 100
        
        if drawdown_percent >= self.max_drawdown * 100:
            return True, f"🚨 CIRCUIT BREAKER: Drawdown {drawdown_percent:.1f}% >= {self.max_drawdown*100:.1f}%"
        
        return False, f"✅ Drawdown OK: {drawdown_percent:.1f}%"
    
    def can_add_position(self,
                        symbol: str,
                        position_value: float,
                        position_risk: float,
                        positions: Dict) -> Tuple[bool, str]:
        """
        Check if can add new position based on risk limits
        
        Args:
            symbol: Stock symbol
            position_value: Value of new position
            position_risk: Risk amount of new position
            positions: Current positions
        
        Returns:
            (can_add, reason)
        """
        # Calculate current risk
        current_metrics = self.calculate_portfolio_risk(positions)
        
        # Check 1: Portfolio risk limit
        new_total_risk = current_metrics.total_risk_amount + position_risk
        new_risk_percent = (new_total_risk / self.total_capital) * 100
        
        if new_risk_percent > self.max_portfolio_risk * 100:
            return False, f"Portfolio risk would exceed limit: {new_risk_percent:.1f}% > {self.max_portfolio_risk*100:.1f}%"
        
        # Check 2: Sector exposure
        sector = get_sector_for_symbol(symbol)
        current_sector_exp = current_metrics.sector_exposure.get(sector, 0)
        new_total_value = current_metrics.total_portfolio_value + position_value
        new_sector_value = (current_metrics.sector_exposure.get(sector, 0) / 100 * current_metrics.total_portfolio_value) + position_value
        new_sector_exp = (new_sector_value / new_total_value * 100) if new_total_value > 0 else 0
        
        if new_sector_exp > self.max_sector_exposure * 100:
            return False, f"Sector {sector} exposure would exceed limit: {new_sector_exp:.1f}% > {self.max_sector_exposure*100:.1f}%"
        
        # Check 3: Correlation (simplified - would need correlation matrix)
        # This is a simplified check - full implementation would calculate correlation
        if len(positions) > 0:
            # Check if too many positions in same sector
            same_sector_count = sum(
                1 for s in positions.keys()
                if get_sector_for_symbol(s) == sector
            )
            if same_sector_count >= 3:  # Max 3 positions per sector
                return False, f"Too many positions in sector {sector} ({same_sector_count})"
        
        return True, "OK to add position"
    
    def get_risk_summary(self, positions: Dict) -> str:
        """Get formatted risk summary"""
        metrics = self.calculate_portfolio_risk(positions)
        
        lines = []
        lines.append("📊 **PORTFOLIO RISK ANALYSIS**")
        lines.append("=" * 50)
        lines.append(f"💰 Portfolio Value: {metrics.total_portfolio_value:,.0f} VNĐ")
        lines.append(f"⚠️ Total Risk: {metrics.total_risk_amount:,.0f} VNĐ ({metrics.portfolio_risk_percent:.1f}%)")
        lines.append(f"📈 Max Position Risk: {metrics.max_position_risk:,.0f} VNĐ")
        lines.append(f"📦 Positions: {metrics.num_positions}")
        lines.append(f"🎯 Diversification Score: {metrics.diversification_score:.2f}")
        lines.append(f"📊 Risk Status: {metrics.risk_status}")
        
        if metrics.sector_exposure:
            lines.append(f"\n🏢 **SECTOR EXPOSURE:**")
            for sector, exp in sorted(metrics.sector_exposure.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  {sector}: {exp:.1f}%")
        
        lines.append(f"\n🔗 **CORRELATION:**")
        lines.append(f"  Avg Correlation: {metrics.correlation_risk:.2f}")
        lines.append(f"  Max Correlation: {metrics.max_correlation:.2f}")
        
        # Circuit breaker check
        can_trade, cb_reason = self.check_circuit_breaker(metrics.total_portfolio_value)
        lines.append(f"\n🔒 **CIRCUIT BREAKER:**")
        lines.append(f"  {cb_reason}")
        
        return "\n".join(lines)


# Singleton
_risk_manager = None

def get_portfolio_risk_manager(total_capital: float = 100_000_000) -> PortfolioRiskManager:
    """Get portfolio risk manager singleton"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = PortfolioRiskManager(total_capital=total_capital)
    return _risk_manager

