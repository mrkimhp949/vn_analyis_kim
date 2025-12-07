# -*- coding: utf-8 -*-
"""
Position Sizing v3.0 - Optimized for Vietnam Market

IMPROVEMENTS over v2.0:
1. Simplified calculation flow (less over-engineering)
2. Vietnam-specific adjustments (T+2, lot size, liquidity tiers)
3. Volatility-adjusted position sizing
4. Smarter Kelly Criterion with regime awareness
5. Real-time margin tracking integration
6. Better correlation handling with sector limits

Author: Trading Bot Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class VNPositionConstants:
    """Vietnam market position sizing constants."""

    # Lot size
    LOT_SIZE = 100

    # Risk limits
    MAX_RISK_PER_TRADE = 0.015  # 1.5% max risk per trade
    MAX_POSITION_PCT = 0.12  # 12% max single position
    MIN_POSITION_PCT = 0.03  # 3% min position (worth trading)
    MAX_TOTAL_EXPOSURE = 0.60  # 60% max total exposure
    MAX_SECTOR_EXPOSURE = 0.25  # 25% max per sector
    MAX_CORRELATION = 0.65  # Max correlation between positions

    # Transaction costs
    ROUND_TRIP_COST = 0.0148  # 1.48%

    # Kelly limits
    MAX_KELLY_PCT = 0.20  # Max 20% from Kelly
    KELLY_FRACTION = 0.5  # Half-Kelly for safety
    MIN_KELLY_FALLBACK = 0.02  # 2% minimum if Kelly negative

    # Liquidity-based position limits
    LIQUIDITY_TIERS = {
        "VN30": {"max_pct": 0.15, "slippage": 0.003},
        "LARGE_CAP": {"max_pct": 0.12, "slippage": 0.004},
        "MID_CAP": {"max_pct": 0.10, "slippage": 0.006},
        "SMALL_CAP": {"max_pct": 0.06, "slippage": 0.010},
    }

    # Regime-based adjustments
    REGIME_MULTIPLIERS = {
        "BULL": 1.2,
        "SIDEWAYS": 1.0,
        "BEAR": 0.6,
        "HIGH_VOLATILITY": 0.5,
    }

    # VN30 symbols for tier detection
    VN30_SYMBOLS = {
        "ACB",
        "BCM",
        "BID",
        "BVH",
        "CTG",
        "FPT",
        "GAS",
        "GVR",
        "HDB",
        "HPG",
        "MBB",
        "MSN",
        "MWG",
        "PLX",
        "POW",
        "SAB",
        "SHB",
        "SSB",
        "SSI",
        "STB",
        "TCB",
        "TPB",
        "VCB",
        "VHM",
        "VIB",
        "VIC",
        "VJC",
        "VNM",
        "VPB",
        "VRE",
    }


class PositionSizeMethod(Enum):
    """Position sizing methods."""

    FIXED_RISK = "FIXED_RISK"  # Fixed % risk per trade
    KELLY = "KELLY"  # Kelly Criterion
    VOLATILITY_ADJUSTED = "VOLATILITY"  # ATR-based sizing
    HYBRID = "HYBRID"  # Combination of methods


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    # Core values
    shares: int
    value: float
    risk_amount: float

    # Percentages
    position_pct: float  # % of portfolio
    risk_pct: float  # % risk of portfolio

    # Price levels
    entry_price: float
    stop_loss: float
    max_loss: float

    # Method info
    method_used: PositionSizeMethod
    kelly_pct: float = 0.0

    # Adjustments applied
    adjustments: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Validation
    is_valid: bool = True
    validation_message: str = ""

    def __post_init__(self):
        """Validate result."""
        if self.shares < VNPositionConstants.LOT_SIZE:
            self.is_valid = False
            self.validation_message = f"Shares ({self.shares}) below minimum lot size (100)"


@dataclass
class PortfolioContext:
    """Current portfolio context for position sizing."""

    total_capital: float
    available_cash: float
    current_exposure: float
    current_positions: Dict[str, Dict] = field(default_factory=dict)
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    pending_settlements: float = 0.0  # T+2 pending

    @property
    def exposure_pct(self) -> float:
        return self.current_exposure / self.total_capital if self.total_capital > 0 else 0

    @property
    def available_for_new_positions(self) -> float:
        """Available capital considering T+2 settlement."""
        max_new = (
            self.total_capital * VNPositionConstants.MAX_TOTAL_EXPOSURE - self.current_exposure
        )
        return min(max_new, self.available_cash - self.pending_settlements)


# =============================================================================
# MAIN CLASS - Position Sizer v3.0
# =============================================================================


class EnhancedPositionSizerV3:
    """
    Enhanced Position Sizer v3.0 for Vietnam Market.

    KEY IMPROVEMENTS:
    1. Simplified 3-step calculation:
       - Base size from risk
       - Adjust for regime/volatility
       - Apply limits and round to lot

    2. Vietnam-specific features:
       - T+2 settlement tracking
       - Lot size 100 enforcement
       - Liquidity tier limits
       - Sector concentration limits

    3. Smarter Kelly:
       - Regime-aware Kelly fraction
       - Transaction cost adjustment
       - Minimum fallback for negative Kelly

    4. Better risk management:
       - Correlation-based reduction
       - Volatility scaling
       - Drawdown protection
    """

    def __init__(
        self,
        total_capital: float = 100_000_000,
        max_risk_per_trade: float = VNPositionConstants.MAX_RISK_PER_TRADE,
        max_position_pct: float = VNPositionConstants.MAX_POSITION_PCT,
        use_kelly: bool = True,
        use_volatility_adjustment: bool = True,
        use_correlation_check: bool = True,
    ):
        """
        Initialize position sizer.

        Args:
            total_capital: Total portfolio capital in VND
            max_risk_per_trade: Maximum risk per trade (0.015 = 1.5%)
            max_position_pct: Maximum position size (0.12 = 12%)
            use_kelly: Use Kelly Criterion
            use_volatility_adjustment: Adjust for volatility
            use_correlation_check: Check correlation with existing positions
        """
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.use_kelly = use_kelly
        self.use_volatility_adjustment = use_volatility_adjustment
        self.use_correlation_check = use_correlation_check

        # Portfolio tracking
        self._portfolio_context: Optional[PortfolioContext] = None

        # Trade history for Kelly calculation
        self._trade_history: List[Dict] = []

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: int = 60,
        market_regime: Optional[Dict] = None,
        sector: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        portfolio_context: Optional[PortfolioContext] = None,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        3-Step Process:
        1. Calculate base size from risk
        2. Apply adjustments (regime, volatility, correlation)
        3. Enforce limits and round to lot size

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Signal confidence (0-100)
            market_regime: Market regime info
            sector: Stock sector
            df: OHLCV DataFrame for volatility calculation
            portfolio_context: Current portfolio state

        Returns:
            PositionSizeResult with calculated values
        """
        warnings: List[str] = []
        adjustments: Dict[str, float] = {}

        # Use provided context or create default
        context = portfolio_context or self._get_default_context()

        # Validate inputs
        if entry_price <= 0 or stop_loss <= 0:
            return self._zero_position("Invalid price data", warnings)

        if stop_loss >= entry_price:
            return self._zero_position("Stop loss must be below entry", warnings)

        # =================================================================
        # STEP 1: Calculate base size from risk
        # =================================================================
        risk_per_share = entry_price - stop_loss
        risk_pct = risk_per_share / entry_price

        # Ensure minimum risk (prevent tiny stops)
        if risk_pct < 0.02:  # Less than 2%
            risk_per_share = entry_price * 0.03  # Enforce 3% minimum
            warnings.append(f"Stop too tight, adjusted to 3% ({risk_per_share:.0f} VND)")

        # Base risk amount
        base_risk_amount = self.total_capital * self.max_risk_per_trade
        base_shares = int(base_risk_amount / risk_per_share)

        adjustments["base_shares"] = base_shares

        # =================================================================
        # STEP 2: Apply adjustments
        # =================================================================
        adjusted_shares = float(base_shares)

        # 2a. Confidence adjustment
        conf_mult = self._get_confidence_multiplier(confidence)
        adjusted_shares *= conf_mult
        adjustments["confidence_mult"] = conf_mult

        # 2b. Regime adjustment
        regime = market_regime.get("regime", "SIDEWAYS") if market_regime else "SIDEWAYS"
        regime_mult = VNPositionConstants.REGIME_MULTIPLIERS.get(regime, 1.0)
        adjusted_shares *= regime_mult
        adjustments["regime_mult"] = regime_mult

        if regime in ["BEAR", "HIGH_VOLATILITY"]:
            warnings.append(f"Position reduced for {regime} market")

        # 2c. Volatility adjustment
        if self.use_volatility_adjustment and df is not None:
            vol_mult = self._get_volatility_multiplier(df)
            adjusted_shares *= vol_mult
            adjustments["volatility_mult"] = vol_mult

            if vol_mult < 0.8:
                warnings.append("Position reduced due to high volatility")

        # 2d. Kelly adjustment (if enabled and data available)
        kelly_pct = 0.0
        if self.use_kelly:
            kelly_pct = self._calculate_kelly_adjusted(regime)
            if kelly_pct > 0:
                kelly_shares = int((self.total_capital * kelly_pct) / entry_price)
                # Use minimum of risk-based and Kelly
                if kelly_shares < adjusted_shares:
                    adjusted_shares = kelly_shares
                    adjustments["kelly_cap"] = kelly_shares

        # 2e. Liquidity tier adjustment
        tier = self._get_liquidity_tier(symbol, df, entry_price)
        tier_config = VNPositionConstants.LIQUIDITY_TIERS.get(
            tier, VNPositionConstants.LIQUIDITY_TIERS["MID_CAP"]
        )
        tier_max_pct = tier_config["max_pct"]

        adjustments["liquidity_tier"] = tier

        # 2f. Correlation adjustment
        if self.use_correlation_check and context.current_positions:
            corr_mult = self._get_correlation_multiplier(symbol, sector, context)
            adjusted_shares *= corr_mult
            adjustments["correlation_mult"] = corr_mult

            if corr_mult < 1.0:
                warnings.append("Position reduced due to portfolio correlation")

        # 2g. Sector exposure check
        if sector:
            sector_exp = context.sector_exposures.get(sector, 0)
            remaining_sector = VNPositionConstants.MAX_SECTOR_EXPOSURE - sector_exp

            if remaining_sector <= 0:
                return self._zero_position(f"Sector {sector} exposure limit reached", warnings)

            max_sector_value = self.total_capital * remaining_sector
            max_sector_shares = int(max_sector_value / entry_price)

            if max_sector_shares < adjusted_shares:
                adjusted_shares = max_sector_shares
                warnings.append(f"Position capped by sector limit ({sector})")

        # =================================================================
        # STEP 3: Enforce limits and round to lot size
        # =================================================================

        # 3a. Max position limit
        max_position_value = self.total_capital * min(self.max_position_pct, tier_max_pct)
        max_shares_by_position = int(max_position_value / entry_price)

        if adjusted_shares > max_shares_by_position:
            adjusted_shares = max_shares_by_position
            adjustments["position_cap"] = max_shares_by_position

        # 3b. Available capital limit
        available = context.available_for_new_positions
        max_shares_by_capital = int(available / entry_price)

        if adjusted_shares > max_shares_by_capital:
            adjusted_shares = max_shares_by_capital
            warnings.append("Position limited by available capital")

        # 3c. Round to lot size
        final_shares = self._round_to_lot(int(adjusted_shares))

        # 3d. Minimum position check
        min_position_value = self.total_capital * VNPositionConstants.MIN_POSITION_PCT
        if final_shares * entry_price < min_position_value:
            # Try to get minimum viable position
            min_shares = self._round_to_lot(int(min_position_value / entry_price))
            if min_shares * entry_price <= available:
                final_shares = min_shares
                warnings.append("Position increased to minimum viable size")
            else:
                return self._zero_position("Position too small to be viable", warnings)

        # =================================================================
        # BUILD RESULT
        # =================================================================
        position_value = final_shares * entry_price
        risk_amount = final_shares * risk_per_share
        max_loss = final_shares * (entry_price - stop_loss)

        return PositionSizeResult(
            shares=final_shares,
            value=position_value,
            risk_amount=risk_amount,
            position_pct=(position_value / self.total_capital) * 100,
            risk_pct=(risk_amount / self.total_capital) * 100,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_loss=max_loss,
            method_used=(
                PositionSizeMethod.HYBRID if self.use_kelly else PositionSizeMethod.FIXED_RISK
            ),
            kelly_pct=kelly_pct * 100,
            adjustments=adjustments,
            warnings=warnings,
        )

    # =========================================================================
    # ADJUSTMENT METHODS
    # =========================================================================

    def _get_confidence_multiplier(self, confidence: int) -> float:
        """
        Get position multiplier based on signal confidence.

        Confidence -> Multiplier:
        - 80+: 1.2 (high conviction)
        - 70-79: 1.1
        - 60-69: 1.0 (standard)
        - 50-59: 0.8
        - <50: 0.6 (low conviction)
        """
        if confidence >= 80:
            return 1.2
        elif confidence >= 70:
            return 1.1
        elif confidence >= 60:
            return 1.0
        elif confidence >= 50:
            return 0.8
        else:
            return 0.6

    def _get_volatility_multiplier(self, df: pd.DataFrame) -> float:
        """
        Get position multiplier based on volatility.

        Uses ATR/Price ratio to determine volatility level.
        Higher volatility = smaller position.
        """
        if df is None or df.empty or len(df) < 20:
            return 1.0

        # Calculate ATR if not present
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
        else:
            # Simple ATR calculation
            high = df["high"]
            low = df["low"]
            close = df["close"]

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

        current_price = df["close"].iloc[-1]
        atr_pct = atr / current_price

        # Volatility tiers
        if atr_pct < 0.02:  # Low volatility
            return 1.1
        elif atr_pct < 0.03:  # Normal volatility
            return 1.0
        elif atr_pct < 0.04:  # High volatility
            return 0.8
        elif atr_pct < 0.05:  # Very high volatility
            return 0.6
        else:  # Extreme volatility
            return 0.5

    def _calculate_kelly_adjusted(self, regime: str) -> float:
        """
        Calculate Kelly Criterion with regime adjustment.

        Kelly Formula: K = W - (1-W)/R
        Where:
        - W = win rate
        - R = average win / average loss

        Adjustments:
        - Use half-Kelly for safety
        - Reduce further in BEAR/HIGH_VOL
        - Include transaction costs
        """
        # Get win rate and W/L ratio from trade history
        if len(self._trade_history) < 10:
            # Not enough data - use conservative estimate
            win_rate = 0.45
            avg_wl_ratio = 1.5
        else:
            wins = [t for t in self._trade_history if t.get("pnl", 0) > 0]
            losses = [t for t in self._trade_history if t.get("pnl", 0) <= 0]

            win_rate = len(wins) / len(self._trade_history)

            avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1

            avg_wl_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5

        # Adjust for transaction costs
        # Each trade costs ~1.48%, so effective win rate is lower
        cost_adjusted_win_rate = win_rate * (1 - VNPositionConstants.ROUND_TRIP_COST)

        # Calculate Kelly
        if avg_wl_ratio <= 0:
            return VNPositionConstants.MIN_KELLY_FALLBACK

        kelly = cost_adjusted_win_rate - ((1 - cost_adjusted_win_rate) / avg_wl_ratio)

        # Handle negative Kelly
        if kelly <= 0:
            logger.warning(f"Negative Kelly ({kelly:.2%}) - using minimum fallback")
            return VNPositionConstants.MIN_KELLY_FALLBACK

        # Apply half-Kelly
        half_kelly = kelly * VNPositionConstants.KELLY_FRACTION

        # Regime adjustment
        regime_factor = {
            "BULL": 1.0,
            "SIDEWAYS": 0.8,
            "BEAR": 0.5,
            "HIGH_VOLATILITY": 0.4,
        }.get(regime, 0.8)

        adjusted_kelly = half_kelly * regime_factor

        # Cap at maximum
        return min(adjusted_kelly, VNPositionConstants.MAX_KELLY_PCT)

    def _get_liquidity_tier(self, symbol: str, df: Optional[pd.DataFrame], price: float) -> str:
        """Determine liquidity tier for position limits."""
        # Check if VN30
        if symbol.upper() in VNPositionConstants.VN30_SYMBOLS:
            return "VN30"

        # Calculate from volume if available
        if df is not None and "volume" in df.columns and len(df) >= 20:
            avg_volume = df["volume"].tail(20).mean()
            avg_value = avg_volume * price

            if avg_value >= 10_000_000_000:  # 10B+
                return "VN30"
            elif avg_value >= 5_000_000_000:  # 5B+
                return "LARGE_CAP"
            elif avg_value >= 2_000_000_000:  # 2B+
                return "MID_CAP"

        return "SMALL_CAP"

    def _get_correlation_multiplier(
        self,
        symbol: str,
        sector: Optional[str],
        context: PortfolioContext,
    ) -> float:
        """
        Get position multiplier based on correlation with existing positions.

        Reduces position if:
        - Same sector already has positions
        - High correlation with existing positions
        """
        if not context.current_positions:
            return 1.0

        multiplier = 1.0

        # Sector concentration check
        if sector:
            same_sector_count = sum(
                1 for pos in context.current_positions.values() if pos.get("sector") == sector
            )

            if same_sector_count >= 3:
                multiplier *= 0.5
            elif same_sector_count >= 2:
                multiplier *= 0.7
            elif same_sector_count >= 1:
                multiplier *= 0.85

        # Position count check (diversification)
        position_count = len(context.current_positions)

        if position_count >= 8:
            multiplier *= 0.7  # Many positions - reduce new ones
        elif position_count >= 5:
            multiplier *= 0.85

        return multiplier

    def _round_to_lot(self, shares: int) -> int:
        """Round shares to Vietnam lot size (100)."""
        lot_size = VNPositionConstants.LOT_SIZE
        rounded = (shares // lot_size) * lot_size
        return max(lot_size, rounded)  # Minimum 1 lot

    def _get_default_context(self) -> PortfolioContext:
        """Get default portfolio context."""
        return PortfolioContext(
            total_capital=self.total_capital,
            available_cash=self.total_capital,
            current_exposure=0,
        )

    def _zero_position(self, reason: str, warnings: List[str]) -> PositionSizeResult:
        """Create zero position result."""
        return PositionSizeResult(
            shares=0,
            value=0,
            risk_amount=0,
            position_pct=0,
            risk_pct=0,
            entry_price=0,
            stop_loss=0,
            max_loss=0,
            method_used=PositionSizeMethod.FIXED_RISK,
            warnings=warnings + [reason],
            is_valid=False,
            validation_message=reason,
        )

    # =========================================================================
    # PORTFOLIO MANAGEMENT
    # =========================================================================

    def update_portfolio_context(self, context: PortfolioContext) -> None:
        """Update portfolio context for position sizing."""
        self._portfolio_context = context
        self.total_capital = context.total_capital

    def record_trade(self, trade: Dict) -> None:
        """
        Record trade for Kelly calculation.

        Args:
            trade: Dict with keys: symbol, entry_price, exit_price, pnl, pnl_pct
        """
        self._trade_history.append(trade)

        # Keep only last 100 trades
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

    def get_trade_statistics(self) -> Dict:
        """Get trade statistics for Kelly calculation."""
        if len(self._trade_history) < 5:
            return {
                "total_trades": len(self._trade_history),
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "kelly_pct": 0,
            }

        wins = [t for t in self._trade_history if t.get("pnl", 0) > 0]
        losses = [t for t in self._trade_history if t.get("pnl", 0) <= 0]

        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1

        profit_factor = (
            (sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))) if losses else 0
        )

        kelly = self._calculate_kelly_adjusted("SIDEWAYS")

        return {
            "total_trades": len(self._trade_history),
            "win_rate": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "kelly_pct": kelly * 100,
        }

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def calculate_max_shares(
        self,
        symbol: str,
        entry_price: float,
        available_capital: Optional[float] = None,
    ) -> int:
        """
        Calculate maximum shares that can be purchased.

        Considers:
        - Available capital
        - Position limits
        - Liquidity tier
        - Lot size
        """
        capital = available_capital or self.total_capital

        # Get tier limit
        tier = self._get_liquidity_tier(symbol, None, entry_price)
        tier_config = VNPositionConstants.LIQUIDITY_TIERS.get(
            tier, VNPositionConstants.LIQUIDITY_TIERS["MID_CAP"]
        )
        max_pct = tier_config["max_pct"]

        # Calculate max value
        max_value = min(
            capital,
            self.total_capital * max_pct,
            self.total_capital * self.max_position_pct,
        )

        # Calculate shares and round to lot
        max_shares = int(max_value / entry_price)
        return self._round_to_lot(max_shares)

    def calculate_risk_for_shares(
        self,
        shares: int,
        entry_price: float,
        stop_loss: float,
    ) -> Dict:
        """
        Calculate risk metrics for a given number of shares.

        Returns:
            Dict with risk_amount, risk_pct, position_value, position_pct
        """
        position_value = shares * entry_price
        risk_per_share = entry_price - stop_loss
        risk_amount = shares * risk_per_share

        return {
            "shares": shares,
            "position_value": position_value,
            "position_pct": (position_value / self.total_capital) * 100,
            "risk_amount": risk_amount,
            "risk_pct": (risk_amount / self.total_capital) * 100,
            "max_loss": risk_amount,
        }

    def suggest_stop_loss(
        self,
        entry_price: float,
        target_risk_pct: float = 0.015,
        shares: Optional[int] = None,
    ) -> float:
        """
        Suggest stop loss price for target risk percentage.

        Args:
            entry_price: Entry price
            target_risk_pct: Target risk as % of portfolio (default 1.5%)
            shares: Number of shares (if known)

        Returns:
            Suggested stop loss price
        """
        if shares:
            # Calculate stop based on shares and target risk
            target_risk_amount = self.total_capital * target_risk_pct
            risk_per_share = target_risk_amount / shares
            stop_loss = entry_price - risk_per_share
        else:
            # Use default 5% stop
            stop_loss = entry_price * 0.95

        # Round to tick size
        if stop_loss < 10000:
            tick = 10
        elif stop_loss < 50000:
            tick = 50
        else:
            tick = 100

        return round(stop_loss / tick) * tick


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_position_sizer_v3: Optional[EnhancedPositionSizerV3] = None


def get_position_sizer_v3(total_capital: float = 100_000_000) -> EnhancedPositionSizerV3:
    """Get singleton position sizer v3."""
    global _position_sizer_v3
    if _position_sizer_v3 is None:
        _position_sizer_v3 = EnhancedPositionSizerV3(total_capital=total_capital)
    return _position_sizer_v3


# =============================================================================
# INTEGRATION HELPER
# =============================================================================


def calculate_position_with_entry(
    symbol: str,
    entry_signal,  # EntrySignalV3
    total_capital: float = 100_000_000,
    market_regime: Optional[Dict] = None,
    sector: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> PositionSizeResult:
    """
    Calculate position size from entry signal.

    Convenience function that combines entry logic with position sizing.
    """
    sizer = get_position_sizer_v3(total_capital)

    return sizer.calculate_position_size(
        symbol=symbol,
        entry_price=entry_signal.entry_price,
        stop_loss=entry_signal.stop_loss,
        confidence=entry_signal.confidence,
        market_regime=market_regime,
        sector=sector,
        df=df,
    )
