"""
Enhanced Position Sizing với Kelly Criterion và Portfolio Risk
Cải thiện từ improved_position_sizing.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging
from exceptions import PositionSizingError, RiskManagementError

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPositionSize:
    """Container cho kết quả position sizing với Kelly"""

    shares: int
    value: float
    risk_amount: float
    risk_percent: float
    max_loss: float
    position_percent: float
    kelly_percent: float  # Kelly percentage
    recommended_entries: List[Dict]
    warnings: List[str]
    adjustments: Dict[str, float]  # Track all adjustments


class EnhancedPositionSizer:
    """
    Enhanced Position Sizing với:
    1. Kelly Criterion (half-Kelly for safety)
    2. Portfolio-level risk limits
    3. Correlation-based adjustments
    4. Sector exposure limits
    5. Win rate based sizing
    """

    def __init__(
        self,
        total_capital: float = 100_000_000,
        max_risk_per_trade: float = 0.02,  # 2% max risk
        max_position_size: float = 0.15,  # 15% max position
        min_position_size: float = 0.05,  # 5% min position
        max_total_exposure: float = 0.60,  # 60% max exposure
        max_portfolio_risk: float = 0.20,  # 20% max portfolio risk
        max_sector_exposure: float = 0.40,  # 40% max per sector
        use_kelly: bool = True,
        kelly_fraction: float = 0.5,
    ):  # Half-Kelly
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_total_exposure = max_total_exposure
        self.max_portfolio_risk = max_portfolio_risk
        self.max_sector_exposure = max_sector_exposure
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction

        # Tracking
        self.current_positions = {}  # {symbol: position_data}
        self.trade_history = []  # Track trades for win rate calculation
        self.sector_exposure = {}  # Track sector exposure

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        signal_strength: str = "MODERATE",
        market_regime: Optional[Dict] = None,
        sector: Optional[str] = None,
        portfolio_risk: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
    ) -> EnhancedPositionSize:
        """
        Calculate position size với Kelly Criterion và portfolio context

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit target
            confidence: 0-100
            signal_strength: Signal strength
            market_regime: Market regime info
            sector: Stock sector
            portfolio_risk: Current portfolio risk percentage
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio

        Returns:
            EnhancedPositionSize
        """
        warnings = []
        adjustments = {}

        # =================================================================
        # CHECK 1: Portfolio Risk Limit
        # =================================================================
        if portfolio_risk is not None and portfolio_risk >= self.max_portfolio_risk:
            raise RiskManagementError(
                f"Portfolio risk ({portfolio_risk*100:.1f}%) exceeds limit ({self.max_portfolio_risk*100:.1f}%)",
                context={
                    "portfolio_risk": portfolio_risk,
                    "limit": self.max_portfolio_risk,
                },
            )

        # =================================================================
        # CHECK 2: Sector Exposure
        # =================================================================
        if sector:
            sector_exp = self._get_sector_exposure(sector)
            if sector_exp >= self.max_sector_exposure:
                raise RiskManagementError(
                    f"Sector {sector} exposure ({sector_exp*100:.1f}%) exceeds limit ({self.max_sector_exposure*100:.1f}%)",
                    context={
                        "sector": sector,
                        "exposure": sector_exp,
                        "limit": self.max_sector_exposure,
                    },
                )

        # =================================================================
        # CHECK 3: Available Capital
        # =================================================================
        current_exposure = self._calculate_current_exposure()
        available_capital = (
            self.total_capital * self.max_total_exposure - current_exposure
        )

        if available_capital <= 0:
            return self._zero_position(
                f"Exposure limit reached ({current_exposure:,.0f} VNĐ)", warnings
            )

        # =================================================================
        # CALCULATE BASE POSITION SIZE
        # =================================================================
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(take_profit - entry_price)

        if risk_per_share <= 0:
            return self._zero_position("Invalid stop loss", warnings)

        risk_reward_ratio = (
            reward_per_share / risk_per_share if risk_per_share > 0 else 0
        )

        # =================================================================
        # METHOD 1: Kelly Criterion (if win rate available)
        # =================================================================
        kelly_percent = 0.0
        if self.use_kelly and win_rate is not None and avg_win_loss_ratio is not None:
            kelly_percent = self._calculate_kelly(
                win_rate=win_rate, avg_win_loss_ratio=avg_win_loss_ratio
            )
            adjustments["kelly"] = kelly_percent

        # =================================================================
        # METHOD 2: Risk-based (fallback or combine)
        # =================================================================
        base_risk_amount = self.total_capital * self.max_risk_per_trade

        # Adjust risk by confidence and signal strength
        risk_multiplier = self._calculate_risk_multiplier(
            confidence, signal_strength, market_regime
        )
        adjustments["risk_multiplier"] = risk_multiplier

        adjusted_risk_amount = base_risk_amount * risk_multiplier

        # Shares by risk
        shares_by_risk = int(adjusted_risk_amount / risk_per_share)

        # =================================================================
        # METHOD 3: Kelly-based (if available)
        # =================================================================
        shares_by_kelly = 0
        if kelly_percent > 0:
            kelly_capital = self.total_capital * kelly_percent
            shares_by_kelly = int(kelly_capital / entry_price)
            adjustments["kelly_shares"] = shares_by_kelly

        # =================================================================
        # COMBINE METHODS
        # =================================================================
        if shares_by_kelly > 0:
            # Use minimum of Kelly and Risk-based (conservative)
            base_shares = min(shares_by_risk, shares_by_kelly)
        else:
            base_shares = shares_by_risk

        # =================================================================
        # ADJUSTMENTS
        # =================================================================

        # 1. Portfolio risk adjustment
        portfolio_adj = 1.0
        if portfolio_risk is not None:
            remaining_risk = self.max_portfolio_risk - portfolio_risk
            portfolio_adj = min(1.0, remaining_risk / self.max_portfolio_risk)
            adjustments["portfolio_risk_adj"] = portfolio_adj

        # 2. Sector exposure adjustment
        sector_adj = 1.0
        if sector:
            sector_exp = self._get_sector_exposure(sector)
            remaining_sector = self.max_sector_exposure - sector_exp
            sector_adj = min(1.0, remaining_sector / self.max_sector_exposure)
            adjustments["sector_adj"] = sector_adj

        # 3. Correlation adjustment (simplified - would need correlation matrix)
        correlation_adj = 1.0
        if len(self.current_positions) > 0:
            # Reduce size if too many positions in similar sectors
            correlation_adj = self._correlation_adjustment(symbol, sector)
            adjustments["correlation_adj"] = correlation_adj

        # Apply all adjustments
        final_shares = int(base_shares * portfolio_adj * sector_adj * correlation_adj)

        # =================================================================
        # ENFORCE LIMITS
        # =================================================================

        # Max by capital
        max_shares_by_capital = int(
            (self.total_capital * self.max_position_size) / entry_price
        )

        # Max by available
        max_shares_by_available = int(available_capital / entry_price)

        # Min position
        min_shares = int((self.total_capital * self.min_position_size) / entry_price)

        # Final shares
        shares = min(final_shares, max_shares_by_capital, max_shares_by_available)
        shares = max(shares, min_shares) if shares > 0 else 0

        # Round to lot of 100
        if shares > 0:
            shares = max((shares // 100) * 100, 100)
        else:
            return self._zero_position("Position size = 0 after calculations", warnings)

        # =================================================================
        # FINAL VALIDATION
        # =================================================================
        position_value = shares * entry_price
        position_percent = (position_value / self.total_capital) * 100
        max_loss = shares * risk_per_share
        risk_percent = (max_loss / self.total_capital) * 100

        # Check risk limit
        if risk_percent > self.max_risk_per_trade * 100:
            max_safe_shares = int(
                (self.total_capital * self.max_risk_per_trade) / risk_per_share
            )
            shares = min(shares, max_safe_shares)
            shares = max((shares // 100) * 100, 100)

            position_value = shares * entry_price
            position_percent = (position_value / self.total_capital) * 100
            max_loss = shares * risk_per_share
            risk_percent = (max_loss / self.total_capital) * 100

            warnings.append(
                f"Reduced shares to keep risk <= {self.max_risk_per_trade*100}%"
            )

        # DCA entries
        recommended_entries = self._calculate_dca_entries(entry_price, shares)

        # Warnings
        if position_percent > self.max_position_size * 100 * 0.8:
            warnings.append(f"Large position: {position_percent:.1f}%")

        if risk_percent > self.max_risk_per_trade * 100 * 0.8:
            warnings.append(f"High risk: {risk_percent:.2f}%")

        return EnhancedPositionSize(
            shares=shares,
            value=position_value,
            risk_amount=max_loss,
            risk_percent=risk_percent,
            max_loss=max_loss,
            position_percent=position_percent,
            kelly_percent=kelly_percent * 100,
            recommended_entries=recommended_entries,
            warnings=warnings,
            adjustments=adjustments,
        )

    def _calculate_kelly(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion percentage

        Formula: K = W - (1-W)/R
        Where:
            W = win rate
            R = average win / average loss

        Returns:
            Kelly percentage (0-1), using half-Kelly for safety
        """
        # VALIDATION: Check avg_win_loss_ratio
        if avg_win_loss_ratio <= 0:
            logger.warning(
                f"⚠️ Invalid avg_win_loss_ratio: {avg_win_loss_ratio:.3f}. "
                f"Must be > 0. Using conservative sizing (Kelly = 0)."
            )
            return 0.0

        # VALIDATION: Check win_rate range
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(
                f"⚠️ Invalid win_rate: {win_rate:.3f}. "
                f"Must be between 0 and 1. Using conservative sizing (Kelly = 0)."
            )
            return 0.0

        # VALIDATION: Check if win_rate is suspiciously low
        if win_rate < 0.3:
            logger.warning(
                f"⚠️ Low win rate detected: {win_rate:.1%}. "
                f"Consider reviewing strategy before trading."
            )

        # Calculate Kelly
        kelly = win_rate - ((1 - win_rate) / avg_win_loss_ratio)

        # Log raw Kelly before applying fraction
        logger.debug(
            f"📊 Kelly Calculation: win_rate={win_rate:.1%}, "
            f"win/loss_ratio={avg_win_loss_ratio:.2f}, "
            f"raw_kelly={kelly:.1%}"
        )

        # Use half-Kelly for safety
        half_kelly = kelly * self.kelly_fraction

        # VALIDATION: Warn if Kelly suggests negative or very large position
        if kelly < 0:
            logger.warning(
                f"⚠️ Negative Kelly ({kelly:.1%}) suggests unfavorable odds. "
                f"Win rate too low or win/loss ratio unfavorable. "
                f"Using Kelly = 0."
            )
            return 0.0

        if kelly > 0.5:
            logger.warning(
                f"⚠️ Very high Kelly ({kelly:.1%}) detected. "
                f"Clamping to max 25% for safety."
            )

        # Clamp to reasonable range
        final_kelly = max(0.0, min(half_kelly, 0.25))  # Max 25% of capital

        logger.info(
            f"✅ Kelly position sizing: {final_kelly:.1%} of capital "
            f"(win_rate={win_rate:.1%}, W/L={avg_win_loss_ratio:.2f})"
        )

        return final_kelly

    def _calculate_risk_multiplier(
        self, confidence: int, signal_strength: str, market_regime: Optional[Dict]
    ) -> float:
        """Calculate risk multiplier"""
        # Base from confidence
        if confidence >= 80:
            base = 1.1
        elif confidence >= 70:
            base = 1.0
        elif confidence >= 60:
            base = 0.8
        else:
            base = 0.6

        # Signal strength
        strength_mult = {
            "VERY_STRONG": 1.1,
            "STRONG": 1.0,
            "MODERATE": 0.9,
            "WEAK": 0.7,
            "VERY_WEAK": 0.5,
        }.get(signal_strength, 0.9)

        # Market regime
        regime_mult = 1.0
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BULL":
                regime_mult = 1.1
            elif regime == "BEAR":
                regime_mult = 0.5
            elif regime == "HIGH_VOLATILITY":
                regime_mult = 0.6
            else:
                regime_mult = 0.8

        return max(0.5, min(base * strength_mult * regime_mult, 1.2))

    def _calculate_correlation(
        self, symbol1: str, symbol2: str, days: int = 60
    ) -> float:
        """
        Calculate correlation coefficient between two stocks

        Args:
            symbol1: First stock symbol
            symbol2: Second stock symbol
            days: Number of days to use for correlation calculation

        Returns:
            Correlation coefficient (-1 to 1), or 0 if calculation fails
        """
        try:
            from src.data.loader import TCBSDataLoader

            loader = TCBSDataLoader()

            # Load data for both symbols
            df1 = loader.load_data(symbol1, days=days)
            df2 = loader.load_data(symbol2, days=days)

            if df1 is None or df2 is None or len(df1) < 10 or len(df2) < 10:
                logger.warning(
                    f"Insufficient data for correlation: {symbol1}-{symbol2}"
                )
                return 0.0

            # Merge on date to align time series
            merged = pd.merge(
                df1[["date", "close"]],
                df2[["date", "close"]],
                on="date",
                suffixes=("_1", "_2"),
            )

            if len(merged) < 10:
                logger.warning(
                    f"Insufficient overlapping dates for {symbol1}-{symbol2}: {len(merged)}"
                )
                return 0.0

            # Calculate correlation
            corr = merged["close_1"].corr(merged["close_2"])

            if pd.isna(corr):
                return 0.0

            return corr

        except Exception as e:
            logger.warning(f"Error calculating correlation {symbol1}-{symbol2}: {e}")
            return 0.0

    def _correlation_adjustment(self, symbol: str, sector: Optional[str]) -> float:
        """
        ENHANCED: Adjust for correlation using real correlation matrix

        Calculates actual price correlation between the new symbol and
        existing positions, rather than just counting same-sector positions.

        Logic:
        1. Calculate correlation with each existing position
        2. Use average absolute correlation
        3. Higher correlation → smaller position size

        Adjustment formula:
        - avg_corr > 0.7 (high): 0.5x (reduce 50%)
        - avg_corr > 0.5 (medium): 0.75x (reduce 25%)
        - avg_corr <= 0.5 (low): 1.0x (no reduction)

        Fallback to sector-based if correlation calc fails.
        """
        if not self.current_positions:
            return 1.0

        # Try to calculate real correlations
        correlations = []
        successful_calcs = 0

        for pos_symbol, pos_data in self.current_positions.items():
            if pos_symbol == symbol:
                continue

            try:
                corr = self._calculate_correlation(symbol, pos_symbol, days=60)
                correlations.append(abs(corr))  # Use absolute correlation
                successful_calcs += 1

                logger.debug(f"Correlation {symbol}-{pos_symbol}: {corr:.3f}")

            except Exception as e:
                logger.debug(
                    f"Skipping correlation calc for {symbol}-{pos_symbol}: {e}"
                )
                continue

        # If we successfully calculated at least one correlation, use it
        if successful_calcs > 0:
            avg_correlation = sum(correlations) / len(correlations)

            logger.info(
                f"📊 Average correlation for {symbol}: {avg_correlation:.3f} "
                f"(calculated with {successful_calcs} positions)"
            )

            # Determine adjustment based on correlation
            if avg_correlation > 0.7:
                adjustment = 0.5  # High correlation - reduce 50%
                logger.warning(
                    f"⚠️ High correlation detected ({avg_correlation:.2f}) "
                    f"for {symbol}. Reducing position size by 50%."
                )
            elif avg_correlation > 0.5:
                adjustment = 0.75  # Medium correlation - reduce 25%
                logger.info(
                    f"Medium correlation ({avg_correlation:.2f}) for {symbol}. "
                    f"Reducing position size by 25%."
                )
            else:
                adjustment = 1.0  # Low correlation - no reduction

            return adjustment

        # FALLBACK: Use sector-based correlation (original logic)
        else:
            logger.info(
                f"Using sector-based correlation for {symbol} (data unavailable for real correlation)"
            )

            if not sector:
                return 1.0

            # Count positions in same sector
            same_sector_count = sum(
                1
                for pos in self.current_positions.values()
                if pos.get("sector") == sector
            )

            # Reduce size if too many in same sector
            if same_sector_count >= 3:
                logger.warning(
                    f"⚠️ {same_sector_count} positions in {sector} sector. "
                    f"Reducing position size by 30%."
                )
                return 0.7  # Reduce 30%
            elif same_sector_count >= 2:
                logger.info(
                    f"{same_sector_count} positions in {sector} sector. "
                    f"Reducing position size by 15%."
                )
                return 0.85  # Reduce 15%

            return 1.0

    def _get_sector_exposure(self, sector: str) -> float:
        """Get current sector exposure as percentage of capital"""
        if not sector:
            return 0.0

        sector_value = sum(
            pos["shares"] * pos["current_price"]
            for pos in self.current_positions.values()
            if pos.get("sector") == sector
        )

        return sector_value / self.total_capital if self.total_capital > 0 else 0.0

    def _calculate_current_exposure(self) -> float:
        """Calculate current total exposure"""
        return sum(
            pos["shares"] * pos["current_price"]
            for pos in self.current_positions.values()
        )

    def _calculate_dca_entries(
        self, base_price: float, total_shares: int
    ) -> List[Dict]:
        """Calculate DCA entry levels"""
        return [
            {
                "level": 1,
                "price": round(base_price * 0.99, -2),
                "shares": int((total_shares * 0.5 // 100) * 100),
                "percent": 50,
            },
            {
                "level": 2,
                "price": round(base_price * 0.98, -2),
                "shares": int((total_shares * 0.3 // 100) * 100),
                "percent": 30,
            },
            {
                "level": 3,
                "price": round(base_price * 0.97, -2),
                "shares": int((total_shares * 0.2 // 100) * 100),
                "percent": 20,
            },
        ]

    def _zero_position(self, reason: str, warnings: List[str]) -> EnhancedPositionSize:
        """Return zero position"""
        warnings.append(reason)
        return EnhancedPositionSize(
            shares=0,
            value=0,
            risk_amount=0,
            risk_percent=0,
            max_loss=0,
            position_percent=0,
            kelly_percent=0,
            recommended_entries=[],
            warnings=warnings,
            adjustments={},
        )
