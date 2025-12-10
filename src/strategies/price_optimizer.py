"""
Price Optimizer Module

Contains price calculation, risk/reward analysis, and entry optimization logic.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


@dataclass
class PriceCalculationResult:
    """Result of price and risk calculations."""

    success: bool
    error_message: str
    stop_loss: float
    reward: float
    take_profit_targets: List[float]
    risk_reward: float
    entry_price: float
    entry_type: str
    optimization_reason: Optional[str] = None


class PriceOptimizer:
    """
    Handles price optimization and risk/reward calculations.

    Features:
    - Stop loss calculation with ATR and support levels
    - Take profit target calculation
    - Entry price optimization for Vietnam market
    - Transaction cost-aware R:R calculations
    """

    def __init__(
        self,
        min_risk_reward: float = 1.5,
        atr_multiplier: float = 2.0,
        limit_order_discount: float = 0.005,  # 0.5% below current for limit orders
        max_wait_bars: int = 3,
    ):
        """
        Initialize PriceOptimizer.

        Args:
            min_risk_reward: Minimum acceptable R:R ratio
            atr_multiplier: Multiplier for ATR-based stop loss
            limit_order_discount: Discount percentage for limit orders
            max_wait_bars: Maximum bars to wait for limit order fill
        """
        self.min_risk_reward = min_risk_reward
        self.atr_multiplier = atr_multiplier
        self.limit_order_discount = limit_order_discount
        self.max_wait_bars = max_wait_bars

    def calculate_prices_and_risk(
        self, df: pd.DataFrame, entry_price: float, sr_check: Dict
    ) -> PriceCalculationResult:
        """
        Calculate stop loss, take profit targets, and risk/reward ratio.

        Accounts for Vietnam market transaction costs:
        - Entry costs: commission + slippage on entry_price
        - Exit costs: commission + slippage on take_profit price
        - Stop loss costs: commission + slippage if stopped out

        Risk calculation:
            Risk = (Entry - StopLoss) + Entry_Costs + StopLoss_Exit_Costs

        Reward calculation:
            Reward = (TakeProfit - Entry) - Entry_Costs - Exit_Costs

        Args:
            df: DataFrame with price data and indicators
            entry_price: Proposed entry price
            sr_check: Support/resistance levels

        Returns:
            PriceCalculationResult with all calculations
        """
        from src.config.constants import TOTAL_TRANSACTION_COST, DEFAULT_SLIPPAGE
        from src.utils.indicators import StopLossCalculator, IndicatorUtils

        atr = IndicatorUtils.get_atr(df)
        support_level = sr_check.get("support_level", None)

        # Calculate stop loss
        try:
            stop_loss, sl_reason = StopLossCalculator.calculate_stop_loss(
                entry_price=entry_price,
                atr=atr,
                support_level=support_level,
                atr_multiplier=self.atr_multiplier,
            )
            logger.debug(f"Stop loss calculated: {stop_loss:.0f} ({sl_reason})")

            # Validate stop loss
            if stop_loss is None or stop_loss <= 0:
                return PriceCalculationResult(
                    success=False,
                    error_message=f"Stop loss invalid: {stop_loss}",
                    stop_loss=0,
                    reward=0,
                    take_profit_targets=[],
                    risk_reward=0,
                    entry_price=entry_price,
                    entry_type="MARKET",
                )

            # Ensure stop loss is below entry (for long positions)
            if stop_loss >= entry_price:
                return PriceCalculationResult(
                    success=False,
                    error_message=f"Stop loss ({stop_loss:.0f}) must be below entry ({entry_price:.0f})",
                    stop_loss=0,
                    reward=0,
                    take_profit_targets=[],
                    risk_reward=0,
                    entry_price=entry_price,
                    entry_type="MARKET",
                )

            # Enforce minimum stop loss distance (3% of entry price)
            min_stop_distance = entry_price * 0.03
            if (entry_price - stop_loss) < min_stop_distance:
                stop_loss = entry_price - min_stop_distance
                logger.warning(
                    f"⚠️ Stop loss too tight, adjusted to 3% below entry: {stop_loss:.0f}"
                )

            # Enforce maximum stop loss distance (10% of entry price)
            max_stop_distance = entry_price * 0.10
            if (entry_price - stop_loss) > max_stop_distance:
                stop_loss = entry_price - max_stop_distance
                logger.warning(
                    f"⚠️ Stop loss too wide, adjusted to 10% below entry: {stop_loss:.0f}"
                )

        except ValueError as e:
            return PriceCalculationResult(
                success=False,
                error_message=f"Stop loss calculation failed: {str(e)}",
                stop_loss=0,
                reward=0,
                take_profit_targets=[],
                risk_reward=0,
                entry_price=entry_price,
                entry_type="MARKET",
            )

        # Calculate take profit targets
        try:
            take_profit_targets = StopLossCalculator.calculate_take_profit_targets(
                entry_price=entry_price, atr=atr, risk_reward_ratios=[1.5, 3.0, 5.0]
            )
        except ValueError as e:
            return PriceCalculationResult(
                success=False,
                error_message=f"Take profit calculation failed: {str(e)}",
                stop_loss=0,
                reward=0,
                take_profit_targets=[],
                risk_reward=0,
                entry_price=entry_price,
                entry_type="MARKET",
            )

        # Validate take profit targets
        if len(take_profit_targets) < 2:
            return PriceCalculationResult(
                success=False,
                error_message="Không đủ take profit targets để tính reward",
                stop_loss=stop_loss,
                reward=0,
                take_profit_targets=take_profit_targets,
                risk_reward=0,
                entry_price=entry_price,
                entry_type="MARKET",
            )

        # Use TP2 (second target) for R:R calculation
        take_profit = take_profit_targets[1]

        # Transaction cost calculations
        entry_cost_pct = TOTAL_TRANSACTION_COST + DEFAULT_SLIPPAGE
        entry_cost = entry_price * entry_cost_pct

        exit_cost_pct = TOTAL_TRANSACTION_COST + DEFAULT_SLIPPAGE
        exit_cost = take_profit * exit_cost_pct

        stop_loss_exit_cost = stop_loss * (TOTAL_TRANSACTION_COST + DEFAULT_SLIPPAGE)

        # Calculate risk and reward
        price_risk = entry_price - stop_loss
        risk = price_risk + entry_cost + stop_loss_exit_cost

        if risk <= 0:
            return PriceCalculationResult(
                success=False,
                error_message=f"Risk calculation error: risk={risk:.0f}",
                stop_loss=stop_loss,
                reward=0,
                take_profit_targets=take_profit_targets,
                risk_reward=0,
                entry_price=entry_price,
                entry_type="MARKET",
            )

        reward_before_costs = take_profit - entry_price
        reward = reward_before_costs - entry_cost - exit_cost

        if reward <= 0:
            return PriceCalculationResult(
                success=False,
                error_message=f"Reward không hợp lệ sau khi trừ phí: {reward:.0f}",
                stop_loss=stop_loss,
                reward=0,
                take_profit_targets=take_profit_targets,
                risk_reward=0,
                entry_price=entry_price,
                entry_type="MARKET",
            )

        # Calculate R:R ratio
        risk_reward = reward / risk

        if risk_reward < self.min_risk_reward:
            return PriceCalculationResult(
                success=False,
                error_message=f"R:R ratio thấp: {risk_reward:.2f} < {self.min_risk_reward:.2f}",
                stop_loss=stop_loss,
                reward=reward,
                take_profit_targets=take_profit_targets,
                risk_reward=risk_reward,
                entry_price=entry_price,
                entry_type="MARKET",
            )

        logger.debug(
            f"✅ R:R calculation: " f"risk={risk:.0f}, reward={reward:.0f}, R:R={risk_reward:.2f}"
        )

        return PriceCalculationResult(
            success=True,
            error_message="",
            stop_loss=stop_loss,
            reward=reward,
            take_profit_targets=take_profit_targets,
            risk_reward=risk_reward,
            entry_price=entry_price,
            entry_type="MARKET",
        )

    def optimize_entry_price(
        self,
        df: pd.DataFrame,
        current_price: float,
        sr_check: Dict,
        trend_check: Dict,
        volatility_check: Dict,
    ) -> Dict:
        """
        Optimize entry price based on market conditions.

        Uses support/resistance, trend, and volatility to determine:
        1. Market order (immediate entry)
        2. Limit order at better price
        3. Wait for pullback

        Vietnam market specifics:
        - ATO (9:00-9:15): Higher volatility, avoid limit orders
        - Regular session: Limit orders work well
        - ATC (14:30-14:45): Closing auction, market orders preferred

        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            sr_check: Support/resistance analysis
            trend_check: Trend alignment analysis
            volatility_check: Volatility analysis

        Returns:
            Dict with entry_price, entry_type, optimization_reason
        """
        # Default: market order at current price
        result = {
            "entry_price": current_price,
            "entry_type": "MARKET",
            "optimization_reason": None,
        }

        if len(df) < 20:
            return result

        atr = safe_get_latest(df, "atr", 0)
        if atr == 0:
            return result

        support = sr_check.get("support_level", 0)
        resistance = sr_check.get("resistance_level", 0)
        near_support = sr_check.get("near_support", False)
        bouncing = sr_check.get("bouncing_from_support", False)

        trend_strength = trend_check.get("strength", 50)
        volatility = volatility_check.get("value", 0)

        # High volatility: use market order to avoid missing entry
        if volatility > 4:
            result["optimization_reason"] = "High volatility - market order recommended"
            return result

        # Strong trend: use market order to catch momentum
        if trend_strength >= 75:
            result["optimization_reason"] = "Strong trend - market order to catch momentum"
            return result

        # Near support and bouncing: limit order slightly above support
        if bouncing and support > 0:
            limit_price = support * 1.01  # 1% above support
            if limit_price < current_price:
                result["entry_price"] = limit_price
                result["entry_type"] = "LIMIT"
                result["optimization_reason"] = f"Limit order at support bounce: {limit_price:.0f}"
                return result

        # Normal conditions: limit order with small discount
        if trend_strength >= 50:
            limit_price = current_price * (1 - self.limit_order_discount)
            # Ensure limit price is above support
            if support > 0 and limit_price < support * 1.02:
                limit_price = support * 1.02

            if limit_price < current_price:
                result["entry_price"] = limit_price
                result["entry_type"] = "LIMIT"
                result["optimization_reason"] = f"Limit order for better entry: {limit_price:.0f}"

        return result

    def get_atr_based_limit_threshold(self, df: pd.DataFrame, current_price: float) -> float:
        """
        Calculate dynamic limit order threshold based on ATR.

        Higher volatility → larger discount for limit orders
        Lower volatility → smaller discount

        Args:
            df: DataFrame with ATR data
            current_price: Current market price

        Returns:
            Threshold percentage for limit order (0.5% to 2%)
        """
        atr = safe_get_latest(df, "atr", 0)
        if atr == 0 or current_price == 0:
            return self.limit_order_discount

        volatility_pct = (atr / current_price) * 100

        # Map volatility to discount
        # Low vol (< 2%): 0.5% discount
        # Medium vol (2-3%): 1% discount
        # High vol (3-4%): 1.5% discount
        # Very high vol (> 4%): 2% discount
        if volatility_pct < 2:
            return 0.005
        elif volatility_pct < 3:
            return 0.01
        elif volatility_pct < 4:
            return 0.015
        else:
            return 0.02


class RiskRewardCalculator:
    """
    Calculates risk/reward metrics with transaction cost awareness.
    """

    def __init__(
        self,
        commission_rate: float = 0.0015,  # 0.15%
        slippage_rate: float = 0.001,  # 0.1%
        min_risk_reward: float = 1.5,
    ):
        """
        Initialize calculator.

        Args:
            commission_rate: Trading commission rate
            slippage_rate: Expected slippage rate
            min_risk_reward: Minimum acceptable R:R ratio
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_risk_reward = min_risk_reward
        self.total_cost_rate = commission_rate + slippage_rate

    def calculate_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        include_costs: bool = True,
    ) -> Dict:
        """
        Calculate risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            include_costs: Whether to include transaction costs

        Returns:
            Dict with risk, reward, ratio, and breakdown
        """
        price_risk = entry_price - stop_loss
        price_reward = take_profit - entry_price

        if include_costs:
            entry_cost = entry_price * self.total_cost_rate
            exit_cost_win = take_profit * self.total_cost_rate
            exit_cost_loss = stop_loss * self.total_cost_rate

            total_risk = price_risk + entry_cost + exit_cost_loss
            total_reward = price_reward - entry_cost - exit_cost_win
        else:
            entry_cost = 0
            exit_cost_win = 0
            exit_cost_loss = 0
            total_risk = price_risk
            total_reward = price_reward

        if total_risk <= 0 or total_reward <= 0:
            return {
                "valid": False,
                "risk": total_risk,
                "reward": total_reward,
                "ratio": 0,
                "reason": "Invalid risk or reward value",
            }

        ratio = total_reward / total_risk

        return {
            "valid": ratio >= self.min_risk_reward,
            "risk": total_risk,
            "reward": total_reward,
            "ratio": ratio,
            "price_risk": price_risk,
            "price_reward": price_reward,
            "entry_cost": entry_cost,
            "exit_cost_win": exit_cost_win,
            "exit_cost_loss": exit_cost_loss,
            "meets_minimum": ratio >= self.min_risk_reward,
        }

    def calculate_position_size(
        self,
        account_value: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
    ) -> Dict:
        """
        Calculate position size based on risk parameters.

        Args:
            account_value: Total account value
            risk_percent: Maximum risk per trade (e.g., 0.02 = 2%)
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Dict with shares, position_value, risk_amount
        """
        risk_amount = account_value * risk_percent
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return {
                "shares": 0,
                "position_value": 0,
                "risk_amount": risk_amount,
                "valid": False,
                "reason": "Stop loss must be below entry price",
            }

        # Account for transaction costs in position sizing
        entry_cost_per_share = entry_price * self.total_cost_rate
        exit_cost_per_share = stop_loss * self.total_cost_rate
        total_risk_per_share = risk_per_share + entry_cost_per_share + exit_cost_per_share

        shares = int(risk_amount / total_risk_per_share)
        position_value = shares * entry_price

        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_per_share": total_risk_per_share,
            "valid": shares > 0,
        }
