"""Portfolio regime-based exposure control."""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RegimeAdjustment:
    regime: str
    target_cash_ratio: float
    target_exposure_ratio: float
    required_cash_increase: float
    suggested_sales: List[Dict]
    notes: Optional[str] = None


class PortfolioRegimeAdjuster:
    """Adjust portfolio exposure based on market regime signals."""

    def __init__(
        self,
        cash_targets: Optional[Dict[str, float]] = None,
        min_sell_lot: int = 1,
    ):
        self.cash_targets = cash_targets or {
            "BULL": 0.15,
            "SIDEWAYS": 0.35,
            "BEAR": 0.70,
            "HIGH_VOLATILITY": 0.60,
            "UNKNOWN": 0.50,
        }
        self.min_sell_lot = max(1, min_sell_lot)

    def _get_target_cash_ratio(self, regime: Optional[str]) -> float:
        if not regime:
            return self.cash_targets.get("UNKNOWN", 0.5)
        regime = regime.upper()
        return self.cash_targets.get(regime, self.cash_targets.get("UNKNOWN", 0.5))

    def evaluate_adjustment(
        self,
        holdings: Dict[str, Dict],
        market_regime: Optional[Dict] = None,
        current_cash: float = 0.0,
    ) -> RegimeAdjustment:
        total_value = float(
            sum(position.get("current_value", 0.0) for position in holdings.values())
        )
        effective_total = total_value + max(current_cash, 0.0)

        regime_name = (market_regime or {}).get("regime", "UNKNOWN")
        target_cash_ratio = self._get_target_cash_ratio(regime_name)
        target_cash = effective_total * target_cash_ratio
        current_cash = max(current_cash, 0.0)
        required_cash_increase = max(0.0, target_cash - current_cash)

        if required_cash_increase <= 1e-6:
            return RegimeAdjustment(
                regime=regime_name,
                target_cash_ratio=target_cash_ratio,
                target_exposure_ratio=1.0 - target_cash_ratio,
                required_cash_increase=0.0,
                suggested_sales=[],
                notes="Danh mục đã đáp ứng mục tiêu tiền mặt theo regime.",
            )

        sell_plan, shortfall = self._build_sell_plan(holdings, required_cash_increase)
        note = "Đề xuất bán để tăng tiền mặt phù hợp với regime."
        if shortfall > 0:
            note += f" Không đủ tài sản để giải ngân thêm {shortfall:,.0f} VNĐ."
        return RegimeAdjustment(
            regime=regime_name,
            target_cash_ratio=target_cash_ratio,
            target_exposure_ratio=1.0 - target_cash_ratio,
            required_cash_increase=required_cash_increase,
            suggested_sales=sell_plan,
            notes=note,
        )

    def _build_sell_plan(
        self,
        holdings: Dict[str, Dict],
        amount_needed: float,
    ) -> Tuple[List[Dict], float]:
        if amount_needed <= 0:
            return [], 0.0

        plan: List[Dict] = []
        remaining = amount_needed

        def _priority(item):
            symbol, data = item
            recommendation_score = 0
            if data.get("recommendation") == "SELL":
                recommendation_score += 2
            pnl = data.get("pnl_percent", 0.0)
            return (-recommendation_score, pnl)

        sorted_holdings = sorted(holdings.items(), key=_priority)

        for symbol, data in sorted_holdings:
            price = data.get("current_price") or 0.0
            shares = int(data.get("shares") or 0)
            if price <= 0 or shares <= 0 or remaining <= 0:
                continue

            max_value = price * shares
            if max_value <= 0:
                continue

            value_to_sell = min(max_value, remaining)
            shares_to_sell = math.ceil(value_to_sell / price)
            shares_to_sell = min(shares, max(self.min_sell_lot, shares_to_sell))

            value_realized = shares_to_sell * price
            remaining = max(0.0, remaining - value_realized)

            plan.append(
                {
                    "symbol": symbol,
                    "shares_to_sell": shares_to_sell,
                    "approx_value": value_realized,
                    "current_price": price,
                    "pnl_percent": data.get("pnl_percent"),
                    "recommendation": data.get("recommendation"),
                }
            )

            if remaining <= 0:
                break

        return plan, remaining

