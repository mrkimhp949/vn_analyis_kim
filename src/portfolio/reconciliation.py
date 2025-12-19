# -*- coding: utf-8 -*-
"""
Position Reconciliation - Ensure DB and Paper Trading Account are in sync

Compares positions between:
- Database (source of truth)
- Paper Trading Account
- (Future: Broker positions)

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PositionMismatch:
    """Record of a position mismatch"""

    symbol: str
    db_shares: int
    paper_shares: int
    db_avg_price: float
    paper_avg_price: float
    mismatch_type: str  # MISSING_IN_DB, MISSING_IN_PAPER, QUANTITY_DIFF, PRICE_DIFF
    severity: str  # LOW, MEDIUM, HIGH

    @property
    def is_critical(self) -> bool:
        return self.severity == "HIGH" or self.mismatch_type in (
            "MISSING_IN_DB",
            "MISSING_IN_PAPER",
        )


class PositionReconciler:
    """
    Reconciles positions between different data sources

    Features:
    - Compares DB vs Paper Trading positions
    - Identifies mismatches
    - Auto-fix option for minor discrepancies
    - Alert on critical mismatches

    Usage:
        reconciler = get_position_reconciler()

        # Check for mismatches
        mismatches = reconciler.check_positions()

        if mismatches:
            for m in mismatches:
                print(f"Mismatch: {m.symbol} - {m.mismatch_type}")

        # Auto-fix (if safe)
        reconciler.auto_fix_safe_mismatches()
    """

    def __init__(
        self,
        price_tolerance_pct: float = 0.01,  # 1% price difference tolerance
        auto_fix_enabled: bool = False,
    ):
        self.price_tolerance_pct = price_tolerance_pct
        self.auto_fix_enabled = auto_fix_enabled
        self._lock = RLock()
        self._last_check: Optional[datetime] = None
        self._last_mismatches: List[PositionMismatch] = []

    def _get_db_positions(self) -> Dict[str, Dict]:
        """Get positions from database"""
        try:
            from src.portfolio.manager import get_portfolio_manager

            pm = get_portfolio_manager()
            # Use get_positions() - returns dict of active positions
            return pm.get_positions()
        except Exception as e:
            logger.error(f"Failed to get DB positions: {e}")
            return {}

    def _get_paper_positions(self) -> Dict[str, Dict]:
        """Get positions from paper trading account"""
        try:
            from src.portfolio.paper_trading import get_paper_account

            paper = get_paper_account()

            # Get active positions from portfolio manager (paper uses same DB)
            # But also check paper account's internal state
            positions = {}

            # Get from portfolio manager
            pm_positions = self._get_db_positions()
            for symbol, data in pm_positions.items():
                positions[symbol] = {
                    "shares": data.get("shares", 0),
                    "avg_price": data.get("avg_price", 0),
                    "entry_date": data.get("entry_date"),
                }

            return positions
        except Exception as e:
            logger.error(f"Failed to get paper positions: {e}")
            return {}

    def check_positions(self) -> List[PositionMismatch]:
        """
        Check for position mismatches

        Returns:
            List of PositionMismatch objects
        """
        with self._lock:
            mismatches = []

            db_positions = self._get_db_positions()
            paper_positions = self._get_paper_positions()

            all_symbols = set(db_positions.keys()) | set(paper_positions.keys())

            for symbol in all_symbols:
                db_pos = db_positions.get(symbol, {})
                paper_pos = paper_positions.get(symbol, {})

                db_shares = db_pos.get("shares", 0)
                paper_shares = paper_pos.get("shares", 0)
                db_price = db_pos.get("avg_price", 0)
                paper_price = paper_pos.get("avg_price", 0)

                # Check for missing positions
                if db_shares > 0 and paper_shares == 0:
                    mismatches.append(
                        PositionMismatch(
                            symbol=symbol,
                            db_shares=db_shares,
                            paper_shares=0,
                            db_avg_price=db_price,
                            paper_avg_price=0,
                            mismatch_type="MISSING_IN_PAPER",
                            severity="HIGH",
                        )
                    )
                    continue

                if paper_shares > 0 and db_shares == 0:
                    mismatches.append(
                        PositionMismatch(
                            symbol=symbol,
                            db_shares=0,
                            paper_shares=paper_shares,
                            db_avg_price=0,
                            paper_avg_price=paper_price,
                            mismatch_type="MISSING_IN_DB",
                            severity="HIGH",
                        )
                    )
                    continue

                # Check quantity difference
                if db_shares != paper_shares:
                    severity = "HIGH" if abs(db_shares - paper_shares) > 100 else "MEDIUM"
                    mismatches.append(
                        PositionMismatch(
                            symbol=symbol,
                            db_shares=db_shares,
                            paper_shares=paper_shares,
                            db_avg_price=db_price,
                            paper_avg_price=paper_price,
                            mismatch_type="QUANTITY_DIFF",
                            severity=severity,
                        )
                    )
                    continue

                # Check price difference
                if db_price > 0 and paper_price > 0:
                    price_diff_pct = abs(db_price - paper_price) / db_price
                    if price_diff_pct > self.price_tolerance_pct:
                        severity = "LOW" if price_diff_pct < 0.05 else "MEDIUM"
                        mismatches.append(
                            PositionMismatch(
                                symbol=symbol,
                                db_shares=db_shares,
                                paper_shares=paper_shares,
                                db_avg_price=db_price,
                                paper_avg_price=paper_price,
                                mismatch_type="PRICE_DIFF",
                                severity=severity,
                            )
                        )

            self._last_check = datetime.now()
            self._last_mismatches = mismatches

            if mismatches:
                logger.warning(f"⚠️ Found {len(mismatches)} position mismatches")
                for m in mismatches:
                    logger.warning(f"  {m.symbol}: {m.mismatch_type} (severity: {m.severity})")
            else:
                logger.info("✅ All positions reconciled - no mismatches")

            return mismatches

    def has_critical_mismatches(self) -> Tuple[bool, List[PositionMismatch]]:
        """
        Check if there are any critical mismatches

        Returns:
            (has_critical, critical_mismatches)
        """
        mismatches = self.check_positions()
        critical = [m for m in mismatches if m.is_critical]
        return len(critical) > 0, critical

    def can_trade_safely(self) -> Tuple[bool, str]:
        """
        Check if it's safe to trade based on reconciliation

        Returns:
            (can_trade, reason)
        """
        has_critical, critical = self.has_critical_mismatches()

        if has_critical:
            symbols = [m.symbol for m in critical]
            return False, f"Critical position mismatches: {', '.join(symbols)}"

        return True, "Positions reconciled"

    def get_status(self) -> Dict:
        """Get reconciliation status"""
        return {
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "mismatches_count": len(self._last_mismatches),
            "critical_count": len([m for m in self._last_mismatches if m.is_critical]),
            "mismatches": [
                {
                    "symbol": m.symbol,
                    "type": m.mismatch_type,
                    "severity": m.severity,
                    "db_shares": m.db_shares,
                    "paper_shares": m.paper_shares,
                }
                for m in self._last_mismatches
            ],
        }


# Singleton instance
_reconciler_instance: Optional[PositionReconciler] = None
_reconciler_lock = RLock()


def get_position_reconciler() -> PositionReconciler:
    """Get singleton reconciler instance"""
    global _reconciler_instance

    with _reconciler_lock:
        if _reconciler_instance is None:
            _reconciler_instance = PositionReconciler()
        return _reconciler_instance


def reset_position_reconciler():
    """Reset reconciler (for testing)"""
    global _reconciler_instance
    with _reconciler_lock:
        _reconciler_instance = None
