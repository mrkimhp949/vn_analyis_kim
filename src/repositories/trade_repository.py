"""
Trade Repository

Implements Repository Pattern for Trade data access.

Usage:
    from src.repositories import TradeRepository
    from src.data.database import get_db_manager

    db = get_db_manager()
    repo = TradeRepository(db)

    # Get recent trades
    trades = repo.get_recent_trades(limit=10)

    # Get trades for a symbol
    symbol_trades = repo.get_by_symbol("VNM")

    # Create new trade
    trade_id = repo.create_trade({
        "symbol": "VNM",
        "action": "BUY",
        "shares": 100,
        "price": 100000,
        ...
    })
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade entity"""

    id: Optional[int] = None
    symbol: str = ""
    action: str = ""  # BUY, SELL
    shares: int = 0
    price: float = 0.0
    total_value: float = 0.0
    commission: float = 0.0
    trade_date: Optional[datetime] = None
    strategy: str = ""
    notes: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    @staticmethod
    def from_db_row(row: Dict[str, Any]) -> "Trade":
        """Create Trade from database row"""
        return Trade(
            id=row.get("id"),
            symbol=row.get("symbol", ""),
            action=row.get("action", ""),
            shares=row.get("shares", 0),
            price=row.get("price", 0.0),
            total_value=row.get("total_value", 0.0),
            commission=row.get("commission", 0.0),
            trade_date=row.get("trade_date"),
            strategy=row.get("strategy", ""),
            notes=row.get("notes"),
            pnl=row.get("pnl"),
            pnl_percent=row.get("pnl_percent"),
        )


class TradeRepository:
    """
    Repository for Trade data access

    Provides all database operations for trades.
    """

    def __init__(self, db_manager):
        """
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """
        Create new trade record

        Args:
            trade_data: Dict with trade information

        Returns:
            Trade ID if created, None if failed
        """
        query = """
            INSERT INTO trades (
                symbol, action, shares, price, total_value,
                commission, trade_date, strategy, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            trade_data.get("symbol"),
            trade_data.get("action"),
            trade_data.get("shares"),
            trade_data.get("price"),
            trade_data.get("total_value"),
            trade_data.get("commission", 0),
            trade_data.get("trade_date", datetime.now()),
            trade_data.get("strategy", ""),
            trade_data.get("notes", ""),
        )

        try:
            trade_id = self.db.execute_update(query, params)
            logger.info(
                f"✅ Created trade: {trade_data.get('action')} {trade_data.get('shares')} "
                f"{trade_data.get('symbol')} @ {trade_data.get('price'):,.0f}"
            )
            return trade_id
        except Exception as e:
            logger.error(f"Error creating trade: {e}")
            return None

    def get_recent_trades(self, limit: int = 100) -> List[Trade]:
        """
        Get recent trades

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of Trade objects
        """
        query = """
            SELECT * FROM trades
            ORDER BY trade_date DESC
            LIMIT ?
        """

        try:
            rows = self.db.execute_query(query, (limit,))
            trades = [Trade.from_db_row(dict(row)) for row in rows]
            logger.debug(f"Retrieved {len(trades)} recent trades")
            return trades
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

    def get_by_symbol(self, symbol: str, limit: int = 50) -> List[Trade]:
        """
        Get trades for a specific symbol

        Args:
            symbol: Stock symbol
            limit: Maximum number of trades to return

        Returns:
            List of Trade objects
        """
        query = """
            SELECT * FROM trades
            WHERE symbol = ?
            ORDER BY trade_date DESC
            LIMIT ?
        """

        try:
            rows = self.db.execute_query(query, (symbol, limit))
            return [Trade.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {e}")
            return []

    def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trade]:
        """
        Get trades within date range

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of Trade objects
        """
        query = """
            SELECT * FROM trades
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date DESC
        """

        try:
            rows = self.db.execute_query(query, (start_date, end_date))
            return [Trade.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error getting trades by date range: {e}")
            return []

    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trade statistics for recent period

        Args:
            days: Number of days to analyze

        Returns:
            Dict with statistics
        """
        since_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sells,
                SUM(CASE WHEN action = 'BUY' THEN total_value ELSE 0 END) as total_bought,
                SUM(CASE WHEN action = 'SELL' THEN total_value ELSE 0 END) as total_sold,
                SUM(commission) as total_commission,
                AVG(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END) as avg_pnl
            FROM trades
            WHERE trade_date >= ?
        """

        try:
            rows = self.db.execute_query(query, (since_date,))
            if rows:
                row = dict(rows[0])
                return {
                    "total_trades": row.get("total_trades", 0),
                    "buys": row.get("buys", 0),
                    "sells": row.get("sells", 0),
                    "total_bought": row.get("total_bought", 0) or 0.0,
                    "total_sold": row.get("total_sold", 0) or 0.0,
                    "total_commission": row.get("total_commission", 0) or 0.0,
                    "avg_pnl": row.get("avg_pnl", 0) or 0.0,
                    "period_days": days,
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting trade statistics: {e}")
            return {}

    def get_win_rate(self, days: int = 90) -> Dict[str, float]:
        """
        Calculate win rate from closed trades

        Args:
            days: Number of days to analyze

        Returns:
            Dict with win rate statistics
        """
        since_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as avg_win,
                AVG(CASE WHEN pnl <= 0 THEN pnl ELSE 0 END) as avg_loss
            FROM trades
            WHERE action = 'SELL' AND pnl IS NOT NULL AND trade_date >= ?
        """

        try:
            rows = self.db.execute_query(query, (since_date,))
            if rows:
                row = dict(rows[0])
                total = row.get("total", 0)
                wins = row.get("wins", 0)
                losses = row.get("losses", 0)
                avg_win = row.get("avg_win", 0) or 0.0
                avg_loss = abs(row.get("avg_loss", 0) or 0.0)

                win_rate = (wins / total) if total > 0 else 0.0
                win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

                return {
                    "total_trades": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "win_loss_ratio": win_loss_ratio,
                }
            return {}
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return {}
