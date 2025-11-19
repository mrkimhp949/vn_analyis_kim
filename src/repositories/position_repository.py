"""
Position Repository

Implements Repository Pattern for Position data access.

This abstracts database operations from business logic, making the code:
- Easier to test (can inject mock repository)
- Easier to maintain (all queries in one place)
- Easier to optimize (can add caching, batch operations, etc.)

Usage:
    from src.repositories import PositionRepository
    from src.data.database import get_db_manager

    db = get_db_manager()
    repo = PositionRepository(db)

    # Get all active positions
    positions = repo.get_all_active()

    # Get position by symbol
    position = repo.get_by_symbol("VNM")

    # Update position
    repo.update_position(position_id, current_price=110000)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Position:
    """
    Position entity

    Represents a trading position with all its properties.
    """

    def __init__(
        self,
        id: Optional[int] = None,
        symbol: str = "",
        shares: int = 0,
        average_price: float = 0.0,
        current_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        entry_date: Optional[datetime] = None,
        status: str = "ACTIVE",
        strategy: str = "",
        sector: Optional[str] = None,
        notes: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id
        self.symbol = symbol
        self.shares = shares
        self.average_price = average_price
        self.current_price = current_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_date = entry_date or datetime.now()
        self.status = status
        self.strategy = strategy
        self.sector = sector
        self.notes = notes
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    @property
    def value(self) -> float:
        """Current value of position"""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost of position"""
        return self.shares * self.average_price

    @property
    def pnl(self) -> float:
        """Profit/Loss in VND"""
        return self.value - self.cost_basis

    @property
    def pnl_percent(self) -> float:
        """Profit/Loss percentage"""
        if self.cost_basis > 0:
            return (self.pnl / self.cost_basis) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "shares": self.shares,
            "average_price": self.average_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "status": self.status,
            "strategy": self.strategy,
            "sector": self.sector,
            "notes": self.notes,
            "value": self.value,
            "cost_basis": self.cost_basis,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
        }

    @staticmethod
    def from_db_row(row: Dict[str, Any]) -> "Position":
        """Create Position from database row"""
        return Position(
            id=row.get("id"),
            symbol=row.get("symbol", ""),
            shares=row.get("shares", 0),
            average_price=row.get("average_price", 0.0),
            current_price=row.get("current_price", 0.0),
            stop_loss=row.get("stop_loss", 0.0),
            take_profit=row.get("take_profit", 0.0),
            entry_date=row.get("entry_date"),
            status=row.get("status", "ACTIVE"),
            strategy=row.get("strategy", ""),
            sector=row.get("sector"),
            notes=row.get("notes"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


class PositionRepository:
    """
    Repository for Position data access

    Provides all database operations for positions.
    """

    def __init__(self, db_manager):
        """
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def get_all_active(self) -> List[Position]:
        """
        Get all active positions

        Returns:
            List of Position objects
        """
        query = """
            SELECT * FROM positions
            WHERE status = 'ACTIVE'
            ORDER BY updated_at DESC
        """

        try:
            rows = self.db.execute_query(query)
            positions = [Position.from_db_row(dict(row)) for row in rows]
            logger.debug(f"Retrieved {len(positions)} active positions")
            return positions
        except Exception:
            logger.error("Error getting active positions")
            return []

    def get_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Get active position by symbol

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None if not found
        """
        query = """
            SELECT * FROM positions
            WHERE symbol = ? AND status = 'ACTIVE'
            LIMIT 1
        """

        try:
            rows = self.db.execute_query(query, (symbol,))
            if rows:
                position = Position.from_db_row(dict(rows[0]))
                logger.debug(f"Retrieved position for {symbol}")
                return position
            return None
        except Exception:
            logger.error(f"Error getting position for {symbol}")
            return None

    def get_by_id(self, position_id: int) -> Optional[Position]:
        """
        Get position by ID

        Args:
            position_id: Position ID

        Returns:
            Position object or None if not found
        """
        query = "SELECT * FROM positions WHERE id = ?"

        try:
            rows = self.db.execute_query(query, (position_id,))
            if rows:
                return Position.from_db_row(dict(rows[0]))
            return None
        except Exception:
            logger.error(f"Error getting position {position_id}")
            return None

    def create(self, position: Position) -> Optional[int]:
        """
        Create new position

        Args:
            position: Position object

        Returns:
            Position ID if created, None if failed
        """
        query = """
            INSERT INTO positions (
                symbol, shares, average_price, current_price,
                stop_loss, take_profit, entry_date, status,
                strategy, sector, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            position.symbol,
            position.shares,
            position.average_price,
            position.current_price,
            position.stop_loss,
            position.take_profit,
            position.entry_date,
            position.status,
            position.strategy,
            position.sector,
            position.notes,
        )

        try:
            position_id = self.db.execute_update(query, params)
            logger.info(f"✅ Created position {position.symbol} (ID: {position_id})")
            return position_id
        except Exception:
            logger.error(f"Error creating position {position.symbol}")
            return None

    def update_price(self, symbol: str, current_price: float) -> bool:
        """
        Update current price for a position

        Args:
            symbol: Stock symbol
            current_price: New price

        Returns:
            True if updated, False otherwise
        """
        query = """
            UPDATE positions
            SET current_price = ?, updated_at = CURRENT_TIMESTAMP
            WHERE symbol = ? AND status = 'ACTIVE'
        """

        try:
            self.db.execute_update(query, (current_price, symbol))
            logger.debug(f"Updated price for {symbol}: {current_price:,.0f}")
            return True
        except Exception:
            logger.error(f"Error updating price for {symbol}")
            return False

    def update_shares(self, symbol: str, shares: int, average_price: float) -> bool:
        """
        Update shares and average price (for adding to position)

        Args:
            symbol: Stock symbol
            shares: New share count
            average_price: New average price

        Returns:
            True if updated, False otherwise
        """
        query = """
            UPDATE positions
            SET shares = ?, average_price = ?, updated_at = CURRENT_TIMESTAMP
            WHERE symbol = ? AND status = 'ACTIVE'
        """

        try:
            self.db.execute_update(query, (shares, average_price, symbol))
            logger.info(f"✅ Updated {symbol}: {shares} shares @ {average_price:,.0f}")
            return True
        except Exception:
            logger.error(f"Error updating shares for {symbol}")
            return False

    def close_position(self, symbol: str, exit_price: float) -> bool:
        """
        Close a position (mark as CLOSED)

        Args:
            symbol: Stock symbol
            exit_price: Exit price

        Returns:
            True if closed, False otherwise
        """
        query = """
            UPDATE positions
            SET status = 'CLOSED', current_price = ?, updated_at = CURRENT_TIMESTAMP
            WHERE symbol = ? AND status = 'ACTIVE'
        """

        try:
            self.db.execute_update(query, (exit_price, symbol))
            logger.info(f"✅ Closed position {symbol} @ {exit_price:,.0f}")
            return True
        except Exception:
            logger.error(f"Error closing position {symbol}")
            return False

    def get_total_value(self) -> float:
        """
        Get total value of all active positions

        Returns:
            Total value in VND
        """
        query = """
            SELECT SUM(shares * current_price) as total
            FROM positions
            WHERE status = 'ACTIVE'
        """

        try:
            rows = self.db.execute_query(query)
            if rows and rows[0]["total"]:
                return float(rows[0]["total"])
            return 0.0
        except Exception:
            logger.error("Error getting total value")
            return 0.0

    def get_by_sector(self, sector: str) -> List[Position]:
        """
        Get all active positions in a sector

        Args:
            sector: Sector name

        Returns:
            List of Position objects
        """
        query = """
            SELECT * FROM positions
            WHERE sector = ? AND status = 'ACTIVE'
            ORDER BY symbol
        """

        try:
            rows = self.db.execute_query(query, (sector,))
            return [Position.from_db_row(dict(row)) for row in rows]
        except Exception:
            logger.error(f"Error getting positions for sector {sector}")
            return []

    def get_sector_exposure(self, sector: str) -> float:
        """
        Get total exposure (value) for a sector

        Args:
            sector: Sector name

        Returns:
            Total value in VND
        """
        query = """
            SELECT SUM(shares * current_price) as total
            FROM positions
            WHERE sector = ? AND status = 'ACTIVE'
        """

        try:
            rows = self.db.execute_query(query, (sector,))
            if rows and rows[0]["total"]:
                return float(rows[0]["total"])
            return 0.0
        except Exception:
            logger.error(f"Error getting sector exposure for {sector}")
            return 0.0

    def batch_update_prices(self, price_updates: Dict[str, float]) -> int:
        """
        Batch update prices for multiple positions

        Args:
            price_updates: Dict of {symbol: current_price}

        Returns:
            Number of positions updated
        """
        updated_count = 0

        # Start a transaction for better performance
        for symbol, price in price_updates.items():
            if self.update_price(symbol, price):
                updated_count += 1

        logger.info(f"✅ Batch updated {updated_count} positions")
        return updated_count
