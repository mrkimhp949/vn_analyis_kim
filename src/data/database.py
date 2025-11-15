"""
SQLite Database for Trading Bot
Uses a dedicated DatabaseManager for safe concurrent access.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from db_manager import db_manager  # Import the singleton instance


class TradingDB:
    """
    Trading database with SQLite.
    All database operations are delegated to the DatabaseManager.
    """

    def __init__(self):
        self.db_manager = db_manager
        self.create_tables()

    def create_tables(self):
        """Create all necessary tables if they don't exist."""

        # Use a list of queries to be executed by the writer thread
        queries = [
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                shares INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                entry_date TEXT NOT NULL,
                entry_value REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_value REAL NOT NULL,
                total_cost REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                num_positions INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                shares INTEGER NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                trade_date TEXT NOT NULL,
                reason TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                shares INTEGER NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                trade_date TEXT NOT NULL,
                pnl REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS signals_cache (
                symbol TEXT PRIMARY KEY,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_history(date)",
        ]

        for query in queries:
            self.db_manager.execute_write(query)

    # ===== POSITIONS =====

    def get_positions(self) -> Dict[str, Dict]:
        """Get all active positions using the read connection."""
        rows = self.db_manager.execute_read("SELECT * FROM positions")
        positions = {}
        for row in rows:
            # Assuming row is a tuple from sqlite3
            (
                symbol,
                shares,
                avg_price,
                entry_date,
                entry_value,
                stop_loss,
                take_profit,
                metadata_str,
                _,
                _,
            ) = row
            positions[symbol] = {
                "shares": shares,
                "avg_price": avg_price,
                "entry_date": entry_date,
                "entry_value": entry_value,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "metadata": json.loads(metadata_str) if metadata_str else {},
            }
        return positions

    def save_position(
        self,
        symbol: str,
        shares: int,
        avg_price: float,
        entry_date: str,
        entry_value: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Save or update a position via the write queue."""
        query = """
            INSERT OR REPLACE INTO positions 
            (symbol, shares, avg_price, entry_date, entry_value, stop_loss, take_profit, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        params = (
            symbol,
            shares,
            avg_price,
            entry_date,
            entry_value,
            stop_loss,
            take_profit,
            json.dumps(metadata) if metadata else None,
        )
        self.db_manager.execute_write(query, params)

    def delete_position(self, symbol: str):
        """Delete a position via the write queue."""
        query = "DELETE FROM positions WHERE symbol = ?"
        params = (symbol,)
        self.db_manager.execute_write(query, params)

    # ===== PORTFOLIO HISTORY =====

    def save_portfolio_snapshot(
        self,
        date: str,
        total_value: float,
        total_cost: float,
        pnl: float,
        pnl_percent: float,
        num_positions: int,
        metadata: Optional[Dict] = None,
    ):
        """Save portfolio snapshot via the write queue."""
        query = """
            INSERT INTO portfolio_history 
            (date, total_value, total_cost, pnl, pnl_percent, num_positions, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            date,
            total_value,
            total_cost,
            pnl,
            pnl_percent,
            num_positions,
            json.dumps(metadata) if metadata else None,
        )
        self.db_manager.execute_write(query, params)

    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio history using the read connection."""
        query = "SELECT * FROM portfolio_history ORDER BY date DESC LIMIT ?"
        params = (days,)
        rows = self.db_manager.execute_read(query, params)
        # Convert tuple rows to dicts
        return [
            {
                "id": r[0],
                "date": r[1],
                "total_value": r[2],
                "total_cost": r[3],
                "pnl": r[4],
                "pnl_percent": r[5],
                "num_positions": r[6],
                "metadata": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    def get_last_portfolio_snapshot(self) -> Optional[Dict]:
        """Get the most recent portfolio snapshot using the read connection."""
        query = "SELECT * FROM portfolio_history ORDER BY date DESC LIMIT 1"
        rows = self.db_manager.execute_read(query)
        if not rows:
            return None

        r = rows[0]
        return {
            "id": r[0],
            "date": r[1],
            "total_value": r[2],
            "total_cost": r[3],
            "pnl": r[4],
            "pnl_percent": r[5],
            "num_positions": r[6],
            "metadata": json.loads(r[7]) if r[7] else {},
        }

    # ===== TRADES =====

    def save_trade(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        total_value: float,
        trade_date: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Save a trade via the write queue."""
        query = """
            INSERT INTO trades 
            (symbol, action, shares, price, total_value, trade_date, reason, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            symbol,
            action,
            shares,
            price,
            total_value,
            trade_date,
            reason,
            json.dumps(metadata) if metadata else None,
        )
        self.db_manager.execute_write(query, params)

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trade history using the read connection."""
        if symbol:
            query = (
                "SELECT * FROM trades WHERE symbol = ? ORDER BY trade_date DESC LIMIT ?"
            )
            params = (symbol, limit)
        else:
            query = "SELECT * FROM trades ORDER BY trade_date DESC LIMIT ?"
            params = (limit,)

        rows = self.db_manager.execute_read(query, params)
        # Convert tuple rows to dicts
        return [
            {
                "id": r[0],
                "symbol": r[1],
                "action": r[2],
                "shares": r[3],
                "price": r[4],
                "total_value": r[5],
                "trade_date": r[6],
                "reason": r[7],
                "metadata": json.loads(r[8]) if r[8] else {},
            }
            for r in rows
        ]

    # ===== SIGNALS CACHE =====

    def save_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        reason: Optional[str] = None,
        metadata: Optional[Dict] = None,
        ttl_hours: int = 24,
    ):
        """Save signal to cache via the write queue."""
        from datetime import timedelta

        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
        query = """
            INSERT OR REPLACE INTO signals_cache
            (symbol, signal, confidence, reason, metadata, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            symbol,
            signal,
            confidence,
            reason,
            json.dumps(metadata) if metadata else None,
            expires_at,
        )
        self.db_manager.execute_write(query, params)

    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get cached signal if not expired using the read connection."""
        query = "SELECT * FROM signals_cache WHERE symbol = ? AND expires_at > ?"
        params = (symbol, datetime.now().isoformat())

        rows = self.db_manager.execute_read(query, params)
        if not rows:
            return None

        row = rows[0]
        return {
            "symbol": row[0],
            "signal": row[1],
            "confidence": row[2],
            "reason": row[3],
            "metadata": json.loads(row[4]) if row[4] else {},
            "created_at": row[5],
        }

    def clear_expired_signals(self):
        """Clear expired signals via the write queue."""
        query = "DELETE FROM signals_cache WHERE expires_at <= ?"
        params = (datetime.now().isoformat(),)
        self.db_manager.execute_write(query, params)
        # Note: We can't return rowcount directly in this async model,
        # but we could implement a callback if needed.

    # ===== MIGRATION (can be run standalone) =====

    def migrate_from_json(self, json_files: Dict[str, str]):
        """
        Migrate data from JSON files to SQLite.
        This should be run as a separate script, not during normal bot operation.
        """
        import os

        print("Starting JSON migration...")
        for table, filepath in json_files.items():
            if not os.path.exists(filepath):
                print(f"⏭️ {filepath} not found, skipping")
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if table == "positions" and isinstance(data, dict):
                    for symbol, pos in data.items():
                        self.save_position(
                            symbol=symbol,
                            shares=pos.get("shares", 0),
                            avg_price=pos.get("avg_price", 0),
                            entry_date=pos.get(
                                "entry_date", datetime.now().isoformat()
                            ),
                            entry_value=pos.get("entry_value", 0),
                            stop_loss=pos.get("stop_loss"),
                            take_profit=pos.get("take_profit"),
                            metadata=pos,
                        )
                    print(
                        f"✅ Queued migration for {len(data)} positions from {filepath}"
                    )

                elif table == "portfolio_history" and isinstance(data, list):
                    for snapshot in data:
                        self.save_portfolio_snapshot(
                            date=snapshot.get("date", datetime.now().isoformat()),
                            total_value=snapshot.get("total_value", 0),
                            total_cost=snapshot.get("total_cost", 0),
                            pnl=snapshot.get("pnl", 0),
                            pnl_percent=snapshot.get("pnl_percent", 0),
                            num_positions=snapshot.get("num_positions", 0),
                            metadata=snapshot,
                        )
                    print(
                        f"✅ Queued migration for {len(data)} snapshots from {filepath}"
                    )

            except Exception as e:
                print(f"❌ Error queuing migration for {filepath}: {e}")

        print(
            "Migration commands have been queued. Please allow time for the writer to process them."
        )


# Singleton instance
_db_instance = None


def get_db() -> TradingDB:
    """Get database singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradingDB()
    return _db_instance


# Test (can be run as a script)
if __name__ == "__main__":
    import time

    print("Testing database with DatabaseManager...")

    db = get_db()

    # Test save position (write)
    print("Saving positions for VCB and FPT...")
    db.save_position("VCB", 100, 60000, "2025-11-01", 6000000)
    db.save_position("FPT", 50, 100000, "2025-11-02", 5000000)

    # Test save trade (write)
    print("Saving a trade for VCB...")
    db.save_trade("VCB", "BUY", 100, 60000, 6000000, "2025-11-01", "Entry signal")

    # Give the writer thread a moment to process the writes
    print("Waiting for writer thread to process...")
    time.sleep(2)

    # Test get positions (read)
    print("Fetching positions...")
    positions = db.get_positions()
    print(f"Positions: {positions}")
    assert "VCB" in positions
    assert "FPT" in positions

    # Test get trades (read)
    print("Fetching trades...")
    trades = db.get_trades()
    print(f"Trades: {trades}")
    assert len(trades) > 0

    # Test delete (write)
    print("Deleting position FPT...")
    db.delete_position("FPT")

    time.sleep(1)

    # Test get positions again (read)
    print("Fetching positions after delete...")
    positions = db.get_positions()
    print(f"Positions: {positions}")
    assert "FPT" not in positions

    print("\n✅ Database test with DatabaseManager completed!")

    # Important: In a real app, you'd call this on shutdown.
    db.db_manager.close()
