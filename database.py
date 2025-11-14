"""
SQLite Database for Trading Bot
Replaces JSON files with proper database
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


class TradingDB:
    """
    Trading database with SQLite

    Tables:
    - positions: Active trading positions
    - portfolio_history: Historical portfolio snapshots
    - trades: Trade history
    - paper_trades: Paper trading history
    - signals: Trading signals cache
    """

    def __init__(self, db_path="trading.db"):
        self.db_path = db_path
        self._enable_wal_mode()
        self.create_tables()

    def _enable_wal_mode(self):
        """Enable WAL mode for better concurrency"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.close()
        except Exception as e:
            import logging

            logging.warning(f"Could not enable WAL mode: {e}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections

        Improvements:
        - Increased timeout to 10s to handle concurrent access
        - WAL mode enabled for better concurrent reads
        - Automatic retry on database locked errors
        """
        max_retries = 3
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                # Increased timeout from default 5s to 10s
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                conn.row_factory = sqlite3.Row  # Return rows as dicts

                try:
                    yield conn
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    conn.rollback()
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        import time

                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time

                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise

    def create_tables(self):
        """Create all tables"""
        with self.get_connection() as conn:
            # Positions table
            conn.execute(
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
            """
            )

            # Portfolio history table
            conn.execute(
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
            """
            )

            # Trades table
            conn.execute(
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
            """
            )

            # Paper trades table
            conn.execute(
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
            """
            )

            # Signals cache table
            conn.execute(
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
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_history(date)"
            )

    # ===== POSITIONS =====

    def get_positions(self) -> Dict[str, Dict]:
        """Get all active positions"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM positions")
            positions = {}
            for row in cursor:
                symbol = row["symbol"]
                positions[symbol] = {
                    "shares": row["shares"],
                    "avg_price": row["avg_price"],
                    "entry_date": row["entry_date"],
                    "entry_value": row["entry_value"],
                    "stop_loss": row["stop_loss"],
                    "take_profit": row["take_profit"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
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
        """Save or update a position"""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions 
                (symbol, shares, avg_price, entry_date, entry_value, stop_loss, take_profit, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    symbol,
                    shares,
                    avg_price,
                    entry_date,
                    entry_value,
                    stop_loss,
                    take_profit,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def delete_position(self, symbol: str):
        """Delete a position"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

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
        """Save portfolio snapshot"""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_history 
                (date, total_value, total_cost, pnl, pnl_percent, num_positions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date,
                    total_value,
                    total_cost,
                    pnl,
                    pnl_percent,
                    num_positions,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio history"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM portfolio_history 
                ORDER BY date DESC 
                LIMIT ?
            """,
                (days,),
            )
            return [dict(row) for row in cursor]

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
        """Save a trade"""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO trades 
                (symbol, action, shares, price, total_value, trade_date, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    action,
                    shares,
                    price,
                    total_value,
                    trade_date,
                    reason,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute(
                    """
                    SELECT * FROM trades 
                    WHERE symbol = ?
                    ORDER BY trade_date DESC 
                    LIMIT ?
                """,
                    (symbol, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM trades 
                    ORDER BY trade_date DESC 
                    LIMIT ?
                """,
                    (limit,),
                )
            return [dict(row) for row in cursor]

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
        """Save signal to cache"""
        from datetime import datetime, timedelta

        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO signals_cache
                (symbol, signal, confidence, reason, metadata, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    signal,
                    confidence,
                    reason,
                    json.dumps(metadata) if metadata else None,
                    expires_at,
                ),
            )

    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get cached signal if not expired"""
        from datetime import datetime

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM signals_cache
                WHERE symbol = ? AND expires_at > ?
            """,
                (symbol, datetime.now().isoformat()),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "symbol": row["symbol"],
                    "signal": row["signal"],
                    "confidence": row["confidence"],
                    "reason": row["reason"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                }
        return None

    def clear_expired_signals(self):
        """Clear expired signals"""
        from datetime import datetime

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM signals_cache
                WHERE expires_at <= ?
            """,
                (datetime.now().isoformat(),),
            )
            return cursor.rowcount

    # ===== MIGRATION =====

    def migrate_from_json(self, json_files: Dict[str, str]):
        """
        Migrate data from JSON files to SQLite

        Args:
            json_files: Dict mapping table name to JSON file path
                e.g. {'positions': 'active_positions.json'}
        """
        import os

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
                    print(f"✅ Migrated {len(data)} positions from {filepath}")

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
                    print(f"✅ Migrated {len(data)} snapshots from {filepath}")

            except Exception as e:
                print(f"❌ Error migrating {filepath}: {e}")


# Singleton instance
_db_instance = None


def get_db() -> TradingDB:
    """Get database singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradingDB()
    return _db_instance


# Test
if __name__ == "__main__":
    print("Testing database...")

    db = TradingDB("test_trading.db")

    # Test save position
    db.save_position("VCB", 100, 60000, "2025-11-01", 6000000)
    db.save_position("FPT", 50, 100000, "2025-11-02", 5000000)

    # Test get positions
    positions = db.get_positions()
    print(f"Positions: {positions}")

    # Test save trade
    db.save_trade("VCB", "BUY", 100, 60000, 6000000, "2025-11-01", "Entry signal")

    # Test get trades
    trades = db.get_trades()
    print(f"Trades: {trades}")

    print("\n✅ Database test completed!")
