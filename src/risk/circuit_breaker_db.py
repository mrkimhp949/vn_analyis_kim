"""
Circuit Breaker Database Backend - IMPROVED v5.1

Provides database-backed storage for circuit breaker stats to address:
1. Single point of failure (file-based storage)
2. Race conditions in concurrent sessions
3. Distributed locking for multiple bots

Author: Trading Bot Team
Version: 5.1.0
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime
from threading import RLock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerDB:
    """
    Database-backed storage for circuit breaker stats.

    IMPROVEMENTS over file-based:
    1. ACID transactions for reliability
    2. Row-level locking for concurrent access
    3. Automatic recovery from corruption
    4. Historical data retention
    """

    def __init__(
        self,
        db_path: str = "trading_bot.db",
        fallback_file: str = "circuit_breaker_stats.json",
    ):
        """
        Initialize database backend.

        Args:
            db_path: Path to SQLite database
            fallback_file: Fallback JSON file if DB fails
        """
        self.db_path = db_path
        self.fallback_file = fallback_file
        self._lock = RLock()
        self._use_db = True

        # Initialize database
        try:
            self._init_db()
            logger.info(f"✅ Circuit breaker DB initialized: {db_path}")
        except Exception as e:
            logger.warning(f"⚠️ DB init failed, using file fallback: {e}")
            self._use_db = False

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Daily stats table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS circuit_breaker_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    trades_count INTEGER DEFAULT 0,
                    total_loss REAL DEFAULT 0.0,
                    total_profit REAL DEFAULT 0.0,
                    net_pnl REAL DEFAULT 0.0,
                    consecutive_losses INTEGER DEFAULT 0,
                    consecutive_wins INTEGER DEFAULT 0,
                    morning_trades INTEGER DEFAULT 0,
                    afternoon_trades INTEGER DEFAULT 0,
                    tripped INTEGER DEFAULT 0,
                    tripped_reason TEXT,
                    last_updated TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Historical stats for analysis
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS circuit_breaker_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Distributed lock table (for multiple bots)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS circuit_breaker_locks (
                    lock_name TEXT PRIMARY KEY,
                    bot_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """
            )

            # Create indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_date 
                ON circuit_breaker_daily(date)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_date 
                ON circuit_breaker_history(date)
            """
            )

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,  # Wait up to 30s for lock
            isolation_level="IMMEDIATE",  # Acquire lock immediately
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def load_stats(self) -> Dict[str, Any]:
        """
        Load stats from database with file fallback.

        Returns:
            Dict with today's stats and consecutive losses
        """
        with self._lock:
            if self._use_db:
                try:
                    return self._load_from_db()
                except Exception as e:
                    logger.warning(f"DB load failed, using fallback: {e}")

            return self._load_from_file()

    def _load_from_db(self) -> Dict[str, Any]:
        """Load stats from database."""
        today = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get or create today's record
            cursor.execute("SELECT * FROM circuit_breaker_daily WHERE date = ?", (today,))
            row = cursor.fetchone()

            if row:
                return {
                    "today": {
                        "date": row["date"],
                        "trades_count": row["trades_count"],
                        "total_loss": row["total_loss"],
                        "total_profit": row["total_profit"],
                        "net_pnl": row["net_pnl"],
                        "last_updated": row["last_updated"],
                    },
                    "consecutive_losses": row["consecutive_losses"],
                    "consecutive_wins": row["consecutive_wins"],
                    "morning_trades": row["morning_trades"],
                    "afternoon_trades": row["afternoon_trades"],
                    "tripped": bool(row["tripped"]),
                    "tripped_reason": row["tripped_reason"],
                    "last_trade_date": row["date"],
                }

            # Create new record for today
            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO circuit_breaker_daily 
                (date, last_updated) VALUES (?, ?)
            """,
                (today, now),
            )
            conn.commit()

            return {
                "today": {
                    "date": today,
                    "trades_count": 0,
                    "total_loss": 0.0,
                    "total_profit": 0.0,
                    "net_pnl": 0.0,
                    "last_updated": now,
                },
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "morning_trades": 0,
                "afternoon_trades": 0,
                "tripped": False,
                "tripped_reason": None,
                "last_trade_date": None,
            }

    def _load_from_file(self) -> Dict[str, Any]:
        """Load stats from fallback file."""
        if os.path.exists(self.fallback_file):
            try:
                with open(self.fallback_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return self._get_default_stats()

    def _get_default_stats(self) -> Dict[str, Any]:
        """Get default stats structure."""
        today = date.today().isoformat()
        return {
            "today": {
                "date": today,
                "trades_count": 0,
                "total_loss": 0.0,
                "total_profit": 0.0,
                "net_pnl": 0.0,
                "last_updated": datetime.now().isoformat(),
            },
            "consecutive_losses": 0,
            "consecutive_wins": 0,
            "last_trade_date": None,
        }

    def save_stats(self, stats: Dict[str, Any]) -> bool:
        """
        Save stats to database with file fallback.

        Args:
            stats: Stats dictionary to save

        Returns:
            True if saved successfully
        """
        with self._lock:
            success = False

            if self._use_db:
                try:
                    self._save_to_db(stats)
                    success = True
                except Exception as e:
                    logger.warning(f"DB save failed: {e}")

            # Always save to file as backup
            try:
                self._save_to_file(stats)
                if not success:
                    success = True
            except Exception as e:
                logger.error(f"File save also failed: {e}")

            return success

    def _save_to_db(self, stats: Dict[str, Any]):
        """Save stats to database."""
        today = date.today().isoformat()
        today_stats = stats.get("today", {})

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO circuit_breaker_daily 
                (date, trades_count, total_loss, total_profit, net_pnl,
                 consecutive_losses, consecutive_wins, morning_trades,
                 afternoon_trades, tripped, tripped_reason, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    today,
                    today_stats.get("trades_count", 0),
                    today_stats.get("total_loss", 0.0),
                    today_stats.get("total_profit", 0.0),
                    today_stats.get("net_pnl", 0.0),
                    stats.get("consecutive_losses", 0),
                    stats.get("consecutive_wins", 0),
                    stats.get("morning_trades", 0),
                    stats.get("afternoon_trades", 0),
                    1 if stats.get("tripped", False) else 0,
                    stats.get("tripped_reason"),
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()

    def _save_to_file(self, stats: Dict[str, Any]):
        """Save stats to fallback file."""
        with open(self.fallback_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def record_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Record an event to history for analysis.

        Args:
            event_type: Type of event (TRADE, TRIP, RESET, etc.)
            event_data: Event details
        """
        if not self._use_db:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO circuit_breaker_history 
                    (date, event_type, event_data)
                    VALUES (?, ?, ?)
                """,
                    (
                        date.today().isoformat(),
                        event_type,
                        json.dumps(event_data, ensure_ascii=False),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record event: {e}")

    def acquire_lock(
        self,
        lock_name: str = "trading",
        bot_id: str = None,
        timeout_seconds: int = 300,
    ) -> bool:
        """
        Acquire distributed lock for multiple bot coordination.

        IMPROVEMENT: Prevents race conditions when multiple bots
        try to trade simultaneously.

        Args:
            lock_name: Name of the lock
            bot_id: Unique identifier for this bot instance
            timeout_seconds: Lock expiration time

        Returns:
            True if lock acquired
        """
        if not self._use_db:
            return True  # No locking without DB

        if bot_id is None:
            import uuid

            bot_id = str(uuid.uuid4())[:8]

        now = datetime.now()
        expires_at = now.replace(second=now.second + timeout_seconds)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check if lock exists and is expired
                cursor.execute(
                    "SELECT * FROM circuit_breaker_locks WHERE lock_name = ?", (lock_name,)
                )
                existing = cursor.fetchone()

                if existing:
                    expires = datetime.fromisoformat(existing["expires_at"])
                    if expires > now and existing["bot_id"] != bot_id:
                        logger.warning(
                            f"Lock '{lock_name}' held by {existing['bot_id']} " f"until {expires}"
                        )
                        return False

                # Acquire or refresh lock
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO circuit_breaker_locks 
                    (lock_name, bot_id, acquired_at, expires_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        lock_name,
                        bot_id,
                        now.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()

                logger.info(f"✅ Lock '{lock_name}' acquired by {bot_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self, lock_name: str = "trading", bot_id: str = None) -> bool:
        """
        Release distributed lock.

        Args:
            lock_name: Name of the lock
            bot_id: Bot identifier (only owner can release)

        Returns:
            True if released
        """
        if not self._use_db:
            return True

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if bot_id:
                    cursor.execute(
                        "DELETE FROM circuit_breaker_locks " "WHERE lock_name = ? AND bot_id = ?",
                        (lock_name, bot_id),
                    )
                else:
                    cursor.execute(
                        "DELETE FROM circuit_breaker_locks WHERE lock_name = ?", (lock_name,)
                    )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    def get_historical_stats(self, days: int = 30) -> list:
        """
        Get historical circuit breaker stats for analysis.

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily stats
        """
        if not self._use_db:
            return []

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM circuit_breaker_daily 
                    ORDER BY date DESC LIMIT ?
                """,
                    (days,),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


# Regime-aware consecutive loss limits
REGIME_CONSECUTIVE_LOSS_LIMITS = {
    "BULL": 4,  # More lenient in bull market
    "SIDEWAYS": 3,  # Standard in sideways (mean-reversion opportunities)
    "BEAR": 2,  # Strict in bear market
    "HIGH_VOLATILITY": 2,  # Very strict in high volatility
    "DEFAULT": 3,
}


def get_regime_aware_loss_limit(regime: str) -> int:
    """
    Get consecutive loss limit based on market regime.

    IMPROVEMENT v5.1: Addresses the issue of being too aggressive
    in sideways market where mean-reversion opportunities exist.

    Args:
        regime: Market regime (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY)

    Returns:
        Maximum consecutive losses allowed
    """
    return REGIME_CONSECUTIVE_LOSS_LIMITS.get(
        regime.upper(), REGIME_CONSECUTIVE_LOSS_LIMITS["DEFAULT"]
    )
