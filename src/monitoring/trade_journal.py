# -*- coding: utf-8 -*-
"""
Trade Journal - Persistent Trade Recording and Analysis

Records all trades with full details for:
- Performance analysis
- Tax reporting
- Strategy optimization
- Compliance auditing

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade status"""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"


class TradeDirection(Enum):
    """Trade direction"""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class JournalEntry:
    """A single trade journal entry"""

    # Identification
    trade_id: str
    symbol: str

    # Direction
    direction: TradeDirection = TradeDirection.LONG
    status: TradeStatus = TradeStatus.OPEN

    # Entry details
    entry_date: str = ""
    entry_price: float = 0.0
    entry_quantity: int = 0
    entry_value: float = 0.0
    entry_reason: str = ""
    entry_confidence: float = 0.0

    # Exit details
    exit_date: Optional[str] = None
    exit_price: float = 0.0
    exit_quantity: int = 0
    exit_value: float = 0.0
    exit_reason: str = ""

    # P&L
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    commission_paid: float = 0.0
    net_pnl: float = 0.0

    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0
    risk_reward_ratio: float = 0.0

    # Context
    market_regime: str = ""
    sector: str = ""
    strategy: str = ""

    # Timing
    holding_days: int = 0

    # ML/Technical signals
    ml_signal: str = ""
    ml_confidence: float = 0.0
    technical_score: float = 0.0

    # Tags for filtering
    tags: List[str] = field(default_factory=list)

    # Notes
    notes: str = ""

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["direction"] = self.direction.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalEntry":
        """Create from dictionary"""
        data["direction"] = TradeDirection(data.get("direction", "LONG"))
        data["status"] = TradeStatus(data.get("status", "OPEN"))
        return cls(**data)


class TradeJournal:
    """
    Persistent trade journal with SQLite storage

    Usage:
        journal = TradeJournal()

        # Record entry
        entry = journal.record_entry(
            symbol="VNM",
            entry_price=85000,
            entry_quantity=100,
            stop_loss=80000,
            take_profit=95000,
            entry_reason="Breakout above resistance",
            confidence=75,
        )

        # Record exit
        journal.record_exit(
            trade_id=entry.trade_id,
            exit_price=92000,
            exit_quantity=100,
            exit_reason="Take profit hit",
        )

        # Get statistics
        stats = journal.get_statistics()
    """

    def __init__(
        self,
        db_path: str = "data/trade_journal.db",
        json_backup_path: str = "data/trade_journal_backup.json",
    ):
        self.db_path = db_path
        self.json_backup_path = json_backup_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT DEFAULT 'LONG',
                status TEXT DEFAULT 'OPEN',
                
                entry_date TEXT,
                entry_price REAL,
                entry_quantity INTEGER,
                entry_value REAL,
                entry_reason TEXT,
                entry_confidence REAL,
                
                exit_date TEXT,
                exit_price REAL,
                exit_quantity INTEGER,
                exit_value REAL,
                exit_reason TEXT,
                
                realized_pnl REAL,
                realized_pnl_pct REAL,
                commission_paid REAL,
                net_pnl REAL,
                
                stop_loss REAL,
                take_profit REAL,
                risk_amount REAL,
                risk_reward_ratio REAL,
                
                market_regime TEXT,
                sector TEXT,
                strategy TEXT,
                holding_days INTEGER,
                
                ml_signal TEXT,
                ml_confidence REAL,
                technical_score REAL,
                
                tags TEXT,
                notes TEXT,
                
                created_at TEXT,
                updated_at TEXT
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_date ON trades(entry_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy)")

        conn.commit()
        conn.close()

        logger.info(f"Trade journal initialized: {self.db_path}")

    def _generate_trade_id(self, symbol: str) -> str:
        """Generate unique trade ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{symbol}_{timestamp}"

    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        entry_quantity: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        entry_reason: str = "",
        confidence: float = 0.0,
        direction: TradeDirection = TradeDirection.LONG,
        market_regime: str = "",
        sector: str = "",
        strategy: str = "",
        ml_signal: str = "",
        ml_confidence: float = 0.0,
        technical_score: float = 0.0,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> JournalEntry:
        """Record trade entry"""

        trade_id = self._generate_trade_id(symbol)
        now = datetime.now().isoformat()

        entry_value = entry_price * entry_quantity

        # Calculate risk
        risk_amount = abs(entry_price - stop_loss) * entry_quantity if stop_loss > 0 else 0

        # Calculate R:R
        if stop_loss > 0 and take_profit > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
        else:
            risk_reward_ratio = 0

        entry = JournalEntry(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            status=TradeStatus.OPEN,
            entry_date=now,
            entry_price=entry_price,
            entry_quantity=entry_quantity,
            entry_value=entry_value,
            entry_reason=entry_reason,
            entry_confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            risk_reward_ratio=risk_reward_ratio,
            market_regime=market_regime,
            sector=sector,
            strategy=strategy,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            technical_score=technical_score,
            tags=tags or [],
            notes=notes,
            created_at=now,
            updated_at=now,
        )

        self._save_entry(entry)

        logger.info(
            f"📝 Trade entry recorded: {symbol} {entry_quantity} @ {entry_price:,.0f} "
            f"(ID: {trade_id})"
        )

        return entry

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_quantity: int,
        exit_reason: str = "",
        commission: float = 0.0,
    ) -> Optional[JournalEntry]:
        """Record trade exit"""

        entry = self.get_trade(trade_id)
        if not entry:
            logger.error(f"Trade not found: {trade_id}")
            return None

        now = datetime.now().isoformat()
        exit_value = exit_price * exit_quantity

        # Calculate P&L
        if entry.direction == TradeDirection.LONG:
            realized_pnl = (exit_price - entry.entry_price) * exit_quantity
        else:
            realized_pnl = (entry.entry_price - exit_price) * exit_quantity

        realized_pnl_pct = (realized_pnl / entry.entry_value * 100) if entry.entry_value > 0 else 0
        net_pnl = realized_pnl - commission

        # Calculate holding days
        entry_dt = datetime.fromisoformat(entry.entry_date)
        exit_dt = datetime.now()
        holding_days = (exit_dt - entry_dt).days

        # Update status
        if exit_quantity >= entry.entry_quantity:
            status = TradeStatus.CLOSED
        else:
            status = TradeStatus.PARTIAL

        # Update entry
        entry.exit_date = now
        entry.exit_price = exit_price
        entry.exit_quantity = exit_quantity
        entry.exit_value = exit_value
        entry.exit_reason = exit_reason
        entry.realized_pnl = realized_pnl
        entry.realized_pnl_pct = realized_pnl_pct
        entry.commission_paid = commission
        entry.net_pnl = net_pnl
        entry.holding_days = holding_days
        entry.status = status
        entry.updated_at = now

        self._save_entry(entry)

        logger.info(
            f"📝 Trade exit recorded: {entry.symbol} {exit_quantity} @ {exit_price:,.0f} "
            f"P&L: {realized_pnl:+,.0f} ({realized_pnl_pct:+.2f}%)"
        )

        return entry

    def _save_entry(self, entry: JournalEntry):
        """Save entry to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = entry.to_dict()
        data["tags"] = json.dumps(data["tags"])

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())

        cursor.execute(f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})", values)

        conn.commit()
        conn.close()

    def get_trade(self, trade_id: str) -> Optional[JournalEntry]:
        """Get trade by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            data = dict(row)
            data["tags"] = json.loads(data["tags"]) if data["tags"] else []
            return JournalEntry.from_dict(data)
        return None

    def get_open_trades(self) -> List[JournalEntry]:
        """Get all open trades"""
        return self._query_trades("status = 'OPEN'")

    def get_trades_by_symbol(self, symbol: str) -> List[JournalEntry]:
        """Get all trades for a symbol"""
        return self._query_trades(f"symbol = '{symbol}'")

    def get_trades_by_date_range(
        self,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[JournalEntry]:
        """Get trades within date range"""
        end_date = end_date or date.today()
        return self._query_trades(
            f"entry_date >= '{start_date.isoformat()}' AND entry_date <= '{end_date.isoformat()}'"
        )

    def _query_trades(self, where_clause: str = "") -> List[JournalEntry]:
        """Query trades with optional filter"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM trades"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += " ORDER BY entry_date DESC"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            data = dict(row)
            data["tags"] = json.loads(data["tags"]) if data["tags"] else []
            entries.append(JournalEntry.from_dict(data))

        return entries

    def get_statistics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get trading statistics"""

        # Build filter
        filters = ["status = 'CLOSED'"]
        if start_date:
            filters.append(f"entry_date >= '{start_date.isoformat()}'")
        if end_date:
            filters.append(f"entry_date <= '{end_date.isoformat()}'")
        if symbol:
            filters.append(f"symbol = '{symbol}'")
        if strategy:
            filters.append(f"strategy = '{strategy}'")

        where_clause = " AND ".join(filters)
        trades = self._query_trades(where_clause)

        if not trades:
            return {"total_trades": 0}

        # Calculate statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl <= 0]

        total_pnl = sum(t.net_pnl for t in trades)
        gross_profit = sum(t.realized_pnl for t in winning_trades)
        gross_loss = sum(t.realized_pnl for t in losing_trades)

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = abs(gross_loss / len(losing_trades)) if losing_trades else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

        avg_holding_days = sum(t.holding_days for t in trades) / total_trades if trades else 0
        avg_risk_reward = sum(t.risk_reward_ratio for t in trades) / total_trades if trades else 0

        # Best and worst trades
        best_trade = max(trades, key=lambda t: t.realized_pnl_pct)
        worst_trade = min(trades, key=lambda t: t.realized_pnl_pct)

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else 0,
            "avg_holding_days": avg_holding_days,
            "avg_risk_reward": avg_risk_reward,
            "best_trade": {
                "symbol": best_trade.symbol,
                "pnl_pct": best_trade.realized_pnl_pct,
            },
            "worst_trade": {
                "symbol": worst_trade.symbol,
                "pnl_pct": worst_trade.realized_pnl_pct,
            },
        }

    def export_to_csv(self, filepath: str = "trade_journal_export.csv"):
        """Export journal to CSV"""
        trades = self._query_trades()

        if not trades:
            logger.warning("No trades to export")
            return

        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(filepath, index=False)

        logger.info(f"Exported {len(trades)} trades to {filepath}")

    def backup_to_json(self):
        """Backup journal to JSON file"""
        trades = self._query_trades()

        data = {
            "exported_at": datetime.now().isoformat(),
            "total_trades": len(trades),
            "trades": [t.to_dict() for t in trades],
        }

        with open(self.json_backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Backed up {len(trades)} trades to {self.json_backup_path}")


# Singleton instance
_trade_journal: Optional[TradeJournal] = None


def get_trade_journal() -> TradeJournal:
    """Get singleton trade journal instance"""
    global _trade_journal
    if _trade_journal is None:
        _trade_journal = TradeJournal()
    return _trade_journal
