"""
DB maintenance utilities: purge test artifacts and basic cleanup.

Usage (programmatic):
    from src.utils.db_maintenance import purge_test_artifacts
    purge_test_artifacts()

Usage (CLI):
    python -m src.utils.db_maintenance purge-tests
"""
import os
import sqlite3
from pathlib import Path


def _get_db_path() -> str:
    env_path = os.getenv("TRADING_DB_PATH")
    if env_path:
        return env_path
    return str(Path("data/database") / "trading.db")


def purge_test_artifacts() -> int:
    """Delete positions and trades with symbols created by tests (TEST*).

    Returns: number of rows deleted across tables.
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        deleted_total = 0

        # Delete test positions
        cur.execute("DELETE FROM positions WHERE symbol LIKE 'TEST%'")
        deleted_total += cur.rowcount if cur.rowcount is not None else 0

        # Delete test trades
        cur.execute("DELETE FROM trades WHERE symbol LIKE 'TEST%'")
        deleted_total += cur.rowcount if cur.rowcount is not None else 0

        conn.commit()
        return deleted_total
    finally:
        conn.close()


def vacuum() -> None:
    """VACUUM the database to reclaim space after deletions."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "purge-tests":
        n = purge_test_artifacts()
        print(f"Deleted {n} test rows (positions + trades)")
        vacuum()
        print("VACUUM completed")
    else:
        print("Usage: python -m src.utils.db_maintenance purge-tests")
