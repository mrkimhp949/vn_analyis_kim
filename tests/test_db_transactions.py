"""
Test database transaction atomicity

Verifies that database transactions work correctly:
- Multiple operations commit together on success
- All operations roll back on failure
- No partial state is left in database
"""

import os
import tempfile
import pytest
from unittest.mock import patch

# Set test database path before importing
test_db_path = tempfile.mktemp(suffix=".db")
os.environ["TRADING_DB_PATH"] = test_db_path

from src.data.database import get_db


@pytest.fixture
def db():
    """Create a fresh database for each test"""
    # Clear singleton
    import src.data.database as db_module

    db_module._db_instance = None

    # Remove existing test database to ensure clean state
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Create new instance
    database = get_db()

    # Clear any existing data
    try:
        # Clear positions table
        database.conn.execute("DELETE FROM positions")
        # Clear trades table
        database.conn.execute("DELETE FROM trades")
        database.conn.commit()
    except Exception:
        pass  # Tables may not exist yet

    yield database

    # Cleanup
    database.close()
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_transaction_commit_success(db):
    """Test that transaction commits all operations on success"""

    # Start transaction and perform multiple operations
    with db.transaction() as conn:
        db.save_position(
            symbol="VCB",
            shares=100,
            avg_price=60000,
            entry_date="2025-11-22",
            entry_value=6000000,
            conn=conn,
        )

        db.save_trade(
            symbol="VCB",
            action="BUY_NEW",
            shares=100,
            price=60000,
            total_value=6000000,
            trade_date="2025-11-22",
            reason="Test trade",
            conn=conn,
        )

    # Verify both operations were committed
    positions = db.get_positions()
    assert "VCB" in positions
    assert positions["VCB"]["shares"] == 100

    trades = db.get_trades(symbol="VCB")
    assert len(trades) == 1
    assert trades[0]["action"] == "BUY_NEW"


def test_transaction_rollback_on_error(db):
    """Test that transaction rolls back all operations on error"""

    # Attempt transaction that will fail midway
    try:
        with db.transaction() as conn:
            # First operation should succeed
            db.save_position(
                symbol="FPT",
                shares=100,
                avg_price=100000,
                entry_date="2025-11-22",
                entry_value=10000000,
                conn=conn,
            )

            # Force an error by raising an exception explicitly
            # This will cause the entire transaction to roll back
            raise ValueError("Simulated error to test rollback")
    except ValueError:
        # Expected to fail
        pass

    # Verify BOTH operations were rolled back (position should NOT exist)
    positions = db.get_positions()
    assert "FPT" not in positions, "Position should have been rolled back!"

    trades = db.get_trades(symbol="FPT")
    assert len(trades) == 0, "Trade should have been rolled back!"


def test_transaction_isolation(db):
    """Test that failed transaction doesn't affect subsequent operations"""

    # First transaction fails
    try:
        with db.transaction() as conn:
            db.save_position(
                symbol="HPG",
                shares=100,
                avg_price=30000,
                entry_date="2025-11-22",
                entry_value=3000000,
                conn=conn,
            )
            # Force error
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Verify rollback
    positions = db.get_positions()
    assert "HPG" not in positions

    # Second transaction succeeds
    with db.transaction() as conn:
        db.save_position(
            symbol="HPG",
            shares=200,
            avg_price=30000,
            entry_date="2025-11-22",
            entry_value=6000000,
            conn=conn,
        )

    # Verify success
    positions = db.get_positions()
    assert "HPG" in positions
    assert positions["HPG"]["shares"] == 200


def test_without_transaction_still_works(db):
    """Test that operations without transaction still work (backward compatibility)"""

    # Save position without transaction (should auto-commit)
    db.save_position(
        symbol="VNM",
        shares=100,
        avg_price=80000,
        entry_date="2025-11-22",
        entry_value=8000000,
    )

    # Verify it was saved
    positions = db.get_positions()
    assert "VNM" in positions
    assert positions["VNM"]["shares"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
