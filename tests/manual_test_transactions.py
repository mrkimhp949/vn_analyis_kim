"""
Manual test for database transaction atomicity

Run directly without pytest to verify transaction behavior.
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test database path
test_db_path = tempfile.mktemp(suffix=".db")
os.environ["TRADING_DB_PATH"] = test_db_path

from src.data.database import get_db


def test_transaction_commit_success():
    """Test that transaction commits all operations on success"""
    print("\n1️⃣ Testing successful transaction commit...")

    db = get_db()

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
    assert "VCB" in positions, "Position was not saved!"
    assert positions["VCB"]["shares"] == 100

    trades = db.get_trades(symbol="VCB")
    assert len(trades) == 1, "Trade was not saved!"
    assert trades[0]["action"] == "BUY_NEW"

    print("   ✅ SUCCESS: Both operations committed together")
    return True


def test_transaction_rollback_on_error():
    """Test that transaction rolls back all operations on error"""
    print("\n2️⃣ Testing transaction rollback on error...")

    db = get_db()

    # Attempt transaction that will fail midway
    try:
        with db.transaction() as conn:
            # First operation
            db.save_position(
                symbol="FPT",
                shares=100,
                avg_price=100000,
                entry_date="2025-11-22",
                entry_value=10000000,
                conn=conn,
            )

            # Force an error
            raise ValueError("Simulated error to test rollback")

    except ValueError:
        # Expected to fail
        pass

    # Verify BOTH operations were rolled back
    positions = db.get_positions()
    assert "FPT" not in positions, "Position should have been rolled back!"

    trades = db.get_trades(symbol="FPT")
    assert len(trades) == 0, "Trade should have been rolled back!"

    print("   ✅ SUCCESS: Operations were rolled back on error")
    return True


def test_transaction_isolation():
    """Test that failed transaction doesn't affect subsequent operations"""
    print("\n3️⃣ Testing transaction isolation...")

    db = get_db()

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
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Verify rollback
    positions = db.get_positions()
    assert "HPG" not in positions, "Failed transaction should not save data"

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
    assert "HPG" in positions, "Second transaction should succeed"
    assert positions["HPG"]["shares"] == 200

    print("   ✅ SUCCESS: Failed transaction didn't affect subsequent operations")
    return True


def test_backward_compatibility():
    """Test that operations without transaction still work"""
    print("\n4️⃣ Testing backward compatibility (no transaction)...")

    db = get_db()

    # Save without transaction (should auto-commit)
    db.save_position(
        symbol="VNM",
        shares=100,
        avg_price=80000,
        entry_date="2025-11-22",
        entry_value=8000000,
    )

    # Verify it was saved
    positions = db.get_positions()
    assert "VNM" in positions, "Position should be saved without transaction"
    assert positions["VNM"]["shares"] == 100

    print("   ✅ SUCCESS: Operations work without transaction (backward compatible)")
    return True


def cleanup():
    """Cleanup test database"""
    from src.data.database import get_db

    db = get_db()
    db.close()

    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"\n🧹 Cleaned up test database: {test_db_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATABASE TRANSACTION ATOMICITY")
    print("=" * 60)

    try:
        # Run tests
        test_transaction_commit_success()
        test_transaction_rollback_on_error()
        test_transaction_isolation()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nDatabase transactions are working correctly:")
        print("  • Multiple operations commit together on success")
        print("  • All operations roll back on error")
        print("  • No partial state left in database")
        print("  • Backward compatible with non-transactional code")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()
