"""
Migration Script: JSON → SQLite
Migrate all JSON data to database
"""

import json
import os
from datetime import datetime

from src.data.database import get_db
from src.portfolio.manager import get_portfolio_manager


def migrate_positions():
    """Migrate active_positions.json to database"""
    json_file = "active_positions.json"

    if not os.path.exists(json_file):
        print(f"⏭️ {json_file} not found")
        return 0

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            positions = json.load(f)

        manager = get_portfolio_manager()
        count = 0

        for symbol, pos in positions.items():
            try:
                manager.add_position(
                    symbol=symbol,
                    shares=pos.get("shares", 0),
                    entry_price=pos.get("avg_price", pos.get("entry_price", 0)),
                    stop_loss=pos.get("stop_loss"),
                    take_profit=(
                        pos.get("take_profit_targets", [None])[0]
                        if pos.get("take_profit_targets")
                        else None
                    ),
                    metadata=pos,
                )
                count += 1
            except Exception:
                print(f"❌ Error migrating {symbol}")

        print(f"✅ Migrated {count} positions from {json_file}")
        return count

    except Exception:
        print(f"❌ Error reading {json_file}")
        return 0


def migrate_portfolio_history():
    """Migrate portfolio_history.json to database"""
    json_file = "portfolio_history.json"

    if not os.path.exists(json_file):
        print(f"⏭️ {json_file} not found")
        return 0

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        db = get_db()
        count = 0

        for snapshot in history:
            try:
                db.save_portfolio_snapshot(
                    date=snapshot.get("date", datetime.now().isoformat()),
                    total_value=snapshot.get("total_value", 0),
                    total_cost=snapshot.get("total_cost", 0),
                    pnl=snapshot.get("pnl", 0),
                    pnl_percent=snapshot.get("pnl_percent", 0),
                    num_positions=snapshot.get("num_positions", 0),
                    metadata=snapshot,
                )
                count += 1
            except Exception:
                print("❌ Error migrating snapshot")

        print(f"✅ Migrated {count} snapshots from {json_file}")
        return count

    except Exception:
        print(f"❌ Error reading {json_file}")
        return 0


def backup_json_files():
    """Backup JSON files before migration"""
    backup_dir = "json_backup"
    os.makedirs(backup_dir, exist_ok=True)

    json_files = [
        "active_positions.json",
        "portfolio_history.json",
        "selected_tickers.json",
    ]

    backed_up = []
    for filename in json_files:
        if os.path.exists(filename):
            backup_path = os.path.join(
                backup_dir, f"{filename}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            try:
                import shutil

                shutil.copy2(filename, backup_path)
                backed_up.append(filename)
                print(f"📦 Backed up: {filename} → {backup_path}")
            except Exception:
                print(f"❌ Error backing up {filename}")

    return backed_up


def verify_migration():
    """Verify migration was successful"""
    db = get_db()
    manager = get_portfolio_manager()

    print("\n🔍 VERIFICATION:")
    print("=" * 50)

    # Check positions
    positions = db.get_positions()
    print(f"✓ Positions in DB: {len(positions)}")
    for symbol, pos in positions.items():
        print(f"  - {symbol}: {pos['shares']} shares @ {pos['avg_price']:,.0f}")

    # Check history
    history = db.get_portfolio_history(days=30)
    print(f"✓ Portfolio snapshots: {len(history)}")

    # Check trades
    trades = db.get_trades(limit=10)
    print(f"✓ Trade records: {len(trades)}")

    # Portfolio value
    portfolio = manager.get_portfolio_value()
    print(f"✓ Portfolio value: {portfolio['total_value']:,.0f} VNĐ")
    print(f"✓ P&L: {portfolio['pnl']:+,.0f} VNĐ ({portfolio['pnl_percent']:+.1f}%)")


def main():
    """Main migration process"""
    print("=" * 60)
    print("🔄 MIGRATION: JSON → SQLite")
    print("=" * 60)

    # Step 1: Backup
    print("\n📦 Step 1: Backing up JSON files...")
    backed_up = backup_json_files()

    if not backed_up:
        print("⚠️ No JSON files to migrate")
        return

    # Step 2: Migrate
    print("\n🔄 Step 2: Migrating data...")
    positions_count = migrate_positions()
    history_count = migrate_portfolio_history()

    # Step 3: Verify
    print("\n✅ Step 3: Verifying migration...")
    verify_migration()

    # Summary
    print("\n" + "=" * 60)
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    print(f"✓ Positions migrated: {positions_count}")
    print(f"✓ History snapshots: {history_count}")
    print(f"✓ Backup files: {len(backed_up)}")

    print("\n" + "=" * 60)
    print("💡 NEXT STEPS:")
    print("=" * 60)
    print("1. Verify data in database:")
    print(
        "   python -c 'from src.portfolio.manager import get_portfolio_manager; "
        "print(get_portfolio_manager().get_detailed_analysis())'"
    )
    print("")
    print("2. Update bot_runner_improved.py to use portfolio_manager")
    print("")
    print("3. Test thoroughly before deleting JSON files")
    print("")
    print("4. JSON backups are in: json_backup/")
    print("=" * 60)


if __name__ == "__main__":
    main()
