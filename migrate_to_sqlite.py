"""
Migration script: JSON files → SQLite database
"""
from database import get_db
import os

def migrate():
    """Migrate all JSON files to SQLite"""
    print("="*60)
    print("🔄 MIGRATION: JSON → SQLite")
    print("="*60)
    
    db = get_db()
    
    # Define JSON files to migrate
    json_files = {
        'positions': 'active_positions.json',
        'portfolio_history': 'portfolio_history.json'
    }
    
    print("\n📦 Migrating data...")
    db.migrate_from_json(json_files)
    
    print("\n✅ Migration completed!")
    print("\n📊 Current data in database:")
    
    # Show positions
    positions = db.get_positions()
    print(f"  Positions: {len(positions)}")
    for symbol, pos in positions.items():
        print(f"    - {symbol}: {pos['shares']} shares @ {pos['avg_price']:,.0f}")
    
    # Show history
    history = db.get_portfolio_history(days=7)
    print(f"  Portfolio History: {len(history)} snapshots")
    
    # Show trades
    trades = db.get_trades(limit=10)
    print(f"  Trades: {len(trades)} records")
    
    print("\n" + "="*60)
    print("💡 NEXT STEPS:")
    print("="*60)
    print("1. Backup JSON files:")
    print("   mkdir json_backup")
    print("   mv *.json json_backup/")
    print("")
    print("2. Update code to use database.py instead of JSON")
    print("")
    print("3. Test thoroughly before deleting JSON files")
    print("="*60)

if __name__ == "__main__":
    migrate()
