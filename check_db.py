import sqlite3
import os

db_path = "data/database/trading.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", [t[0] for t in tables])

    try:
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        print(f"Total trades: {count}")

        cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 5")
        trades = cursor.fetchall()
        if trades:
            print("Recent trades:")
            for t in trades:
                print(f"  {t}")
        else:
            print("No trades found in DB")
    except Exception as e:
        print(f"Trades error: {e}")

    try:
        cursor.execute("SELECT COUNT(*) FROM positions")
        count = cursor.fetchone()[0]
        print(f"Total positions: {count}")
    except Exception as e:
        print(f"Positions error: {e}")

    conn.close()
else:
    print("Database not found")
