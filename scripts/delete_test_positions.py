#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to delete test positions from database
Xóa các positions do test tạo ra
"""

import sys
import os

# Fix encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

import sqlite3
from pathlib import Path


def delete_positions_by_symbols(symbols: list):
    """Xóa positions theo danh sách symbols"""
    db_dir = Path("data/database")
    db_path = db_dir / "trading.db"

    if not db_path.exists():
        print(f"❌ Database không tồn tại tại: {db_path}")
        return

    print(f"📊 Database path: {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Count positions before deletion
        placeholders = ",".join(["?"] * len(symbols))
        cursor.execute(f"SELECT COUNT(*) FROM positions WHERE symbol IN ({placeholders})", symbols)
        count_before = cursor.fetchone()[0]
        print(f"📋 Số positions sẽ xóa: {count_before}")

        if count_before == 0:
            print("✅ Không có positions nào để xóa")
            conn.close()
            return

        # Get list of symbols before deletion
        cursor.execute(f"SELECT symbol FROM positions WHERE symbol IN ({placeholders})", symbols)
        symbols_to_delete = [row[0] for row in cursor.fetchall()]

        # Delete positions
        cursor.execute(f"DELETE FROM positions WHERE symbol IN ({placeholders})", symbols)

        # Also delete related trades
        cursor.execute(f"DELETE FROM trades WHERE symbol IN ({placeholders})", symbols)

        deleted_positions = cursor.rowcount if cursor.rowcount >= 0 else count_before
        cursor.execute(f"SELECT COUNT(*) FROM trades WHERE symbol IN ({placeholders})", symbols)
        deleted_trades = cursor.fetchone()[0]
        if deleted_trades > 0:
            cursor.execute(f"DELETE FROM trades WHERE symbol IN ({placeholders})", symbols)

        # Commit
        conn.commit()

        # Verify deletion
        cursor.execute(f"SELECT COUNT(*) FROM positions WHERE symbol IN ({placeholders})", symbols)
        count_after = cursor.fetchone()[0]

        conn.close()

        print(f"✅ Đã xóa {deleted_positions} positions và related trades:")
        print(f"   Các mã: {', '.join(symbols_to_delete)}")
        print(f"📊 Số positions còn lại sau khi xóa: {count_after}")

        if count_after == 0:
            print("✅ Tất cả positions đã được xóa!")
        else:
            print(f"⚠️ Còn {count_after} positions trong DB")

    except Exception as e:
        print(f"❌ Lỗi khi xóa positions: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()


def delete_all_positions():
    """Xóa tất cả positions (cẩn thận!)"""
    db_dir = Path("data/database")
    db_path = db_dir / "trading.db"

    if not db_path.exists():
        print(f"❌ Database không tồn tại tại: {db_path}")
        return

    print(f"📊 Database path: {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Count positions before deletion
        cursor.execute("SELECT COUNT(*) FROM positions")
        count_before = cursor.fetchone()[0]
        print(f"📋 Số positions hiện tại: {count_before}")

        if count_before == 0:
            print("✅ Database đã trống, không cần xóa")
            conn.close()
            return

        # Get list of symbols before deletion
        cursor.execute("SELECT symbol FROM positions")
        symbols = [row[0] for row in cursor.fetchall()]

        # Delete all positions
        cursor.execute("DELETE FROM positions")
        cursor.execute("DELETE FROM trades")

        # Commit
        conn.commit()

        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM positions")
        count_after = cursor.fetchone()[0]

        conn.close()

        print(f"✅ Đã xóa {count_before} positions và tất cả trades:")
        if symbols:
            print(f"   Các mã đã xóa: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                print(f"   ... và {len(symbols) - 10} mã khác")

        print(f"📊 Số positions sau khi xóa: {count_after}")

        if count_after == 0:
            print("✅ Database đã được clean hoàn toàn!")
        else:
            print(f"⚠️ Còn {count_after} positions trong DB (có thể là lỗi)")

    except Exception as e:
        print(f"❌ Lỗi khi xóa positions: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("🧹 DELETE TEST POSITIONS")
    print("=" * 70)
    print()

    # Check arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            print("⚠️  Chế độ: XÓA TẤT CẢ POSITIONS")
            print("⚠️  Cảnh báo: Thao tác này sẽ xóa toàn bộ positions trong DB!")
            print()
            try:
                response = input("Bạn có chắc chắn muốn xóa TẤT CẢ? (yes/no): ")
                if response.lower() in ["yes", "y", "có"]:
                    delete_all_positions()
                else:
                    print("❌ Đã hủy. Không có gì bị xóa.")
            except EOFError:
                print("✅ Non-interactive mode: Xóa tất cả positions...")
                delete_all_positions()
        else:
            # Delete specific symbols
            symbols = [s.upper() for s in sys.argv[1:]]
            print(f"🔍 Chế độ: Xóa positions theo symbols")
            print(f"📋 Các symbols sẽ xóa: {', '.join(symbols)}")
            print()
            delete_positions_by_symbols(symbols)
    else:
        print("Usage:")
        print("  python scripts/delete_test_positions.py SYMBOL1 SYMBOL2 ...")
        print("  python scripts/delete_test_positions.py --all")
        print()
        print("Examples:")
        print("  python scripts/delete_test_positions.py ABB ACV AFX")
        print("  python scripts/delete_test_positions.py --all")
