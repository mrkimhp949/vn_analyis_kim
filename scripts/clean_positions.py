#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to clean all positions from database
Xóa tất cả positions trong DB để reset
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


def clean_all_positions():
    """Xóa tất cả positions từ database"""
    # Find database path
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
        deleted_count = cursor.rowcount if cursor.rowcount >= 0 else count_before

        # Commit
        conn.commit()

        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM positions")
        count_after = cursor.fetchone()[0]

        conn.close()

        print(f"✅ Đã xóa {deleted_count} positions:")
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


def clean_zero_share_positions():
    """Xóa chỉ các positions có shares = 0 hoặc âm"""
    db_dir = Path("data/database")
    db_path = db_dir / "trading.db"

    if not db_path.exists():
        print(f"❌ Database không tồn tại tại: {db_path}")
        return

    print(f"📊 Database path: {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Count zero-share positions
        cursor.execute("SELECT COUNT(*) FROM positions WHERE shares <= 0")
        count_zero = cursor.fetchone()[0]
        print(f"📋 Số positions có shares <= 0: {count_zero}")

        if count_zero == 0:
            print("✅ Không có positions nào có shares <= 0")
            conn.close()
            return

        # Get list before deletion
        cursor.execute("SELECT symbol FROM positions WHERE shares <= 0")
        symbols = [row[0] for row in cursor.fetchall()]

        # Delete zero-share positions
        cursor.execute("DELETE FROM positions WHERE shares <= 0")
        deleted_count = cursor.rowcount if cursor.rowcount >= 0 else count_zero

        conn.commit()
        conn.close()

        print(f"✅ Đã xóa {deleted_count} positions với shares <= 0:")
        print(f"   Các mã: {', '.join(symbols)}")

    except Exception as e:
        print(f"❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("🧹 CLEAN POSITIONS FROM DATABASE")
    print("=" * 70)
    print()

    # Check arguments
    auto_confirm = len(sys.argv) > 1 and sys.argv[1] in ["--yes", "-y", "--force"]
    zero_only = len(sys.argv) > 1 and sys.argv[1] == "--zero-only"

    if zero_only:
        print("🔍 Chế độ: Chỉ xóa positions có shares <= 0")
        print()
        clean_zero_share_positions()
    else:
        print("⚠️  Chế độ: XÓA TẤT CẢ POSITIONS")
        print("⚠️  Cảnh báo: Thao tác này sẽ xóa toàn bộ positions trong DB!")
        print()

        if auto_confirm:
            print("✅ Auto-confirm: Xóa tất cả positions...")
            clean_all_positions()
        else:
            try:
                response = input("Bạn có chắc chắn muốn xóa TẤT CẢ? (yes/no): ")
                if response.lower() in ["yes", "y", "có"]:
                    clean_all_positions()
                else:
                    print("❌ Đã hủy. Không có gì bị xóa.")
                    print()
                    print("💡 Để xóa tự động (không cần xác nhận):")
                    print("   python scripts/clean_positions.py --yes")
            except EOFError:
                # Non-interactive mode - auto confirm
                print("✅ Non-interactive mode: Xóa tất cả positions...")
                clean_all_positions()
