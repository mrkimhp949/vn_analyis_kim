"""
Telegram Subscription Manager
Quản lý đăng ký nhận tin theo symbol/sector
"""

import json
import os
from typing import Dict, List, Set
from datetime import datetime

SUBSCRIPTIONS_FILE = "telegram_subscriptions.json"


class SubscriptionManager:
    """Quản lý subscriptions cho Telegram users"""

    def __init__(self):
        self.subscriptions = self._load_subscriptions()

    def _load_subscriptions(self) -> Dict:
        """Load subscriptions từ file"""
        if not os.path.exists(SUBSCRIPTIONS_FILE):
            return {
                "users": {},  # {user_id: {symbols: set(), sectors: set()}}
                "symbol_subscribers": {},  # {symbol: set(user_ids)}
                "sector_subscribers": {},  # {sector: set(user_ids)}
            }

        try:
            with open(SUBSCRIPTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert lists back to sets
                for user_id, user_data in data.get("users", {}).items():
                    user_data["symbols"] = set(user_data.get("symbols", []))
                    user_data["sectors"] = set(user_data.get("sectors", []))

                for key, value in data.get("symbol_subscribers", {}).items():
                    data["symbol_subscribers"][key] = set(value)

                for key, value in data.get("sector_subscribers", {}).items():
                    data["sector_subscribers"][key] = set(value)

                return data
        except Exception:
            return {
                "users": {},
                "symbol_subscribers": {},
                "sector_subscribers": {},
            }

    def _save_subscriptions(self):
        """Lưu subscriptions vào file"""
        # Convert sets to lists for JSON serialization
        data = {
            "users": {},
            "symbol_subscribers": {},
            "sector_subscribers": {},
        }

        for user_id, user_data in self.subscriptions["users"].items():
            data["users"][user_id] = {
                "symbols": list(user_data.get("symbols", set())),
                "sectors": list(user_data.get("sectors", set())),
            }

        for symbol, user_ids in self.subscriptions["symbol_subscribers"].items():
            data["symbol_subscribers"][symbol] = list(user_ids)

        for sector, user_ids in self.subscriptions["sector_subscribers"].items():
            data["sector_subscribers"][sector] = list(user_ids)

        with open(SUBSCRIPTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def subscribe_symbol(self, user_id: int, symbol: str) -> bool:
        """Đăng ký nhận tin cho một symbol"""
        symbol = symbol.upper()
        user_id_str = str(user_id)

        if user_id_str not in self.subscriptions["users"]:
            self.subscriptions["users"][user_id_str] = {
                "symbols": set(),
                "sectors": set(),
            }

        self.subscriptions["users"][user_id_str]["symbols"].add(symbol)

        if symbol not in self.subscriptions["symbol_subscribers"]:
            self.subscriptions["symbol_subscribers"][symbol] = set()
        self.subscriptions["symbol_subscribers"][symbol].add(user_id_str)

        self._save_subscriptions()
        return True

    def unsubscribe_symbol(self, user_id: int, symbol: str) -> bool:
        """Hủy đăng ký nhận tin cho một symbol"""
        symbol = symbol.upper()
        user_id_str = str(user_id)

        if user_id_str in self.subscriptions["users"]:
            self.subscriptions["users"][user_id_str]["symbols"].discard(symbol)

        if symbol in self.subscriptions["symbol_subscribers"]:
            self.subscriptions["symbol_subscribers"][symbol].discard(user_id_str)
            if not self.subscriptions["symbol_subscribers"][symbol]:
                del self.subscriptions["symbol_subscribers"][symbol]

        self._save_subscriptions()
        return True

    def subscribe_sector(self, user_id: int, sector: str) -> bool:
        """Đăng ký nhận tin cho một sector"""
        sector = sector.upper()
        user_id_str = str(user_id)

        if user_id_str not in self.subscriptions["users"]:
            self.subscriptions["users"][user_id_str] = {
                "symbols": set(),
                "sectors": set(),
            }

        self.subscriptions["users"][user_id_str]["sectors"].add(sector)

        if sector not in self.subscriptions["sector_subscribers"]:
            self.subscriptions["sector_subscribers"][sector] = set()
        self.subscriptions["sector_subscribers"][sector].add(user_id_str)

        self._save_subscriptions()
        return True

    def unsubscribe_sector(self, user_id: int, sector: str) -> bool:
        """Hủy đăng ký nhận tin cho một sector"""
        sector = sector.upper()
        user_id_str = str(user_id)

        if user_id_str in self.subscriptions["users"]:
            self.subscriptions["users"][user_id_str]["sectors"].discard(sector)

        if sector in self.subscriptions["sector_subscribers"]:
            self.subscriptions["sector_subscribers"][sector].discard(user_id_str)
            if not self.subscriptions["sector_subscribers"][sector]:
                del self.subscriptions["sector_subscribers"][sector]

        self._save_subscriptions()
        return True

    def get_user_subscriptions(self, user_id: int) -> Dict:
        """Lấy danh sách subscriptions của user"""
        user_id_str = str(user_id)
        user_data = self.subscriptions["users"].get(
            user_id_str, {"symbols": set(), "sectors": set()}
        )
        return {
            "symbols": sorted(list(user_data.get("symbols", set()))),
            "sectors": sorted(list(user_data.get("sectors", set()))),
        }

    def get_symbol_subscribers(self, symbol: str) -> List[int]:
        """Lấy danh sách users đăng ký nhận tin cho symbol"""
        symbol = symbol.upper()
        user_ids = self.subscriptions["symbol_subscribers"].get(symbol, set())
        return [int(uid) for uid in user_ids]

    def get_sector_subscribers(self, sector: str) -> List[int]:
        """Lấy danh sách users đăng ký nhận tin cho sector"""
        sector = sector.upper()
        user_ids = self.subscriptions["sector_subscribers"].get(sector, set())
        return [int(uid) for uid in user_ids]

    def list_all_subscriptions(self) -> Dict:
        """Lấy tất cả subscriptions (for admin/debugging)"""
        return {
            "total_users": len(self.subscriptions["users"]),
            "total_symbol_subscriptions": len(self.subscriptions["symbol_subscribers"]),
            "total_sector_subscriptions": len(self.subscriptions["sector_subscribers"]),
        }
