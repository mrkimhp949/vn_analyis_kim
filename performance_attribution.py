# -*- coding: utf-8 -*-
"""
Performance Attribution Analysis
Phân tích nguồn gốc lợi nhuận - biết lợi nhuận đến từ đâu
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from database import get_db
from risk_metrics import get_sector_for_symbol
import logging

logger = logging.getLogger(__name__)


class PerformanceAttributionAnalyzer:
    """
    Phân tích performance attribution

    Trả lời các câu hỏi:
    - Ngành nào profitable nhất?
    - Signal strength nào tốt nhất?
    - Market regime nào trade tốt nhất?
    - Hold period optimal là bao nhiêu?
    - Entry time nào tốt nhất?
    """

    def __init__(self):
        self.db = get_db()

    def analyze_full_attribution(self, days: int = 90) -> Dict:
        """Phân tích toàn diện performance attribution"""
        trades = self.db.get_trades(limit=1000)

        if not trades:
            return {"total_trades": 0, "message": "Chưa có giao dịch nào để phân tích"}

        # Filter by date
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        trades = [t for t in trades if t["trade_date"] >= cutoff_date]

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)

        # Calculate PnL for each trade
        df = self._calculate_trade_pnl(df)

        return {
            "total_trades": len(df),
            "total_pnl": float(df["pnl"].sum()),
            "avg_pnl": float(df["pnl"].mean()),
            "win_rate": float((df["pnl"] > 0).sum() / len(df) * 100),
            "by_sector": self._analyze_by_sector(df),
            "by_signal_strength": self._analyze_by_signal_strength(df),
            "by_market_regime": self._analyze_by_market_regime(df),
            "by_hold_period": self._analyze_by_hold_period(df),
            "by_entry_time": self._analyze_by_entry_time(df),
            "by_month": self._analyze_by_month(df),
            "top_winners": self._get_top_trades(df, top_n=5, best=True),
            "top_losers": self._get_top_trades(df, top_n=5, best=False),
        }

    def _calculate_trade_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính PnL cho mỗi trade"""
        # Group by symbol to match BUY with SELL
        df["pnl"] = 0.0
        df["hold_days"] = 0

        for symbol in df["symbol"].unique():
            symbol_trades = df[df["symbol"] == symbol].sort_values("trade_date")

            buy_trades = symbol_trades[symbol_trades["action"] == "BUY"]
            sell_trades = symbol_trades[symbol_trades["action"] == "SELL"]

            for _, sell in sell_trades.iterrows():
                # Find matching buy
                matching_buys = buy_trades[
                    buy_trades["trade_date"] < sell["trade_date"]
                ]
                if not matching_buys.empty:
                    buy = matching_buys.iloc[-1]

                    # Calculate PnL
                    pnl = (sell["price"] - buy["price"]) * sell["shares"]
                    df.loc[sell.name, "pnl"] = pnl

                    # Calculate hold days
                    try:
                        buy_date = pd.to_datetime(buy["trade_date"], format="mixed")
                        sell_date = pd.to_datetime(sell["trade_date"], format="mixed")
                        hold_days = (sell_date - buy_date).days
                        df.loc[sell.name, "hold_days"] = hold_days
                    except Exception:
                        df.loc[sell.name, "hold_days"] = 0

        return df

    def _analyze_by_sector(self, df: pd.DataFrame) -> Dict:
        """Phân tích theo ngành"""
        df["sector"] = df["symbol"].apply(get_sector_for_symbol)

        sector_stats = {}
        for sector in df["sector"].unique():
            sector_df = df[df["sector"] == sector]
            sector_stats[sector] = {
                "total_trades": len(sector_df),
                "total_pnl": float(sector_df["pnl"].sum()),
                "avg_pnl": float(sector_df["pnl"].mean()),
                "win_rate": (
                    float((sector_df["pnl"] > 0).sum() / len(sector_df) * 100)
                    if len(sector_df) > 0
                    else 0
                ),
                "best_stock": (
                    sector_df.groupby("symbol")["pnl"].sum().idxmax()
                    if len(sector_df) > 0
                    else None
                ),
            }

        # Sort by total PnL
        sector_stats = dict(
            sorted(sector_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        )

        return sector_stats

    def _analyze_by_signal_strength(self, df: pd.DataFrame) -> Dict:
        """Phân tích theo signal strength"""
        # Extract signal strength from metadata
        df["signal_strength"] = df["metadata"].apply(
            lambda x: eval(x).get("signal_strength", "UNKNOWN") if x else "UNKNOWN"
        )

        strength_stats = {}
        for strength in df["signal_strength"].unique():
            strength_df = df[df["signal_strength"] == strength]
            strength_stats[strength] = {
                "total_trades": len(strength_df),
                "total_pnl": float(strength_df["pnl"].sum()),
                "avg_pnl": float(strength_df["pnl"].mean()),
                "win_rate": (
                    float((strength_df["pnl"] > 0).sum() / len(strength_df) * 100)
                    if len(strength_df) > 0
                    else 0
                ),
            }

        return strength_stats

    def _analyze_by_market_regime(self, df: pd.DataFrame) -> Dict:
        """Phân tích theo market regime"""
        # Extract market regime from metadata
        df["market_regime"] = df["metadata"].apply(
            lambda x: eval(x).get("market_regime", "UNKNOWN") if x else "UNKNOWN"
        )

        regime_stats = {}
        for regime in df["market_regime"].unique():
            regime_df = df[df["market_regime"] == regime]
            regime_stats[regime] = {
                "total_trades": len(regime_df),
                "total_pnl": float(regime_df["pnl"].sum()),
                "avg_pnl": float(regime_df["pnl"].mean()),
                "win_rate": (
                    float((regime_df["pnl"] > 0).sum() / len(regime_df) * 100)
                    if len(regime_df) > 0
                    else 0
                ),
            }

        return regime_stats

    def _analyze_by_hold_period(self, df: pd.DataFrame) -> Dict:
        """Phân tích theo hold period"""
        # Categorize hold periods
        df["hold_category"] = pd.cut(
            df["hold_days"],
            bins=[0, 3, 7, 14, 30, 60, 999],
            labels=[
                "0-3 days",
                "3-7 days",
                "7-14 days",
                "14-30 days",
                "30-60 days",
                "60+ days",
            ],
        )

        hold_stats = {}
        for category in df["hold_category"].unique():
            if pd.isna(category):
                continue
            category_df = df[df["hold_category"] == category]
            hold_stats[str(category)] = {
                "total_trades": len(category_df),
                "total_pnl": float(category_df["pnl"].sum()),
                "avg_pnl": float(category_df["pnl"].mean()),
                "win_rate": (
                    float((category_df["pnl"] > 0).sum() / len(category_df) * 100)
                    if len(category_df) > 0
                    else 0
                ),
            }

        return hold_stats

    def _analyze_by_entry_time(self, df: pd.DataFrame) -> Dict:
        """Phân tích theo thời gian entry"""
        df["entry_hour"] = pd.to_datetime(
            df["trade_date"], format="mixed", errors="coerce"
        ).dt.hour

        # Categorize by trading session
        def get_session(hour):
            if 9 <= hour < 11:
                return "Morning (9-11h)"
            elif 11 <= hour < 13:
                return "Pre-Lunch (11-13h)"
            elif 13 <= hour < 15:
                return "Afternoon (13-15h)"
            else:
                return "Other"

        df["session"] = df["entry_hour"].apply(get_session)

        session_stats = {}
        for session in df["session"].unique():
            session_df = df[df["session"] == session]
            session_stats[session] = {
                "total_trades": len(session_df),
                "total_pnl": float(session_df["pnl"].sum()),
                "avg_pnl": float(session_df["pnl"].mean()),
                "win_rate": (
                    float((session_df["pnl"] > 0).sum() / len(session_df) * 100)
                    if len(session_df) > 0
                    else 0
                ),
            }

        return session_stats

    def _analyze_by_month(self, df: pd.DataFrame) -> Dict:
        """Phân tích theo tháng"""
        df["month"] = pd.to_datetime(
            df["trade_date"], format="mixed", errors="coerce"
        ).dt.to_period("M")

        month_stats = {}
        for month in df["month"].unique():
            month_df = df[df["month"] == month]
            month_stats[str(month)] = {
                "total_trades": len(month_df),
                "total_pnl": float(month_df["pnl"].sum()),
                "avg_pnl": float(month_df["pnl"].mean()),
                "win_rate": (
                    float((month_df["pnl"] > 0).sum() / len(month_df) * 100)
                    if len(month_df) > 0
                    else 0
                ),
            }

        return month_stats

    def _get_top_trades(
        self, df: pd.DataFrame, top_n: int = 5, best: bool = True
    ) -> List[Dict]:
        """Lấy top trades (best hoặc worst)"""
        sorted_df = df.sort_values("pnl", ascending=not best)
        top_trades = sorted_df.head(top_n)

        return [
            {
                "symbol": row["symbol"],
                "pnl": float(row["pnl"]),
                "pnl_percent": (
                    float((row["pnl"] / row["total_value"]) * 100)
                    if row["total_value"] > 0
                    else 0
                ),
                "hold_days": int(row["hold_days"]),
                "entry_date": row["trade_date"],
            }
            for _, row in top_trades.iterrows()
        ]

    def format_attribution_report(self, attribution: Dict) -> str:
        """Format báo cáo attribution"""
        if attribution.get("total_trades", 0) == 0:
            return "📊 Chưa có dữ liệu để phân tích"

        lines = []
        lines.append("📊 **PERFORMANCE ATTRIBUTION ANALYSIS**")
        lines.append("=" * 50)
        lines.append("")

        # Overall stats
        lines.append("📈 **TỔNG QUAN**")
        lines.append(f"• Tổng giao dịch: {attribution['total_trades']}")
        lines.append(f"• Tổng P&L: {attribution['total_pnl']:+,.0f} VNĐ")
        lines.append(f"• P&L trung bình: {attribution['avg_pnl']:+,.0f} VNĐ")
        lines.append(f"• Win rate: {attribution['win_rate']:.1f}%")
        lines.append("")

        # By sector
        if attribution.get("by_sector"):
            lines.append("🏢 **THEO NGÀNH**")
            for sector, stats in list(attribution["by_sector"].items())[:5]:
                lines.append(f"• {sector}:")
                lines.append(
                    f"  └ P&L: {stats['total_pnl']:+,.0f} VNĐ | Win rate: {stats['win_rate']:.1f}% | {stats['total_trades']} trades"
                )
            lines.append("")

        # By signal strength
        if attribution.get("by_signal_strength"):
            lines.append("💪 **THEO SIGNAL STRENGTH**")
            for strength, stats in attribution["by_signal_strength"].items():
                lines.append(f"• {strength}:")
                lines.append(
                    f"  └ P&L: {stats['total_pnl']:+,.0f} VNĐ | Win rate: {stats['win_rate']:.1f}%"
                )
            lines.append("")

        # By hold period
        if attribution.get("by_hold_period"):
            lines.append("⏱️ **THEO HOLD PERIOD**")
            for period, stats in attribution["by_hold_period"].items():
                lines.append(f"• {period}:")
                lines.append(
                    f"  └ P&L: {stats['total_pnl']:+,.0f} VNĐ | Win rate: {stats['win_rate']:.1f}%"
                )
            lines.append("")

        # Top winners
        if attribution.get("top_winners"):
            lines.append("🏆 **TOP WINNERS**")
            for trade in attribution["top_winners"]:
                lines.append(
                    f"• {trade['symbol']}: {trade['pnl']:+,.0f} VNĐ ({trade['pnl_percent']:+.1f}%) - {trade['hold_days']} days"
                )
            lines.append("")

        # Top losers
        if attribution.get("top_losers"):
            lines.append("💔 **TOP LOSERS**")
            for trade in attribution["top_losers"]:
                lines.append(
                    f"• {trade['symbol']}: {trade['pnl']:+,.0f} VNĐ ({trade['pnl_percent']:+.1f}%) - {trade['hold_days']} days"
                )
            lines.append("")

        return "\n".join(lines)


# Singleton
_analyzer = None


def get_attribution_analyzer() -> PerformanceAttributionAnalyzer:
    """Get analyzer singleton"""
    global _analyzer
    if _analyzer is None:
        _analyzer = PerformanceAttributionAnalyzer()
    return _analyzer


# Test
if __name__ == "__main__":
    print("Testing Performance Attribution...")

    analyzer = PerformanceAttributionAnalyzer()
    attribution = analyzer.analyze_full_attribution(days=90)
    report = analyzer.format_attribution_report(attribution)

    print(report)
    print("\n✅ Test completed!")
