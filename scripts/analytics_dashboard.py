# -*- coding: utf-8 -*-
"""
Analytics Dashboard
Tổng hợp tất cả analytics và monitoring
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """
    Dashboard tổng hợp analytics

    Features:
    - Performance attribution
    - Portfolio analysis
    - Cache statistics
    - System monitoring
    - Real-time alerts
    """

    def __init__(self):
        self.last_update = None

    async def get_full_dashboard(self) -> Dict:
        """Lấy toàn bộ dashboard data"""
        print("📊 Generating analytics dashboard...")

        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "performance_attribution": self._get_performance_attribution(),
            "portfolio_analysis": self._get_portfolio_analysis(),
            "cache_stats": self._get_cache_stats(),
            "system_health": self._get_system_health(),
            "monitoring_status": self._get_monitoring_status(),
        }

        self.last_update = datetime.now()
        return dashboard

    def _get_performance_attribution(self) -> Dict:
        """Get performance attribution"""
        try:
            from performance_attribution import get_attribution_analyzer

            analyzer = get_attribution_analyzer()
            attribution = analyzer.analyze_full_attribution(days=90)

            return {"status": "success", "data": attribution}
        except Exception as e:
            logger.error(f"Error getting performance attribution: {e}")
            return {"status": "error", "error": str(e)}

    def _get_portfolio_analysis(self) -> Dict:
        """Get portfolio analysis"""
        try:
            from portfolio_manager import get_portfolio_manager

            pm = get_portfolio_manager()
            positions = pm.get_positions()
            portfolio_value = pm.get_portfolio_value()

            return {
                "status": "success",
                "data": {
                    "num_positions": len(positions),
                    "total_value": portfolio_value["total_value"],
                    "total_pnl": portfolio_value["total_pnl"],
                    "total_return_pct": portfolio_value["total_return_pct"],
                },
            }
        except Exception as e:
            logger.error(f"Error getting portfolio analysis: {e}")
            return {"status": "error", "error": str(e)}

    def _get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            from smart_cache import get_cache

            cache = get_cache()
            stats = cache.get_stats()

            return {"status": "success", "data": stats}
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}

    def _get_system_health(self) -> Dict:
        """Get system health"""
        try:
            from monitoring import get_system_monitor

            monitor = get_system_monitor()
            api_stats = monitor.get_api_stats()

            return {
                "status": "success",
                "data": {"api_stats": api_stats, "error_count": len(monitor.errors)},
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"status": "error", "error": str(e)}

    def _get_monitoring_status(self) -> Dict:
        """Get monitoring status"""
        try:
            from realtime_monitor import get_realtime_monitor

            monitor = get_realtime_monitor()
            status = monitor.get_monitoring_status()

            return {"status": "success", "data": status}
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {"status": "error", "error": str(e)}

    def format_dashboard_report(self, dashboard: Dict) -> str:
        """Format dashboard report"""
        lines = []
        lines.append("📊 **ANALYTICS DASHBOARD**")
        lines.append("=" * 60)
        lines.append(f"Generated: {dashboard['generated_at'][:19]}")
        lines.append("")

        # Performance Attribution
        perf = dashboard.get("performance_attribution", {})
        if perf.get("status") == "success" and perf.get("data"):
            data = perf["data"]
            lines.append("📈 **PERFORMANCE ATTRIBUTION**")
            lines.append(f"• Total trades: {data.get('total_trades', 0)}")
            lines.append(f"• Total P&L: {data.get('total_pnl', 0):+,.0f} VNĐ")
            lines.append(f"• Win rate: {data.get('win_rate', 0):.1f}%")

            # Top sector
            by_sector = data.get("by_sector", {})
            if by_sector:
                top_sector = max(by_sector.items(), key=lambda x: x[1]["total_pnl"])
                lines.append(
                    f"• Best sector: {top_sector[0]} ({top_sector[1]['total_pnl']:+,.0f} VNĐ)"
                )
            lines.append("")

        # Portfolio
        portfolio = dashboard.get("portfolio_analysis", {})
        if portfolio.get("status") == "success" and portfolio.get("data"):
            data = portfolio["data"]
            lines.append("💼 **PORTFOLIO**")
            lines.append(f"• Positions: {data.get('num_positions', 0)}")
            lines.append(f"• Total value: {data.get('total_value', 0):,.0f} VNĐ")
            lines.append(
                f"• Total P&L: {data.get('total_pnl', 0):+,.0f} VNĐ ({data.get('total_return_pct', 0):+.1f}%)"
            )
            lines.append("")

        # Cache Stats
        cache = dashboard.get("cache_stats", {})
        if cache.get("status") == "success" and cache.get("data"):
            data = cache["data"]
            lines.append("💾 **CACHE PERFORMANCE**")
            lines.append(f"• Hit rate: {data.get('hit_rate', 0):.1f}%")
            lines.append(
                f"• Hits: {data.get('hits', 0)} | Misses: {data.get('misses', 0)}"
            )
            lines.append(f"• Memory entries: {data.get('memory_entries', 0)}")
            lines.append("")

        # System Health
        health = dashboard.get("system_health", {})
        if health.get("status") == "success" and health.get("data"):
            data = health["data"]
            lines.append("🏥 **SYSTEM HEALTH**")

            api_stats = data.get("api_stats", {})
            if api_stats:
                for api_name, stats in list(api_stats.items())[:3]:
                    lines.append(
                        f"• {api_name}: {stats.get('success_rate', 0):.1f}% success"
                    )

            error_count = data.get("error_count", 0)
            lines.append(f"• Recent errors: {error_count}")
            lines.append("")

        # Monitoring
        monitoring = dashboard.get("monitoring_status", {})
        if monitoring.get("status") == "success" and monitoring.get("data"):
            data = monitoring["data"]
            lines.append("🔍 **REAL-TIME MONITORING**")
            lines.append(
                f"• Status: {'🟢 Running' if data.get('is_running') else '🔴 Stopped'}"
            )
            lines.append(f"• Monitored symbols: {data.get('monitored_symbols', 0)}")
            lines.append(f"• Check interval: {data.get('check_interval', 0)}s")
            lines.append("")

        return "\n".join(lines)

    async def send_dashboard_to_telegram(self, chat_id: str):
        """Gửi dashboard qua Telegram"""
        try:
            from telegram import Bot

            from config import TELEGRAM_TOKEN

            bot = Bot(token=TELEGRAM_TOKEN)

            # Get dashboard
            dashboard = await self.get_full_dashboard()

            # Format report
            report = self.format_dashboard_report(dashboard)

            # Send
            if len(report) > 4000:
                # Split into chunks
                chunks = [report[i : i + 4000] for i in range(0, len(report), 4000)]
                for chunk in chunks:
                    await bot.send_message(chat_id, chunk, parse_mode="Markdown")
            else:
                await bot.send_message(chat_id, report, parse_mode="Markdown")

            print("✅ Dashboard sent to Telegram")

        except Exception as e:
            logger.error(f"Error sending dashboard to Telegram: {e}")


# Singleton
_dashboard = None


def get_dashboard() -> AnalyticsDashboard:
    """Get dashboard singleton"""
    global _dashboard
    if _dashboard is None:
        _dashboard = AnalyticsDashboard()
    return _dashboard


# CLI command
async def show_dashboard():
    """Show dashboard in console"""
    dashboard_obj = get_dashboard()
    dashboard = await dashboard_obj.get_full_dashboard()
    report = dashboard_obj.format_dashboard_report(dashboard)
    print(report)


# Test
if __name__ == "__main__":
    print("Testing Analytics Dashboard...")

    asyncio.run(show_dashboard())

    print("\n✅ Test completed!")
