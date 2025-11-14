#!/usr/bin/env python
"""
Performance Metrics Viewer
Display trading bot performance metrics
"""
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_portfolio_metrics() -> Dict:
    """Get portfolio performance metrics"""
    from portfolio_manager import get_portfolio_manager

    manager = get_portfolio_manager()
    positions = manager.get_positions()
    portfolio = manager.get_portfolio_value()

    return {
        "num_positions": portfolio["num_positions"],
        "total_value": portfolio["total_value"],
        "total_cost": portfolio["total_cost"],
        "pnl": portfolio["pnl"],
        "pnl_percent": portfolio["pnl_percent"],
        "positions": positions,
    }


def get_trade_statistics() -> Dict:
    """Get trade statistics from database"""
    from database import get_db

    db = get_db()

    with db.get_connection() as conn:
        # Get total trades
        cursor = conn.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]

        if total_trades == 0:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "recent_trades": [],
            }

        # Get buy/sell counts
        cursor = conn.execute("SELECT action, COUNT(*) FROM trades GROUP BY action")
        action_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Get recent trades
        cursor = conn.execute(
            """
            SELECT symbol, action, shares, price, trade_date, reason
            FROM trades
            ORDER BY trade_date DESC
            LIMIT 10
            """
        )
        recent_trades = [
            {
                "symbol": row[0],
                "action": row[1],
                "shares": row[2],
                "price": row[3],
                "date": row[4],
                "reason": row[5],
            }
            for row in cursor.fetchall()
        ]

    return {
        "total_trades": total_trades,
        "buy_trades": action_counts.get("BUY", 0),
        "sell_trades": action_counts.get("SELL", 0),
        "recent_trades": recent_trades,
    }


def get_paper_trading_stats() -> Dict:
    """Get paper trading statistics"""
    try:
        from paper_trading import get_paper_account

        account = get_paper_account()
        stats = account.get_statistics()

        return stats if stats else {}
    except Exception:
        return {}


def display_dashboard():
    """Display metrics dashboard"""
    print("=" * 80)
    print("📊 TRADING BOT PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Portfolio Metrics
    try:
        print("💼 PORTFOLIO METRICS")
        print("-" * 80)

        portfolio = get_portfolio_metrics()

        print(f"Positions:     {portfolio['num_positions']}")
        print(f"Total Value:   {portfolio['total_value']:,.0f} VNĐ")
        print(f"Total Cost:    {portfolio['total_cost']:,.0f} VNĐ")
        print(
            f"P&L:           {portfolio['pnl']:+,.0f} VNĐ ({portfolio['pnl_percent']:+.2f}%)"
        )

        if portfolio["positions"]:
            print(f"\nCurrent Positions:")
            for symbol, pos in portfolio["positions"].items():
                shares = pos["shares"]
                avg_price = pos["avg_price"]
                print(f"  • {symbol:6s} {shares:>6,} shares @ {avg_price:>10,.0f} VNĐ")

    except Exception as e:
        print(f"❌ Error loading portfolio metrics: {e}")

    print()

    # Trade Statistics
    try:
        print("📈 TRADE STATISTICS")
        print("-" * 80)

        trades = get_trade_statistics()

        print(f"Total Trades:  {trades['total_trades']}")
        print(f"  - Buy:       {trades['buy_trades']}")
        print(f"  - Sell:      {trades['sell_trades']}")

        if trades["recent_trades"]:
            print(f"\nRecent Trades (last 10):")
            for trade in trades["recent_trades"]:
                action_emoji = "🟢" if trade["action"] == "BUY" else "🔴"
                date = trade["date"][:10] if trade["date"] else "N/A"
                print(
                    f"  {action_emoji} {date} {trade['symbol']:6s} "
                    f"{trade['action']:4s} {trade['shares']:>4} @ {trade['price']:>10,.0f} "
                    f"- {trade['reason']}"
                )

    except Exception as e:
        print(f"❌ Error loading trade statistics: {e}")

    print()

    # Paper Trading
    try:
        print("📝 PAPER TRADING")
        print("-" * 80)

        paper = get_paper_trading_stats()

        if paper:
            print(f"Balance:       {paper.get('balance', 0):,.0f} VNĐ")
            print(f"Total P&L:     {paper.get('total_pnl', 0):+,.0f} VNĐ")
            print(f"Total Trades:  {paper.get('total_trades', 0)}")
            print(f"Win Rate:      {paper.get('win_rate', 0):.1f}%")
            print(f"Avg Profit:    {paper.get('avg_profit', 0):,.0f} VNĐ")
            print(f"Avg Loss:      {paper.get('avg_loss', 0):,.0f} VNĐ")
        else:
            print("No paper trading data available")

    except Exception as e:
        print(f"❌ Error loading paper trading stats: {e}")

    print()

    # System Info
    try:
        print("🖥️  SYSTEM INFO")
        print("-" * 80)

        # Check models
        from ml_models import MLPredictor

        predictor = MLPredictor()
        loaded = predictor.load_models()

        model_status = "Loaded" if loaded and predictor.rf_model else "Dummy Mode"
        print(f"ML Models:     {model_status}")

        # Check database size
        import os

        if os.path.exists("trading.db"):
            db_size = os.path.getsize("trading.db") / (1024 * 1024)  # MB
            print(f"Database Size: {db_size:.2f} MB")

        # Check cache
        if os.path.exists("data_cache"):
            cache_files = [f for f in os.listdir("data_cache") if f.endswith(".pkl")]
            print(f"Cached Tickers: {len(cache_files)}")

    except Exception as e:
        print(f"❌ Error loading system info: {e}")

    print("\n" + "=" * 80)


def export_metrics_json():
    """Export metrics as JSON"""
    import json

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "portfolio": get_portfolio_metrics(),
        "trades": get_trade_statistics(),
        "paper_trading": get_paper_trading_stats(),
    }

    return json.dumps(metrics, indent=2, default=str)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="View trading bot metrics")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    parser.add_argument("--export", metavar="FILE", help="Export metrics to JSON file")

    args = parser.parse_args()

    if args.json or args.export:
        json_output = export_metrics_json()

        if args.export:
            with open(args.export, "w") as f:
                f.write(json_output)
            print(f"✅ Metrics exported to {args.export}")
        else:
            print(json_output)
    else:
        display_dashboard()


if __name__ == "__main__":
    main()
