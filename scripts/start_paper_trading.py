#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Start Safe Paper Trading Session

This script initializes all safety guards and starts a paper trading session.
Run this before starting any trading activities.

Usage:
    python scripts/start_paper_trading.py
    python scripts/start_paper_trading.py --check-only  # Health check only
    python scripts/start_paper_trading.py --reset       # Reset all states

Author: Trading Bot Team
Version: 1.0.0
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/paper_trading_{datetime.now().strftime("%Y%m%d")}.log'),
    ],
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner"""
    print(
        """
╔═══════════════════════════════════════════════════════════════╗
║                  SAFE PAPER TRADING SYSTEM                     ║
║                    Vietnam Stock Market                        ║
╠═══════════════════════════════════════════════════════════════╣
║  Safety Guards:                                                ║
║  ✓ Kill Switch - Emergency stop                                ║
║  ✓ Order Guard - Duplicate prevention                          ║
║  ✓ Audit Logger - Complete trade history                       ║
║  ✓ Circuit Breaker - Loss protection                           ║
║  ✓ Position Reconciliation - Data integrity                    ║
╚═══════════════════════════════════════════════════════════════╝
    """
    )


def run_health_check() -> bool:
    """Run comprehensive health check"""
    print("\n🔍 Running Health Check...")
    print("=" * 50)

    all_ok = True

    # Check 1: Database
    print("\n1. Database Connection:")
    try:
        from src.data.database import get_db

        db = get_db()
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        all_ok = False

    # Check 2: Kill Switch
    print("\n2. Kill Switch:")
    try:
        from src.risk.kill_switch import get_kill_switch

        ks = get_kill_switch()
        can_trade, reason = ks.can_trade()
        if can_trade:
            print("   ✅ Kill switch OK - Trading enabled")
        else:
            print(f"   ⚠️ Kill switch: {reason}")
    except Exception as e:
        print(f"   ❌ Kill switch error: {e}")
        all_ok = False

    # Check 3: Circuit Breaker
    print("\n3. Circuit Breaker:")
    try:
        from src.risk.circuit_breaker import get_circuit_breaker

        cb = get_circuit_breaker()
        cb_ok, cb_reason = cb.can_trade()
        if cb_ok:
            print("   ✅ Circuit breaker OK")
        else:
            print(f"   ⚠️ Circuit breaker: {cb_reason}")
    except Exception as e:
        print(f"   ❌ Circuit breaker error: {e}")
        all_ok = False

    # Check 4: Order Guard
    print("\n4. Order Guard:")
    try:
        from src.risk.order_guard import get_order_guard

        og = get_order_guard()
        stats = og.get_statistics()
        print(f"   ✅ Order guard OK - {stats['pending_count']} pending orders")
    except Exception as e:
        print(f"   ❌ Order guard error: {e}")
        all_ok = False

    # Check 5: Audit Logger
    print("\n5. Audit Logger:")
    try:
        from src.monitoring.audit_logger import get_audit_logger

        audit = get_audit_logger()
        print(f"   ✅ Audit logger OK - Session: {audit.session_id}")
    except Exception as e:
        print(f"   ❌ Audit logger error: {e}")
        all_ok = False

    # Check 6: Paper Trading Account
    print("\n6. Paper Trading Account:")
    try:
        from src.portfolio.paper_trading import get_paper_account

        paper = get_paper_account()
        cash = paper.account.get("cash", 0)
        initial = paper.account.get("initial_capital", 0)
        print(f"   ✅ Paper account OK")
        print(f"      Cash: {cash:,.0f} VND")
        print(f"      Initial: {initial:,.0f} VND")
    except Exception as e:
        print(f"   ❌ Paper account error: {e}")
        all_ok = False

    # Check 7: Position Reconciliation
    print("\n7. Position Reconciliation:")
    try:
        from src.portfolio.reconciliation import get_position_reconciler

        recon = get_position_reconciler()
        mismatches = recon.check_positions()
        if len(mismatches) == 0:
            print("   ✅ All positions reconciled")
        else:
            print(f"   ⚠️ Found {len(mismatches)} mismatches")
            for m in mismatches[:3]:
                print(f"      - {m.symbol}: {m.mismatch_type}")
    except Exception as e:
        print(f"   ❌ Reconciliation error: {e}")
        all_ok = False

    # Check 8: Portfolio Manager
    print("\n8. Portfolio Manager:")
    try:
        from src.portfolio.manager import get_portfolio_manager

        pm = get_portfolio_manager()
        positions = pm.get_positions()  # Use get_positions() not get_active_positions()
        print(f"   ✅ Portfolio OK - {len(positions)} active positions")
    except Exception as e:
        print(f"   ❌ Portfolio error: {e}")
        all_ok = False

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ ALL HEALTH CHECKS PASSED - Ready for paper trading!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before trading")

    return all_ok


def reset_all_states():
    """Reset all trading states"""
    print("\n⚠️ Resetting all trading states...")

    # Reset kill switch
    try:
        from src.risk.kill_switch import get_kill_switch

        ks = get_kill_switch()
        ks.resume("System reset")
        print("   ✅ Kill switch reset")
    except Exception as e:
        print(f"   ❌ Kill switch reset failed: {e}")

    # Reset order guard
    try:
        from src.risk.order_guard import get_order_guard

        og = get_order_guard()
        cancelled = og.cancel_all_pending()
        print(f"   ✅ Order guard reset - {cancelled} orders cancelled")
    except Exception as e:
        print(f"   ❌ Order guard reset failed: {e}")

    # Reset circuit breaker
    try:
        from src.risk.circuit_breaker import get_circuit_breaker

        cb = get_circuit_breaker()
        cb.reset_daily_stats()
        print("   ✅ Circuit breaker reset")
    except Exception as e:
        print(f"   ❌ Circuit breaker reset failed: {e}")

    print("\n✅ All states reset")


def show_status():
    """Show current trading status"""
    print("\n📊 Current Trading Status")
    print("=" * 50)

    try:
        from src.portfolio.safe_paper_trading import get_safe_paper_trader

        trader = get_safe_paper_trader()
        status = trader.get_safety_status()

        print(f"\n🕐 Timestamp: {status['timestamp']}")
        print(f"🚦 Can Trade: {'✅ YES' if status['can_trade'] else '❌ NO'}")

        if status.get("reasons"):
            print("\n⚠️ Issues:")
            for reason in status["reasons"]:
                print(f"   - {reason}")

        if status.get("kill_switch"):
            ks = status["kill_switch"]
            print(f"\n🔴 Kill Switch: {ks['state']}")

        if status.get("circuit_breaker"):
            cb = status["circuit_breaker"]
            print(f"⚡ Circuit Breaker: {'OK' if cb['can_trade'] else cb['reason']}")

        if status.get("paper_account"):
            pa = status["paper_account"]
            print(f"\n💰 Paper Account:")
            print(f"   Cash: {pa['cash']:,.0f} VND")
            print(f"   Initial: {pa['initial_capital']:,.0f} VND")

        if status.get("order_guard"):
            og = status["order_guard"]
            print(f"\n📝 Order Guard:")
            print(f"   Pending: {og['pending_count']}")
            print(f"   Filled: {og['filled_count']}")

    except Exception as e:
        print(f"❌ Error getting status: {e}")


def start_trading_session():
    """Start a new trading session"""
    print_banner()

    # Run health check first
    if not run_health_check():
        print("\n❌ Health check failed. Fix issues before starting.")
        return False

    # Show status
    show_status()

    # Log session start
    try:
        from src.monitoring.audit_logger import get_audit_logger

        audit = get_audit_logger()
        logger.info(f"📋 Paper trading session started: {audit.session_id}")
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("✅ PAPER TRADING SESSION STARTED")
    print("=" * 50)
    print("\nCommands available:")
    print("  - Use get_safe_paper_trader() for safe trading")
    print("  - Call trader.pause_trading() to pause")
    print("  - Call trader.kill_trading() for emergency stop")
    print("  - Call trader.get_safety_status() for status")
    print("\nHappy trading! 🚀")

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Safe Paper Trading Manager")
    parser.add_argument(
        "--check-only", action="store_true", help="Run health check only, don't start session"
    )
    parser.add_argument("--reset", action="store_true", help="Reset all trading states")
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    Path("logs/audit").mkdir(exist_ok=True)

    if args.reset:
        reset_all_states()
    elif args.check_only:
        run_health_check()
    elif args.status:
        show_status()
    else:
        start_trading_session()


if __name__ == "__main__":
    main()
