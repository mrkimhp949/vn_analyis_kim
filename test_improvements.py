"""
Test script for improvements
Quick test of database, config, and monitoring
"""

import sys


def test_config():
    """Test centralized configuration"""
    print("=" * 60)
    print("1️⃣ TESTING CONFIGURATION")
    print("=" * 60)

    try:
        from trading_config import get_config

        config = get_config()
        print("✅ Config loaded successfully")

        # Test access
        print(f"  - Min Confidence: {config.trading.min_confidence}%")
        print(f"  - Lookback: {config.data.lookback}")
        print(f"  - Telegram Enabled: {config.telegram.enabled}")

        # Validate
        try:
            config.validate()
            print("✅ Config validation passed")
        except ValueError as e:
            print(f"⚠️ Config validation warning: {e}")

        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_database():
    """Test database functionality"""
    print("\n" + "=" * 60)
    print("2️⃣ TESTING DATABASE")
    print("=" * 60)

    try:
        from database import get_db

        db = get_db()
        print("✅ Database initialized")

        # Test save position
        db.save_position("TEST", 100, 50000, "2025-11-13", 5000000)
        print("✅ Save position works")

        # Test get positions
        positions = db.get_positions()
        print(f"✅ Get positions works: {len(positions)} positions")

        # Test save trade
        db.save_trade("TEST", "BUY", 100, 50000, 5000000, "2025-11-13", "Test trade")
        print("✅ Save trade works")

        # Test get trades
        trades = db.get_trades(limit=5)
        print(f"✅ Get trades works: {len(trades)} trades")

        # Cleanup test data
        db.delete_position("TEST")
        print("✅ Delete position works")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_portfolio_manager():
    """Test portfolio manager"""
    print("\n" + "=" * 60)
    print("3️⃣ TESTING PORTFOLIO MANAGER")
    print("=" * 60)

    try:
        from portfolio_manager import get_portfolio_manager

        manager = get_portfolio_manager()
        print("✅ Portfolio manager initialized")

        # Test add position
        manager.add_position("TEST2", 100, 60000)
        print("✅ Add position works")

        # Test get positions
        positions = manager.get_positions()
        print(f"✅ Get positions works: {len(positions)} positions")

        # Test portfolio value
        portfolio = manager.get_portfolio_value()
        print(f"✅ Portfolio value: {portfolio['total_value']:,.0f} VNĐ")

        # Test analysis
        analysis = manager.get_detailed_analysis()
        print("✅ Detailed analysis works")
        print("\n" + analysis)

        # Cleanup
        if "TEST2" in positions:
            manager.close_position("TEST2", 61000, "Test cleanup")
            print("✅ Close position works")

        return True

    except Exception as e:
        print(f"❌ Portfolio manager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_monitoring():
    """Test monitoring"""
    print("\n" + "=" * 60)
    print("4️⃣ TESTING MONITORING")
    print("=" * 60)

    try:
        from monitoring import get_performance_monitor, get_system_monitor

        # Performance monitor
        perf = get_performance_monitor()
        print("✅ Performance monitor initialized")

        # Track test trade
        perf.track_trade("TEST", 50000, 55000, 100, "2025-11-01", "2025-11-10")
        print("✅ Track trade works")

        # Get metrics
        metrics = perf.get_metrics()
        print(
            f"✅ Metrics: {metrics['total_trades']} trades, {metrics['win_rate']:.1f}% win rate"
        )

        # System monitor
        sys_mon = get_system_monitor()
        print("✅ System monitor initialized")

        # Track API call
        sys_mon.track_api_call("test_api", 0.5, True)
        print("✅ Track API call works")

        # Get stats
        stats = sys_mon.get_api_stats()
        print(f"✅ API stats: {len(stats)} APIs tracked")

        return True

    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility with old code"""
    print("\n" + "=" * 60)
    print("5️⃣ TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)

    try:
        # Test old config import
        from config import TELEGRAM_TOKEN, CHAT_ID, TICKERS

        print("✅ Old config imports work")
        print(f"  - Tickers: {len(TICKERS)} mã")

        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 TESTING IMPROVEMENTS")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Configuration", test_config()))
    results.append(("Database", test_database()))
    results.append(("Portfolio Manager", test_portfolio_manager()))
    results.append(("Monitoring", test_monitoring()))
    results.append(("Backward Compatibility", test_backward_compatibility()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n💡 Next steps:")
        print("  1. Run migration: python migrate_json_to_db.py")
        print("  2. Test bot: python bot_runner_improved.py")
        print("  3. Check MIGRATION_GUIDE.md for details")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Check errors above and fix issues")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
