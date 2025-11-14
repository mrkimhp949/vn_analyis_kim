# -*- coding: utf-8 -*-
"""
Test Analytics Features
Kiểm tra tất cả tính năng analytics mới
"""
import asyncio
import sys


def test_imports():
    """Test tất cả imports"""
    print("🧪 Testing imports...")

    try:
        from performance_attribution import get_attribution_analyzer

        print("  ✅ performance_attribution")
    except Exception as e:
        print(f"  ❌ performance_attribution: {e}")
        return False

    try:
        from smart_cache import get_cache, cached

        print("  ✅ smart_cache")
    except Exception as e:
        print(f"  ❌ smart_cache: {e}")
        return False

    try:
        from parallel_scanner import ParallelScanner

        print("  ✅ parallel_scanner")
    except Exception as e:
        print(f"  ❌ parallel_scanner: {e}")
        return False

    try:
        from realtime_monitor import get_realtime_monitor

        print("  ✅ realtime_monitor")
    except Exception as e:
        print(f"  ❌ realtime_monitor: {e}")
        return False

    try:
        from walk_forward_test import WalkForwardTester

        print("  ✅ walk_forward_test")
    except Exception as e:
        print(f"  ❌ walk_forward_test: {e}")
        return False

    try:
        from analytics_dashboard import get_dashboard

        print("  ✅ analytics_dashboard")
    except Exception as e:
        print(f"  ❌ analytics_dashboard: {e}")
        return False

    return True


def test_cache():
    """Test cache functionality"""
    print("\n🧪 Testing cache...")

    try:
        from smart_cache import get_cache

        cache = get_cache()

        # Test set/get
        cache.set("test_key", {"value": 123})
        result = cache.get("test_key", ttl=60)

        if result and result["value"] == 123:
            print("  ✅ Cache set/get works")
        else:
            print("  ❌ Cache set/get failed")
            return False

        # Test stats
        stats = cache.get_stats()
        if "hits" in stats and "misses" in stats:
            print(f"  ✅ Cache stats: {stats['hits']} hits, {stats['misses']} misses")
        else:
            print("  ❌ Cache stats failed")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Cache test failed: {e}")
        return False


def test_parallel_scanner():
    """Test parallel scanner"""
    print("\n🧪 Testing parallel scanner...")

    try:
        from parallel_scanner import ParallelScanner
        import time

        def test_scan(symbol):
            time.sleep(0.01)  # Simulate work
            return {"symbol": symbol, "value": len(symbol)}

        scanner = ParallelScanner(max_workers=3, timeout=5)
        test_symbols = ["VCB", "FPT", "VNM"]

        results = scanner.scan_symbols(test_symbols, test_scan)

        if len(results) == len(test_symbols):
            success_count = sum(1 for r in results if r.success)
            print(f"  ✅ Parallel scanner: {success_count}/{len(test_symbols)} success")
            return True
        else:
            print("  ❌ Parallel scanner failed")
            return False

    except Exception as e:
        print(f"  ❌ Parallel scanner test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_attribution():
    """Test performance attribution"""
    print("\n🧪 Testing performance attribution...")

    try:
        from performance_attribution import get_attribution_analyzer

        analyzer = get_attribution_analyzer()

        # Try to analyze (may have no data)
        attribution = analyzer.analyze_full_attribution(days=30)

        if "total_trades" in attribution:
            print(f"  ✅ Attribution analysis: {attribution['total_trades']} trades")
            return True
        else:
            print("  ❌ Attribution analysis failed")
            return False

    except Exception as e:
        print(f"  ❌ Attribution test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_dashboard():
    """Test dashboard"""
    print("\n🧪 Testing dashboard...")

    try:
        from analytics_dashboard import get_dashboard

        dashboard_obj = get_dashboard()
        dashboard = await dashboard_obj.get_full_dashboard()

        if "generated_at" in dashboard:
            print("  ✅ Dashboard generation works")

            # Test report formatting
            report = dashboard_obj.format_dashboard_report(dashboard)
            if len(report) > 0:
                print("  ✅ Dashboard report formatting works")
                return True
            else:
                print("  ❌ Dashboard report formatting failed")
                return False
        else:
            print("  ❌ Dashboard generation failed")
            return False

    except Exception as e:
        print(f"  ❌ Dashboard test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sector_map():
    """Test SECTOR_MAP"""
    print("\n🧪 Testing SECTOR_MAP...")

    try:
        from risk_metrics import SECTOR_MAP, get_sector_for_symbol

        # Test some known symbols
        test_symbols = ["VCB", "FPT", "VNM", "HPG", "SSI"]

        for symbol in test_symbols:
            sector = get_sector_for_symbol(symbol)
            if sector and sector != "UNCLASSIFIED":
                print(f"  ✅ {symbol} -> {sector}")
            else:
                print(f"  ⚠️ {symbol} -> UNCLASSIFIED")

        print(f"  ✅ SECTOR_MAP has {len(SECTOR_MAP)} symbols")
        return True

    except Exception as e:
        print(f"  ❌ SECTOR_MAP test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🚀 TESTING ANALYTICS FEATURES")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test cache
    results.append(("Cache", test_cache()))

    # Test parallel scanner
    results.append(("Parallel Scanner", test_parallel_scanner()))

    # Test performance attribution
    results.append(("Performance Attribution", test_performance_attribution()))

    # Test dashboard
    results.append(("Dashboard", await test_dashboard()))

    # Test SECTOR_MAP
    results.append(("SECTOR_MAP", test_sector_map()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def test_threshold_tools():
    """Test threshold adjustment tools"""
    print("\n🧪 Testing threshold tools...")

    try:
        # Test imports
        import adjust_thresholds
        import analyze_no_signals

        print("  ✅ Threshold tools imported successfully")

        # Test RECOMMENDED_THRESHOLDS
        if hasattr(adjust_thresholds, "RECOMMENDED_THRESHOLDS"):
            profiles = adjust_thresholds.RECOMMENDED_THRESHOLDS
            if "balanced" in profiles and "conservative" in profiles:
                print(f"  ✅ Found {len(profiles)} threshold profiles")
                return True
            else:
                print("  ❌ Missing threshold profiles")
                return False
        else:
            print("  ❌ RECOMMENDED_THRESHOLDS not found")
            return False

    except Exception as e:
        print(f"  ❌ Threshold tools test failed: {e}")
        return False


# Update run_all_tests to include new test
async def run_all_tests_updated():
    """Run all tests including new ones"""
    print("=" * 60)
    print("🚀 TESTING ANALYTICS FEATURES")
    print("=" * 60)

    results = []

    # Existing tests
    results.append(("Imports", test_imports()))
    results.append(("Cache", test_cache()))
    results.append(("Parallel Scanner", test_parallel_scanner()))
    results.append(("Performance Attribution", test_performance_attribution()))
    results.append(("Dashboard", await test_dashboard()))
    results.append(("SECTOR_MAP", test_sector_map()))

    # New test
    results.append(("Threshold Tools", test_threshold_tools()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return False
