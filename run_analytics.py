# -*- coding: utf-8 -*-
"""
Run Analytics
Script để chạy các analytics và monitoring
"""
import asyncio
import argparse
from datetime import datetime


async def run_performance_attribution(days: int = 90):
    """Chạy performance attribution analysis"""
    print(f"\n📊 PERFORMANCE ATTRIBUTION ANALYSIS ({days} days)")
    print("=" * 60)
    
    from performance_attribution import get_attribution_analyzer
    
    analyzer = get_attribution_analyzer()
    attribution = analyzer.analyze_full_attribution(days=days)
    report = analyzer.format_attribution_report(attribution)
    
    print(report)
    
    # Save to file
    filename = f"reports/attribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        import os
        os.makedirs('reports', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ Saved to {filename}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")


async def run_walk_forward_test(symbols: list = None):
    """Chạy walk-forward test"""
    print("\n🔄 WALK-FORWARD TEST")
    print("=" * 60)
    
    from walk_forward_test import WalkForwardTester, example_strategy
    
    if symbols is None:
        symbols = ['VCB', 'FPT', 'VNM', 'HPG']
    
    tester = WalkForwardTester(
        train_period=180,
        test_period=30,
        step=30
    )
    
    results = tester.run_walk_forward(
        symbols=symbols,
        strategy_function=example_strategy,
        initial_capital=100_000_000
    )
    
    report = tester.format_report(results)
    print(report)
    
    # Save to file
    filename = f"reports/walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        import os
        os.makedirs('reports', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ Saved to {filename}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")


async def show_cache_stats():
    """Hiển thị cache statistics"""
    print("\n💾 CACHE STATISTICS")
    print("=" * 60)
    
    from smart_cache import get_cache
    
    cache = get_cache()
    stats = cache.get_stats()
    
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.1f}%")
    print(f"Memory Entries: {stats['memory_entries']}")
    print(f"Total Saves: {stats['saves']}")


async def cleanup_cache():
    """Cleanup expired cache"""
    print("\n🧹 CLEANING UP CACHE")
    print("=" * 60)
    
    from smart_cache import get_cache
    
    cache = get_cache()
    
    print("Cleaning up expired entries...")
    cache.cleanup_expired(ttl=86400)  # 24 hours
    
    stats = cache.get_stats()
    print(f"✅ Done. Current entries: {stats['memory_entries']}")


async def start_realtime_monitoring(interval: int = 300):
    """Bắt đầu real-time monitoring"""
    print(f"\n🔍 STARTING REAL-TIME MONITORING (interval: {interval}s)")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    
    from realtime_monitor import get_realtime_monitor
    
    monitor = get_realtime_monitor(check_interval=interval, with_telegram=True)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitoring...")
        monitor.stop_monitoring()


async def show_dashboard():
    """Hiển thị analytics dashboard"""
    print("\n📊 ANALYTICS DASHBOARD")
    print("=" * 60)
    
    from analytics_dashboard import get_dashboard
    
    dashboard_obj = get_dashboard()
    dashboard = await dashboard_obj.get_full_dashboard()
    report = dashboard_obj.format_dashboard_report(dashboard)
    
    print(report)


async def send_dashboard_telegram():
    """Gửi dashboard qua Telegram"""
    print("\n📱 SENDING DASHBOARD TO TELEGRAM")
    print("=" * 60)
    
    from analytics_dashboard import get_dashboard
    from config import CHAT_ID
    
    dashboard_obj = get_dashboard()
    await dashboard_obj.send_dashboard_to_telegram(CHAT_ID)


async def test_parallel_scanning():
    """Test parallel scanning"""
    print("\n⚡ TESTING PARALLEL SCANNING")
    print("=" * 60)
    
    from parallel_scanner import ParallelScanner
    from config import TICKERS
    
    def test_scan(symbol: str):
        from data_loader import load_data
        df = load_data(symbol, lookback=100)
        return {'symbol': symbol, 'rows': len(df)}
    
    scanner = ParallelScanner(max_workers=5)
    
    # Test with first 20 symbols
    test_symbols = TICKERS[:20]
    
    results = scanner.scan_symbols(test_symbols, test_scan)
    summary = scanner.get_summary(results)
    
    print(f"\nSummary:")
    print(f"  Total: {summary['total']}")
    print(f"  Success: {summary['success']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    print(f"  Avg Duration: {summary['avg_duration']:.2f}s")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Analytics and Monitoring')
    parser.add_argument('command', choices=[
        'attribution',
        'walkforward',
        'cache-stats',
        'cache-cleanup',
        'monitor',
        'dashboard',
        'send-dashboard',
        'test-parallel',
        'all'
    ], help='Command to run')
    parser.add_argument('--days', type=int, default=90, help='Days for attribution analysis')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval in seconds')
    parser.add_argument('--symbols', nargs='+', help='Symbols for walk-forward test')
    
    args = parser.parse_args()
    
    if args.command == 'attribution':
        await run_performance_attribution(args.days)
    
    elif args.command == 'walkforward':
        await run_walk_forward_test(args.symbols)
    
    elif args.command == 'cache-stats':
        await show_cache_stats()
    
    elif args.command == 'cache-cleanup':
        await cleanup_cache()
    
    elif args.command == 'monitor':
        await start_realtime_monitoring(args.interval)
    
    elif args.command == 'dashboard':
        await show_dashboard()
    
    elif args.command == 'send-dashboard':
        await send_dashboard_telegram()
    
    elif args.command == 'test-parallel':
        await test_parallel_scanning()
    
    elif args.command == 'all':
        await show_dashboard()
        await run_performance_attribution(args.days)
        await show_cache_stats()


if __name__ == "__main__":
    print("🚀 Analytics & Monitoring Tool")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
