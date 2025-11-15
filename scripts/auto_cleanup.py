"""
Automatic Cleanup Script - Removes old files without confirmation
USE WITH CAUTION!
"""

import os
import shutil
from pathlib import Path

# Files to remove
OLD_FILES = [
    "orchestrator.py",
    "bot_runner_improved.py",
    "improved_entry_logic.py",
    "improved_exit_logic.py",
    "position_sizing_enhanced.py",
    "enhanced_risk_management.py",
    "strategy_manager.py",
    "improved_position_sizing.py",
    "ml_models.py",
    "ml_models_enhanced.py",
    "ml_lstm_model.py",
    "features.py",
    "features_enhanced.py",
    "ml_signals.py",
    "ml_signals_enhanced.py",
    "train_models.py",
    "train_enhanced_models.py",
    "ml_model_monitor.py",
    "data_loader.py",
    "database.py",
    "ticker_loader.py",
    "smart_cache.py",
    "incremental_cache.py",
    "data_quality.py",
    "portfolio_manager.py",
    "portfolio_analyzer.py",
    "portfolio_lock.py",
    "paper_trading.py",
    "portfolio_optimizer.py",
    "portfolio_risk_manager.py",
    "portfolio.py",
    "portfolio_history.py",
    "portfolio_regime_adjuster.py",
    "circuit_breaker.py",
    "emergency_stop.py",
    "risk_metrics.py",
    "risk_management.py",
    "market_regime.py",
    "market_regime_proxy.py",
    "improved_sector_analysis.py",
    "vn_trading_schedule.py",
    "monitoring.py",
    "monitoring_enhanced.py",
    "health_check.py",
    "prometheus_metrics.py",
    "model_monitor.py",
    "api_monitor.py",
    "telegram_notifications.py",
    "telegram_subscriptions.py",
    "tg_listener.py",
    "main.py",
    "auth.py",
    "logging_config.py",
    "suppress_warnings.py",
    "rate_limiter.py",
    "backup_manager.py",
    "config.py",
    "trading_config.py",
    "exceptions.py",
    "migrate_json_to_db.py",
    "analytics_dashboard.py",
    "dashboard_app.py",
    "walk_forward_test.py",
    "backtest.py",
    "run_backtest.py",
    "validate_list_csv.py",
    "analyze_no_signals.py",
    "adjust_thresholds.py",
    "run_analytics.py",
    "test_analytics.py",
    "test_improvements.py",
    "test_ml_improvements.py",
    "retrain_models.py",
    "live_runner.py",
    "realtime_data_stream.py",
    "realtime_monitor.py",
    "parallel_scanner.py",
    "performance_attribution.py",
    "multi_timeframe.py",
    "exit_strategy_enhanced.py",
    "db_manager.py",
    "news_analyzer.py",
]


def main():
    """Remove old files automatically"""
    print("=" * 70)
    print("🧹 AUTOMATIC CLEANUP")
    print("=" * 70)

    removed = 0
    not_found = 0
    errors = []

    for file_path in OLD_FILES:
        file = Path(file_path)

        if not file.exists():
            not_found += 1
            continue

        try:
            file.unlink()
            print(f"🗑️  Removed: {file_path}")
            removed += 1
        except Exception as e:
            error_msg = f"❌ Error: {file_path} - {e}"
            print(error_msg)
            errors.append(error_msg)

    # Remove old directories if empty
    old_dirs = ["services", "utils"]
    for dir_path in old_dirs:
        dir = Path(dir_path)
        if dir.exists():
            try:
                contents = list(dir.iterdir())
                if not contents or all(item.name == "__pycache__" for item in contents):
                    shutil.rmtree(dir)
                    print(f"🗑️  Removed directory: {dir_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {dir_path}: {e}")

    print("\n" + "=" * 70)
    print("✅ CLEANUP COMPLETED!")
    print("=" * 70)
    print(f"Removed: {removed} files")
    print(f"Not found: {not_found} files")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
