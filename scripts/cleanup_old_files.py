"""
Cleanup Old Files Script
Removes old files that have been migrated to src/ directory
"""

import os
from pathlib import Path

# Files to remove (old locations that have been migrated)
OLD_FILES = [
    # Core
    'orchestrator.py',
    'bot_runner_improved.py',
    
    # Strategies
    'improved_entry_logic.py',
    'improved_exit_logic.py',
    'position_sizing_enhanced.py',
    'enhanced_risk_management.py',
    'strategy_manager.py',
    'improved_position_sizing.py',
    
    # ML
    'ml_models.py',
    'ml_models_enhanced.py',
    'ml_lstm_model.py',
    'features.py',
    'features_enhanced.py',
    'ml_signals.py',
    'ml_signals_enhanced.py',
    'train_models.py',
    'train_enhanced_models.py',
    'ml_model_monitor.py',
    
    # Data
    'data_loader.py',
    'database.py',
    'ticker_loader.py',
    'smart_cache.py',
    'incremental_cache.py',
    'data_quality.py',
    
    # Portfolio
    'portfolio_manager.py',
    'portfolio_analyzer.py',
    'portfolio_lock.py',
    'paper_trading.py',
    'portfolio_optimizer.py',
    'portfolio_risk_manager.py',
    'portfolio.py',
    'portfolio_history.py',
    'portfolio_regime_adjuster.py',
    
    # Risk
    'circuit_breaker.py',
    'emergency_stop.py',
    'risk_metrics.py',
    'risk_management.py',
    
    # Market
    'market_regime.py',
    'market_regime_proxy.py',
    'improved_sector_analysis.py',
    'vn_trading_schedule.py',
    
    # Monitoring
    'monitoring.py',
    'monitoring_enhanced.py',
    'health_check.py',
    'prometheus_metrics.py',
    'model_monitor.py',
    'api_monitor.py',
    
    # Notifications
    'telegram_notifications.py',
    'telegram_subscriptions.py',
    'tg_listener.py',
    
    # API
    'main.py',
    'auth.py',
    
    # Utils
    'logging_config.py',
    'suppress_warnings.py',
    'rate_limiter.py',
    'backup_manager.py',
    
    # Config
    'config.py',
    'trading_config.py',
    'exceptions.py',
    
    # Scripts (moved to scripts/)
    'migrate_json_to_db.py',
    'analytics_dashboard.py',
    'dashboard_app.py',
    'walk_forward_test.py',
    'backtest.py',
    'run_backtest.py',
    'validate_list_csv.py',
    'analyze_no_signals.py',
    'adjust_thresholds.py',
    'run_analytics.py',
    'test_analytics.py',
    'test_improvements.py',
    'test_ml_improvements.py',
    'retrain_models.py',
    'live_runner.py',
    
    # Other
    'realtime_data_stream.py',
    'realtime_monitor.py',
    'parallel_scanner.py',
    'performance_attribution.py',
    'multi_timeframe.py',
    'exit_strategy_enhanced.py',
    'db_manager.py',
    'news_analyzer.py',
]

# Directories to remove (old structure)
OLD_DIRS = [
    'services',  # Moved to src/services
    'utils',     # Moved to src/utils
]

def cleanup_files(dry_run=True):
    """Remove old files"""
    print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning up old files...")
    
    removed = 0
    not_found = 0
    errors = []
    
    for file_path in OLD_FILES:
        file = Path(file_path)
        
        if not file.exists():
            print(f"⏭️  Skip: {file_path} (not found)")
            not_found += 1
            continue
        
        try:
            if not dry_run:
                file.unlink()
                print(f"🗑️  Removed: {file_path}")
            else:
                print(f"📋 Would remove: {file_path}")
            
            removed += 1
        except Exception as e:
            error_msg = f"❌ Error removing {file_path}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    print(f"\n📊 Summary:")
    print(f"  Removed: {removed}")
    print(f"  Not found: {not_found}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ Errors:")
        for error in errors:
            print(f"  {error}")
    
    return removed, not_found, errors

def cleanup_directories(dry_run=True):
    """Remove old directories"""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning up old directories...")
    
    removed = 0
    
    for dir_path in OLD_DIRS:
        dir = Path(dir_path)
        
        if not dir.exists():
            print(f"⏭️  Skip: {dir_path} (not found)")
            continue
        
        # Check if directory is empty or only has __pycache__
        contents = list(dir.iterdir())
        pycache_only = all(item.name == '__pycache__' for item in contents)
        
        if not contents or pycache_only:
            try:
                if not dry_run:
                    import shutil
                    shutil.rmtree(dir)
                    print(f"🗑️  Removed directory: {dir_path}")
                else:
                    print(f"📋 Would remove directory: {dir_path}")
                
                removed += 1
            except Exception as e:
                print(f"❌ Error removing {dir_path}: {e}")
        else:
            print(f"⚠️  Skip: {dir_path} (not empty)")
    
    print(f"\nRemoved {removed} directories")
    return removed

def main():
    """Main cleanup function"""
    print("="*70)
    print("🧹 CLEANUP OLD FILES")
    print("="*70)
    
    # Dry run first
    print("\n" + "="*70)
    print("DRY RUN - No files will be removed")
    print("="*70)
    removed, not_found, errors = cleanup_files(dry_run=True)
    cleanup_directories(dry_run=True)
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Proceed with actual cleanup? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\n" + "="*70)
        print("ACTUAL CLEANUP - Removing files...")
        print("="*70)
        removed, not_found, errors = cleanup_files(dry_run=False)
        cleanup_directories(dry_run=False)
        
        print("\n" + "="*70)
        print("✅ CLEANUP COMPLETED!")
        print("="*70)
        print(f"\nRemoved {removed} files")
        
        if errors:
            print(f"\n⚠️  {len(errors)} errors occurred")
            return 1
        
        print("\n✅ All old files cleaned up successfully!")
        return 0
    else:
        print("\n❌ Cleanup cancelled")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
