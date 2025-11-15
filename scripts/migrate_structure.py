"""
Project Structure Migration Script
Automatically reorganizes project files into new structure
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

# Define file mappings: {old_path: new_path}
FILE_MAPPINGS = {
    # Core
    'orchestrator.py': 'src/core/orchestrator.py',
    'orchestrator_v2.py': 'src/core/orchestrator_v2.py',
    'bot_runner_improved.py': 'src/core/bot_runner.py',
    
    # Services (already in services/)
    'services/risk_service.py': 'src/services/risk_service.py',
    'services/entry_service.py': 'src/services/entry_service.py',
    'services/exit_service.py': 'src/services/exit_service.py',
    'services/notification_service.py': 'src/services/notification_service.py',
    'services/__init__.py': 'src/services/__init__.py',
    
    # Strategies
    'improved_entry_logic.py': 'src/strategies/entry_logic.py',
    'improved_exit_logic.py': 'src/strategies/exit_logic.py',
    'position_sizing_enhanced.py': 'src/strategies/position_sizing.py',
    'enhanced_risk_management.py': 'src/strategies/risk_management.py',
    'risk_management.py': 'src/strategies/risk_management_base.py',
    'strategy_manager.py': 'src/strategies/manager.py',
    
    # ML
    'ml_models.py': 'src/ml/models/predictor.py',
    'ml_models_enhanced.py': 'src/ml/models/ensemble.py',
    'ml_lstm_model.py': 'src/ml/models/lstm.py',
    'features.py': 'src/ml/features/technical.py',
    'features_enhanced.py': 'src/ml/features/enhanced.py',
    'ml_signals.py': 'src/ml/signals/generator.py',
    'ml_signals_enhanced.py': 'src/ml/signals/enhanced.py',
    'train_models.py': 'src/ml/training/trainer.py',
    'train_enhanced_models.py': 'src/ml/training/enhanced_trainer.py',
    'ml_model_monitor.py': 'src/ml/monitor.py',
    
    # Data
    'data_loader.py': 'src/data/loader.py',
    'database.py': 'src/data/database.py',
    'ticker_loader.py': 'src/data/ticker_loader.py',
    'smart_cache.py': 'src/data/cache.py',
    'incremental_cache.py': 'src/data/incremental_cache.py',
    
    # Portfolio
    'portfolio_manager.py': 'src/portfolio/manager.py',
    'portfolio_analyzer.py': 'src/portfolio/analyzer.py',
    'portfolio_lock.py': 'src/portfolio/lock.py',
    'paper_trading.py': 'src/portfolio/paper_trading.py',
    'portfolio_optimizer.py': 'src/portfolio/optimizer.py',
    'portfolio_risk_manager.py': 'src/portfolio/risk_manager.py',
    
    # Risk
    'circuit_breaker.py': 'src/risk/circuit_breaker.py',
    'emergency_stop.py': 'src/risk/emergency_stop.py',
    'risk_metrics.py': 'src/risk/metrics.py',
    
    # Market
    'market_regime.py': 'src/market/regime.py',
    'market_regime_proxy.py': 'src/market/regime_proxy.py',
    'improved_sector_analysis.py': 'src/market/sector_analysis.py',
    'vn_trading_schedule.py': 'src/market/schedule.py',
    
    # Monitoring
    'monitoring.py': 'src/monitoring/performance.py',
    'monitoring_enhanced.py': 'src/monitoring/enhanced.py',
    'health_check.py': 'src/monitoring/health.py',
    'prometheus_metrics.py': 'src/monitoring/prometheus.py',
    
    # Notifications
    'telegram_notifications.py': 'src/notifications/telegram.py',
    'telegram_subscriptions.py': 'src/notifications/subscriptions.py',
    'tg_listener.py': 'src/notifications/listener.py',
    
    # API
    'main.py': 'src/api/main.py',
    'auth.py': 'src/api/auth.py',
    
    # Utils (already in utils/)
    'utils/indicators.py': 'src/utils/indicators.py',
    'utils/validation.py': 'src/utils/validation.py',
    'utils/__init__.py': 'src/utils/__init__.py',
    'logging_config.py': 'src/utils/logging_config.py',
    'suppress_warnings.py': 'src/utils/suppress_warnings.py',
    'rate_limiter.py': 'src/utils/rate_limiter.py',
    
    # Config
    'config.py': 'src/config/legacy_config.py',
    'trading_config.py': 'src/config/trading_config.py',
    'exceptions.py': 'src/config/exceptions.py',
    
    # Backup & Migration
    'backup_manager.py': 'src/utils/backup_manager.py',
    'migrate_json_to_db.py': 'scripts/migrate_json_to_db.py',
    
    # Analytics & Testing
    'analytics_dashboard.py': 'scripts/analytics_dashboard.py',
    'dashboard_app.py': 'scripts/dashboard_app.py',
    'walk_forward_test.py': 'scripts/walk_forward_test.py',
    'backtest.py': 'scripts/backtest.py',
    'run_backtest.py': 'scripts/run_backtest.py',
    
    # Validation & Analysis
    'validate_list_csv.py': 'scripts/validate_list_csv.py',
    'analyze_no_signals.py': 'scripts/analyze_no_signals.py',
    'adjust_thresholds.py': 'scripts/adjust_thresholds.py',
    
    # Tests (already in tests/)
    'tests/unit/test_critical_fixes.py': 'tests/unit/test_critical_fixes.py',
    'tests/unit/test_services.py': 'tests/unit/test_services.py',
    'tests/conftest.py': 'tests/conftest.py',
}

# Directories to create
DIRECTORIES = [
    'src',
    'src/core',
    'src/services',
    'src/strategies',
    'src/ml',
    'src/ml/models',
    'src/ml/features',
    'src/ml/signals',
    'src/ml/training',
    'src/data',
    'src/portfolio',
    'src/risk',
    'src/market',
    'src/monitoring',
    'src/notifications',
    'src/api',
    'src/api/routes',
    'src/utils',
    'src/config',
    'tests/unit',
    'tests/integration',
    'tests/fixtures',
    'scripts',
    'docs',
    'docs/analysis',
    'docs/guides',
    'data',
    'data/tickers',
    'data/models',
    'data/cache',
    'logs',
    'backups',
    'notebooks',
]


def create_directories():
    """Create all necessary directories"""
    print("📁 Creating directory structure...")
    for directory in DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith('src/') or directory.startswith('tests/'):
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('"""Package initialization"""\n')
    
    print(f"✅ Created {len(DIRECTORIES)} directories")


def move_files(dry_run=True):
    """Move files to new locations"""
    print(f"\n📦 {'[DRY RUN] ' if dry_run else ''}Moving files...")
    
    moved = 0
    skipped = 0
    errors = []
    
    for old_path, new_path in FILE_MAPPINGS.items():
        old_file = Path(old_path)
        new_file = Path(new_path)
        
        if not old_file.exists():
            print(f"⏭️  Skip: {old_path} (not found)")
            skipped += 1
            continue
        
        if new_file.exists():
            print(f"⏭️  Skip: {new_path} (already exists)")
            skipped += 1
            continue
        
        try:
            if not dry_run:
                # Ensure parent directory exists
                new_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file (don't move yet, for safety)
                shutil.copy2(old_file, new_file)
                print(f"✅ Copied: {old_path} → {new_path}")
            else:
                print(f"📋 Would copy: {old_path} → {new_path}")
            
            moved += 1
        except Exception as e:
            error_msg = f"❌ Error moving {old_path}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    print(f"\n📊 Summary:")
    print(f"  Moved: {moved}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ Errors:")
        for error in errors:
            print(f"  {error}")
    
    return moved, skipped, errors


def create_init_files():
    """Create __init__.py files with proper imports"""
    print("\n📝 Creating __init__.py files...")
    
    init_files = {
        'src/__init__.py': '"""VN Trading Bot - Main Package"""\n__version__ = "2.0.0"\n',
        
        'src/core/__init__.py': '''"""Core orchestration logic"""
from .bot_runner import run_bot_sync
from .orchestrator_v2 import TradingOrchestratorV2

__all__ = ['run_bot_sync', 'TradingOrchestratorV2']
''',
        
        'src/ml/__init__.py': '''"""Machine Learning components"""
from .models import *
from .features import *
from .signals import *

__all__ = []
''',
        
        'src/ml/models/__init__.py': '''"""ML Models"""
# Import models here when ready
''',
        
        'src/ml/features/__init__.py': '''"""Feature engineering"""
# Import features here when ready
''',
        
        'src/ml/signals/__init__.py': '''"""Signal generation"""
# Import signals here when ready
''',
        
        'src/api/routes/__init__.py': '''"""API Routes"""
# Import routes here when ready
''',
    }
    
    for file_path, content in init_files.items():
        file = Path(file_path)
        if not file.exists():
            file.write_text(content)
            print(f"✅ Created: {file_path}")


def move_documentation():
    """Move documentation files"""
    print("\n📚 Moving documentation...")
    
    doc_files = {
        'README.md': 'README.md',  # Keep at root
        'QUICK_START.md': 'docs/guides/QUICK_START.md',
        'INSTALLATION_GUIDE.md': 'docs/guides/INSTALLATION.md',
        'DEPLOYMENT_GUIDE_V2.md': 'docs/DEPLOYMENT.md',
        'PROJECT_STRUCTURE.md': 'docs/ARCHITECTURE.md',
        'PHAN_TICH_DU_AN.md': 'docs/analysis/PHAN_TICH_DU_AN.md',
        'ALL_IMPROVEMENTS_SUMMARY.md': 'docs/analysis/improvements/ALL_IMPROVEMENTS.md',
        'CRITICAL_FIXES_COMPLETED.md': 'docs/analysis/improvements/CRITICAL_FIXES.md',
        'REFACTORING_COMPLETED.md': 'docs/analysis/improvements/REFACTORING.md',
    }
    
    for old_path, new_path in doc_files.items():
        old_file = Path(old_path)
        new_file = Path(new_path)
        
        if old_file.exists() and not new_file.exists():
            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_file, new_file)
            print(f"✅ Copied: {old_path} → {new_path}")


def create_setup_py():
    """Create setup.py for package installation"""
    setup_content = '''"""
Setup configuration for VN Trading Bot
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vn-trading-bot",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated trading bot for Vietnamese stock market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vn-trading-bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "trading-bot=src.api.main:main",
        ],
    },
)
'''
    
    Path('setup.py').write_text(setup_content)
    print("✅ Created setup.py")


def main():
    """Main migration function"""
    print("="*70)
    print("🚀 PROJECT STRUCTURE MIGRATION")
    print("="*70)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Create __init__.py files
    create_init_files()
    
    # Step 3: Move files (dry run first)
    print("\n" + "="*70)
    print("DRY RUN - No files will be moved")
    print("="*70)
    moved, skipped, errors = move_files(dry_run=True)
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Proceed with actual migration? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\n" + "="*70)
        print("ACTUAL MIGRATION - Moving files...")
        print("="*70)
        moved, skipped, errors = move_files(dry_run=False)
        
        # Step 4: Move documentation
        move_documentation()
        
        # Step 5: Create setup.py
        create_setup_py()
        
        print("\n" + "="*70)
        print("✅ MIGRATION COMPLETED!")
        print("="*70)
        print("\nNext steps:")
        print("1. Update imports in moved files")
        print("2. Run tests: pytest tests/")
        print("3. Update documentation")
        print("4. Commit changes")
    else:
        print("\n❌ Migration cancelled")


if __name__ == "__main__":
    main()
