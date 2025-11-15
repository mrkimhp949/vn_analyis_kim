"""
Update Imports Script
Automatically updates imports in all Python files to use new structure
"""

import os
import re
from pathlib import Path

# Import mappings: {old_pattern: new_pattern}
IMPORT_MAPPINGS = {
    # Core
    r"from orchestrator import": "from src.core.orchestrator import",
    r"from orchestrator_v2 import": "from src.core.orchestrator_v2 import",
    r"from bot_runner_improved import": "from src.core.bot_runner import",
    # Services
    r"from services\.": "from src.services.",
    # Strategies
    r"from improved_entry_logic import": "from src.strategies.entry_logic import",
    r"from improved_exit_logic import": "from src.strategies.exit_logic import",
    r"from position_sizing_enhanced import": "from src.strategies.position_sizing import",
    r"from enhanced_risk_management import": "from src.strategies.risk_management import",
    r"from strategy_manager import": "from src.strategies.manager import",
    # ML
    r"from ml_models import": "from src.ml.models.predictor import",
    r"from ml_models_enhanced import": "from src.ml.models.ensemble import",
    r"from ml_lstm_model import": "from src.ml.models.lstm import",
    r"from features import": "from src.ml.features.technical import",
    r"from features_enhanced import": "from src.ml.features.enhanced import",
    r"from ml_signals import": "from src.ml.signals.generator import",
    r"from ml_signals_enhanced import": "from src.ml.signals.enhanced import",
    r"from ml_model_monitor import": "from src.ml.monitor import",
    # Data
    r"from data_loader import": "from src.data.loader import",
    r"from database import": "from src.data.database import",
    r"from ticker_loader import": "from src.data.ticker_loader import",
    r"from smart_cache import": "from src.data.cache import",
    # Portfolio
    r"from portfolio_manager import": "from src.portfolio.manager import",
    r"from portfolio_analyzer import": "from src.portfolio.analyzer import",
    r"from portfolio_lock import": "from src.portfolio.lock import",
    r"from paper_trading import": "from src.portfolio.paper_trading import",
    r"from portfolio_optimizer import": "from src.portfolio.optimizer import",
    r"from portfolio_risk_manager import": "from src.portfolio.risk_manager import",
    # Risk
    r"from circuit_breaker import": "from src.risk.circuit_breaker import",
    r"from emergency_stop import": "from src.risk.emergency_stop import",
    r"from risk_metrics import": "from src.risk.metrics import",
    # Market
    r"from market_regime import": "from src.market.regime import",
    r"from market_regime_proxy import": "from src.market.regime_proxy import",
    r"from improved_sector_analysis import": "from src.market.sector_analysis import",
    r"from vn_trading_schedule import": "from src.market.schedule import",
    # Monitoring
    r"from monitoring import": "from src.monitoring.performance import",
    r"from monitoring_enhanced import": "from src.monitoring.enhanced import",
    r"from health_check import": "from src.monitoring.health import",
    r"from prometheus_metrics import": "from src.monitoring.prometheus import",
    # Notifications
    r"from telegram_notifications import": "from src.notifications.telegram import",
    r"from telegram_subscriptions import": "from src.notifications.subscriptions import",
    r"from tg_listener import": "from src.notifications.listener import",
    # Utils
    r"from utils\.": "from src.utils.",
    r"from logging_config import": "from src.utils.logging_config import",
    r"from suppress_warnings import": "from src.utils.suppress_warnings import",
    r"from rate_limiter import": "from src.utils.rate_limiter import",
    r"from backup_manager import": "from src.utils.backup_manager import",
    # Config
    r"from config import": "from src.config.legacy_config import",
    r"from trading_config import": "from src.config.trading_config import",
    r"from exceptions import": "from src.config.exceptions import",
}


def update_file_imports(file_path: Path, dry_run=True):
    """Update imports in a single file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        changes = []

        # Apply all import mappings
        for old_pattern, new_pattern in IMPORT_MAPPINGS.items():
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                changes.append(f"{old_pattern} → {new_pattern}")

        # Only write if changes were made
        if content != original_content:
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Updated: {file_path}")
            else:
                print(f"📋 Would update: {file_path}")

            for change in changes:
                print(f"   {change}")

            return True

        return False

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False


def update_all_imports(directories=["src", "tests", "scripts"], dry_run=True):
    """Update imports in all Python files"""
    print(f"{'[DRY RUN] ' if dry_run else ''}Updating imports...")

    updated = 0
    skipped = 0
    errors = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"⏭️  Skip: {directory} (not found)")
            continue

        # Find all Python files
        for py_file in dir_path.rglob("*.py"):
            # Skip __pycache__ and .pyc files
            if "__pycache__" in str(py_file):
                continue

            try:
                if update_file_imports(py_file, dry_run):
                    updated += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"❌ Error: {py_file} - {e}")
                errors += 1

    print(f"\n📊 Summary:")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    return updated, skipped, errors


def main():
    """Main function"""
    print("=" * 70)
    print("🔄 UPDATE IMPORTS")
    print("=" * 70)

    # Dry run first
    print("\n" + "=" * 70)
    print("DRY RUN - No files will be modified")
    print("=" * 70)
    updated, skipped, errors = update_all_imports(dry_run=True)

    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("Proceed with actual update? (yes/no): ")

    if response.lower() == "yes":
        print("\n" + "=" * 70)
        print("UPDATING IMPORTS...")
        print("=" * 70)
        updated, skipped, errors = update_all_imports(dry_run=False)

        print("\n" + "=" * 70)
        print("✅ IMPORT UPDATE COMPLETED!")
        print("=" * 70)
        print(f"\nUpdated {updated} files")

        if errors > 0:
            print(f"\n⚠️  {errors} errors occurred")
            return 1

        print("\n✅ All imports updated successfully!")
        print("\nNext steps:")
        print("1. Run tests: pytest tests/")
        print("2. Check for any remaining issues")
        print("3. Commit changes")

        return 0
    else:
        print("\n❌ Update cancelled")
        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
