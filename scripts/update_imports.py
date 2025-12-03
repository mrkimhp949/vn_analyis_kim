"""
Update Imports Script
Automatically updates imports in all Python files to use new structure
"""

import re
from pathlib import Path

# Import mappings: {old_pattern: new_pattern}
IMPORT_MAPPINGS = {
    # Core
    r"from src.core.orchestrator import": "from src.core.orchestrator import",
    r"from src.core.orchestrator_v2 import": "from src.core.orchestrator_v2 import",
    r"from src.core.bot_runner import": "from src.core.bot_runner import",
    # Services
    r"from services\.": "from src.services.",
    # Strategies
    r"from src.strategies.entry_logic import": "from src.strategies.entry_logic import",
    r"from src.strategies.exit_logic import": "from src.strategies.exit_logic import",
    r"from src.strategies.position_sizing import": "from src.strategies.position_sizing import",
    r"from src.strategies.risk_management import": "from src.strategies.risk_management import",
    r"from src.strategies.manager import": "from src.strategies.manager import",
    # ML
    r"from src.ml.models.predictor import": "from src.ml.models.predictor import",
    r"from src.ml.models.ensemble import": "from src.ml.models.ensemble import",
    r"from src.ml.models.lstm import": "from src.ml.models.lstm import",
    r"from src.ml.features.technical import": "from src.ml.features.technical import",
    r"from src.ml.features.enhanced import": "from src.ml.features.enhanced import",
    r"from src.ml.signals.generator import": "from src.ml.signals.generator import",
    r"from src.ml.signals.enhanced import": "from src.ml.signals.enhanced import",
    r"from src.ml.monitor import": "from src.ml.monitor import",
    # Data
    r"from src.data.loader import": "from src.data.loader import",
    r"from src.data.database import": "from src.data.database import",
    r"from src.data.ticker_loader import": "from src.data.ticker_loader import",
    r"from src.data.cache import": "from src.data.cache import",
    # Portfolio
    r"from src.portfolio.manager import": "from src.portfolio.manager import",
    r"from src.portfolio.analyzer import": "from src.portfolio.analyzer import",
    r"from src.portfolio.lock import": "from src.portfolio.lock import",
    r"from src.portfolio.paper_trading import": "from src.portfolio.paper_trading import",
    r"from src.portfolio.optimizer import": "from src.portfolio.optimizer import",
    r"from src.portfolio.risk_manager import": "from src.portfolio.risk_manager import",
    # Risk
    r"from src.risk.circuit_breaker import": "from src.risk.circuit_breaker import",
    r"from src.risk.emergency_stop import": "from src.risk.emergency_stop import",
    r"from src.risk.metrics import": "from src.risk.metrics import",
    # Market
    r"from src.market.regime import": "from src.market.regime_detector import",
    r"from src.market.regime_proxy import": "from src.market.regime_proxy import",
    r"from src.market.sector_analysis import": "from src.market.sector_analysis import",
    r"from src.market.schedule import": "from src.market.schedule import",
    # Monitoring
    r"from src.monitoring.performance import": "from src.monitoring.performance import",
    r"from src.monitoring.enhanced import": "from src.monitoring.enhanced import",
    r"from src.monitoring.health import": "from src.monitoring.health import",
    r"from src.monitoring.prometheus import": "from src.monitoring.prometheus import",
    # Notifications
    r"from src.notifications.telegram import": "from src.notifications.telegram import",
    r"from src.notifications.subscriptions import": "from src.notifications.subscriptions import",
    r"from src.notifications.listener import": "from src.notifications.listener import",
    # Utils
    r"from utils\.": "from src.utils.",
    r"from src.utils.logging_config import": "from src.utils.logging_config import",
    r"from src.utils.suppress_warnings import": "from src.utils.suppress_warnings import",
    r"from src.utils.rate_limiter import": "from src.utils.rate_limiter import",
    r"from src.utils.backup_manager import": "from src.utils.backup_manager import",
    # Config
    r"from src.config.legacy_config import": "from src.config.legacy_config import",
    r"from src.config.trading_config import": "from src.config.trading_config import",
    r"from src.config.exceptions import": "from src.config.exceptions import",
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
                print("✅ Updated: {file_path}")
            else:
                print("📋 Would update: {file_path}")

            for change in changes:
                print("   {change}")

            return True

        return False

    except Exception:
        print("❌ Error updating {file_path}")
        return False


def update_all_imports(directories=["src", "tests", "scripts"], dry_run=True):
    """Update imports in all Python files"""
    print("{'[DRY RUN] ' if dry_run else ''}Updating imports...")

    updated = 0
    skipped = 0
    errors = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print("⏭️  Skip: {directory} (not found)")
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
            except Exception:
                print("❌{py_file} -")
                errors += 1

    print("\n📊 Summary:")
    print("  Updated: {updated}")
    print("  Skipped: {skipped}")
    print("  Errors: {errors}")

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
        print("\nUpdated {updated} files")

        if errors > 0:
            print("\n⚠️  {errors} errors occurred")
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
