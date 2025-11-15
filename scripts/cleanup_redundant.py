"""
Cleanup Redundant Files and Directories
Removes unnecessary files after restructuring
"""

import os
import shutil
from pathlib import Path

# Files to remove (redundant/old/test files)
REDUNDANT_FILES = [
    # Old orchestrator (moved to src/)
    'orchestrator_v2.py',
    
    # Example/test files
    'type_hints_example.py',
    'run_tests.py',
    
    # Old README
    'README_NEW.md',
    
    # Deployment files (if not using)
    'cloudbuild.yaml',
    'deploy-cloudrun.sh',
    'Procfile',
    'runtime.txt',
    'start.sh',
    
    # Docker (keep if using, remove if not)
    # 'Dockerfile',
    # 'docker-compose.yml',
    # '.dockerignore',
    
    # Old analysis documents (moved to docs/)
    'PHAN_TICH_DU_AN.md',
    'PHAN_TICH_DU_AN_P2.md',
    'PHAN_TICH_DU_AN_P4.md',
    'PHAN_TICH_DU_AN_P5.md',
    'PHAN_TICH_DU_AN_SUMMARY.md',
    'PHAN_TICH_CAI_TIEN_LOGIC_NGHIEP_VU.md',
    'LUONG_HOAT_DONG_VA_CHUC_NANG.md',
    
    # Old improvement docs (consolidated)
    'CRITICAL_IMPROVEMENTS_SUMMARY.md',
    'IMPORTANT_IMPROVEMENTS_SUMMARY.md',
    'IMPLEMENTATION_GUIDE.md',
    'FINAL_SUMMARY.md',
    
    # Old installation guides (moved to docs/)
    'INSTALLATION_GUIDE.md',
    'DEPLOYMENT_GUIDE_V2.md',
    
    # Cleanup messages
    'cleanup_messages.md',
    
    # Old structure docs
    'ALL_IMPROVEMENTS_SUMMARY.md',
    'CRITICAL_FIXES_COMPLETED.md',
    'SUMMARY_CRITICAL_FIXES.md',
    'REFACTORING_COMPLETED.md',
    'RESTRUCTURING_SUMMARY.md',
]

# Directories to remove (empty or redundant)
REDUNDANT_DIRS = [
    'services',      # Moved to src/services
    'utils',         # Moved to src/utils
    'strategies',    # Moved to src/strategies (if empty)
    'trading_bot',   # Old structure (if empty)
    'backtest_results',  # Old results
    'json_backup',   # Old backups
    'intraday_cache',  # Old cache
    'smart_cache',   # Old cache
    'reports',       # Old reports
    'grafana',       # If not using
]

# Cache/temp files to remove
CACHE_FILES = [
    'ml_predictions.json',
    'ticker_validation_cache.json',
    'circuit_breaker_stats.json',
    'emergency_events.json',
    'metrics.json',
    '.coverage',
]

def cleanup_files():
    """Remove redundant files"""
    print("🗑️  Cleaning up redundant files...")
    
    removed = 0
    not_found = 0
    
    for file_path in REDUNDANT_FILES:
        file = Path(file_path)
        
        if not file.exists():
            not_found += 1
            continue
        
        try:
            file.unlink()
            print(f"✅ Removed: {file_path}")
            removed += 1
        except Exception as e:
            print(f"❌ Error removing {file_path}: {e}")
    
    print(f"\n📊 Files: Removed {removed}, Not found {not_found}")
    return removed

def cleanup_cache():
    """Remove cache files"""
    print("\n🗑️  Cleaning up cache files...")
    
    removed = 0
    
    for file_path in CACHE_FILES:
        file = Path(file_path)
        
        if file.exists():
            try:
                file.unlink()
                print(f"✅ Removed: {file_path}")
                removed += 1
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print(f"\n📊 Cache: Removed {removed} files")
    return removed

def cleanup_directories():
    """Remove redundant directories"""
    print("\n🗑️  Cleaning up redundant directories...")
    
    removed = 0
    
    for dir_path in REDUNDANT_DIRS:
        dir = Path(dir_path)
        
        if not dir.exists():
            continue
        
        try:
            # Check if empty or only has __pycache__
            contents = list(dir.iterdir())
            
            if not contents:
                # Empty directory
                dir.rmdir()
                print(f"✅ Removed empty: {dir_path}")
                removed += 1
            elif all(item.name == '__pycache__' for item in contents):
                # Only __pycache__
                shutil.rmtree(dir)
                print(f"✅ Removed: {dir_path}")
                removed += 1
            else:
                # Has files
                print(f"⏭️  Skip: {dir_path} (not empty - {len(contents)} items)")
        
        except Exception as e:
            print(f"❌ Error removing {dir_path}: {e}")
    
    print(f"\n📊 Directories: Removed {removed}")
    return removed

def cleanup_pycache():
    """Remove all __pycache__ directories"""
    print("\n🗑️  Cleaning up __pycache__ directories...")
    
    removed = 0
    
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = Path(root) / '__pycache__'
            try:
                shutil.rmtree(pycache_path)
                print(f"✅ Removed: {pycache_path}")
                removed += 1
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print(f"\n📊 __pycache__: Removed {removed} directories")
    return removed

def main():
    """Main cleanup function"""
    print("="*70)
    print("🧹 CLEANUP REDUNDANT FILES")
    print("="*70)
    
    total_removed = 0
    
    # Cleanup files
    total_removed += cleanup_files()
    
    # Cleanup cache
    total_removed += cleanup_cache()
    
    # Cleanup directories
    total_removed += cleanup_directories()
    
    # Cleanup __pycache__
    total_removed += cleanup_pycache()
    
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETED!")
    print("="*70)
    print(f"\nTotal removed: {total_removed} items")
    print("\n✅ Project is now clean and organized!")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
