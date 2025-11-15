"""
Automatic Migration Script - No user input required
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.migrate_structure import (
    create_directories,
    create_init_files,
    move_files,
    move_documentation,
    create_setup_py
)

def main():
    """Run migration automatically"""
    print("="*70)
    print("🚀 AUTOMATIC PROJECT MIGRATION")
    print("="*70)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Create __init__.py files
    create_init_files()
    
    # Step 3: Move files (actual migration)
    print("\n" + "="*70)
    print("MOVING FILES...")
    print("="*70)
    moved, skipped, errors = move_files(dry_run=False)
    
    # Step 4: Move documentation
    move_documentation()
    
    # Step 5: Create setup.py
    create_setup_py()
    
    print("\n" + "="*70)
    print("✅ MIGRATION COMPLETED!")
    print("="*70)
    print(f"\nMoved: {moved} files")
    print(f"Skipped: {skipped} files")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\n❌ Errors occurred:")
        for error in errors:
            print(f"  {error}")
        return 1
    
    print("\n✅ All files migrated successfully!")
    print("\nNext steps:")
    print("1. Update imports in moved files")
    print("2. Run tests: pytest tests/")
    print("3. Verify everything works")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
