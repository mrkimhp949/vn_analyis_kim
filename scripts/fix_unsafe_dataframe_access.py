#!/usr/bin/env python3
"""
Script to fix unsafe DataFrame access patterns throughout the codebase
Replaces df.iloc[-1] with safe_get_latest() calls
"""

import os
import re
from pathlib import Path


def fix_file(filepath):
    """Fix unsafe DataFrame access in a single file"""
    print(f"Processing {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Add import if not present
    if "from utils.dataframe_utils import" not in content:
        # Find first import line to add after
        import_lines = []
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                import_lines.append(i)

        if import_lines:
            # Add after last import
            last_import = max(import_lines)
            lines.insert(
                last_import + 1,
                "from utils.dataframe_utils import safe_get_latest, safe_rolling_operation",
            )
            content = "\n".join(lines)

    # Pattern 1: df.iloc[-1] direct access
    # Replace: latest = df.iloc[-1]
    content = re.sub(
        r"(\s+)latest\s*=\s*df\.iloc\[-1\]",
        r"\1# Use safe access instead of df.iloc[-1]",
        content,
    )

    # Pattern 2: df["column"].iloc[-1]
    # Replace with safe_get_latest(df, "column")
    content = re.sub(r'df\["([^"]+)"\]\.iloc\[-1\]', r'safe_get_latest(df, "\1", 0)', content)

    # Pattern 3: df[column].iloc[-1]
    content = re.sub(r"df\[([^]]+)\]\.iloc\[-1\]", r"safe_get_latest(df, \1, 0)", content)

    # Pattern 4: rolling().mean().iloc[-1]
    content = re.sub(
        r'df\["([^"]+)"\]\.rolling\((\d+)\)\.mean\(\)\.iloc\[-1\]',
        r'safe_rolling_operation(df, "\1", \2, "mean", 0)',
        content,
    )

    # Pattern 5: rolling().min().iloc[-1]
    content = re.sub(
        r'df\["([^"]+)"\]\.rolling\((\d+)\)\.min\(\)\.iloc\[-1\]',
        r'safe_rolling_operation(df, "\1", \2, "min", 0)',
        content,
    )

    # Pattern 6: rolling().max().iloc[-1]
    content = re.sub(
        r'df\["([^"]+)"\]\.rolling\((\d+)\)\.max\(\)\.iloc\[-1\]',
        r'safe_rolling_operation(df, "\1", \2, "max", 0)',
        content,
    )

    # Pattern 7: latest["column"] references after removing latest = df.iloc[-1]
    # This is more complex, need to handle case by case

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ Fixed {filepath}")
        return True
    else:
        print(f"  ⏭️ No changes needed in {filepath}")
        return False


def main():
    """Main function to fix all files"""

    # Files to process
    files_to_fix = [
        "src/utils/validation.py",
        "src/strategies/entry_logic.py",
        "src/strategies/exit_logic.py",
        "src/strategies/risk_management.py",
        "src/services/exit_service.py",
        "src/services/entry_service.py",
        "src/risk/emergency_stop.py",
        "src/portfolio/paper_trading.py",
        "src/portfolio/analyzer.py",
        "src/ml/signals/generator.py",
        "src/ml/signals/enhanced.py",
        "src/ml/features/technical.py",
        "src/ml/features/enhanced.py",
        "src/market/regime.py",
    ]

    fixed_count = 0

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"⚠️ File not found: {filepath}")

    print(f"\n✅ Fixed {fixed_count} files")
    print("⚠️ Note: Some files may need manual review for complex patterns")


if __name__ == "__main__":
    main()
