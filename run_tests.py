#!/usr/bin/env python
"""
Test Runner
Run all unit tests for the trading bot
"""
import sys
import pytest


def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 RUNNING TRADING BOT TESTS")
    print("=" * 70)

    # Run pytest with verbose output
    args = [
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes",
        "-ra",  # Show summary of all test outcomes
    ]

    # Add coverage if available
    try:
        import pytest_cov

        args.extend(["--cov=.", "--cov-report=term-missing"])
    except ImportError:
        print("⚠️ pytest-cov not installed. Install with: pip install pytest-cov")

    exit_code = pytest.main(args)

    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ TESTS FAILED (exit code: {exit_code})")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
