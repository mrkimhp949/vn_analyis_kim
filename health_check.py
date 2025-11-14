#!/usr/bin/env python
"""
Health Check Script
Monitor trading bot status and critical components
"""
import sys
import os
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple


class HealthChecker:
    """Check health of trading bot components"""

    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []

    def check_api_server(self) -> Tuple[bool, str]:
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return True, "API server is healthy"
            else:
                return False, f"API server returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to API server"
        except Exception as e:
            return False, f"API check failed: {e}"

    def check_database(self) -> Tuple[bool, str]:
        """Check database connectivity and integrity"""
        try:
            from database import get_db

            db = get_db()

            with db.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM positions")
                count = cursor.fetchone()[0]

            return True, f"Database OK ({count} active positions)"
        except Exception as e:
            return False, f"Database check failed: {e}"

    def check_models(self) -> Tuple[bool, str]:
        """Check if ML models are loaded"""
        try:
            from ml_models import MLPredictor

            predictor = MLPredictor()
            loaded = predictor.load_models()

            if loaded and predictor.rf_model is not None:
                return True, "ML models loaded successfully"
            else:
                self.warnings.append("ML models using dummy fallback")
                return True, "ML models available (dummy mode)"
        except Exception as e:
            return False, f"Model check failed: {e}"

    def check_configuration(self) -> Tuple[bool, str]:
        """Check configuration validity"""
        try:
            from trading_config import get_config
            from exceptions import ConfigurationError

            config = get_config(validate=True)
            return True, "Configuration is valid"
        except ConfigurationError as e:
            return False, f"Configuration invalid: {e}"
        except Exception as e:
            return False, f"Configuration check failed: {e}"

    def check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space"""
        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            free_percent = (free / total) * 100

            if free_percent < 5:
                return False, f"Low disk space: {free_gb}GB ({free_percent:.1f}%)"
            elif free_percent < 10:
                self.warnings.append(f"Disk space getting low: {free_gb}GB")
                return True, f"Disk space OK: {free_gb}GB ({free_percent:.1f}%)"
            else:
                return True, f"Disk space OK: {free_gb}GB ({free_percent:.1f}%)"
        except Exception as e:
            return False, f"Disk space check failed: {e}"

    def check_data_freshness(self) -> Tuple[bool, str]:
        """Check if cached data is fresh"""
        try:
            cache_dir = "data_cache"
            if not os.path.exists(cache_dir):
                self.warnings.append("No cached data found")
                return True, "No cache directory (will fetch fresh data)"

            # Check newest file in cache
            cache_files = [
                os.path.join(cache_dir, f)
                for f in os.listdir(cache_dir)
                if f.endswith(".pkl")
            ]

            if not cache_files:
                self.warnings.append("Cache directory empty")
                return True, "Cache empty (will fetch fresh data)"

            newest_file = max(cache_files, key=os.path.getmtime)
            mtime = datetime.fromtimestamp(os.path.getmtime(newest_file))
            age_hours = (datetime.now() - mtime).total_seconds() / 3600

            if age_hours > 24:
                self.warnings.append(f"Cached data is {age_hours:.1f} hours old")
                return True, f"Cache exists but old ({age_hours:.1f}h)"
            else:
                return True, f"Cache is fresh ({age_hours:.1f}h old)"
        except Exception as e:
            return False, f"Data freshness check failed: {e}"

    def check_portfolio_risk(self) -> Tuple[bool, str]:
        """Check portfolio risk metrics"""
        try:
            from portfolio_manager import get_portfolio_manager

            manager = get_portfolio_manager()
            positions = manager.get_positions()

            if not positions:
                return True, "No positions (no risk)"

            portfolio = manager.get_portfolio_value()
            num_positions = portfolio["num_positions"]
            pnl_percent = portfolio["pnl_percent"]

            status = f"{num_positions} positions, P&L: {pnl_percent:+.1f}%"

            # Check for concerning drawdown
            if pnl_percent < -15:
                self.warnings.append(f"Large drawdown: {pnl_percent:.1f}%")

            return True, status
        except Exception as e:
            # Portfolio check is optional
            self.warnings.append(f"Could not check portfolio: {e}")
            return True, "Portfolio check skipped"

    def run_all_checks(self) -> Dict:
        """Run all health checks"""
        checks = [
            ("API Server", self.check_api_server),
            ("Database", self.check_database),
            ("ML Models", self.check_models),
            ("Configuration", self.check_configuration),
            ("Disk Space", self.check_disk_space),
            ("Data Freshness", self.check_data_freshness),
            ("Portfolio Risk", self.check_portfolio_risk),
        ]

        results = {}

        print("=" * 70)
        print("🏥 HEALTH CHECK REPORT")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for name, check_func in checks:
            try:
                passed, message = check_func()
                results[name] = {"passed": passed, "message": message}

                emoji = "✅" if passed else "❌"
                print(f"{emoji} {name:20s} {message}")

                if passed:
                    self.checks_passed += 1
                else:
                    self.checks_failed += 1
            except Exception as e:
                results[name] = {"passed": False, "message": f"Error: {e}"}
                print(f"❌ {name:20s} Error: {e}")
                self.checks_failed += 1

        # Print warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")

        # Summary
        print("\n" + "=" * 70)
        total = self.checks_passed + self.checks_failed
        print(f"📊 SUMMARY: {self.checks_passed}/{total} checks passed")

        if self.checks_failed == 0 and not self.warnings:
            print("🎉 All systems operational!")
            results["overall_status"] = "healthy"
        elif self.checks_failed == 0:
            print("⚠️  System operational with warnings")
            results["overall_status"] = "warning"
        else:
            print("❌ System has issues that need attention")
            results["overall_status"] = "unhealthy"

        print("=" * 70)

        return results


def main():
    """Run health check"""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot Health Check")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="API server URL (default: http://localhost:8080)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    checker = HealthChecker(api_url=args.api_url)
    results = checker.run_all_checks()

    if args.json:
        import json

        print(json.dumps(results, indent=2))

    # Exit code: 0 if healthy, 1 if warnings, 2 if unhealthy
    if results["overall_status"] == "healthy":
        sys.exit(0)
    elif results["overall_status"] == "warning":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
