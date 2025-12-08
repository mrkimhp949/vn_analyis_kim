#!/usr/bin/env python
"""
Performance Benchmarking Script
Measure and report performance of critical bot components
"""
import os
import sys
import time
from typing import Callable, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Benchmark:
    """Performance benchmark runner"""

    def __init__(self):
        self.results = {}

    def measure(self, name: str, func: Callable, iterations: int = 100) -> Dict:
        """
        Measure function performance

        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations

        Returns:
            Performance statistics
        """
        times = []

        print(f"Running benchmark: {name} ({iterations} iterations)...", end=" ")

        for _ in range(iterations):
            start = time.time()
            func()
            duration = time.time() - start
            times.append(duration)

        times_arr = np.array(times)

        result = {
            "name": name,
            "iterations": iterations,
            "mean": np.mean(times_arr),
            "median": np.median(times_arr),
            "std": np.std(times_arr),
            "min": np.min(times_arr),
            "max": np.max(times_arr),
            "p95": np.percentile(times_arr, 95),
            "p99": np.percentile(times_arr, 99),
        }

        self.results[name] = result
        print(f"✅ Done ({result['mean']*1000:.2f}ms avg)")

        return result

    def report(self):
        """Print benchmark report"""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        df = pd.DataFrame(self.results.values())

        # Sort by mean time
        df = df.sort_values("mean", ascending=False)

        print(f"\n{'Benchmark':<40} {'Avg (ms)':<12} {'p95 (ms)':<12} {'Max (ms)':<12}")
        print("-" * 80)

        for _, row in df.iterrows():
            print(
                f"{row['name']:<40} "
                f"{row['mean']*1000:<12.2f} "
                f"{row['p95']*1000:<12.2f} "
                f"{row['max']*1000:<12.2f}"
            )

        print("\n" + "=" * 80)


def benchmark_ml_prediction():
    """Benchmark ML model prediction"""
    from src.ml.models.predictor import MLPredictor

    predictor = MLPredictor()
    predictor.load_models()

    X_test = np.random.randn(10, 18)

    def predict():
        predictor.predict(X_test)

    return predict


def benchmark_data_loading():
    """Benchmark data loading"""
    from src.data.loader import load_data

    def load():
        try:
            load_data("VCB", lookback=100, use_cache=True)
        except Exception:
            pass

    return load


def benchmark_technical_indicators():
    """Benchmark technical indicators calculation"""
    from src.ml.features.enhanced_v2 import add_ml_features

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "time": dates,
            "close": np.random.randn(100).cumsum() + 50000,
            "high": np.random.randn(100).cumsum() + 51000,
            "low": np.random.randn(100).cumsum() + 49000,
            "open": np.random.randn(100).cumsum() + 50000,
            "volume": np.random.randint(100000, 1000000, 100),
        }
    )

    def calculate():
        add_ml_features(df.copy())

    return calculate


def benchmark_entry_logic():
    """Benchmark entry logic analysis"""
    from src.strategies.entry_logic import ImprovedEntryLogic

    logic = ImprovedEntryLogic()

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "time": dates,
            "close": np.linspace(50000, 60000, 100),
            "high": np.linspace(51000, 61000, 100),
            "low": np.linspace(49000, 59000, 100),
            "volume": np.random.randint(100000, 1000000, 100),
            "sma20": np.linspace(50000, 60000, 100),
            "ema20": np.linspace(50000, 60000, 100),
            "ema50": np.linspace(48000, 58000, 100),
            "rsi": np.full(100, 50),
            "atr": np.full(100, 1000),
            "macd": np.full(100, 100),
            "macd_signal": np.full(100, 50),
            "volume_ratio": np.full(100, 1.2),
        }
    )

    ml_signal = {"signal": "BUY", "confidence": 75, "reason": "Test"}

    def analyze():
        logic.analyze_entry(df, ml_signal)

    return analyze


def benchmark_database_operations():
    """Benchmark database operations"""
    from src.data.database import get_db

    db = get_db()

    def db_operation():
        with db.get_connection() as conn:
            conn.execute("SELECT COUNT(*) FROM positions")
            conn.execute("SELECT * FROM trades LIMIT 10")

    return db_operation


def benchmark_portfolio_calculation():
    """Benchmark portfolio value calculation"""
    from src.portfolio.manager import get_portfolio_manager

    manager = get_portfolio_manager()

    def calculate():
        manager.get_portfolio_value()

    return calculate


def main():
    """Run all benchmarks"""
    print("=" * 80)
    print("🚀 STARTING PERFORMANCE BENCHMARKS")
    print("=" * 80)

    bench = Benchmark()

    # ML benchmarks
    bench.measure("ML Model Prediction (10 samples)", benchmark_ml_prediction(), iterations=100)

    # Data benchmarks
    bench.measure("Data Loading (with cache)", benchmark_data_loading(), iterations=20)
    bench.measure(
        "Technical Indicators Calculation",
        benchmark_technical_indicators(),
        iterations=50,
    )

    # Logic benchmarks
    bench.measure("Entry Logic Analysis", benchmark_entry_logic(), iterations=50)

    # Database benchmarks
    bench.measure("Database Query Operations", benchmark_database_operations(), iterations=100)

    # Portfolio benchmarks
    bench.measure("Portfolio Value Calculation", benchmark_portfolio_calculation(), iterations=100)

    # Generate report
    bench.report()

    # Performance targets
    print("\n📋 PERFORMANCE TARGETS:")
    print("-" * 80)
    print("Component                     Target       Status")
    print("-" * 80)

    targets = {
        "ML Model Prediction": 100,  # ms
        "Data Loading": 500,
        "Entry Logic Analysis": 50,
        "Database Query": 10,
        "Portfolio Calculation": 20,
    }

    for component, target in targets.items():
        # Find matching benchmark
        matching = [r for r in bench.results.values() if component.lower() in r["name"].lower()]
        if matching:
            actual = matching[0]["mean"] * 1000
            status = "✅ PASS" if actual < target else "❌ FAIL"
            print(f"{component:<30} {target:>6.0f}ms    {status} ({actual:.1f}ms)")
        else:
            print(f"{component:<30} {target:>6.0f}ms    ⚠️  Not measured")

    print("\n" + "=" * 80)
    print("✅ Benchmarks complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
