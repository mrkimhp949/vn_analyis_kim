#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script for Vietnam Market Improvements v4.0

Demonstrates:
1. Enhanced Market Regime Detection (VN30, HNX, margin debt)
2. Session Trading (ATO/ATC timing)
3. Fundamental Analysis
4. Walk-Forward Validation
5. Monte Carlo Simulation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def demo_enhanced_regime():
    """Demo enhanced market regime detection"""
    print("\n" + "=" * 70)
    print("1️⃣  ENHANCED MARKET REGIME DETECTION")
    print("=" * 70)

    from src.market.regime_detector import MarketRegimeDetector as EnhancedRegimeDetector

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")

    # VNINDEX - uptrend
    vnindex_prices = 1200 + np.cumsum(np.random.randn(250) * 8 + 0.5)
    vnindex_df = pd.DataFrame(
        {
            "open": vnindex_prices * 0.99,
            "high": vnindex_prices * 1.01,
            "low": vnindex_prices * 0.98,
            "close": vnindex_prices,
            "volume": np.random.randint(100000000, 500000000, 250),
        }
    )

    # VN30 - similar trend
    vn30_prices = 1300 + np.cumsum(np.random.randn(250) * 10 + 0.6)
    vn30_df = pd.DataFrame(
        {
            "open": vn30_prices * 0.99,
            "high": vn30_prices * 1.01,
            "low": vn30_prices * 0.98,
            "close": vn30_prices,
            "volume": np.random.randint(50000000, 200000000, 250),
        }
    )

    # Detect regime
    detector = EnhancedRegimeDetector()
    regime = detector.detect(vnindex_df, vn30_df)

    print(f"\n📊 Market Regime: {regime.regime}")
    print(f"   Confidence: {regime.confidence:.1f}%")
    print(f"   Tradeable: {regime.tradeable}")
    print(f"\n   Component Scores:")
    print(f"   - VNINDEX: {regime.vnindex_score:.3f}")
    print(f"   - VN30: {regime.vn30_score:.3f}")
    print(f"   - HNX: {regime.hnx_score:.3f}")
    print(f"\n   Additional Signals:")
    print(f"   - Foreign Flow: {regime.foreign_flow_signal}")
    print(f"   - Margin Debt: {regime.margin_debt_signal}")
    print(f"   - Volatility Percentile: {regime.volatility_percentile:.0f}th")
    print(f"\n   Description: {regime.description}")
    print(f"\n   Recommendations:")
    for rec in regime.recommendations:
        print(f"   {rec}")


def demo_session_trading():
    """Demo session trading logic"""
    print("\n" + "=" * 70)
    print("2️⃣  SESSION TRADING (ATO/ATC)")
    print("=" * 70)

    from src.market.session_trading import SessionTradingManager
    import pytz

    manager = SessionTradingManager()
    VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

    # Test different times
    test_times = [
        (9, 5, "ATO Session"),
        (10, 0, "Morning Optimal"),
        (11, 15, "Pre-lunch"),
        (13, 45, "Afternoon Optimal"),
        (14, 35, "ATC Session"),
    ]

    print("\n📅 Session Analysis for Different Times:")
    print("-" * 60)

    for hour, minute, label in test_times:
        test_time = datetime(2024, 12, 3, hour, minute, tzinfo=VN_TZ)
        session = manager.get_current_session(test_time)
        timing = manager.analyze_entry_timing(test_time)

        print(f"\n⏰ {hour:02d}:{minute:02d} - {label}")
        print(f"   Session: {session.session_type.value}")
        print(f"   Entry Quality: {session.entry_quality}")
        print(f"   Risk Level: {session.risk_level}")
        print(f"   Order Type: {session.recommended_order_type.value}")
        print(f"   Quality Score: {timing.quality_score:.0f}/100")
        print(f"   Position Multiplier: {timing.position_size_multiplier:.2f}x")

        if session.warnings:
            print(f"   Warnings: {session.warnings[0]}")


def demo_fundamental_analysis():
    """Demo fundamental analysis"""
    print("\n" + "=" * 70)
    print("3️⃣  FUNDAMENTAL ANALYSIS")
    print("=" * 70)

    from src.data.fundamental_analyzer import (
        FundamentalAnalyzer,
        FundamentalMetrics,
        EarningsEvent,
    )

    analyzer = FundamentalAnalyzer()

    # Test stocks with different fundamentals
    test_stocks = [
        {
            "symbol": "VNM",
            "sector": "Thực phẩm",
            "pe_ratio": 18.5,
            "pb_ratio": 3.2,
            "roe": 25.0,
            "debt_to_equity": 0.4,
            "revenue_growth": 8.0,
        },
        {
            "symbol": "HPG",
            "sector": "Thép",
            "pe_ratio": 6.5,
            "pb_ratio": 0.9,
            "roe": 12.0,
            "debt_to_equity": 0.8,
            "revenue_growth": -5.0,
        },
        {
            "symbol": "FPT",
            "sector": "Công nghệ",
            "pe_ratio": 22.0,
            "pb_ratio": 4.5,
            "roe": 22.0,
            "debt_to_equity": 0.3,
            "revenue_growth": 18.0,
        },
    ]

    print("\n📊 Fundamental Analysis Results:")
    print("-" * 60)

    for stock in test_stocks:
        # Create and cache metrics
        metrics = FundamentalMetrics(
            symbol=stock["symbol"],
            pe_ratio=stock["pe_ratio"],
            pb_ratio=stock["pb_ratio"],
            roe=stock["roe"],
            debt_to_equity=stock["debt_to_equity"],
            revenue_growth=stock["revenue_growth"],
            last_updated=datetime.now(),
        )
        analyzer._metrics_cache[stock["symbol"]] = metrics

        # Calculate score
        score = analyzer.calculate_fundamental_score(stock["symbol"], stock["sector"])

        print(f"\n📈 {stock['symbol']} ({stock['sector']})")
        print(f"   Total Score: {score.total_score:.1f}/100")
        print(f"   - Valuation: {score.valuation_score:.1f}")
        print(f"   - Profitability: {score.profitability_score:.1f}")
        print(f"   - Financial Health: {score.financial_health_score:.1f}")
        print(f"   - Growth: {score.growth_score:.1f}")
        print(f"   Recommendation: {score.recommendation}")

        if score.warnings:
            print(f"   Warnings: {', '.join(score.warnings[:2])}")


def demo_walk_forward():
    """Demo walk-forward validation"""
    print("\n" + "=" * 70)
    print("4️⃣  WALK-FORWARD VALIDATION")
    print("=" * 70)

    from backtesting.walk_forward import WalkForwardValidator

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
    prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.015 + 0.0003)

    data = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, 500),
        },
        index=dates,
    )

    # Simple momentum strategy
    def momentum_strategy(df, params):
        returns = df["close"].pct_change().dropna()

        # Simple momentum: buy when 20-day return > 0
        momentum = df["close"].pct_change(20)
        signals = (momentum > 0).shift(1).fillna(False)

        strategy_returns = returns * signals

        total_return = (1 + strategy_returns).prod() - 1
        sharpe = (
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            if strategy_returns.std() > 0
            else 0
        )
        win_rate = (
            (strategy_returns > 0).sum() / (strategy_returns != 0).sum()
            if (strategy_returns != 0).sum() > 0
            else 0
        )

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
        }

    # Run validation
    validator = WalkForwardValidator(num_windows=4)
    result = validator.validate(data, momentum_strategy)

    print(f"\n📊 Walk-Forward Validation Results:")
    print("-" * 60)
    print(f"   Total Windows: {result.total_windows}")
    print(f"   Avg Train Return: {result.avg_train_return:.2%}")
    print(f"   Avg Test Return: {result.avg_test_return:.2%}")
    print(f"   Test/Train Ratio: {result.test_vs_train_ratio:.2f}")
    print(f"   Consistency Score: {result.consistency_score:.1%}")
    print(f"   Avg Degradation: {result.avg_degradation:.1%}")
    print(f"\n   Verdict: {result.verdict}")

    print(f"\n   Window Details:")
    for w in result.windows:
        print(f"   - Window {w.window_id}: Train={w.train_return:.2%}, Test={w.test_return:.2%}")


def demo_monte_carlo():
    """Demo Monte Carlo simulation"""
    print("\n" + "=" * 70)
    print("5️⃣  MONTE CARLO SIMULATION")
    print("=" * 70)

    from backtesting.walk_forward import MonteCarloSimulator

    # Sample trade returns (realistic distribution)
    np.random.seed(42)

    # Mix of wins and losses
    wins = np.random.uniform(0.02, 0.15, 60)  # 60 winning trades: 2-15%
    losses = np.random.uniform(-0.07, -0.01, 40)  # 40 losing trades: -1% to -7%
    trade_returns = list(np.concatenate([wins, losses]))
    np.random.shuffle(trade_returns)

    # Run simulation
    simulator = MonteCarloSimulator(num_simulations=10000)
    result = simulator.simulate(trade_returns, initial_capital=100_000_000)

    print(f"\n📊 Monte Carlo Simulation Results (10,000 simulations):")
    print("-" * 60)
    print(f"   Expected Return: {result.mean_return:.2%}")
    print(f"   Median Return: {result.median_return:.2%}")
    print(f"   Std Deviation: {result.std_return:.2%}")
    print(f"\n   Return Distribution:")
    print(f"   - 5th Percentile (Worst): {result.percentile_5:.2%}")
    print(f"   - 25th Percentile: {result.percentile_25:.2%}")
    print(f"   - 75th Percentile: {result.percentile_75:.2%}")
    print(f"   - 95th Percentile (Best): {result.percentile_95:.2%}")
    print(f"\n   Risk Metrics:")
    print(f"   - Probability of Loss: {result.probability_of_loss:.1%}")
    print(f"   - Probability of Ruin (>50% DD): {result.probability_of_ruin:.1%}")
    print(f"   - Expected Max Drawdown: {result.expected_max_drawdown:.2%}")
    print(f"   - VaR 95%: {result.var_95:.2%}")
    print(f"   - CVaR 95%: {result.cvar_95:.2%}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("🚀 VIETNAM MARKET IMPROVEMENTS v4.0 - DEMO")
    print("=" * 70)

    try:
        demo_enhanced_regime()
    except Exception as e:
        print(f"❌ Enhanced Regime Demo failed: {e}")

    try:
        demo_session_trading()
    except Exception as e:
        print(f"❌ Session Trading Demo failed: {e}")

    try:
        demo_fundamental_analysis()
    except Exception as e:
        print(f"❌ Fundamental Analysis Demo failed: {e}")

    try:
        demo_walk_forward()
    except Exception as e:
        print(f"❌ Walk-Forward Demo failed: {e}")

    try:
        demo_monte_carlo()
    except Exception as e:
        print(f"❌ Monte Carlo Demo failed: {e}")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
