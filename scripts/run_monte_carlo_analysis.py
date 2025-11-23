#!/usr/bin/env python3
"""
Monte Carlo Risk Analysis Script

Integrates Monte Carlo simulation with actual trading performance data.

Usage:
    # Analyze ML signals
    python scripts/run_monte_carlo_analysis.py --source ml

    # Analyze technical-only signals
    python scripts/run_monte_carlo_analysis.py --source technical_only

    # Analyze with custom parameters
    python scripts/run_monte_carlo_analysis.py --simulations 20000 --trades 200

    # Save results to file
    python scripts/run_monte_carlo_analysis.py --output monte_carlo_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.monte_carlo import MonteCarloSimulator
from src.monitoring.signal_performance_tracker import get_signal_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_from_tracker(
    signal_source: str = "ml",
    num_simulations: int = 10000,
    num_trades: int = 100,
    initial_capital: float = 100_000_000,
    use_kelly: bool = False,
    output_file: str = None,
):
    """
    Run Monte Carlo analysis using actual performance data

    Args:
        signal_source: 'ml' or 'technical_only'
        num_simulations: Number of Monte Carlo simulations
        num_trades: Trades per simulation
        initial_capital: Starting capital
        use_kelly: Use Kelly Criterion
        output_file: Save results to JSON file
    """
    print("\n" + "=" * 80)
    print("🎲 MONTE CARLO RISK ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Analyzing {signal_source.upper()} signal performance...")
    print(f"   Simulations: {num_simulations:,}")
    print(f"   Trades per sim: {num_trades}")
    print(f"   Initial capital: {initial_capital:,.0f} VND")
    print(f"   Position sizing: {'Kelly Criterion' if use_kelly else 'Fixed 10%'}")
    print()

    # Get performance data from tracker
    tracker = get_signal_tracker()
    perf = tracker.get_performance(signal_source)

    if perf is None:
        print(f"❌ No performance data found for {signal_source} signals")
        print("   Run some trades first to collect performance data")
        return

    if perf.executed_trades < 10:
        print(f"⚠️  WARNING: Only {perf.executed_trades} trades recorded")
        print("   Monte Carlo requires >= 10 trades for reliable results")
        print("   Results may not be statistically significant")
        print()

    # Display current performance stats
    print("📈 CURRENT PERFORMANCE STATISTICS:")
    print(f"   Total signals: {perf.total_signals}")
    print(f"   Executed trades: {perf.executed_trades}")
    print(f"   Win rate: {perf.win_rate:.1%}")
    print(f"   Average return: {perf.avg_return:+.2f}%")
    print(f"   Average win: {perf.avg_win:+.2f}%")
    print(f"   Average loss: {perf.avg_loss:+.2f}%")
    print(f"   Win/Loss ratio: {perf.win_loss_ratio:.2f}")
    print(f"   Sharpe ratio: {perf.sharpe_ratio:.2f}")
    print()

    # Validate data
    if perf.win_rate <= 0 or perf.win_rate >= 1:
        print(f"❌ Invalid win rate: {perf.win_rate:.1%}")
        return

    if perf.avg_win <= 0:
        print(f"❌ Invalid average win: {perf.avg_win:.2f}%")
        return

    if perf.avg_loss >= 0:
        print(f"❌ Invalid average loss: {perf.avg_loss:.2f}%")
        return

    # Run Monte Carlo simulation
    print("🎲 Running Monte Carlo simulation...")
    print()

    simulator = MonteCarloSimulator(
        win_rate=perf.win_rate,
        avg_win_pct=perf.avg_win,
        avg_loss_pct=perf.avg_loss,
        num_simulations=num_simulations,
        num_trades_per_sim=num_trades,
        initial_capital=initial_capital,
        use_kelly=use_kelly,
    )

    result = simulator.run_simulation()
    report = simulator.generate_report(result)
    print(report)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        result_dict = {
            "signal_source": signal_source,
            "current_performance": {
                "win_rate": perf.win_rate,
                "avg_win": perf.avg_win,
                "avg_loss": perf.avg_loss,
                "executed_trades": perf.executed_trades,
                "sharpe_ratio": perf.sharpe_ratio,
            },
            "monte_carlo": {
                "risk_of_ruin_pct": result.risk_of_ruin_pct,
                "expected_return_pct": result.expected_return_pct,
                "avg_max_drawdown": result.avg_max_drawdown,
                "percentiles": {
                    "5th": result.percentile_5th,
                    "25th": result.percentile_25th,
                    "50th": result.percentile_50th,
                    "75th": result.percentile_75th,
                    "95th": result.percentile_95th,
                },
                "extremes": {
                    "worst_case": result.worst_case,
                    "best_case": result.best_case,
                },
            },
            "parameters": {
                "num_simulations": num_simulations,
                "num_trades_per_sim": num_trades,
                "initial_capital": initial_capital,
                "use_kelly": use_kelly,
            },
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

    # Print recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")

    if result.risk_of_ruin_pct < 2.0:
        print("   ✅ Risk of ruin is EXCELLENT (<2%)")
        print("   → Strategy is safe for live trading")
    elif result.risk_of_ruin_pct < 5.0:
        print("   ✅ Risk of ruin is ACCEPTABLE (<5%)")
        print("   → Strategy meets A+ requirements")
    else:
        print(f"   ⚠️  Risk of ruin is HIGH ({result.risk_of_ruin_pct:.2f}%)")
        print("   → Reduce position sizes or improve win rate")

    if result.expected_return_pct > 10:
        print("   ✅ Expected return is EXCELLENT (>10%)")
    elif result.expected_return_pct > 0:
        print("   ✅ Expected return is POSITIVE")
    else:
        print("   ❌ Expected return is NEGATIVE - DO NOT TRADE")

    if result.avg_max_drawdown < 0.10:
        print("   ✅ Average drawdown is LOW (<10%)")
    elif result.avg_max_drawdown < 0.15:
        print("   ✅ Average drawdown is ACCEPTABLE (<15%)")
    else:
        print(f"   ⚠️  Average drawdown is HIGH ({result.avg_max_drawdown*100:.1f}%)")
        print("   → Consider tighter stop losses")

    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Risk Analysis for Trading Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ML signals
  python scripts/run_monte_carlo_analysis.py --source ml

  # Analyze with more simulations
  python scripts/run_monte_carlo_analysis.py --simulations 20000

  # Use Kelly Criterion
  python scripts/run_monte_carlo_analysis.py --use-kelly

  # Save results
  python scripts/run_monte_carlo_analysis.py --output results.json
        """
    )

    parser.add_argument(
        "--source",
        choices=["ml", "technical_only"],
        default="ml",
        help="Signal source to analyze (default: ml)"
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)"
    )

    parser.add_argument(
        "--trades",
        type=int,
        default=100,
        help="Number of trades per simulation (default: 100)"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=100_000_000,
        help="Initial capital in VND (default: 100,000,000)"
    )

    parser.add_argument(
        "--use-kelly",
        action="store_true",
        help="Use Kelly Criterion for position sizing"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    try:
        analyze_from_tracker(
            signal_source=args.source,
            num_simulations=args.simulations,
            num_trades=args.trades,
            initial_capital=args.capital,
            use_kelly=args.use_kelly,
            output_file=args.output,
        )
    except Exception as e:
        logger.error(f"Error running Monte Carlo analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
