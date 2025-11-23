# Monte Carlo Risk Analysis - Sample Output

This document shows expected output from the Monte Carlo simulator.

## Example 1: Profitable Strategy (52% win rate)

```
🎲 MONTE CARLO RISK ANALYSIS

📊 Simulation Parameters:
   Simulations: 10,000
   Trades per simulation: 100
   Initial capital: 100,000,000 VND
   Win rate: 52.0%
   Avg win: +2.50%
   Avg loss: -1.50%
   Position sizing: Fixed (10.0%)

🎯 RISK METRICS:
   Risk of Ruin (>50% loss): 0.85%
   ✅ EXCELLENT: Risk of ruin <2%
   Risk of 30% loss: 3.20%
   Risk of 20% loss: 8.50%

💰 EXPECTED VALUE:
   Expected return: +12.35%
   Median return: +11.80%
   Expected final capital: 112,350,000 VND
   ✅ Positive expected value

📊 CONFIDENCE INTERVALS:
   5th percentile: -5.20% (94,800,000 VND)
   25th percentile: +5.50% (105,500,000 VND)
   50th percentile: +11.80% (111,800,000 VND)
   75th percentile: +18.90% (118,900,000 VND)
   95th percentile: +31.50% (131,500,000 VND)

🔺 EXTREMES:
   Best case: +45.80% (145,800,000 VND)
   Worst case: -28.50% (71,500,000 VND)

📉 DRAWDOWN STATISTICS:
   Average max drawdown: 8.50%
   Worst drawdown: 22.30%
   Simulations with DD >10%: 28.5%
   Simulations with DD >20%: 3.2%
   ✅ Average drawdown <15%

🎲 WIN/LOSS DISTRIBUTION:
   Average win rate: 52.1%
   Average profit factor: 1.73

================================================================================
📋 OVERALL ASSESSMENT:
   Grade: A (Very Good)

   ✅ STRATEGY APPROVED FOR TRADING
   - Low risk of ruin (0.85%)
   - Positive expected value (+12.35%)
   - Acceptable drawdown (8.5%)
================================================================================
```

## Example 2: With Kelly Criterion

```
🎲 MONTE CARLO RISK ANALYSIS

📊 Simulation Parameters:
   Simulations: 10,000
   Trades per simulation: 100
   Initial capital: 100,000,000 VND
   Win rate: 52.0%
   Avg win: +2.50%
   Avg loss: -1.50%
   Position sizing: Kelly (14.0%)

🎯 RISK METRICS:
   Risk of Ruin (>50% loss): 0.35%
   ✅ EXCELLENT: Risk of ruin <2%
   Risk of 30% loss: 1.80%
   Risk of 20% loss: 5.20%

💰 EXPECTED VALUE:
   Expected return: +18.75%
   Median return: +17.90%
   Expected final capital: 118,750,000 VND
   ✅ Positive expected value

📊 CONFIDENCE INTERVALS:
   5th percentile: -2.50% (97,500,000 VND)
   25th percentile: +8.80% (108,800,000 VND)
   50th percentile: +17.90% (117,900,000 VND)
   75th percentile: +27.50% (127,500,000 VND)
   95th percentile: +42.30% (142,300,000 VND)

🔺 EXTREMES:
   Best case: +58.90% (158,900,000 VND)
   Worst case: -18.70% (81,300,000 VND)

📉 DRAWDOWN STATISTICS:
   Average max drawdown: 6.80%
   Worst drawdown: 18.50%
   Simulations with DD >10%: 18.2%
   Simulations with DD >20%: 0.8%
   ✅ Average drawdown <15%

🎲 WIN/LOSS DISTRIBUTION:
   Average win rate: 52.0%
   Average profit factor: 1.82

================================================================================
📋 OVERALL ASSESSMENT:
   Grade: A+ (Excellent)

   ✅ STRATEGY APPROVED FOR TRADING
   - Low risk of ruin (0.35%)
   - Positive expected value (+18.75%)
   - Acceptable drawdown (6.8%)
================================================================================
```

## Key Metrics for A+ Rating

To achieve A+ rating (95-100), your Monte Carlo analysis must show:

### ✅ **REQUIRED** (Must Meet All):
1. **Risk of Ruin < 5%** (preferably < 2%)
2. **Expected Return > 0%** (positive expectancy)
3. **Average Max Drawdown < 20%** (preferably < 15%)

### 🎯 **RECOMMENDED** (For Higher Scores):
1. **Risk of Ruin < 2%** → Excellent risk management
2. **Expected Return > 10%** → Strong performance
3. **Average Max Drawdown < 10%** → Low volatility
4. **95th Percentile > 0%** → Consistent profitability
5. **Profit Factor > 1.5** → Good risk/reward ratio

## Usage Instructions

### Basic Usage:
```bash
# Analyze ML signals
python scripts/run_monte_carlo_analysis.py --source ml

# Analyze technical-only signals
python scripts/run_monte_carlo_analysis.py --source technical_only
```

### Advanced Options:
```bash
# More simulations for better accuracy
python scripts/run_monte_carlo_analysis.py --simulations 20000

# Use Kelly Criterion
python scripts/run_monte_carlo_analysis.py --use-kelly

# Save results to file
python scripts/run_monte_carlo_analysis.py --output results.json

# Run with more trades per simulation
python scripts/run_monte_carlo_analysis.py --trades 200
```

## Integration with Performance Tracker

The Monte Carlo simulator automatically pulls data from your actual trading performance:

```python
from src.analytics.monte_carlo import MonteCarloSimulator
from src.monitoring.signal_performance_tracker import get_signal_tracker

# Get actual performance data
tracker = get_signal_tracker()
perf = tracker.get_performance('ml')

# Run Monte Carlo with real data
sim = MonteCarloSimulator(
    win_rate=perf.win_rate,
    avg_win_pct=perf.avg_win,
    avg_loss_pct=perf.avg_loss,
    num_simulations=10000,
)

result = sim.run_simulation()
```

## Interpreting Results

### Risk of Ruin
- **< 1%**: Excellent - Very safe strategy
- **1-2%**: Good - Acceptable risk
- **2-5%**: Borderline - Consider improvements
- **> 5%**: Too risky - Reduce position sizes

### Expected Return
- **> 20%**: Excellent
- **10-20%**: Very Good
- **5-10%**: Good
- **0-5%**: Marginal
- **< 0%**: Do not trade

### Average Max Drawdown
- **< 10%**: Excellent - Low volatility
- **10-15%**: Good - Acceptable
- **15-20%**: Borderline - May be stressful
- **> 20%**: High - Difficult to maintain discipline

## Next Steps

After running Monte Carlo analysis:

1. **If Risk of Ruin < 5%**: ✅ Proceed to paper trading
2. **If Risk of Ruin 5-10%**: ⚠️ Reduce position sizes
3. **If Risk of Ruin > 10%**: ❌ Improve strategy before trading

See `ROADMAP_TO_A_PLUS.md` for complete A+ requirements.
