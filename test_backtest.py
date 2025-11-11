from run_backtest import Backtester

backtester = Backtester()
result = backtester.run_backtest('ACB', confidence_threshold=30)
print(f"ACB với threshold 30: {result['total_trades']} trades")