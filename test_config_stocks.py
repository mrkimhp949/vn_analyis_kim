# test_config_stocks.py
from config import TICKERS

print(f"SO LUONG MA TRONG TICKERS: {len(TICKERS)}")
print(f"DANH SACH MA: {TICKERS}")

# Test import market regime
from market_regime_proxy import ProxyMarketRegimeAnalyzer

analyzer = ProxyMarketRegimeAnalyzer()
print(f"Market Regime Analyzer duoc khoi tao voi: {len(analyzer.proxy_stocks)} ma")
print(f"Danh sach ma: {analyzer.proxy_stocks}")