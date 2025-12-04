"""Test TCBS API để debug vấn đề insufficient data"""

from datetime import datetime, timedelta
import requests

symbol = "VCB"
lookback = 200

start = datetime.now() - timedelta(days=lookback)
end = datetime.now()

url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
params = {
    "ticker": symbol,
    "type": "stock",
    "resolution": "D",
    "from": int(start.timestamp()),
    "to": int(end.timestamp()),
}

print(f"Request: {start.date()} to {end.date()} ({lookback} days)")
print(f"Params: {params}")

response = requests.get(url, params=params, timeout=10)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    bars = data.get("data", [])
    print(f"Bars returned: {len(bars)}")
    if bars:
        print(f"First bar: {bars[0]['tradingDate']}")
        print(f"Last bar: {bars[-1]['tradingDate']}")
