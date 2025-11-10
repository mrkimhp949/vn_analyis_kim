# test_tcbs.py
import requests
from datetime import datetime, timedelta

symbol = "VNM"
end = datetime.today()
start = end - timedelta(days=400)

url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
params = {
    "ticker": symbol,
    "type": "stock",
    "resolution": "D",
    "from": int(start.timestamp()),
    "to": int(end.timestamp())
}

print(f"URL: {url}")
print(f"Params: {params}")

response = requests.get(url, params=params, timeout=10)
print(f"\nStatus Code: {response.status_code}")
print(f"\nResponse: {response.text[:500]}")

data = response.json()
print(f"\nKeys: {list(data.keys())}")

if 'data' in data:
    print(f"Số dòng data: {len(data['data'])}")
    if data['data']:
        print(f"Mẫu data[0]: {data['data'][0]}")