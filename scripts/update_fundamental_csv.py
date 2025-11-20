"""
Helper Script: Update Fundamental Ratios CSV
Get latest P/E, P/B data from websites and update CSV file

Usage:
  python scripts/update_fundamental_csv.py

This script helps you maintain the CSV file with latest fundamental data.
Run this weekly to keep data fresh.
"""

import pandas as pd
from datetime import datetime

CSV_FILE = "data/fundamental_ratios.csv"

print("=" * 80)
print("📊 FUNDAMENTAL RATIOS CSV UPDATER")
print("=" * 80)

print("\n📝 HOW TO USE THIS SCRIPT:")
print("1. Get latest P/E, P/B data from:")
print("   - https://www.tcbs.com.vn/marketwatch")
print("   - https://finance.vietstock.vn")
print("   - https://s.cafef.vn")
print("\n2. Edit this script and add/update symbols below")
print("3. Run: python scripts/update_fundamental_csv.py")
print("4. Data will be saved to: data/fundamental_ratios.csv")

print("\n" + "=" * 80)

# ============================================================================
# EDIT THIS SECTION WITH YOUR DATA
# ============================================================================

# Update this dict with latest P/E, P/B data
# Get data from tcbs.com.vn, vietstock.vn, or cafef.vn

fundamental_data = {
    # Symbol: (PE, PB, ROE, ROA, DebtToEquity, EPS, MarketCap)

    # Blue chips
    "VNM": (18.5, 5.2, 25.3, 12.1, 0.35, 5200, 95000000000000),
    "VCB": (12.3, 2.8, 22.1, 1.8, 8.5, 8500, 450000000000000),
    "FPT": (15.7, 4.1, 28.5, 15.2, 0.42, 6800, 75000000000000),
    "HPG": (8.2, 1.5, 18.3, 9.5, 1.2, 3200, 62000000000000),
    "VIC": (25.1, 3.8, 12.5, 5.3, 2.5, 4100, 180000000000000),

    # Real estate
    "VHM": (15.8, 2.1, 14.2, 6.8, 1.8, 3500, 125000000000000),
    "VRE": (18.9, 1.8, 9.8, 5.2, 2.1, 2800, 72000000000000),

    # Banks
    "TCB": (9.8, 1.9, 20.5, 1.5, 7.2, 7200, 98000000000000),
    "VPB": (6.5, 1.2, 18.9, 1.2, 9.8, 5600, 75000000000000),

    # Consumer
    "MWG": (12.5, 2.3, 16.8, 8.3, 0.65, 8900, 48000000000000),
    "MSN": (14.3, 3.2, 22.1, 9.7, 0.55, 12000, 88000000000000),
    "SAB": (22.5, 6.5, 28.3, 18.5, 0.15, 18500, 135000000000000),

    # Energy
    "GAS": (11.2, 2.5, 19.5, 12.8, 0.28, 9500, 105000000000000),
    "PLX": (9.8, 1.4, 15.2, 7.8, 0.85, 6200, 58000000000000),
    "POW": (8.5, 1.1, 13.5, 8.2, 1.5, 4500, 42000000000000),

    # Add more symbols here...
    # "SYMBOL": (PE, PB, ROE, ROA, D/E, EPS, MarketCap),
}

# ============================================================================
# END OF EDIT SECTION
# ============================================================================

print("\n📊 Processing data...")

# Convert to DataFrame
data_list = []
for symbol, values in fundamental_data.items():
    pe, pb, roe, roa, de, eps, mcap = values
    data_list.append({
        'Symbol': symbol,
        'PE': pe,
        'PB': pb,
        'ROE': roe,
        'ROA': roa,
        'DebtToEquity': de,
        'EPS': eps,
        'MarketCap': mcap,
        'LastUpdate': datetime.now().strftime('%Y-%m-%d')
    })

df = pd.DataFrame(data_list)

# Sort by symbol
df = df.sort_values('Symbol')

# Save to CSV
df.to_csv(CSV_FILE, index=False)

print(f"\n✅ Updated {len(df)} symbols in {CSV_FILE}")
print(f"📅 Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Show summary
print("\n📊 Summary:")
print(f"   Total symbols: {len(df)}")
print(f"   Avg P/E: {df['PE'].mean():.2f}")
print(f"   Avg P/B: {df['PB'].mean():.2f}")
print(f"   Avg ROE: {df['ROE'].mean():.2f}%")

print("\n💡 Next time:")
print("   1. Get latest data from tcbs.com.vn or vietstock.vn")
print("   2. Update the fundamental_data dict in this script")
print("   3. Run this script again")
print("   4. Recommend: Update weekly")

print("\n" + "=" * 80)
