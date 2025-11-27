"""
Validate tickers in List.csv against TCBS API
Check which tickers have data available

Features:
- Filter by exchange (HSX, HNX, Upcom, OTC)
- Check data quality (volume, gaps, price validity)
- Option to auto-update List.csv (remove invalid tickers)
- Export quality tickers to quality_tickers.txt
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

import time
import pandas as pd
from typing import List, Dict, Tuple, Optional

from src.data.loader import load_data
from src.data.ticker_loader import get_ticker_loader


# Quality thresholds
MIN_AVG_VOLUME = 50_000  # Minimum average volume (shares)
MIN_AVG_VALUE = 500_000_000  # Minimum average value (500M VND)
MAX_GAP_DAYS = 5  # Maximum allowed gap in trading days
MIN_PRICE = 1000  # Minimum price (1,000 VND)
MAX_PRICE = 500_000  # Maximum price (500,000 VND)


def validate_tickers(
    sample_size: Optional[int] = None,
    skip_first: int = 0,
    exchanges: Optional[List[str]] = None,
    check_quality: bool = True,
    auto_update: bool = False,
):
    """
    Validate tickers from List.csv against TCBS

    Args:
        sample_size: Number of tickers to test (None = all)
        skip_first: Skip first N tickers
        exchanges: List of exchanges to include (None = all except OTC)
        check_quality: Whether to check data quality (volume, gaps, etc.)
        auto_update: Whether to update List.csv with valid tickers only
    """
    print("=" * 70)
    print("🔍 VALIDATING TICKERS FROM LIST.CSV")
    print("=" * 70)

    # Load tickers with exchange info
    loader = get_ticker_loader()

    # Get ticker info from List.csv directly for exchange filtering
    ticker_info = load_ticker_info()
    all_tickers = loader.all_tickers

    print(f"\n📊 Total tickers in List.csv: {len(all_tickers)}")

    # Filter by exchange
    if exchanges is None:
        # Default: exclude OTC (usually no data on TCBS)
        exchanges = ["HSX", "HNX", "Upcom"]

    filtered_tickers = []
    excluded_otc = []
    for ticker in all_tickers:
        exchange = ticker_info.get(ticker, {}).get("exchange", "Unknown")
        if exchange in exchanges:
            filtered_tickers.append(ticker)
        elif exchange == "OTC":
            excluded_otc.append(ticker)

    if excluded_otc:
        print(f"⏭️ Excluded {len(excluded_otc)} OTC tickers (no TCBS data)")

    all_tickers = filtered_tickers
    print(f"📋 Tickers after exchange filter: {len(all_tickers)}")

    # Select tickers to test
    if skip_first > 0:
        all_tickers = all_tickers[skip_first:]
        print(f"⏭️ Skipping first {skip_first} tickers")

    if sample_size:
        test_tickers = all_tickers[:sample_size]
        print(f"🧪 Testing {len(test_tickers)} tickers (sample)")
    else:
        test_tickers = all_tickers
        print(f"🧪 Testing ALL {len(test_tickers)} tickers")

    if check_quality:
        print(f"\n📏 Quality thresholds:")
        print(f"   - Min avg volume: {MIN_AVG_VOLUME:,} shares")
        print(f"   - Min avg value: {MIN_AVG_VALUE/1e6:.0f}M VND")
        print(f"   - Max gap days: {MAX_GAP_DAYS}")
        print(f"   - Price range: {MIN_PRICE:,} - {MAX_PRICE:,} VND")

    print("\n" + "=" * 70)
    print("Starting validation...")
    print("=" * 70 + "\n")

    valid_tickers = []
    invalid_tickers = []
    low_quality_tickers = []
    error_tickers = []

    for i, ticker in enumerate(test_tickers, 1):
        try:
            print(f"({i}/{len(test_tickers)}) Testing {ticker}...", end=" ")

            # Try to load data
            df = load_data(ticker, lookback=60, use_cache=False, required_bars=1)

            if df is None or df.empty or len(df) == 0:
                print("❌ No data")
                invalid_tickers.append(ticker)
                time.sleep(0.1)
                continue

            # Basic validation passed
            bars = len(df)

            # Check data quality if enabled
            if check_quality:
                quality_issues = check_data_quality(df, ticker)

                if quality_issues:
                    print(f"⚠️ Low quality ({bars} bars): {', '.join(quality_issues)}")
                    low_quality_tickers.append((ticker, quality_issues))
                else:
                    print(f"✅ OK ({bars} bars)")
                    valid_tickers.append(ticker)
            else:
                print(f"✅ OK ({bars} bars)")
                valid_tickers.append(ticker)

            # Rate limiting
            time.sleep(0.1)

        except ValueError as e:
            error_msg = str(e)
            if "không có dữ liệu" in error_msg or "TCBS không có" in error_msg:
                print("❌ No data in TCBS")
                invalid_tickers.append(ticker)
            else:
                print(f"⚠️ {error_msg[:50]}")
                error_tickers.append((ticker, error_msg))

        except Exception as e:
            print(f"💥 Exception: {str(e)[:50]}")
            error_tickers.append((ticker, str(e)))

    # Summary
    print_summary(test_tickers, valid_tickers, invalid_tickers, low_quality_tickers, error_tickers)

    # Save results
    save_results(valid_tickers, invalid_tickers, low_quality_tickers, error_tickers, excluded_otc)

    # Export quality tickers
    export_quality_tickers(valid_tickers)

    # Auto-update List.csv if requested
    if auto_update and invalid_tickers:
        update_list_csv(valid_tickers + [t for t, _ in low_quality_tickers], ticker_info)

    print("\n" + "=" * 70)
    print("✅ Validation completed!")
    print("=" * 70)

    return valid_tickers, invalid_tickers, low_quality_tickers, error_tickers


def load_ticker_info() -> Dict[str, Dict]:
    """Load ticker info from List.csv"""
    ticker_info = {}
    try:
        df = pd.read_csv(
            "List.csv",
            header=None,
            names=["symbol", "name", "exchange"],
            encoding="utf-8",
            on_bad_lines="skip",
        )
        for _, row in df.iterrows():
            ticker_info[row["symbol"]] = {
                "name": row["name"],
                "exchange": row["exchange"],
            }
    except Exception as e:
        print(f"⚠️ Could not load ticker info: {e}")
    return ticker_info


def check_data_quality(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Check data quality and return list of issues

    Returns:
        List of quality issues (empty if all good)
    """
    issues = []

    try:
        # 1. Check average volume
        avg_volume = df["volume"].mean()
        if avg_volume < MIN_AVG_VOLUME:
            issues.append(f"low_vol({avg_volume/1000:.0f}K)")

        # 2. Check average trading value
        avg_price = df["close"].mean()
        avg_value = avg_volume * avg_price
        if avg_value < MIN_AVG_VALUE:
            issues.append(f"low_value({avg_value/1e6:.0f}M)")

        # 3. Check price range
        latest_price = df.iloc[-1]["close"]
        if latest_price < MIN_PRICE:
            issues.append(f"price_too_low({latest_price:,.0f})")
        elif latest_price > MAX_PRICE:
            issues.append(f"price_too_high({latest_price:,.0f})")

        # 4. Check for gaps in data
        if "time" in df.columns:
            df_sorted = df.sort_values("time")
            time_diffs = df_sorted["time"].diff().dt.days
            max_gap = time_diffs.max()
            if pd.notna(max_gap) and max_gap > MAX_GAP_DAYS:
                issues.append(f"data_gap({int(max_gap)}d)")

        # 5. Check for zero/negative prices
        if (df["close"] <= 0).any():
            issues.append("invalid_price")

        # 6. Check for suspicious volume spikes (potential data errors)
        if len(df) > 5:
            vol_std = df["volume"].std()
            vol_mean = df["volume"].mean()
            if vol_mean > 0 and vol_std / vol_mean > 5:
                issues.append("vol_anomaly")

    except Exception as e:
        issues.append(f"check_error({str(e)[:20]})")

    return issues


def print_summary(test_tickers, valid, invalid, low_quality, errors):
    """Print validation summary"""
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)

    total = len(test_tickers)
    print(f"\n✅ Valid (high quality): {len(valid)}/{total} ({len(valid)/total*100:.1f}%)")
    print(f"⚠️ Low quality: {len(low_quality)}/{total} ({len(low_quality)/total*100:.1f}%)")
    print(f"❌ Invalid (no data): {len(invalid)}/{total} ({len(invalid)/total*100:.1f}%)")
    print(f"💥 Errors: {len(errors)}/{total} ({len(errors)/total*100:.1f}%)")

    # Show invalid tickers
    if invalid:
        print(f"\n❌ Invalid tickers ({len(invalid)}):")
        for ticker in invalid[:20]:
            print(f"  - {ticker}")
        if len(invalid) > 20:
            print(f"  ... and {len(invalid) - 20} more")

    # Show low quality tickers
    if low_quality:
        print(f"\n⚠️ Low quality tickers ({len(low_quality)}):")
        for ticker, issues in low_quality[:15]:
            print(f"  - {ticker}: {', '.join(issues)}")
        if len(low_quality) > 15:
            print(f"  ... and {len(low_quality) - 15} more")

    # Show error tickers
    if errors:
        print(f"\n💥 Error tickers ({len(errors)}):")
        for ticker, error in errors[:10]:
            print(f"  - {ticker}: {error[:50]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


def save_results(valid, invalid, low_quality, errors, excluded_otc):
    """Save validation results to file"""
    try:
        with open("validation_results.txt", "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("TICKER VALIDATION RESULTS\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Valid (high quality): {len(valid)}\n")
            f.write(f"Low quality: {len(low_quality)}\n")
            f.write(f"Invalid (no data): {len(invalid)}\n")
            f.write(f"Errors: {len(errors)}\n")
            f.write(f"Excluded OTC: {len(excluded_otc)}\n\n")

            if valid:
                f.write("=" * 70 + "\n")
                f.write("VALID TICKERS (HIGH QUALITY)\n")
                f.write("=" * 70 + "\n")
                for ticker in sorted(valid):
                    f.write(f"{ticker}\n")
                f.write("\n")

            if low_quality:
                f.write("=" * 70 + "\n")
                f.write("LOW QUALITY TICKERS\n")
                f.write("=" * 70 + "\n")
                for ticker, issues in sorted(low_quality):
                    f.write(f"{ticker}: {', '.join(issues)}\n")
                f.write("\n")

            if invalid:
                f.write("=" * 70 + "\n")
                f.write("INVALID TICKERS (No data in TCBS)\n")
                f.write("=" * 70 + "\n")
                for ticker in sorted(invalid):
                    f.write(f"{ticker}\n")
                f.write("\n")

            if excluded_otc:
                f.write("=" * 70 + "\n")
                f.write("EXCLUDED OTC TICKERS\n")
                f.write("=" * 70 + "\n")
                for ticker in sorted(excluded_otc):
                    f.write(f"{ticker}\n")
                f.write("\n")

            if errors:
                f.write("=" * 70 + "\n")
                f.write("ERROR TICKERS\n")
                f.write("=" * 70 + "\n")
                for ticker, error in sorted(errors):
                    f.write(f"{ticker}: {error}\n")

        print("\n💾 Results saved to: validation_results.txt")

    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")


def export_quality_tickers(valid_tickers: List[str]):
    """Export quality tickers to a simple text file for use in trading"""
    try:
        with open("quality_tickers.txt", "w", encoding="utf-8") as f:
            f.write("# Quality tickers validated against TCBS\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total: {len(valid_tickers)} tickers\n\n")
            for ticker in sorted(valid_tickers):
                f.write(f"{ticker}\n")
        print(f"💾 Quality tickers exported to: quality_tickers.txt ({len(valid_tickers)} tickers)")
    except Exception as e:
        print(f"⚠️ Could not export quality tickers: {e}")


def update_list_csv(valid_tickers: List[str], ticker_info: Dict[str, Dict]):
    """Update List.csv to keep only valid tickers"""
    try:
        # Backup original
        import shutil

        shutil.copy("List.csv", "List.csv.backup")
        print("📦 Backed up List.csv to List.csv.backup")

        # Write new List.csv with only valid tickers
        with open("List.csv", "w", encoding="utf-8") as f:
            for ticker in valid_tickers:
                info = ticker_info.get(ticker, {})
                name = info.get("name", "")
                exchange = info.get("exchange", "")
                f.write(f"{ticker},{name},{exchange}\n")

        print(f"✅ Updated List.csv with {len(valid_tickers)} valid tickers")

    except Exception as e:
        print(f"❌ Could not update List.csv: {e}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate tickers in List.csv")
    parser.add_argument("-n", "--sample", type=int, help="Number of tickers to test")
    parser.add_argument("-s", "--skip", type=int, default=0, help="Skip first N tickers")
    parser.add_argument(
        "-e",
        "--exchanges",
        nargs="+",
        default=None,
        help="Exchanges to include (HSX HNX Upcom OTC). Default: HSX HNX Upcom",
    )
    parser.add_argument(
        "--no-quality", action="store_true", help="Skip quality checks (only check if data exists)"
    )
    parser.add_argument(
        "--auto-update", action="store_true", help="Auto-update List.csv to remove invalid tickers"
    )
    parser.add_argument(
        "--include-otc", action="store_true", help="Include OTC tickers in validation"
    )

    args = parser.parse_args()

    # Handle exchanges
    exchanges = args.exchanges
    if args.include_otc and exchanges is None:
        exchanges = ["HSX", "HNX", "Upcom", "OTC"]

    # Run validation
    try:
        validate_tickers(
            sample_size=args.sample,
            skip_first=args.skip,
            exchanges=exchanges,
            check_quality=not args.no_quality,
            auto_update=args.auto_update,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
    except Exception:
        print("\n\n❌ Error")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
