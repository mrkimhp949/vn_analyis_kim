"""
Validate tickers in List.csv against TCBS API
Check which tickers have data available
"""
import sys
import os

# Fix encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from ticker_loader import get_ticker_loader
from data_loader import load_data
import time


def validate_tickers(sample_size=None, skip_first=0):
    """
    Validate tickers from List.csv against TCBS
    
    Args:
        sample_size: Number of tickers to test (None = all)
        skip_first: Skip first N tickers
    """
    print("=" * 70)
    print("🔍 VALIDATING TICKERS FROM LIST.CSV")
    print("=" * 70)
    
    # Load tickers
    loader = get_ticker_loader()
    all_tickers = loader.all_tickers
    
    print(f"\n📊 Total tickers in List.csv: {len(all_tickers)}")
    
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
    
    print("\n" + "=" * 70)
    print("Starting validation...")
    print("=" * 70 + "\n")
    
    valid_tickers = []
    invalid_tickers = []
    error_tickers = []
    
    for i, ticker in enumerate(test_tickers, 1):
        try:
            print(f"({i}/{len(test_tickers)}) Testing {ticker}...", end=' ')
            
            # Try to load data
            df = load_data(ticker, lookback=10, use_cache=False)
            
            if df is not None and not df.empty and len(df) > 0:
                print(f"✅ OK ({len(df)} bars)")
                valid_tickers.append(ticker)
            else:
                print(f"❌ No data")
                invalid_tickers.append(ticker)
            
            # Rate limiting
            time.sleep(0.1)
            
        except ValueError as e:
            error_msg = str(e)
            if "không có dữ liệu" in error_msg or "TCBS không có" in error_msg:
                print(f"❌ No data in TCBS")
                invalid_tickers.append(ticker)
            else:
                print(f"⚠️ Error: {error_msg[:50]}")
                error_tickers.append((ticker, error_msg))
        
        except Exception as e:
            print(f"💥 Exception: {str(e)[:50]}")
            error_tickers.append((ticker, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Valid tickers: {len(valid_tickers)}/{len(test_tickers)} ({len(valid_tickers)/len(test_tickers)*100:.1f}%)")
    print(f"❌ Invalid tickers: {len(invalid_tickers)}")
    print(f"⚠️ Error tickers: {len(error_tickers)}")
    
    # Show invalid tickers
    if invalid_tickers:
        print(f"\n❌ Invalid tickers ({len(invalid_tickers)}):")
        for ticker in invalid_tickers[:20]:  # Show first 20
            print(f"  - {ticker}")
        if len(invalid_tickers) > 20:
            print(f"  ... and {len(invalid_tickers) - 20} more")
    
    # Show error tickers
    if error_tickers:
        print(f"\n⚠️ Error tickers ({len(error_tickers)}):")
        for ticker, error in error_tickers[:10]:  # Show first 10
            print(f"  - {ticker}: {error[:50]}")
        if len(error_tickers) > 10:
            print(f"  ... and {len(error_tickers) - 10} more")
    
    # Save results
    save_results(valid_tickers, invalid_tickers, error_tickers)
    
    print("\n" + "=" * 70)
    print("✅ Validation completed!")
    print("=" * 70)
    
    return valid_tickers, invalid_tickers, error_tickers


def save_results(valid, invalid, errors):
    """Save validation results to file"""
    try:
        with open('validation_results.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("TICKER VALIDATION RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Valid tickers: {len(valid)}\n")
            f.write(f"Invalid tickers: {len(invalid)}\n")
            f.write(f"Error tickers: {len(errors)}\n\n")
            
            if valid:
                f.write("=" * 70 + "\n")
                f.write("VALID TICKERS\n")
                f.write("=" * 70 + "\n")
                for ticker in valid:
                    f.write(f"{ticker}\n")
                f.write("\n")
            
            if invalid:
                f.write("=" * 70 + "\n")
                f.write("INVALID TICKERS (No data in TCBS)\n")
                f.write("=" * 70 + "\n")
                for ticker in invalid:
                    f.write(f"{ticker}\n")
                f.write("\n")
            
            if errors:
                f.write("=" * 70 + "\n")
                f.write("ERROR TICKERS\n")
                f.write("=" * 70 + "\n")
                for ticker, error in errors:
                    f.write(f"{ticker}: {error}\n")
        
        print(f"\n💾 Results saved to: validation_results.txt")
        
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")


def main():
    """Main function"""
    import sys
    
    # Parse arguments
    sample_size = None
    skip_first = 0
    
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except:
            print("Usage: python validate_list_csv.py [sample_size] [skip_first]")
            print("Example: python validate_list_csv.py 100  # Test first 100")
            print("Example: python validate_list_csv.py 100 500  # Test 100 tickers starting from 500")
            return
    
    if len(sys.argv) > 2:
        try:
            skip_first = int(sys.argv[2])
        except:
            pass
    
    # Run validation
    try:
        validate_tickers(sample_size=sample_size, skip_first=skip_first)
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
