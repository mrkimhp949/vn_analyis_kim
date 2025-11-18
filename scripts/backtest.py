"""
Script chạy backtest - Fixed & Clean Version
"""

import os
import sys
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from run_backtest import Backtester

from src.config.legacy_config import TICKERS

# Force unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def main():
    # === Chuẩn bị thư mục kết quả ===
    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(
        """
    ╔════════════════════════════════════════════════╗
    ║      BACKTESTING TRADING STRATEGY              ║
    ║      ML-Based với Confidence Threshold         ║
    ╚════════════════════════════════════════════════╝
    """
    )

    # === Kiểm tra model ===
    print("🔍 Kiểm tra models...")
    if not os.path.exists("models/random_forest.pkl"):
        print("⚠️  KHÔNG TÌM THẤY models/random_forest.pkl")
        print("💡 Khuyến nghị chạy: python train_models.py\n")
    else:
        print("✅ Models OK\n")

    # === Khởi tạo backtester ===
    try:
        backtester = Backtester(initial_capital=100_000_000, commission=0.0015)
    except Exception:
        print("❌ Lỗi khởi tạo backtester")
        import traceback

        traceback.print_exc()
        sys.exit(1)  # ✅ thay return bằng sys.exit

    # === MENU ===
    print("=" * 60)
    print("CHỌN CHẾ ĐỘ BACKTEST:")
    print("1. Backtest 1 cổ phiếu (chi tiết + biểu đồ)")
    print("2. Backtest nhiều cổ phiếu (tổng quan)")
    print("3. So sánh các threshold khác nhau")
    print("=" * 60)

    choice = input("\nNhập lựa chọn (1-3): ").strip()

    # === OPTION 1: Single Stock ===
    if choice == "1":
        symbol = (
            input(f"Nhập mã cổ phiếu (mặc định: {TICKERS[0] if TICKERS else 'VNM'}): ")
            .strip()
            .upper()
        )
        if not symbol:
            symbol = TICKERS[0] if TICKERS else "VNM"

        lookback = input("Số ngày lịch sử (mặc định: 500): ").strip()
        lookback = int(lookback) if lookback else 500

        threshold = input("Confidence threshold % (mặc định: 50): ").strip()
        threshold = int(threshold) if threshold else 50

        print(f"\n🚀 Bắt đầu backtest {symbol} | {lookback} ngày | threshold {threshold}%\n")

        try:
            result = backtester.run_backtest(
                symbol, lookback=lookback, confidence_threshold=threshold
            )

            if result["total_trades"] == 0:
                print("❌ Không có giao dịch nào được thực hiện!")
            else:
                print("\n" + "=" * 60)
                print("✅ HOÀN THÀNH!")
                print("📊 Return: {result['total_return']:.2f}%")
                print(f"🔄 Trades: {result['total_trades']}")
                print("🎯 Win Rate: {result['win_rate']:.2f}%")
                print(f"⚖️ Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                print("=" * 60)

                # Biểu đồ
                if input("\nVẽ biểu đồ? (y/n): ").strip().lower() == "y":
                    backtester.plot_results(result)
                    print("✅ Đã vẽ biểu đồ!")

                # Giao dịch chi tiết
                if (
                    input("\nHiển thị chi tiết giao dịch? (y/n): ").strip().lower() == "y"
                    and len(result["trades"]) > 0
                ):
                    print("\n📋 CHI TIẾT GIAO DỊCH:")
                    print(result["trades"].to_string(index=False))

                    if input("\nLưu giao dịch vào CSV? (y/n): ").strip().lower() == "y":
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file = f"backtest_results/{symbol}_trades_{timestamp}.csv"
                        result["trades"].to_csv(file, index=False, encoding="utf-8-sig")
                        print(f"💾 Đã lưu: {file}")

        except Exception:
            print("\n❌ Lỗi backtest")
            import traceback

            traceback.print_exc()

    # === OPTION 2: Multiple Stocks ===
    elif choice == "2":
        lookback = input("Số ngày lịch sử (mặc định: 500): ").strip()
        lookback = int(lookback) if lookback else 500

        threshold = input("Confidence threshold % (mặc định: 50): ").strip()
        threshold = int(threshold) if threshold else 50

        print(f"\n🚀 Bắt đầu backtest {len(TICKERS)} cổ phiếu với threshold {threshold}%...\n")

        try:
            # run_multiple_backtest đã được cập nhật để nhận confidence_threshold
            _results = backtester.run_multiple_backtest(  # noqa: F841
                TICKERS[:5], lookback=lookback, confidence_threshold=threshold
            )
            print("\n✅ Kết quả đã lưu trong thư mục: backtest_results/")
        except Exception:
            print("\n❌ Lỗi")
            import traceback

            traceback.print_exc()

    # === OPTION 3: Threshold Comparison ===
    elif choice == "3":
        symbol = (
            input(f"Nhập mã cổ phiếu (mặc định: {TICKERS[0] if TICKERS else 'VNM'}): ")
            .strip()
            .upper()
        )
        if not symbol:
            symbol = TICKERS[0] if TICKERS else "VNM"

        lookback = input("Số ngày lịch sử (mặc định: 500): ").strip()
        lookback = int(lookback) if lookback else 500

        print("\n🧪 So sánh các threshold cho {symbol}...\n")

        thresholds = [30, 40, 50, 60, 70]
        comparison_results = []

        for threshold in thresholds:
            print("⏳ Đang test threshold {threshold}%...")
            try:
                result = backtester.run_backtest(
                    symbol, lookback=lookback, confidence_threshold=threshold
                )
                comparison_results.append(
                    {
                        "Threshold": threshold,
                        "Return (%)": result["total_return"],
                        "Trades": result["total_trades"],
                        "Win Rate (%)": result["win_rate"],
                        "Sharpe": result.get("sharpe_ratio", 0),
                        "Avg Confidence": result.get("avg_confidence", 0),
                    }
                )
                print("  ✅ {threshold}%: Return {result['total_return']:.2f}%")
            except Exception:
                print("  ⚠️ Lỗi threshold {threshold}%")

        if comparison_results:
            df = pd.DataFrame(comparison_results)
            print("\n" + "=" * 80)
            print("📊 SO SÁNH CÁC THRESHOLD")
            print("=" * 80)
            print(df.to_string(index=False))

            # Best theo Return & Sharpe
            best_return = df.loc[df["Return (%)"].idxmax()]
            best_sharpe = df.loc[df["Sharpe"].idxmax()]

            print("\n" + "=" * 80)
            print(
                f"🏆 TỐT NHẤT THEO RETURN: {best_return['Threshold']}% ({best_return['Return (%)']:.2f}%)"
            )
            print(
                f"🏆 TỐT NHẤT THEO SHARPE: {best_sharpe['Threshold']}% (Sharpe: {best_sharpe['Sharpe']:.2f})"
            )
            print("=" * 80)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file = f"backtest_results/threshold_comparison_{symbol}_{timestamp}.csv"
            df.to_csv(file, index=False, encoding="utf-8-sig")
            print(f"\n💾 Đã lưu kết quả: {file}")
        else:
            print("\n❌ Không có kết quả hợp lệ!")

    else:
        print("❌ Lựa chọn không hợp lệ! Vui lòng chọn 1, 2 hoặc 3.")


# === Chạy chương trình ===
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Đã hủy bởi người dùng!")
    except Exception:
        print("\n❌ Lỗi không mong đợi")
        import traceback

        traceback.print_exc()
