#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Quality Tickers for Training
Lọc ~150-200 mã chất lượng cao để train ML models

Tiêu chí:
1. VN30 (30 mã blue chips)
2. VN100 top stocks (HSX large caps)
3. Top HNX stocks
4. Bỏ qua: OTC, Upcom (low liquidity)

Output: quality_tickers.txt
"""

import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# VN30 Index components (30 blue chips - cập nhật 2024)
VN30_TICKERS = [
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "KDH",
    "MBB",
    "MSN",
    "MWG",
    "NVL",
    "PDR",
    "PLX",
    "POW",
    "SAB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
]

# Top VN100 stocks (large caps ngoài VN30)
VN100_ADDITIONAL = [
    # Ngân hàng
    "EIB",
    "LPB",
    "MSB",
    "OCB",
    "SHB",
    "VAB",
    "VIB",
    "BVB",
    "PGB",
    # Bất động sản
    "DIG",
    "DXG",
    "HDC",
    "HDG",
    "HQC",
    "LDG",
    "NLG",
    "PDN",
    "SZC",
    "BCG",
    "CEO",
    "DXS",
    "IJC",
    "KBC",
    "NBB",
    "NTL",
    "PPC",
    "SCR",
    # Công nghiệp
    "DGC",
    "DPM",
    "GEX",
    "GMD",
    "HNG",
    "HSG",
    "NT2",
    "PVD",
    "PVS",
    "PVT",
    "QCG",
    "TLG",
    "VCS",
    "VGC",
    "VHC",
    "VNE",
    "VSC",
    # Tiêu dùng
    "ANV",
    "ASM",
    "BBC",
    "BHN",
    "BMI",
    "CII",
    "DCM",
    "DGW",
    "DPR",
    "DRC",
    "FRT",
    "GIL",
    "HNG",
    "HT1",
    "KDC",
    "MCH",
    "MCP",
    "PAN",
    "SAM",
    "SBT",
    "TNA",
    "VHG",
    "VTO",
    # Thép, vật liệu
    "DTL",
    "HT1",
    "NKG",
    "POM",
    "SMC",
    "TLH",
    "VGS",
    "VIS",
    # Dầu khí
    "BSR",
    "OIL",
    "PVB",
    "PVC",
    "PVG",
    "PVP",
    "PVX",
    # Điện
    "PC1",
    "POW",
    "PPC",
    "REE",
    "SBA",
    "VSH",
    # Chứng khoán
    "AGR",
    "APS",
    "BSI",
    "CTS",
    "FTS",
    "HCM",
    "IVS",
    "MBS",
    "ORS",
    "SHS",
    "VCI",
    "VDS",
    "VIX",
    "VND",
]

# Top HNX stocks (chọn lọc)
TOP_HNX = [
    # Ngân hàng HNX
    "ACB",
    "SHB",
    "VCS",
    "PVB",
    # Large caps HNX
    "CEO",
    "HUT",
    "LAS",
    "NBC",
    "NTP",
    "PLC",
    "PVS",
    "PVI",
    "SHS",
    "TNG",
    "TS4",
    "VC3",
    "VCG",
    "VGC",
    # Mid caps HNX quality
    "AMV",
    "CEO",
    "DBC",
    "DTD",
    "HHC",
    "KLF",
    "L10",
    "MBS",
    "NDN",
    "NRC",
    "PVI",
    "PVL",
    "SHB",
    "SHS",
    "THD",
    "TIG",
    "VCS",
    "VGC",
    "VNR",
    "VRC",
]


def load_all_tickers(csv_path: Path) -> dict:
    """
    Load tất cả tickers từ List.csv

    Returns:
        Dict[ticker] = {"name": str, "exchange": str}
    """
    tickers = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    ticker = row[0].strip().upper()
                    name = row[1].strip()
                    exchange = row[2].strip()

                    tickers[ticker] = {"name": name, "exchange": exchange}

        logger.info(f"✅ Loaded {len(tickers)} tickers from {csv_path}")
        return tickers

    except Exception as e:
        logger.error(f"❌ Error loading {csv_path}: {e}")
        return {}


def filter_quality_tickers(all_tickers: dict) -> list:
    """
    Lọc quality tickers theo tiêu chí

    Returns:
        List of quality ticker symbols
    """
    quality = set()

    # 1. Add VN30 (priority highest)
    logger.info("\n📊 Adding VN30 components...")
    vn30_count = 0
    for ticker in VN30_TICKERS:
        if ticker in all_tickers:
            quality.add(ticker)
            vn30_count += 1
        else:
            logger.warning(f"   ⚠️ {ticker} not found in List.csv")

    logger.info(f"   ✅ Added {vn30_count} VN30 stocks")

    # 2. Add VN100 additional (large caps)
    logger.info("\n📈 Adding VN100 additional stocks...")
    vn100_count = 0
    for ticker in VN100_ADDITIONAL:
        if ticker in all_tickers:
            exchange = all_tickers[ticker]["exchange"]
            # Chỉ lấy HSX và HNX, bỏ Upcom/OTC
            if exchange in ["HSX", "HNX"]:
                quality.add(ticker)
                vn100_count += 1

    logger.info(f"   ✅ Added {vn100_count} VN100 stocks")

    # 3. Add top HNX
    logger.info("\n🏦 Adding top HNX stocks...")
    hnx_count = 0
    for ticker in TOP_HNX:
        if ticker in all_tickers:
            exchange = all_tickers[ticker]["exchange"]
            if exchange == "HNX":
                quality.add(ticker)
                hnx_count += 1

    logger.info(f"   ✅ Added {hnx_count} HNX stocks")

    # 4. Fill with more HSX stocks if needed (to reach ~200)
    target_count = 200
    current_count = len(quality)

    if current_count < target_count:
        logger.info(f"\n📊 Filling with HSX stocks to reach {target_count}...")
        hsx_tickers = [
            ticker
            for ticker, info in all_tickers.items()
            if info["exchange"] == "HSX" and ticker not in quality
        ]

        # Add until reach target
        needed = target_count - current_count
        for ticker in hsx_tickers[:needed]:
            quality.add(ticker)

        logger.info(f"   ✅ Added {min(needed, len(hsx_tickers))} more HSX stocks")

    return sorted(list(quality))


def save_quality_tickers(tickers: list, output_path: Path):
    """Save quality tickers to file"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")

        logger.info(f"\n💾 Saved {len(tickers)} tickers to {output_path}")

    except Exception as e:
        logger.error(f"❌ Error saving to {output_path}: {e}")


def print_summary(quality_tickers: list, all_tickers: dict):
    """Print summary statistics"""
    logger.info("\n" + "=" * 70)
    logger.info("📊 QUALITY TICKERS SUMMARY")
    logger.info("=" * 70)

    # Count by exchange
    exchanges = {}
    for ticker in quality_tickers:
        if ticker in all_tickers:
            exchange = all_tickers[ticker]["exchange"]
            exchanges[exchange] = exchanges.get(exchange, 0) + 1

    logger.info(f"\n✅ Total: {len(quality_tickers)} tickers")
    logger.info(f"\nBreakdown by exchange:")
    for exchange, count in sorted(exchanges.items(), key=lambda x: -x[1]):
        logger.info(f"   {exchange:10s}: {count:3d} tickers")

    # Show first 20
    logger.info(f"\n📝 First 20 tickers:")
    for i, ticker in enumerate(quality_tickers[:20], 1):
        if ticker in all_tickers:
            name = all_tickers[ticker]["name"][:40]
            exchange = all_tickers[ticker]["exchange"]
            logger.info(f"   {i:2d}. {ticker:6s} - {name:40s} [{exchange}]")

    if len(quality_tickers) > 20:
        logger.info(f"   ... and {len(quality_tickers) - 20} more")

    logger.info("\n" + "=" * 70)


def main():
    """Main function"""
    logger.info("\n🚀 QUALITY TICKER FILTER")
    logger.info("=" * 70)

    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "List.csv"
    output_path = project_root / "quality_tickers.txt"

    # Check if List.csv exists
    if not csv_path.exists():
        logger.error(f"❌ List.csv not found at {csv_path}")
        logger.error("   Please ensure List.csv is in project root directory")
        return 1

    # Load all tickers
    all_tickers = load_all_tickers(csv_path)

    if not all_tickers:
        logger.error("❌ No tickers loaded. Exiting.")
        return 1

    # Filter quality tickers
    quality_tickers = filter_quality_tickers(all_tickers)

    # Save to file
    save_quality_tickers(quality_tickers, output_path)

    # Print summary
    print_summary(quality_tickers, all_tickers)

    logger.info("\n✅ SUCCESS!")
    logger.info(f"\n📝 Next steps:")
    logger.info(f"   1. Review quality_tickers.txt")
    logger.info(
        f"   2. Train models: python scripts/train_models.py --ticker-file quality_tickers.txt"
    )
    logger.info(f"   3. Backtest: Use quality tickers for scanning")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
