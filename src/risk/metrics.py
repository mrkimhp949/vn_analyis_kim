import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Sector mapping for Vietnamese stocks
SECTOR_MAP = {
    # Banking
    "VCB": "BANKING",
    "TCB": "BANKING",
    "CTG": "BANKING",
    "BID": "BANKING",
    "MBB": "BANKING",
    "VPB": "BANKING",
    "ACB": "BANKING",
    "STB": "BANKING",
    "HDB": "BANKING",
    "TPB": "BANKING",
    "SHB": "BANKING",
    "VIB": "BANKING",
    "LPB": "BANKING",
    "EIB": "BANKING",
    "MSB": "BANKING",
    "OCB": "BANKING",
    "BAB": "BANKING",
    "BVB": "BANKING",
    "NVB": "BANKING",
    "PGB": "BANKING",
    "VAB": "BANKING",
    "ABB": "BANKING",
    "KLB": "BANKING",
    "NAB": "BANKING",
    "SGB": "BANKING",
    "VBB": "BANKING",
    # Securities
    "SSI": "SECURITIES",
    "VND": "SECURITIES",
    "HCM": "SECURITIES",
    "VCI": "SECURITIES",
    "FTS": "SECURITIES",
    "MBS": "SECURITIES",
    "VIX": "SECURITIES",
    "AGR": "SECURITIES",
    "BSI": "SECURITIES",
    "CTS": "SECURITIES",
    "SHS": "SECURITIES",
    "ORS": "SECURITIES",
    # Real Estate
    "VHM": "REAL_ESTATE",
    "VIC": "REAL_ESTATE",
    "NVL": "REAL_ESTATE",
    "VRE": "REAL_ESTATE",
    "DXG": "REAL_ESTATE",
    "PDR": "REAL_ESTATE",
    "KDH": "REAL_ESTATE",
    "DIG": "REAL_ESTATE",
    "HDG": "REAL_ESTATE",
    "NLG": "REAL_ESTATE",
    "KBC": "REAL_ESTATE",
    "CEO": "REAL_ESTATE",
    "HDC": "REAL_ESTATE",
    "SCR": "REAL_ESTATE",
    "IDC": "REAL_ESTATE",
    "LDG": "REAL_ESTATE",
    "TCH": "REAL_ESTATE",
    "DXS": "REAL_ESTATE",
    "CII": "REAL_ESTATE",
    "IJC": "REAL_ESTATE",
    # Technology
    "FPT": "TECHNOLOGY",
    "CMG": "TECHNOLOGY",
    "VGI": "TECHNOLOGY",
    "SAM": "TECHNOLOGY",
    "ELC": "TECHNOLOGY",
    "ITD": "TECHNOLOGY",
    "CMT": "TECHNOLOGY",
    "SGT": "TECHNOLOGY",
    # Retail
    "MWG": "RETAIL",
    "PNJ": "RETAIL",
    "FRT": "RETAIL",
    "DGW": "RETAIL",
    "VGC": "RETAIL",
    "SCS": "RETAIL",
    "PET": "RETAIL",
    "HAX": "RETAIL",
    # Food & Beverage
    "VNM": "FOOD_BEVERAGE",
    "MSN": "FOOD_BEVERAGE",
    "SAB": "FOOD_BEVERAGE",
    "VHC": "FOOD_BEVERAGE",
    "MCH": "FOOD_BEVERAGE",
    "KDC": "FOOD_BEVERAGE",
    "QNS": "FOOD_BEVERAGE",
    # "SBT": "FOOD_BEVERAGE",  # Duplicate - see below
    "VCF": "FOOD_BEVERAGE",
    # "BAF": "FOOD_BEVERAGE",  # Duplicate - see below
    "ANV": "FOOD_BEVERAGE",
    "MML": "FOOD_BEVERAGE",
    # Oil & Gas
    "GAS": "OIL_GAS",
    "PLX": "OIL_GAS",
    "PVS": "OIL_GAS",
    "PVD": "OIL_GAS",
    # "PVT": "OIL_GAS",  # Duplicate - see below
    "PVC": "OIL_GAS",
    "PVG": "OIL_GAS",
    "BSR": "OIL_GAS",
    # "POW": "OIL_GAS",  # Duplicate - keep UTILITIES
    "PVB": "OIL_GAS",
    # Steel & Materials
    "HPG": "STEEL_MATERIALS",
    "HSG": "STEEL_MATERIALS",
    "NKG": "STEEL_MATERIALS",
    "TLH": "STEEL_MATERIALS",
    "VGS": "STEEL_MATERIALS",
    "POM": "STEEL_MATERIALS",
    "DTL": "STEEL_MATERIALS",
    "VIS": "STEEL_MATERIALS",
    "SMC": "STEEL_MATERIALS",
    # "TNG": "STEEL_MATERIALS",  # Duplicate - see below
    "VCS": "STEEL_MATERIALS",
    # Construction
    "CTD": "CONSTRUCTION",
    "HBC": "CONSTRUCTION",
    # "PC1": "CONSTRUCTION",  # Duplicate - see below
    "LCG": "CONSTRUCTION",
    "HT1": "CONSTRUCTION",
    "VCG": "CONSTRUCTION",
    "FCN": "CONSTRUCTION",
    "C32": "CONSTRUCTION",
    "HU1": "CONSTRUCTION",
    "CTI": "CONSTRUCTION",
    "VC3": "CONSTRUCTION",
    # Utilities
    "POW": "UTILITIES",
    "NT2": "UTILITIES",
    "GEG": "UTILITIES",
    "REE": "UTILITIES",
    # "PC1": "UTILITIES",  # Duplicate - final assignment is CONSTRUCTION (line 189)
    "VSH": "UTILITIES",
    "BWE": "UTILITIES",
    # Healthcare
    "DHG": "HEALTHCARE",
    "DMC": "HEALTHCARE",
    "IMP": "HEALTHCARE",
    "DCL": "HEALTHCARE",
    "DBD": "HEALTHCARE",
    "TNH": "HEALTHCARE",
    "PME": "HEALTHCARE",
    "DP3": "HEALTHCARE",
    # Transportation
    "HVN": "TRANSPORTATION",
    "VJC": "TRANSPORTATION",
    "GMD": "TRANSPORTATION",
    "HAH": "TRANSPORTATION",
    "VOS": "TRANSPORTATION",
    "VSC": "TRANSPORTATION",
    "PVT": "TRANSPORTATION",  # Final assignment for PVT
    # Agriculture
    "HAG": "AGRICULTURE",
    "HNG": "AGRICULTURE",
    "SBT": "AGRICULTURE",  # Final assignment for SBT
    "BAF": "AGRICULTURE",  # Final assignment for BAF
    "NSC": "AGRICULTURE",
    "LSS": "AGRICULTURE",
    "HVG": "AGRICULTURE",
    # Textile
    "MSH": "TEXTILE",
    "TNG": "TEXTILE",  # Final assignment for TNG
    "STK": "TEXTILE",
    "VGT": "TEXTILE",
    "TCM": "TEXTILE",
    "GIL": "TEXTILE",
    # Chemicals
    "DGC": "CHEMICALS",
    "DPM": "CHEMICALS",
    "DCM": "CHEMICALS",
    "BFC": "CHEMICALS",
    "CSV": "CHEMICALS",
    "LAS": "CHEMICALS",
    # Construction (additional)
    "PC1": "CONSTRUCTION",  # Final assignment for PC1
    # Insurance
    "BVH": "INSURANCE",
    "BMI": "INSURANCE",
    "PVI": "INSURANCE",
    "PTI": "INSURANCE",
    "MIG": "INSURANCE",
    "VNR": "INSURANCE",
}


def calculate_sector_exposure(current_holdings: Dict[str, Dict]) -> Dict[str, float]:
    """Tính exposure theo sector"""
    exposure = defaultdict(float)
    total_value = 0.0

    for symbol, data in current_holdings.items():
        value = data.get("current_value") or (data.get("shares", 0) * data.get("current_price", 0))
        total_value += value
        sector = SECTOR_MAP.get(symbol.upper(), "UNCLASSIFIED")
        exposure[sector] += value

    if total_value == 0:
        return {sector: 0.0 for sector in exposure}

    return {sector: (value / total_value) * 100 for sector, value in exposure.items()}


def check_sector_overweight(
    exposure: Dict[str, float], max_sector_pct: float = 40.0
) -> List[Tuple[str, float]]:
    """
    Kiểm tra xem có sector nào bị overweight không

    Returns:
        List of (sector, percentage) tuples that exceed limit
    """
    overweight = []
    for sector, pct in exposure.items():
        if pct > max_sector_pct:
            overweight.append((sector, pct))
    return sorted(overweight, key=lambda x: x[1], reverse=True)


def load_returns_dataframe(symbols: List[str], lookback: int = 60) -> pd.DataFrame:
    """Load daily returns for given symbols."""
    from src.data.loader import load_data

    returns_data = {}

    for symbol in symbols:
        try:
            df = load_data(symbol, lookback=lookback, use_cache=True)
            if not df.empty and "close" in df.columns:
                returns = df["close"].pct_change().dropna()
                if len(returns) >= 20:
                    returns_data[symbol] = returns
        except Exception:
            logger.warning(f"Could not load returns data for {symbol}")
            continue

    if len(returns_data) < 2:
        return pd.DataFrame()

    return pd.DataFrame(returns_data)


def calculate_correlation_matrix(symbols: List[str], lookback: int = 60) -> pd.DataFrame:
    """
    Tính correlation matrix giữa các symbols

    Args:
        symbols: List các mã cổ phiếu
        lookback: Số ngày để tính correlation

    Returns:
        DataFrame với correlation matrix
    """
    returns_df = load_returns_dataframe(symbols, lookback)
    if returns_df.empty:
        return pd.DataFrame()
    return returns_df.corr()


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1:
        x = x.ravel()
    if y.ndim != 1:
        y = y.ravel()

    n = len(x)
    if n != len(y) or n == 0:
        return 0.0

    x = x.reshape(n, 1)
    y = y.reshape(n, 1)

    a = np.abs(x - x.T)
    b = np.abs(y - y.T)

    A = a - a.mean(axis=0) - a.mean(axis=1).reshape(-1, 1) + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1).reshape(-1, 1) + b.mean()

    dcov_xy = np.mean(A * B)
    dcov_xx = np.mean(A * A)
    dcov_yy = np.mean(B * B)

    if dcov_xx <= 0 or dcov_yy <= 0:
        return 0.0

    return math.sqrt(max(dcov_xy, 0.0) / math.sqrt(dcov_xx * dcov_yy))


def calculate_distance_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df.empty or returns_df.shape[1] < 2:
        return pd.DataFrame()

    symbols = returns_df.columns.tolist()
    matrix = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

    for i, sym_i in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            sym_j = symbols[j]
            dcor = _distance_correlation(returns_df[sym_i].values, returns_df[sym_j].values)
            matrix.loc[sym_i, sym_j] = dcor
            matrix.loc[sym_j, sym_i] = dcor

    return matrix


def calculate_copula_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df.empty or returns_df.shape[1] < 2:
        return pd.DataFrame()

    ranks = returns_df.rank(pct=True)
    matrix = ranks.corr()
    return matrix


def check_high_correlation(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.7,
    current_holdings: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Kiểm tra các cặp symbols có correlation cao

    Args:
        correlation_matrix: Correlation matrix từ calculate_correlation_matrix
        threshold: Ngưỡng correlation (default 0.7)
        current_holdings: List symbols đang nắm giữ (nếu None thì check tất cả)

    Returns:
        List of (symbol1, symbol2, correlation) tuples
    """
    if correlation_matrix.empty:
        return []

    high_corr_pairs = []

    # Nếu có current_holdings, chỉ check các cặp trong holdings
    if current_holdings:
        symbols_to_check = [s for s in current_holdings if s in correlation_matrix.columns]
    else:
        symbols_to_check = correlation_matrix.columns.tolist()

    for i, symbol1 in enumerate(symbols_to_check):
        for symbol2 in symbols_to_check[i + 1 :]:
            if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                corr = correlation_matrix.loc[symbol1, symbol2]
                if not np.isnan(corr) and abs(corr) >= threshold:
                    high_corr_pairs.append((symbol1, symbol2, corr))

    return sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)


def check_high_distance_correlation(
    distance_matrix: pd.DataFrame,
    threshold: float = 0.6,
    current_holdings: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    if distance_matrix.empty:
        return []

    symbols = current_holdings or distance_matrix.columns.tolist()
    symbols = [s for s in symbols if s in distance_matrix.columns]

    pairs = []
    for i, sym_i in enumerate(symbols):
        for sym_j in symbols[i + 1 :]:
            if sym_j in distance_matrix.columns:
                value = distance_matrix.loc[sym_i, sym_j]
                if not np.isnan(value) and value >= threshold:
                    pairs.append((sym_i, sym_j, value))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def check_high_copula_correlation(
    copula_matrix: pd.DataFrame,
    threshold: float = 0.7,
    current_holdings: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    if copula_matrix.empty:
        return []

    symbols = current_holdings or copula_matrix.columns.tolist()
    symbols = [s for s in symbols if s in copula_matrix.columns]

    pairs = []
    for i, sym_i in enumerate(symbols):
        for sym_j in symbols[i + 1 :]:
            if sym_j in copula_matrix.columns:
                value = copula_matrix.loc[sym_i, sym_j]
                if not np.isnan(value) and abs(value) >= threshold:
                    pairs.append((sym_i, sym_j, value))
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


def get_sector_for_symbol(symbol: str) -> str:
    """Lấy sector của một symbol"""
    return SECTOR_MAP.get(symbol.upper(), "UNCLASSIFIED")


def calculate_portfolio_correlation_risk(
    current_holdings: List[str], lookback: int = 60, max_avg_correlation: float = 0.5
) -> Dict:
    """
    Tính correlation risk của portfolio

    Returns:
        Dict với:
        - avg_correlation: Correlation trung bình
        - high_corr_pairs: Các cặp có correlation cao
        - risk_score: Điểm rủi ro (0-100)
    """
    if len(current_holdings) < 2:
        return {
            "avg_correlation": 0.0,
            "high_corr_pairs": [],
            "distance_correlation_avg": 0.0,
            "high_distance_pairs": [],
            "copula_correlation_avg": 0.0,
            "high_copula_pairs": [],
            "risk_score": 0,
            "recommendation": "Cần thêm mã để đánh giá correlation",
        }

    returns_df = load_returns_dataframe(current_holdings, lookback)
    if returns_df.empty or returns_df.shape[1] < 2:
        return {
            "avg_correlation": 0.0,
            "high_corr_pairs": [],
            "distance_correlation_avg": 0.0,
            "high_distance_pairs": [],
            "copula_correlation_avg": 0.0,
            "high_copula_pairs": [],
            "risk_score": 50,
            "recommendation": "Không đủ dữ liệu để tính correlation",
        }

    correlation_matrix = returns_df.corr()
    distance_matrix = calculate_distance_correlation_matrix(returns_df)
    copula_matrix = calculate_copula_correlation_matrix(returns_df)

    # Tính correlation trung bình (chỉ lấy upper triangle, loại bỏ diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlations = correlation_matrix.where(mask).stack()
    avg_correlation = correlations.abs().mean() if not correlations.empty else 0.0

    distance_avg = 0.0
    copula_avg = 0.0
    if not distance_matrix.empty:
        mask_dist = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
        distance_values = distance_matrix.where(mask_dist).stack()
        distance_avg = distance_values.mean() if not distance_values.empty else 0.0
    if not copula_matrix.empty:
        mask_cop = np.triu(np.ones_like(copula_matrix, dtype=bool), k=1)
        copula_vals = copula_matrix.where(mask_cop).stack().abs()
        copula_avg = copula_vals.mean() if not copula_vals.empty else 0.0

    # Tìm các cặp có correlation cao
    high_corr_pairs = check_high_correlation(
        correlation_matrix, threshold=0.7, current_holdings=current_holdings
    )
    high_distance_pairs = check_high_distance_correlation(
        distance_matrix, threshold=0.6, current_holdings=current_holdings
    )
    high_copula_pairs = check_high_copula_correlation(
        copula_matrix, threshold=0.7, current_holdings=current_holdings
    )

    # Tính risk score (0-100, càng cao càng rủi ro)
    risk_score = min(100, int(avg_correlation * 100))

    # Recommendation
    if avg_correlation > max_avg_correlation:
        recommendation = (
            f"⚠️ Portfolio có correlation cao ({avg_correlation:.2f}). Nên đa dạng hóa."
        )
    elif len(high_corr_pairs) > 0:
        recommendation = f"⚠️ Có {len(high_corr_pairs)} cặp mã có correlation cao. Nên xem xét."
    else:
        recommendation = "✅ Portfolio đa dạng hóa tốt."

    return {
        "avg_correlation": float(avg_correlation),
        "high_corr_pairs": high_corr_pairs,
        "distance_correlation_avg": float(distance_avg),
        "high_distance_pairs": high_distance_pairs,
        "copula_correlation_avg": float(copula_avg),
        "high_copula_pairs": high_copula_pairs,
        "risk_score": risk_score,
        "recommendation": recommendation,
    }


def summarize_exposure(exposure: Dict[str, float], top_n: int = 5) -> List[str]:
    sorted_items = sorted(exposure.items(), key=lambda x: x[1], reverse=True)
    return [f"{sector}: {pct:.1f}%" for sector, pct in sorted_items[:top_n]]


def get_diversification_recommendation(
    current_holdings: Dict[str, Dict],
    max_sector_pct: float = 40.0,
    min_sectors: int = 3,
) -> Dict:
    """
    Đưa ra khuyến nghị đa dạng hóa portfolio

    Returns:
        Dict với recommendations và warnings
    """
    exposure = calculate_sector_exposure(current_holdings)
    overweight = check_sector_overweight(exposure, max_sector_pct)

    symbols = list(current_holdings.keys())
    correlation_risk = calculate_portfolio_correlation_risk(symbols)

    warnings = []
    recommendations = []

    # Sector exposure warnings
    if overweight:
        for sector, pct in overweight:
            warnings.append(f"⚠️ {sector} chiếm {pct:.1f}% portfolio (vượt {max_sector_pct}%)")
            recommendations.append(f"Nên giảm exposure {sector} hoặc thêm mã từ ngành khác")

    # Correlation warnings
    if correlation_risk["risk_score"] > 50:
        warnings.append(correlation_risk["recommendation"])
        if correlation_risk["high_corr_pairs"]:
            top_pair = correlation_risk["high_corr_pairs"][0]
            recommendations.append(
                f"Nên xem xét giảm một trong hai mã {top_pair[0]} hoặc {top_pair[1]} (correlation: {top_pair[2]:.2f})"
            )

    # Sector count
    num_sectors = len(exposure)
    if num_sectors < min_sectors:
        warnings.append(f"⚠️ Chỉ có {num_sectors} ngành (nên có ít nhất {min_sectors})")
        recommendations.append("Nên thêm mã từ các ngành khác để đa dạng hóa")

    return {
        "exposure": exposure,
        "overweight_sectors": overweight,
        "correlation_risk": correlation_risk,
        "warnings": warnings,
        "recommendations": recommendations,
        "diversification_score": max(0, 100 - len(warnings) * 20),  # Score 0-100
    }
