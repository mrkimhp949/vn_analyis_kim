from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from config import KIM_SECTOR, THUY_SECTOR
except ImportError:
    KIM_SECTOR = {}
    THUY_SECTOR = {}


def _build_sector_map() -> Dict[str, str]:
    sector_map = {}
    for sector_group, codes in KIM_SECTOR.items():
        for code in codes:
            sector_map[code.upper()] = f"KIM::{sector_group.upper()}"
    for sector_group, codes in THUY_SECTOR.items():
        for code in codes:
            sector_map[code.upper()] = f"THUY::{sector_group.upper()}"
    return sector_map


SECTOR_MAP = _build_sector_map()


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


def check_sector_overweight(exposure: Dict[str, float], max_sector_pct: float = 40.0) -> List[Tuple[str, float]]:
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


def calculate_correlation_matrix(symbols: List[str], lookback: int = 60) -> pd.DataFrame:
    """
    Tính correlation matrix giữa các symbols
    
    Args:
        symbols: List các mã cổ phiếu
        lookback: Số ngày để tính correlation
        
    Returns:
        DataFrame với correlation matrix
    """
    from data_loader import load_data
    
    returns_data = {}
    
    for symbol in symbols:
        try:
            df = load_data(symbol, lookback=lookback, use_cache=True)
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                if len(returns) >= 20:  # Cần ít nhất 20 điểm dữ liệu
                    returns_data[symbol] = returns
        except Exception:
            continue
    
    if len(returns_data) < 2:
        # Trả về empty matrix nếu không đủ data
        return pd.DataFrame()
    
    # Tạo DataFrame từ returns
    returns_df = pd.DataFrame(returns_data)
    
    # Tính correlation
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def check_high_correlation(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.7,
    current_holdings: Optional[List[str]] = None
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
        for symbol2 in symbols_to_check[i+1:]:
            if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                corr = correlation_matrix.loc[symbol1, symbol2]
                if not np.isnan(corr) and abs(corr) >= threshold:
                    high_corr_pairs.append((symbol1, symbol2, corr))
    
    return sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)


def get_sector_for_symbol(symbol: str) -> str:
    """Lấy sector của một symbol"""
    return SECTOR_MAP.get(symbol.upper(), "UNCLASSIFIED")


def calculate_portfolio_correlation_risk(
    current_holdings: List[str],
    lookback: int = 60,
    max_avg_correlation: float = 0.5
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
            "risk_score": 0,
            "recommendation": "Cần thêm mã để đánh giá correlation"
        }
    
    correlation_matrix = calculate_correlation_matrix(current_holdings, lookback)
    
    if correlation_matrix.empty:
        return {
            "avg_correlation": 0.0,
            "high_corr_pairs": [],
            "risk_score": 50,
            "recommendation": "Không đủ dữ liệu để tính correlation"
        }
    
    # Tính correlation trung bình (chỉ lấy upper triangle, loại bỏ diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlations = correlation_matrix.where(mask).stack()
    avg_correlation = correlations.abs().mean()
    
    # Tìm các cặp có correlation cao
    high_corr_pairs = check_high_correlation(correlation_matrix, threshold=0.7, current_holdings=current_holdings)
    
    # Tính risk score (0-100, càng cao càng rủi ro)
    risk_score = min(100, int(avg_correlation * 100))
    
    # Recommendation
    if avg_correlation > max_avg_correlation:
        recommendation = f"⚠️ Portfolio có correlation cao ({avg_correlation:.2f}). Nên đa dạng hóa."
    elif len(high_corr_pairs) > 0:
        recommendation = f"⚠️ Có {len(high_corr_pairs)} cặp mã có correlation cao. Nên xem xét."
    else:
        recommendation = "✅ Portfolio đa dạng hóa tốt."
    
    return {
        "avg_correlation": float(avg_correlation),
        "high_corr_pairs": high_corr_pairs,
        "risk_score": risk_score,
        "recommendation": recommendation
    }


def summarize_exposure(exposure: Dict[str, float], top_n: int = 5) -> List[str]:
    sorted_items = sorted(exposure.items(), key=lambda x: x[1], reverse=True)
    return [f"{sector}: {pct:.1f}%" for sector, pct in sorted_items[:top_n]]


def get_diversification_recommendation(
    current_holdings: Dict[str, Dict],
    max_sector_pct: float = 40.0,
    min_sectors: int = 3
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
            recommendations.append(f"Nên xem xét giảm một trong hai mã {top_pair[0]} hoặc {top_pair[1]} (correlation: {top_pair[2]:.2f})")
    
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
        "diversification_score": max(0, 100 - len(warnings) * 20)  # Score 0-100
    }

