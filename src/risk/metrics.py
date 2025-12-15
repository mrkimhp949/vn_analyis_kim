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
        recommendation = f"⚠️ Portfolio có correlation cao ({avg_correlation:.2f}). Nên đa dạng hóa."
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


class DynamicCorrelationMonitor:
    """
    Monitor portfolio correlation dynamically with regime detection and correlation breakdown alerts.

    Features:
    - Rolling correlation to detect correlation shifts
    - Regime-based correlation (crisis periods have higher correlation)
    - Correlation breakdown alerts when relationships change
    - Real-time risk scoring with Vietnam market adjustments
    """

    # Vietnam-specific: banking sector dominates VNINDEX, often shows high correlation
    SECTOR_CORRELATION_ADJUSTMENTS = {
        "BANKING": 0.85,  # Banks are highly correlated in VN market
        "SECURITIES": 0.80,  # Securities firms follow market closely
        "REAL_ESTATE": 0.75,  # RE sector shows sector-wide movements
        "TECHNOLOGY": 0.60,  # Tech shows lower correlation
        "FOOD_BEVERAGE": 0.55,  # Defensive, lower correlation
        "RETAIL": 0.65,
        "UTILITIES": 0.50,  # Most defensive
    }

    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        correlation_threshold: float = 0.7,
        breakdown_threshold: float = 0.3,
    ):
        """
        Initialize dynamic correlation monitor.

        Args:
            lookback_short: Short-term lookback for recent correlation (default 20 days)
            lookback_long: Long-term lookback for baseline correlation (default 60 days)
            correlation_threshold: Threshold for high correlation warning
            breakdown_threshold: Threshold for correlation breakdown alert (change in correlation)
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.correlation_threshold = correlation_threshold
        self.breakdown_threshold = breakdown_threshold
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes

    def calculate_rolling_correlation(
        self,
        returns_df: pd.DataFrame,
        window: int = 20,
    ) -> Dict[str, pd.Series]:
        """
        Calculate rolling correlation between all pairs.

        Returns:
            Dict mapping "SYMBOL1_SYMBOL2" to Series of rolling correlations
        """
        if returns_df.empty or returns_df.shape[1] < 2:
            return {}

        rolling_corr = {}
        symbols = returns_df.columns.tolist()

        for i, sym_i in enumerate(symbols):
            for sym_j in symbols[i + 1 :]:
                key = f"{sym_i}_{sym_j}"
                # Calculate rolling correlation
                rolling = returns_df[sym_i].rolling(window=window).corr(returns_df[sym_j])
                rolling_corr[key] = rolling.dropna()

        return rolling_corr

    def detect_correlation_breakdown(
        self,
        symbols: List[str],
    ) -> Dict:
        """
        Detect correlation breakdown (when correlation relationships change significantly).

        This is critical for Vietnam market where correlation can spike during foreign outflows.

        Returns:
            Dict with breakdown alerts and analysis
        """
        from src.data.loader import load_data

        # Load data for longer period
        returns_data = {}
        max_lookback = max(self.lookback_long, 90)

        for symbol in symbols:
            try:
                df = load_data(symbol, lookback=max_lookback, use_cache=True)
                if not df.empty and "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    if len(returns) >= self.lookback_long:
                        returns_data[symbol] = returns
            except Exception as e:
                logger.warning(f"Could not load data for {symbol}: {e}")
                continue

        if len(returns_data) < 2:
            return {
                "alerts": [],
                "analysis": "Insufficient data for correlation breakdown detection",
            }

        returns_df = pd.DataFrame(returns_data)

        # Calculate short-term and long-term correlation
        short_term_df = returns_df.tail(self.lookback_short)
        long_term_df = returns_df.tail(self.lookback_long)

        short_corr = short_term_df.corr()
        long_corr = long_term_df.corr()

        # Find correlation changes
        alerts = []
        pair_analysis = {}

        symbols_list = returns_df.columns.tolist()
        for i, sym_i in enumerate(symbols_list):
            for sym_j in symbols_list[i + 1 :]:
                short_val = short_corr.loc[sym_i, sym_j]
                long_val = long_corr.loc[sym_i, sym_j]

                if pd.isna(short_val) or pd.isna(long_val):
                    continue

                change = short_val - long_val
                pair_key = f"{sym_i}_{sym_j}"

                pair_analysis[pair_key] = {
                    "short_term": float(short_val),
                    "long_term": float(long_val),
                    "change": float(change),
                }

                # Check for significant breakdown
                if abs(change) >= self.breakdown_threshold:
                    direction = "increased" if change > 0 else "decreased"
                    severity = "high" if abs(change) >= 0.5 else "medium"

                    alerts.append(
                        {
                            "pair": (sym_i, sym_j),
                            "direction": direction,
                            "change": float(change),
                            "short_term_corr": float(short_val),
                            "long_term_corr": float(long_val),
                            "severity": severity,
                            "message": f"⚠️ {sym_i}-{sym_j} correlation {direction} by {abs(change):.2f} "
                            f"({long_val:.2f} → {short_val:.2f})",
                        }
                    )

        # Sort alerts by severity
        alerts.sort(key=lambda x: abs(x["change"]), reverse=True)

        return {
            "alerts": alerts,
            "pair_analysis": pair_analysis,
            "has_breakdown": len(alerts) > 0,
            "num_breakdowns": len(alerts),
        }

    def calculate_regime_adjusted_correlation(
        self,
        returns_df: pd.DataFrame,
        market_regime: str = "normal",
    ) -> pd.DataFrame:
        """
        Calculate correlation adjusted for market regime.

        In crisis/high-volatility regimes, correlations tend to increase.
        In normal/low-volatility regimes, correlations show true diversification.

        Args:
            returns_df: DataFrame of returns
            market_regime: One of "normal", "high_volatility", "crisis"

        Returns:
            Adjusted correlation matrix
        """
        if returns_df.empty:
            return pd.DataFrame()

        base_corr = returns_df.corr()

        # Regime adjustment factors (correlations increase in crisis)
        regime_factors = {
            "normal": 1.0,
            "high_volatility": 1.15,  # Correlation tends to be 15% higher
            "crisis": 1.30,  # Correlation spikes 30% in crisis
            "low_volatility": 0.90,  # Correlation lower in calm markets
        }

        factor = regime_factors.get(market_regime, 1.0)

        if factor == 1.0:
            return base_corr

        # Adjust correlation (but cap at 1.0 and floor at -1.0)
        adjusted = base_corr * factor
        adjusted = adjusted.clip(lower=-1.0, upper=1.0)

        # Keep diagonal as 1.0
        np.fill_diagonal(adjusted.values, 1.0)

        return adjusted

    def get_dynamic_portfolio_risk(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None,
        market_regime: str = "normal",
    ) -> Dict:
        """
        Calculate dynamic portfolio risk considering regime and correlation shifts.

        Args:
            symbols: List of portfolio symbols
            weights: Optional weight per symbol (default: equal weight)
            market_regime: Current market regime

        Returns:
            Dict with comprehensive risk metrics
        """
        import time

        # Check cache
        cache_key = f"{','.join(sorted(symbols))}_{market_regime}"
        current_time = time.time()

        if (
            self._cache_time is not None
            and current_time - self._cache_time < self._cache_ttl
            and cache_key in self._cache
        ):
            return self._cache[cache_key]

        if len(symbols) < 2:
            return {
                "portfolio_variance": 0.0,
                "portfolio_volatility": 0.0,
                "diversification_ratio": 1.0,
                "effective_positions": len(symbols),
                "risk_score": 0,
                "alerts": [],
                "recommendation": "Need at least 2 positions for correlation analysis",
            }

        # Load returns
        returns_df = load_returns_dataframe(symbols, self.lookback_long)
        if returns_df.empty or returns_df.shape[1] < 2:
            return {
                "portfolio_variance": 0.0,
                "portfolio_volatility": 0.0,
                "diversification_ratio": 1.0,
                "effective_positions": len(symbols),
                "risk_score": 50,
                "alerts": ["Insufficient data for correlation analysis"],
                "recommendation": "Cannot calculate portfolio risk without historical data",
            }

        # Use only symbols with available data
        available_symbols = returns_df.columns.tolist()

        # Set equal weights if not provided
        if weights is None:
            weights = {s: 1.0 / len(available_symbols) for s in available_symbols}

        # Normalize weights for available symbols
        total_weight = sum(weights.get(s, 0) for s in available_symbols)
        if total_weight == 0:
            total_weight = 1.0
        w = np.array(
            [weights.get(s, 1.0 / len(available_symbols)) / total_weight for s in available_symbols]
        )

        # Calculate regime-adjusted correlation
        corr_matrix = self.calculate_regime_adjusted_correlation(returns_df, market_regime)

        # Calculate individual volatilities
        individual_vols = returns_df.std().values

        # Calculate portfolio variance using correlation matrix
        # σ²_p = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
        cov_matrix = np.outer(individual_vols, individual_vols) * corr_matrix.values
        portfolio_variance = float(w @ cov_matrix @ w)
        portfolio_volatility = float(np.sqrt(portfolio_variance))

        # Weighted average volatility (for diversification ratio)
        weighted_avg_vol = float(np.sum(w * individual_vols))

        # Diversification ratio = weighted avg vol / portfolio vol
        # Higher is better (more diversification benefit)
        diversification_ratio = (
            weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
        )

        # Effective number of positions (1 / Σwᵢ²) - Herfindahl adjustment
        effective_positions = 1.0 / np.sum(w**2) if np.sum(w**2) > 0 else len(symbols)

        # Detect correlation breakdowns
        breakdown_info = self.detect_correlation_breakdown(available_symbols)

        # Calculate risk score (0-100, higher = more risky)
        # Based on: low diversification + high correlation + breakdowns
        base_corr = returns_df.corr()
        mask = np.triu(np.ones_like(base_corr, dtype=bool), k=1)
        avg_corr = base_corr.where(mask).stack().abs().mean()

        risk_factors = [
            min(40, int(avg_corr * 40)),  # Correlation contribution (max 40)
            min(30, int((2.0 - diversification_ratio) * 20)),  # Diversification (max 30)
            min(30, breakdown_info["num_breakdowns"] * 10),  # Breakdown alerts (max 30)
        ]
        risk_score = min(100, sum(risk_factors))

        # Collect alerts
        alerts = []
        if avg_corr > 0.6:
            alerts.append(f"⚠️ High average correlation: {avg_corr:.2f}")
        if diversification_ratio < 1.2:
            alerts.append(f"⚠️ Low diversification benefit: {diversification_ratio:.2f}")
        alerts.extend(
            [a["message"] for a in breakdown_info["alerts"][:3]]
        )  # Top 3 breakdown alerts

        # Recommendation
        if risk_score >= 70:
            recommendation = "🔴 High portfolio risk - consider reducing correlated positions"
        elif risk_score >= 40:
            recommendation = "🟡 Moderate portfolio risk - monitor correlation changes"
        else:
            recommendation = "🟢 Portfolio well-diversified"

        result = {
            "portfolio_variance": float(portfolio_variance),
            "portfolio_volatility": float(portfolio_volatility),
            "annualized_volatility": float(portfolio_volatility * np.sqrt(252)),
            "diversification_ratio": float(diversification_ratio),
            "effective_positions": float(effective_positions),
            "average_correlation": float(avg_corr),
            "risk_score": risk_score,
            "alerts": alerts,
            "breakdown_analysis": breakdown_info,
            "regime": market_regime,
            "recommendation": recommendation,
        }

        # Cache result
        self._cache[cache_key] = result
        self._cache_time = current_time

        return result

    def get_correlation_heatmap_data(
        self,
        symbols: List[str],
        include_rolling: bool = True,
    ) -> Dict:
        """
        Get data for correlation heatmap visualization.

        Returns:
            Dict with correlation matrices and metadata for visualization
        """
        returns_df = load_returns_dataframe(symbols, self.lookback_long)
        if returns_df.empty:
            return {"error": "No data available"}

        available = returns_df.columns.tolist()

        result = {
            "symbols": available,
            "pearson_correlation": returns_df.corr().to_dict(),
            "distance_correlation": calculate_distance_correlation_matrix(returns_df).to_dict(),
            "copula_correlation": calculate_copula_correlation_matrix(returns_df).to_dict(),
        }

        if include_rolling:
            rolling_corr = self.calculate_rolling_correlation(
                returns_df, window=self.lookback_short
            )
            # Get latest values
            latest_rolling = {
                k: float(v.iloc[-1]) if len(v) > 0 else None for k, v in rolling_corr.items()
            }
            result["rolling_correlation"] = latest_rolling

            # Get trend (increasing/decreasing)
            trends = {}
            for k, v in rolling_corr.items():
                if len(v) >= 10:
                    recent = v.tail(10).mean()
                    earlier = v.head(10).mean()
                    trends[k] = (
                        "increasing"
                        if recent > earlier + 0.05
                        else ("decreasing" if recent < earlier - 0.05 else "stable")
                    )
            result["correlation_trends"] = trends

        return result


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
