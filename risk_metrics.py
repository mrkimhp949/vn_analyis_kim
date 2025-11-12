from collections import defaultdict
from typing import Dict, List

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


def summarize_exposure(exposure: Dict[str, float], top_n: int = 5) -> List[str]:
    sorted_items = sorted(exposure.items(), key=lambda x: x[1], reverse=True)
    return [f"{sector}: {pct:.1f}%" for sector, pct in sorted_items[:top_n]]

