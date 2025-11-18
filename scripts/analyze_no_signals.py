# -*- coding: utf-8 -*-
"""
Analyze No Signals
Phân tích tại sao không có tín hiệu mua
"""
import logging
from collections import defaultdict
from typing import Dict, List


logger = logging.getLogger(__name__)


class NoSignalAnalyzer:
    """
    Phân tích tại sao không có signals

    Giúp hiểu:
    - Bao nhiêu % mã bị loại ở mỗi filter?
    - Filter nào strict nhất?
    - Confidence distribution như thế nào?
    - Nên adjust threshold nào?
    """

    def __init__(self):
        self.filter_stats = defaultdict(int)
        self.confidence_distribution = []
        self.total_scanned = 0
        self.ml_signals = []

    def analyze_scan_results(
        self, symbols: List[str], ml_generator, entry_logic, market_regime: Dict
    ) -> Dict:
        """
        Phân tích kết quả scan để hiểu tại sao không có signals
        """
        from src.data.loader import load_data

        from src.config.legacy_config import LOOKBACK

        print(f"\n🔍 Analyzing {len(symbols)} symbols...")

        results = {
            "total_scanned": 0,
            "data_issues": 0,
            "ml_analysis": {
                "total": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "avg_confidence": 0,
                "confidence_distribution": {},
            },
            "entry_filters": {
                "passed_all": 0,
                "failed_confidence": 0,
                "failed_trend": 0,
                "failed_support": 0,
                "failed_volume": 0,
                "failed_volatility": 0,
                "failed_rsi": 0,
                "failed_other": 0,
            },
            "confidence_ranges": {
                "0-20": 0,
                "20-40": 0,
                "40-60": 0,
                "60-80": 0,
                "80-100": 0,
            },
            "recommendations": [],
        }

        # Preload VNINDEX for ML features
        try:
            index_df = load_data("VNINDEX", lookback=LOOKBACK, is_index=True)
        except Exception:
            index_df = None

        for symbol in symbols[:100]:  # Analyze first 100 for speed
            try:
                # Load data
                df = load_data(symbol, lookback=LOOKBACK)
                if df.empty or len(df) < 50:
                    results["data_issues"] += 1
                    continue

                results["total_scanned"] += 1

                # ML analysis
                ml_signal = ml_generator.analyze(df, index_df=index_df)
                results["ml_analysis"]["total"] += 1

                if ml_signal["signal"] == "BUY":
                    results["ml_analysis"]["buy_signals"] += 1
                elif ml_signal["signal"] == "SELL":
                    results["ml_analysis"]["sell_signals"] += 1
                else:
                    results["ml_analysis"]["hold_signals"] += 1

                # Confidence distribution
                conf = ml_signal["confidence"]
                if conf < 20:
                    results["confidence_ranges"]["0-20"] += 1
                elif conf < 40:
                    results["confidence_ranges"]["20-40"] += 1
                elif conf < 60:
                    results["confidence_ranges"]["40-60"] += 1
                elif conf < 80:
                    results["confidence_ranges"]["60-80"] += 1
                else:
                    results["confidence_ranges"]["80-100"] += 1

                # Entry logic analysis
                entry_signal = entry_logic.analyze_entry(
                    df=df, ml_signal=ml_signal, market_regime=market_regime, symbol=symbol
                )

                if entry_signal.should_enter:
                    results["entry_filters"]["passed_all"] += 1
                else:
                    # Analyze why failed
                    if entry_signal.confidence < entry_logic.min_confidence:
                        results["entry_filters"]["failed_confidence"] += 1

                    # Check warnings for specific filters
                    warnings_str = " ".join(entry_signal.warnings)
                    if "trend" in warnings_str.lower():
                        results["entry_filters"]["failed_trend"] += 1
                    if "support" in warnings_str.lower() or "resistance" in warnings_str.lower():
                        results["entry_filters"]["failed_support"] += 1
                    if "volume" in warnings_str.lower():
                        results["entry_filters"]["failed_volume"] += 1
                    if "volatility" in warnings_str.lower():
                        results["entry_filters"]["failed_volatility"] += 1
                    if "rsi" in warnings_str.lower():
                        results["entry_filters"]["failed_rsi"] += 1
                    if not entry_signal.warnings:
                        results["entry_filters"]["failed_other"] += 1

            except Exception:
                logger.error(f"Error analyzing {symbol}")
                results["data_issues"] += 1

        # Calculate averages
        if results["ml_analysis"]["total"] > 0:
            results["ml_analysis"]["buy_rate"] = (
                results["ml_analysis"]["buy_signals"] / results["ml_analysis"]["total"] * 100
            )

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results, entry_logic)

        return results

    def _generate_recommendations(self, results: Dict, entry_logic) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        total = results["total_scanned"]
        if total == 0:
            return ["Không có dữ liệu để phân tích"]

        # Check ML signals
        buy_rate = results["ml_analysis"].get("buy_rate", 0)
        if buy_rate < 10:
            recommendations.append(
                f"⚠️ ML buy rate rất thấp ({buy_rate:.1f}%). "
                "Cần retrain models hoặc adjust ML threshold."
            )

        # Check confidence distribution
        low_conf = results["confidence_ranges"]["0-20"] + results["confidence_ranges"]["20-40"]
        if low_conf / total > 0.5:
            recommendations.append(
                f"⚠️ {low_conf/total*100:.1f}% mã có confidence <40%. "
                "ML models có thể cần improvement."
            )

        # Check entry filters
        failed_conf = results["entry_filters"]["failed_confidence"]
        if failed_conf / total > 0.3:
            current_threshold = entry_logic.min_confidence
            suggested = max(40, current_threshold - 10)
            recommendations.append(
                f"💡 {failed_conf/total*100:.1f}% mã fail do confidence threshold. "
                f"Hiện tại: {current_threshold}%, đề xuất giảm xuống {suggested}%"
            )

        failed_trend = results["entry_filters"]["failed_trend"]
        if failed_trend / total > 0.3:
            recommendations.append(
                f"💡 {failed_trend/total*100:.1f}% mã fail do trend alignment. "
                "Có thể set require_trend_alignment=False trong sideways market."
            )

        failed_volatility = results["entry_filters"]["failed_volatility"]
        if failed_volatility / total > 0.2:
            recommendations.append(
                f"💡 {failed_volatility/total*100:.1f}% mã fail do volatility. "
                "Market có thể đang volatile, cân nhắc relax volatility filter."
            )

        # Overall recommendation
        passed = results["entry_filters"]["passed_all"]
        if passed == 0:
            recommendations.append(
                "🚨 KHÔNG CÓ MÃ NÀO PASS! Filters quá strict. "
                "Đề xuất: Giảm min_confidence 10-15% và review các filters."
            )
        elif passed / total < 0.05:
            recommendations.append(
                f"⚠️ Chỉ {passed/total*100:.1f}% mã pass filters. "
                "Filters có thể hơi strict, cân nhắc điều chỉnh."
            )

        return recommendations

    def format_report(self, results: Dict) -> str:
        """Format analysis report"""
        lines = []
        lines.append("🔍 **NO SIGNALS ANALYSIS**")
        lines.append("=" * 60)
        lines.append("")

        # Overview
        lines.append("📊 **TỔNG QUAN**")
        lines.append(f"• Tổng mã scan: {results['total_scanned']}")
        lines.append(f"• Lỗi data: {results['data_issues']}")
        lines.append("")

        # ML Analysis
        ml = results["ml_analysis"]
        lines.append("🤖 **ML SIGNALS**")
        total_ml = ml.get("total", 0) or 0
        buy_pct = (ml["buy_signals"] / total_ml * 100) if total_ml > 0 else 0
        sell_pct = (ml["sell_signals"] / total_ml * 100) if total_ml > 0 else 0
        hold_pct = (ml["hold_signals"] / total_ml * 100) if total_ml > 0 else 0
        lines.append(f"• BUY signals: {ml['buy_signals']} ({buy_pct:.1f}%)")
        lines.append(f"• SELL signals: {ml['sell_signals']} ({sell_pct:.1f}%)")
        lines.append(f"• HOLD signals: {ml['hold_signals']} ({hold_pct:.1f}%)")
        lines.append("")

        # Confidence Distribution
        lines.append("📈 **CONFIDENCE DISTRIBUTION**")
        total = results["total_scanned"]
        for range_name, count in results["confidence_ranges"].items():
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"• {range_name}%: {count:3d} ({pct:5.1f}%) {bar}")
        lines.append("")

        # Entry Filters
        lines.append("🚪 **ENTRY FILTERS**")
        filters = results["entry_filters"]
        passed_pct = (filters["passed_all"] / total * 100) if total > 0 else 0
        fail_conf_pct = (filters["failed_confidence"] / total * 100) if total > 0 else 0
        fail_trend_pct = (filters["failed_trend"] / total * 100) if total > 0 else 0
        fail_sr_pct = (filters["failed_support"] / total * 100) if total > 0 else 0
        fail_vol_pct = (filters["failed_volume"] / total * 100) if total > 0 else 0
        fail_vola_pct = (filters["failed_volatility"] / total * 100) if total > 0 else 0
        fail_rsi_pct = (filters["failed_rsi"] / total * 100) if total > 0 else 0
        lines.append(f"• ✅ Passed all: {filters['passed_all']} ({passed_pct:.1f}%)")
        lines.append(
            f"• ❌ Failed confidence: {filters['failed_confidence']} ({fail_conf_pct:.1f}%)"
        )
        lines.append(f"• ❌ Failed trend: {filters['failed_trend']} ({fail_trend_pct:.1f}%)")
        lines.append(
            f"• ❌ Failed support/resistance: {filters['failed_support']} ({fail_sr_pct:.1f}%)"
        )
        lines.append(f"• ❌ Failed volume: {filters['failed_volume']} ({fail_vol_pct:.1f}%)")
        lines.append(
            f"• ❌ Failed volatility: {filters['failed_volatility']} ({fail_vola_pct:.1f}%)"
        )
        lines.append(f"• ❌ Failed RSI: {filters['failed_rsi']} ({fail_rsi_pct:.1f}%)")
        lines.append("")

        # Recommendations
        if results["recommendations"]:
            lines.append("💡 **KHUYẾN NGHỊ**")
            for rec in results["recommendations"]:
                lines.append(f"• {rec}")
            lines.append("")

        return "\n".join(lines)


# CLI function
def analyze_no_signals():
    """Analyze why no signals"""
    print("🔍 Analyzing why no signals...")

    try:
        # Lazy imports to avoid heavy startup
        from src.strategies.entry_logic import ImprovedEntryLogic
        from src.market.regime_proxy import ProxyMarketRegimeAnalyzer
        from src.ml.signals.generator import MLSignalGenerator
        from src.config.trading_config import get_config

        # Initialize
        config = get_config(validate=False)
        ml_generator = MLSignalGenerator()
        entry_logic = ImprovedEntryLogic(min_confidence=config.trading.min_confidence)
        market_analyzer = ProxyMarketRegimeAnalyzer()

        # Get market regime
        market_regime = market_analyzer.analyze_market_regime()
        print(f"Market regime: {market_regime['regime']}")

        # Analyze
        analyzer = NoSignalAnalyzer()

        # Load tickers using legacy helper (parses CSV correctly)
        from src.config.legacy_config import get_tickers

        tickers = get_tickers()
        print(f"📊 Loaded {len(tickers)} mã từ List.csv")

        results = analyzer.analyze_scan_results(
            tickers[:100], ml_generator, entry_logic, market_regime  # First 100 symbols
        )

        # Print report
        report = analyzer.format_report(results)
        print(report)

        # Save to file
        import os
        from datetime import datetime

        os.makedirs("reports", exist_ok=True)
        filename = f"reports/no_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Saved to {filename}")

    except Exception:
        print("❌ Error")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_no_signals()
