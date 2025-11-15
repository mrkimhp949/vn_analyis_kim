# [file name]: portfolio_analyzer.py
# [file content begin]
# -*- coding: utf-8 -*-
"""
Portfolio Analyzer - Phân tích danh mục hiện tại
Kiểm tra cổ phiếu đang nắm giữ, đề xuất mua/bán
"""

import pandas as pd
import json
import os
from datetime import datetime
from data_loader import load_data
from ml_signals import MLSignalGenerator
from improved_entry_logic import ImprovedEntryLogic
from improved_exit_logic import ImprovedExitStrategy
from improved_position_sizing import ConservativePositionSizer
from market_regime_proxy import ProxyMarketRegimeAnalyzer
from portfolio_regime_adjuster import PortfolioRegimeAdjuster
from portfolio_optimizer import PortfolioOptimizer
from config import LOOKBACK
from risk_metrics import calculate_sector_exposure, summarize_exposure
import logging
import sys
import logging

logger = logging.getLogger(__name__)

# Fix encoding for Windows
if sys.platform == "win32":
    try:
        # Force UTF-8 encoding for console
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except:
        pass

    # Set environment variable for UTF-8
    os.environ["PYTHONIOENCODING"] = "utf-8"


def safe_print(message):
    """Print an toàn cho Windows, xử lý Unicode và emoji"""
    try:
        # Thử in với encoding mặc định
        print(message)
    except UnicodeEncodeError:
        try:
            # Thử với UTF-8
            print(message.encode("utf-8", errors="replace").decode("utf-8"))
        except:
            # Fallback: loại bỏ các ký tự không in được
            clean_message = "".join(char for char in message if ord(char) < 128)
            print(clean_message)


def safe_log(message):
    """Ghi log an toàn"""
    safe_print(message)


class PortfolioAnalyzer:
    def __init__(self, portfolio_file="portfolio_status.json"):
        self.portfolio_file = portfolio_file
        self.ml_generator = MLSignalGenerator()
        self.entry_logic = ImprovedEntryLogic()
        self.exit_strategy = ImprovedExitStrategy()
        self.position_sizer = ConservativePositionSizer()
        self.market_analyzer = ProxyMarketRegimeAnalyzer()
        self.regime_adjuster = PortfolioRegimeAdjuster()
        self.optimizer = PortfolioOptimizer()

    def analyze_current_portfolio(self, current_holdings):
        """
        Phân tích portfolio hiện tại - với xử lý JSON serialization
        """
        print("🔍 Đang phân tích portfolio hiện tại...")

        analysis_result = {
            "analyzed_at": datetime.now().isoformat(),
            "market_regime": None,
            "current_holdings": {},
            "sell_recommendations": [],
            "hold_recommendations": [],
            "new_buy_recommendations": [],
            "portfolio_summary": {},
            "cash_available": 0,
        }

        try:
            # Phân tích market regime
            market_regime = self.market_analyzer.analyze_market_regime()
            analysis_result["market_regime"] = market_regime

            # Phân tích từng cổ phiếu đang nắm giữ
            for symbol, holding in current_holdings.items():
                try:
                    stock_analysis = self._analyze_single_stock(
                        symbol, holding, market_regime
                    )
                    analysis_result["current_holdings"][symbol] = stock_analysis

                    # Phân loại recommendation
                    if stock_analysis["recommendation"] == "SELL":
                        analysis_result["sell_recommendations"].append(stock_analysis)
                    elif stock_analysis["recommendation"] == "HOLD":
                        analysis_result["hold_recommendations"].append(stock_analysis)

                except Exception as e:
                    logger.error(f"Lỗi phân tích {symbol}: {e}")
                    analysis_result["current_holdings"][symbol] = {
                        "error": str(e),
                        "recommendation": "HOLD",
                    }

            # Tìm cổ phiếu mới nên mua
            analysis_result["new_buy_recommendations"] = (
                self._find_new_buy_opportunities(current_holdings, market_regime)
            )

            # Tổng kết portfolio
            analysis_result["portfolio_summary"] = self._calculate_portfolio_summary(
                analysis_result["current_holdings"],
                market_regime,
                analysis_result.get("cash_available", 0.0),
            )

            self._save_analysis(analysis_result)

        except Exception as e:
            logger.error(f"Lỗi phân tích portfolio: {e}")

        return analysis_result

    def _analyze_single_stock(self, symbol, holding, market_regime):
        """Phân tích 1 cổ phiếu - KHÔNG trả về bất kỳ object nào"""
        try:
            df = load_data(symbol, lookback=LOOKBACK)
            if df.empty:
                return self._create_error_analysis(symbol, "Không có dữ liệu")
        except ValueError as e:
            error_msg = str(e)
            if "hủy niêm yết" in error_msg or "không tồn tại" in error_msg:
                return self._create_error_analysis(
                    symbol, f"Mã có thể đã bị hủy niêm yết"
                )
            return self._create_error_analysis(symbol, f"Lỗi tải dữ liệu: {error_msg}")
        except Exception as e:
            return self._create_error_analysis(symbol, f"Lỗi: {str(e)}")

        try:

            current_price = df["close"].iloc[-1]
            entry_price = holding["avg_price"]
            shares = holding["shares"]

            # Tính PnL
            current_value = shares * current_price
            entry_value = shares * entry_price
            pnl_amount = current_value - entry_value
            pnl_percent = (pnl_amount / entry_value) * 100 if entry_value > 0 else 0

            # Phân tích ML signal
            ml_signal = self.ml_generator.analyze(df)

            # Kiểm tra exit nhưng chỉ lấy thông tin đơn giản
            exit_decision = self.exit_strategy.check_exit(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss=entry_price * 0.95,
                take_profit_targets=[
                    entry_price * 1.10,
                    entry_price * 1.15,
                    entry_price * 1.25,
                ],
                entry_date=datetime.now(),
                df=df,
                ml_signal=ml_signal,
                market_regime=market_regime,
            )

            # Đề xuất
            recommendation = self._generate_recommendation(
                exit_decision, pnl_percent, ml_signal
            )

            # CHỈ trả về primitive types, không có objects
            return {
                "symbol": symbol,
                "shares": shares,
                "entry_price": entry_price,
                "current_price": current_price,
                "current_value": current_value,
                "entry_value": entry_value,
                "pnl_amount": pnl_amount,
                "pnl_percent": pnl_percent,
                "ml_signal": ml_signal["signal"],
                "ml_confidence": ml_signal["confidence"],
                # Chỉ lấy thông tin từ exit_decision, không lưu object
                "should_exit": exit_decision.should_exit,
                "exit_reason": (
                    exit_decision.exit_reason.value
                    if exit_decision.exit_reason
                    else None
                ),
                "exit_type": exit_decision.exit_type,
                "exit_urgency": exit_decision.urgency,
                "exit_message": exit_decision.message,
                "recommendation": recommendation,
                "recommendation_reason": self._get_recommendation_reason(
                    recommendation, exit_decision, pnl_percent
                ),
            }

        except Exception as e:
            return self._create_error_analysis(symbol, str(e))

    def _find_new_buy_opportunities(
        self, current_holdings, market_regime, max_recommendations=20
    ):
        """Tìm cổ phiếu mới nên mua - CHỈ dùng mã từ config"""
        safe_print("🎯 TÌM CƠ HỘI MUA MỚI - CHỈ DÙNG MÃ TỪ CONFIG")

        try:
            from config import TICKERS

            safe_print(f"   📊 Tổng mã trong config: {len(TICKERS)}")
            safe_print(
                f"   📦 Mã đang nắm: {len(current_holdings)} - {list(current_holdings.keys())}"
            )

            # ⭐⭐⭐ QUAN TRỌNG: Chỉ dùng mã từ config, loại trừ mã đã có ⭐⭐⭐
            symbols_to_scan = [s for s in TICKERS if s not in current_holdings]

            safe_print(
                f"   🔍 Sẽ quét {len(symbols_to_scan)} mã từ config (loại trừ mã đang nắm)"
            )

            if not symbols_to_scan:
                safe_print(
                    "   ⚠️ Không có mã nào để quét - có thể đã nắm hết các mã trong config"
                )
                return []

            buy_opportunities = []

            for i, symbol in enumerate(symbols_to_scan):
                try:
                    safe_print(
                        f"     📈 ({i+1}/{len(symbols_to_scan)}) Phân tích {symbol}..."
                    )

                    try:
                        df = load_data(symbol, lookback=LOOKBACK)
                    except ValueError as e:
                        error_msg = str(e)
                        if "hủy niêm yết" in error_msg or "không tồn tại" in error_msg:
                            safe_print(
                                f"       ⚠️ {symbol}: Mã có thể đã bị hủy niêm yết"
                            )
                            continue
                        safe_print(f"       ❌ {symbol}: Lỗi tải dữ liệu - {error_msg}")
                        continue

                    if df.empty or len(df) < 50:
                        safe_print(f"       ⚠️ {symbol}: Không đủ dữ liệu")
                        continue

                    ml_signal = self.ml_generator.analyze(df)
                    current_price = df["close"].iloc[-1]

                    # Kiểm tra entry signal
                    entry_signal = self.entry_logic.analyze_entry(
                        df=df, ml_signal=ml_signal, market_regime=market_regime
                    )

                    if entry_signal.should_enter:
                        opportunity = {
                            "symbol": symbol,
                            "current_price": current_price,
                            "confidence": entry_signal.confidence,
                            "strength": entry_signal.strength.name,
                            "stop_loss": entry_signal.stop_loss,
                            "take_profit_targets": entry_signal.take_profit_targets,
                            "score": entry_signal.confidence,
                        }
                        buy_opportunities.append(opportunity)
                        safe_print(
                            f"       ✅ {symbol}: BUY ({entry_signal.confidence}%) - {entry_signal.strength.name}"
                        )
                    else:
                        safe_print(
                            f"       ❌ {symbol}: {entry_signal.signal_type} ({entry_signal.confidence}%)"
                        )

                except Exception as e:
                    safe_print(f"       💥 {symbol}: Lỗi - {str(e)[:50]}...")

            safe_print(
                f"🎯 KẾT QUẢ: Tìm thấy {len(buy_opportunities)}/{len(symbols_to_scan)} mã BUY từ config"
            )
            return buy_opportunities[:max_recommendations]

        except Exception as e:
            safe_print(f"💥 LỖI NGHIÊM TRỌNG trong _find_new_buy_opportunities: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _generate_recommendation(self, exit_decision, pnl_percent, ml_signal):
        """Tạo đề xuất dựa trên nhiều yếu tố"""
        if exit_decision.should_exit:
            return "SELL"

        # Nếu ML signal là SELL và đang có lời
        if ml_signal["signal"] == "SELL" and pnl_percent > 5:
            return "SELL"

        # Nếu ML signal là BUY và đang lỗ ít
        if ml_signal["signal"] == "BUY" and pnl_percent > -5:
            return "HOLD"

        return "HOLD"

    def _get_recommendation_reason(self, recommendation, exit_decision, pnl_percent):
        """Lý do cho đề xuất"""
        if recommendation == "SELL":
            if exit_decision.should_exit:
                return exit_decision.message
            elif pnl_percent > 20:
                return f"Đã đạt lợi nhuận cao +{pnl_percent:.1f}% - Nên chốt lời"
            else:
                return "Tín hiệu kỹ thuật yếu - Nên thoát"
        else:
            return "Vẫn tiềm năng - Nên tiếp tục nắm giữ"

    def _calculate_portfolio_summary(
        self, current_holdings, market_regime, current_cash=0.0
    ):
        """Tính tổng kết portfolio"""
        total_value = sum(
            h["current_value"]
            for h in current_holdings.values()
            if "current_value" in h
        )
        total_pnl = sum(
            h["pnl_amount"] for h in current_holdings.values() if "pnl_amount" in h
        )
        total_invested = sum(
            h["entry_value"] for h in current_holdings.values() if "entry_value" in h
        )

        sell_recommendations = sum(
            1 for h in current_holdings.values() if h.get("recommendation") == "SELL"
        )
        hold_recommendations = sum(
            1 for h in current_holdings.values() if h.get("recommendation") == "HOLD"
        )

        sector_exposure = (
            calculate_sector_exposure(current_holdings) if current_holdings else {}
        )

        regime_adjustment = self.regime_adjuster.evaluate_adjustment(
            current_holdings,
            market_regime=market_regime,
            current_cash=current_cash or 0.0,
        )

        optimal_allocation = None
        try:
            optimization = self.optimizer.optimize_weights(
                [
                    symbol
                    for symbol, data in current_holdings.items()
                    if data.get("shares", 0) > 0
                ]
            )
            if optimization:
                optimal_allocation = {
                    "method": optimization.method,
                    "weights": optimization.weights,
                    "annualized_volatility": optimization.annualized_volatility,
                    "notes": optimization.notes,
                }
        except Exception as exc:
            logging_msg = f"Failed to optimize portfolio: {exc}"
            logger.debug(logging_msg)

        return {
            "total_portfolio_value": total_value,
            "total_invested": total_invested,
            "total_pnl": total_pnl,
            "total_return_percent": (
                (total_pnl / total_invested * 100) if total_invested > 0 else 0
            ),
            "number_of_stocks": len(current_holdings),
            "sell_recommendations_count": sell_recommendations,
            "hold_recommendations_count": hold_recommendations,
            "sector_exposure": sector_exposure,
            "regime_adjustment": {
                "regime": regime_adjustment.regime,
                "target_cash_ratio": regime_adjustment.target_cash_ratio,
                "target_exposure_ratio": regime_adjustment.target_exposure_ratio,
                "required_cash_increase": regime_adjustment.required_cash_increase,
                "suggested_sales": regime_adjustment.suggested_sales,
                "notes": regime_adjustment.notes,
            },
            "optimal_allocation": optimal_allocation,
        }

    def _create_error_analysis(self, symbol, error_msg):
        """Tạo phân tích lỗi"""
        return {
            "symbol": symbol,
            "error": error_msg,
            "recommendation": "HOLD",
            "recommendation_reason": f"Lỗi phân tích: {error_msg}",
        }

    def _save_analysis(self, analysis_result):
        """Lưu kết quả phân tích - sử dụng safe_print"""
        try:
            # Tạo bản copy đơn giản
            safe_result = {
                "analyzed_at": analysis_result["analyzed_at"],
                "portfolio_summary": analysis_result["portfolio_summary"],
                "sell_count": len(analysis_result.get("sell_recommendations", [])),
                "hold_count": len(analysis_result.get("hold_recommendations", [])),
                "buy_count": len(analysis_result.get("new_buy_recommendations", [])),
            }

            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(safe_result, f, indent=2, ensure_ascii=False)
            safe_print("✅ Đã lưu phân tích vào portfolio file")

        except Exception as e:
            safe_print(f"❌ Lỗi lưu phân tích: {e}")

    def format_analysis_report(self, analysis_result):
        """Format báo cáo phân tích"""
        report = []

        # Header
        report.append("📊 **BÁO CÁO PHÂN TÍCH PORTFOLIO**")
        report.append("=" * 50)

        # Note
        report.append(
            "ℹ️ **LƯU Ý**: Đề xuất BÁN/GIỮ là cho mã đang nắm giữ, MUA MỚI là cho mã chưa có"
        )
        report.append("")

        # Market Regime
        regime = analysis_result["market_regime"]
        if regime:
            report.append(
                f"📈 **Market Regime**: {regime['regime']} ({regime['confidence']}%)"
            )
            report.append(f"💡 {regime['message']}")
        report.append("")

        # Portfolio Summary
        summary = analysis_result["portfolio_summary"]
        report.append("💰 **TỔNG QUAN PORTFOLIO**")
        report.append(f"• Tổng giá trị: {summary['total_portfolio_value']:,.0f} VNĐ")
        report.append(f"• Tổng đầu tư: {summary['total_invested']:,.0f} VNĐ")
        report.append(
            f"• Lợi nhuận: {summary['total_pnl']:,.0f} VNĐ ({summary['total_return_percent']:.1f}%)"
        )
        report.append(f"• Số mã: {summary['number_of_stocks']}")
        report.append(f"• Đề xuất BÁN: {summary['sell_recommendations_count']} mã")
        exposure_lines = summarize_exposure(summary.get("sector_exposure", {}), top_n=5)
        if exposure_lines:
            report.append("• Phân bổ theo ngành:")
            for line in exposure_lines:
                report.append(f"  └ {line}")
        adjustment = summary.get("regime_adjustment")
        if adjustment:
            report.append("")
            report.append("⚠️ **ĐIỀU CHỈNH THEO REGIME**")
            report.append(f"• Regime: {adjustment.get('regime')}")
            report.append(
                f"• Mục tiêu tiền mặt: {adjustment.get('target_cash_ratio', 0)*100:.0f}%"
            )
            report.append(
                f"• Cần tăng tiền mặt thêm: {adjustment.get('required_cash_increase', 0):,.0f} VNĐ"
            )
            if adjustment.get("suggested_sales"):
                report.append("• Đề xuất bán:")
                for sale in adjustment["suggested_sales"]:
                    report.append(
                        f"  └ {sale['symbol']}: bán {sale['shares_to_sell']} CP (~{sale['approx_value']:,.0f} VNĐ)"
                    )
            if adjustment.get("notes"):
                report.append(f"• Ghi chú: {adjustment['notes']}")
        report.append("")

        optimal = summary.get("optimal_allocation")
        if optimal and optimal.get("weights"):
            report.append("🧮 **PHÂN BỔ TỐI ƯU (HRP/RISK BUDGETING)**")
            report.append(f"• Phương pháp: {optimal.get('method')}")
            if optimal.get("annualized_volatility") is not None:
                report.append(f"• Vol kỳ vọng: {optimal['annualized_volatility']:.2%}")
            for symbol, weight in sorted(
                optimal["weights"].items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"  └ {symbol}: {weight*100:.1f}%")
            report.append("")

        # Sell Recommendations
        if analysis_result["sell_recommendations"]:
            report.append("🔴 **NÊN BÁN** (Mã đang nắm giữ)")
            for rec in analysis_result["sell_recommendations"]:
                report.append(
                    f"• {rec['symbol']}: {rec['pnl_percent']:+.1f}% - {rec['recommendation_reason']}"
                )
            report.append("")

        # Hold Recommendations
        if analysis_result["hold_recommendations"]:
            report.append("🟡 **NÊN GIỮ** (Mã đang nắm giữ)")
            for rec in analysis_result["hold_recommendations"]:
                report.append(
                    f"• {rec['symbol']}: {rec['pnl_percent']:+.1f}% - {rec['ml_signal']} ({rec['ml_confidence']}%)"
                )
            report.append("")

        # New Buy Opportunities
        if analysis_result["new_buy_recommendations"]:
            report.append("🟢 **CƠ HỘI MUA MỚI** (Mã chưa nắm giữ)")
            for opp in analysis_result["new_buy_recommendations"]:
                signal = opp["entry_signal"]
                report.append(f"• {opp['symbol']}: {opp['current_price']:,.0f} VNĐ")
                report.append(
                    f"  └ {signal.confidence}% - {signal.strength.name} - {opp['position_info'].shares:,} CP"
                )
            report.append("")
        return "\n".join(report)


# Utility function để tích hợp nhanh
def analyze_my_portfolio(current_holdings):
    """
    Hàm tiện ích để phân tích portfolio nhanh

    Args:
        current_holdings: Dict {symbol: {'shares': int, 'avg_price': float}}

    Returns:
        Formatted report string
    """
    analyzer = PortfolioAnalyzer()
    analysis = analyzer.analyze_current_portfolio(current_holdings)
    return analyzer.format_analysis_report(analysis)


# Test function
if __name__ == "__main__":
    # Ví dụ portfolio
    sample_portfolio = {
        "ACB": {"shares": 500, "avg_price": 25000},
        "VNM": {"shares": 300, "avg_price": 80000},
        "HPG": {"shares": 400, "avg_price": 45000},
    }

    print("🧪 Testing Portfolio Analyzer...")
    analyzer = PortfolioAnalyzer()
    result = analyzer.analyze_current_portfolio(sample_portfolio)
    report = analyzer.format_analysis_report(result)
    print(report)
# [file content end]
