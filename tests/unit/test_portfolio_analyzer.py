"""
Unit tests for Portfolio Analyzer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np

from src.portfolio.analyzer import PortfolioAnalyzer, safe_print, safe_log


class TestSafePrintAndLog:
    """Test safe print and log functions"""

    def test_safe_print_normal_string(self, capsys):
        """Test safe_print with normal ASCII string"""
        safe_print("Hello World")
        captured = capsys.readouterr()
        assert "Hello World" in captured.out

    def test_safe_print_unicode_string(self, capsys):
        """Test safe_print with Unicode characters"""
        safe_print("Phân tích portfolio 📊")
        captured = capsys.readouterr()
        # Should not raise exception
        assert len(captured.out) > 0

    def test_safe_log_normal_string(self, capsys):
        """Test safe_log with normal string"""
        safe_log("Test log message")
        captured = capsys.readouterr()
        assert "Test log message" in captured.out


class TestPortfolioAnalyzer:
    """Test PortfolioAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with mocked dependencies"""
        with patch("src.portfolio.analyzer.MLSignalGenerator"), patch(
            "src.portfolio.analyzer.ImprovedEntryLogic"
        ), patch("src.portfolio.analyzer.ImprovedExitStrategy"), patch(
            "src.portfolio.analyzer.EnhancedPositionSizer"
        ), patch(
            "src.portfolio.analyzer.ProxyMarketRegimeAnalyzer"
        ), patch(
            "src.portfolio.analyzer.PortfolioOptimizer"
        ):

            analyzer = PortfolioAnalyzer(portfolio_file="test_portfolio.json")
            return analyzer

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.portfolio_file == "test_portfolio.json"
        assert analyzer.ml_generator is not None
        assert analyzer.entry_logic is not None
        assert analyzer.exit_strategy is not None
        assert analyzer.position_sizer is not None
        assert analyzer.market_analyzer is not None

    def test_initialization_default_file(self):
        """Test initialization with default portfolio file"""
        with patch("src.portfolio.analyzer.MLSignalGenerator"), patch(
            "src.portfolio.analyzer.ImprovedEntryLogic"
        ), patch("src.portfolio.analyzer.ImprovedExitStrategy"), patch(
            "src.portfolio.analyzer.EnhancedPositionSizer"
        ), patch(
            "src.portfolio.analyzer.ProxyMarketRegimeAnalyzer"
        ), patch(
            "src.portfolio.analyzer.PortfolioOptimizer"
        ):

            analyzer = PortfolioAnalyzer()
            assert analyzer.portfolio_file == "portfolio_status.json"

    @patch("src.portfolio.analyzer.load_data")
    def test_analyze_single_stock_no_data(self, mock_load_data, analyzer):
        """Test analyzing stock with no data"""
        mock_load_data.return_value = pd.DataFrame()

        holding = {"avg_price": 50000, "shares": 100}
        market_regime = {"regime": "BULLISH"}

        result = analyzer._analyze_single_stock("VCB", holding, market_regime)

        assert "error" in result
        assert "Không có dữ liệu" in result["error"]

    @patch("src.portfolio.analyzer.load_data")
    def test_analyze_single_stock_delisted(self, mock_load_data, analyzer):
        """Test analyzing delisted stock"""
        mock_load_data.side_effect = ValueError("Mã hủy niêm yết")

        holding = {"avg_price": 50000, "shares": 100}
        market_regime = {"regime": "BULLISH"}

        result = analyzer._analyze_single_stock("ABC", holding, market_regime)

        assert "error" in result
        assert "hủy niêm yết" in result["error"]

    @patch("src.portfolio.analyzer.load_data")
    def test_analyze_single_stock_success(self, mock_load_data, analyzer):
        """Test successful stock analysis"""
        # Create mock dataframe
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        mock_df = pd.DataFrame(
            {
                "close": np.random.uniform(45000, 55000, 100),
                "open": np.random.uniform(45000, 55000, 100),
                "high": np.random.uniform(50000, 60000, 100),
                "low": np.random.uniform(40000, 50000, 100),
                "volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

        mock_load_data.return_value = mock_df

        # Mock ML signal
        analyzer.ml_generator.analyze = Mock(
            return_value={"signal": "BUY", "confidence": 0.8}
        )

        # Mock exit decision
        mock_exit_decision = Mock()
        mock_exit_decision.should_exit = False
        mock_exit_decision.exit_reason = None
        analyzer.exit_strategy.check_exit = Mock(return_value=mock_exit_decision)

        holding = {"avg_price": 50000, "shares": 100}
        market_regime = {"regime": "BULLISH"}

        result = analyzer._analyze_single_stock("VCB", holding, market_regime)

        # Check result structure
        assert "symbol" in result
        assert result["symbol"] == "VCB"
        assert "shares" in result
        assert result["shares"] == 100
        assert "entry_price" in result
        assert result["entry_price"] == 50000
        assert "current_price" in result
        assert "recommendation" in result

    def test_generate_recommendation_sell(self, analyzer):
        """Test recommendation generation for SELL"""
        mock_exit_decision = Mock()
        mock_exit_decision.should_exit = True
        mock_exit_decision.exit_reason = "STOP_LOSS"

        recommendation = analyzer._generate_recommendation(
            mock_exit_decision, pnl_percent=-5.0, ml_signal=None
        )

        assert recommendation == "SELL"

    def test_generate_recommendation_hold(self, analyzer):
        """Test recommendation generation for HOLD"""
        mock_exit_decision = Mock()
        mock_exit_decision.should_exit = False

        recommendation = analyzer._generate_recommendation(
            mock_exit_decision, pnl_percent=2.0, ml_signal={"signal": "HOLD"}
        )

        assert recommendation == "HOLD"

    @patch("src.portfolio.analyzer.load_data")
    def test_analyze_current_portfolio_empty(self, mock_load_data, analyzer):
        """Test analyzing empty portfolio"""
        analyzer.market_analyzer.analyze_market_regime = Mock(
            return_value={"regime": "NEUTRAL"}
        )

        result = analyzer.analyze_current_portfolio({})

        assert "analyzed_at" in result
        assert "market_regime" in result
        assert "current_holdings" in result
        assert len(result["current_holdings"]) == 0
        assert "sell_recommendations" in result
        assert "hold_recommendations" in result

    @patch("src.portfolio.analyzer.load_data")
    def test_analyze_current_portfolio_with_holdings(self, mock_load_data, analyzer):
        """Test analyzing portfolio with holdings"""
        # Mock data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        mock_df = pd.DataFrame(
            {
                "close": np.random.uniform(45000, 55000, 100),
                "open": np.random.uniform(45000, 55000, 100),
                "high": np.random.uniform(50000, 60000, 100),
                "low": np.random.uniform(40000, 50000, 100),
                "volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

        mock_load_data.return_value = mock_df

        # Mock dependencies
        analyzer.market_analyzer.analyze_market_regime = Mock(
            return_value={"regime": "BULLISH"}
        )

        analyzer.ml_generator.analyze = Mock(
            return_value={"signal": "BUY", "confidence": 0.7}
        )

        mock_exit_decision = Mock()
        mock_exit_decision.should_exit = False
        analyzer.exit_strategy.check_exit = Mock(return_value=mock_exit_decision)

        analyzer._find_new_buy_opportunities = Mock(return_value=[])
        analyzer._calculate_portfolio_summary = Mock(
            return_value={"total_value": 5000000, "total_pnl": 100000}
        )
        analyzer._save_analysis = Mock()

        holdings = {"VCB": {"avg_price": 50000, "shares": 100}}

        result = analyzer.analyze_current_portfolio(holdings)

        assert "analyzed_at" in result
        assert "market_regime" in result
        assert "current_holdings" in result
        assert "VCB" in result["current_holdings"]
        assert "portfolio_summary" in result

    def test_create_error_analysis(self, analyzer):
        """Test error analysis creation"""
        result = analyzer._create_error_analysis("ABC", "Test error message")

        assert "symbol" in result
        assert result["symbol"] == "ABC"
        assert "error" in result
        assert result["error"] == "Test error message"
        assert "recommendation" in result
        assert result["recommendation"] == "HOLD"

    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_save_analysis(self, mock_json_dump, mock_open, analyzer):
        """Test saving analysis to file"""
        analysis_result = {
            "analyzed_at": datetime.now().isoformat(),
            "portfolio_summary": {"total_value": 1000000},
            "sell_recommendations": [],
            "hold_recommendations": [],
            "new_buy_recommendations": [],
        }

        analyzer._save_analysis(analysis_result)

        # Verify file was opened for writing
        mock_open.assert_called_once_with("test_portfolio.json", "w", encoding="utf-8")
        # Verify json.dump was called
        mock_json_dump.assert_called_once()

    def test_analyze_current_portfolio_handles_exception(self, analyzer):
        """Test that analyze_current_portfolio handles exceptions gracefully"""
        analyzer.market_analyzer.analyze_market_regime = Mock(
            side_effect=Exception("Market analysis failed")
        )

        result = analyzer.analyze_current_portfolio({})

        # Should return result structure even with error
        assert "analyzed_at" in result
        assert "current_holdings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
