"""
Unit tests for deprecated sector analysis module
"""

import pytest
from datetime import datetime
from src.market.sector_analysis import EnhancedSectorAnalyzer


class TestEnhancedSectorAnalyzer:
    """Test deprecated EnhancedSectorAnalyzer class"""

    def test_initialization(self):
        """Test that analyzer can be initialized with default parameters"""
        analyzer = EnhancedSectorAnalyzer()

        assert analyzer.min_volume == 1_000_000
        assert analyzer.min_price == 10_000

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        analyzer = EnhancedSectorAnalyzer(min_volume=500_000, min_price=5_000)

        assert analyzer.min_volume == 500_000
        assert analyzer.min_price == 5_000

    def test_analyze_all_sectors_returns_empty_result(self):
        """Test that analyze_all_sectors returns empty deprecated result"""
        analyzer = EnhancedSectorAnalyzer()

        result = analyzer.analyze_all_sectors()

        # Check structure
        assert "analyzed_at" in result
        assert "sector_scores" in result
        assert "ranked_sectors" in result
        assert "selected_sectors" in result
        assert "selected_tickers" in result
        assert "market_summary" in result

        # Check empty results
        assert result["sector_scores"] == {}
        assert result["ranked_sectors"] == []
        assert result["selected_sectors"] == []
        assert result["selected_tickers"] == []

        # Check market summary
        assert result["market_summary"]["market_sentiment"] == "NEUTRAL"
        assert result["market_summary"]["avg_sector_score"] == 0
        assert "deprecated" in result["market_summary"]["note"].lower()

    def test_analyze_all_sectors_with_parameters(self):
        """Test that analyze_all_sectors ignores parameters and returns empty result"""
        analyzer = EnhancedSectorAnalyzer()

        # Pass parameters that should be ignored
        result = analyzer.analyze_all_sectors(
            sectors_dict={"Technology": ["FPT", "CMG"]}, lookback=50
        )

        # Should still return empty result
        assert result["sector_scores"] == {}
        assert result["ranked_sectors"] == []
        assert result["selected_tickers"] == []

    def test_analyzed_at_timestamp_format(self):
        """Test that analyzed_at timestamp is in ISO format"""
        analyzer = EnhancedSectorAnalyzer()

        result = analyzer.analyze_all_sectors()

        # Should be able to parse the timestamp
        timestamp = result["analyzed_at"]
        parsed = datetime.fromisoformat(timestamp)

        assert isinstance(parsed, datetime)

    def test_deprecation_warning_on_init(self, caplog):
        """Test that deprecation warning is logged on initialization"""
        with caplog.at_level("WARNING"):
            analyzer = EnhancedSectorAnalyzer()

        # Check that warning was logged
        assert any("deprecated" in record.message.lower() for record in caplog.records)

    def test_deprecation_warning_on_analyze(self, caplog):
        """Test that deprecation warning is logged when analyzing"""
        analyzer = EnhancedSectorAnalyzer()

        with caplog.at_level("WARNING"):
            analyzer.analyze_all_sectors()

        # Check that warning was logged
        assert any("deprecated" in record.message.lower() for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
