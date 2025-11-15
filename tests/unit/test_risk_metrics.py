"""
Unit tests for src/risk/metrics.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.risk.metrics import (
    calculate_sector_exposure,
    check_sector_overweight,
    calculate_correlation_matrix,
    check_high_correlation,
    get_sector_for_symbol,
    calculate_portfolio_correlation_risk,
    summarize_exposure,
    get_diversification_recommendation,
    load_returns_dataframe,
    calculate_distance_correlation_matrix,
    calculate_copula_correlation_matrix,
    check_high_distance_correlation,
    check_high_copula_correlation,
    _distance_correlation,
    SECTOR_MAP,
)


class TestSectorExposure:
    """Tests for sector exposure calculation"""

    def test_calculate_sector_exposure_basic(self):
        """Test basic sector exposure calculation"""
        holdings = {
            "VCB": {"shares": 100, "current_price": 100, "current_value": 10000},
            "FPT": {"shares": 100, "current_price": 80, "current_value": 8000},
            "VHM": {"shares": 100, "current_price": 70, "current_value": 7000},
        }

        exposure = calculate_sector_exposure(holdings)

        # Total = 25000
        # BANKING (VCB): 10000 = 40%
        # TECHNOLOGY (FPT): 8000 = 32%
        # REAL_ESTATE (VHM): 7000 = 28%
        assert exposure["BANKING"] == pytest.approx(40.0, rel=0.01)
        assert exposure["TECHNOLOGY"] == pytest.approx(32.0, rel=0.01)
        assert exposure["REAL_ESTATE"] == pytest.approx(28.0, rel=0.01)

    def test_calculate_sector_exposure_unclassified(self):
        """Test exposure with unclassified stocks"""
        holdings = {
            "UNKNOWN": {"shares": 100, "current_price": 50, "current_value": 5000},
            "VCB": {"shares": 100, "current_price": 50, "current_value": 5000},
        }

        exposure = calculate_sector_exposure(holdings)

        assert exposure["UNCLASSIFIED"] == pytest.approx(50.0, rel=0.01)
        assert exposure["BANKING"] == pytest.approx(50.0, rel=0.01)

    def test_calculate_sector_exposure_empty(self):
        """Test exposure with empty holdings"""
        exposure = calculate_sector_exposure({})
        assert exposure == {}

    def test_calculate_sector_exposure_zero_value(self):
        """Test exposure when total value is zero"""
        holdings = {
            "VCB": {"shares": 0, "current_price": 100, "current_value": 0},
            "FPT": {"shares": 0, "current_price": 80, "current_value": 0},
        }

        exposure = calculate_sector_exposure(holdings)

        # All sectors should have 0% when total value is 0
        for pct in exposure.values():
            assert pct == 0.0

    def test_calculate_sector_exposure_without_current_value(self):
        """Test exposure calculation from shares and price"""
        holdings = {
            "VCB": {"shares": 100, "current_price": 100},
            "FPT": {"shares": 100, "current_price": 80},
        }

        exposure = calculate_sector_exposure(holdings)

        # Total = 18000
        # BANKING: 10000 = 55.56%
        # TECHNOLOGY: 8000 = 44.44%
        assert exposure["BANKING"] == pytest.approx(55.56, rel=0.01)
        assert exposure["TECHNOLOGY"] == pytest.approx(44.44, rel=0.01)


class TestSectorOverweight:
    """Tests for sector overweight detection"""

    def test_check_sector_overweight_found(self):
        """Test detection of overweight sectors"""
        exposure = {
            "BANKING": 45.0,
            "TECHNOLOGY": 30.0,
            "REAL_ESTATE": 25.0,
        }

        overweight = check_sector_overweight(exposure, max_sector_pct=40.0)

        assert len(overweight) == 1
        assert overweight[0] == ("BANKING", 45.0)

    def test_check_sector_overweight_multiple(self):
        """Test detection of multiple overweight sectors"""
        exposure = {
            "BANKING": 50.0,
            "TECHNOLOGY": 45.0,
            "REAL_ESTATE": 5.0,
        }

        overweight = check_sector_overweight(exposure, max_sector_pct=40.0)

        assert len(overweight) == 2
        # Should be sorted by percentage descending
        assert overweight[0] == ("BANKING", 50.0)
        assert overweight[1] == ("TECHNOLOGY", 45.0)

    def test_check_sector_overweight_none(self):
        """Test when no sectors are overweight"""
        exposure = {
            "BANKING": 30.0,
            "TECHNOLOGY": 25.0,
            "REAL_ESTATE": 20.0,
        }

        overweight = check_sector_overweight(exposure, max_sector_pct=40.0)

        assert len(overweight) == 0


class TestGetSector:
    """Tests for get_sector_for_symbol"""

    def test_get_sector_known_symbols(self):
        """Test getting sector for known symbols"""
        assert get_sector_for_symbol("VCB") == "BANKING"
        assert get_sector_for_symbol("vcb") == "BANKING"  # Case insensitive
        assert get_sector_for_symbol("FPT") == "TECHNOLOGY"
        assert get_sector_for_symbol("VHM") == "REAL_ESTATE"

    def test_get_sector_unknown_symbol(self):
        """Test getting sector for unknown symbol"""
        assert get_sector_for_symbol("UNKNOWN123") == "UNCLASSIFIED"


class TestLoadReturnsDataframe:
    """Tests for load_returns_dataframe"""

    @patch("src.data.loader.load_data")
    def test_load_returns_dataframe_success(self, mock_load_data):
        """Test successful loading of returns data"""
        # Mock data for two symbols
        mock_df1 = pd.DataFrame({"close": [100, 102, 101, 103, 105] + [100] * 20})
        mock_df2 = pd.DataFrame({"close": [50, 51, 52, 50, 53] + [50] * 20})

        mock_load_data.side_effect = [mock_df1, mock_df2]

        result = load_returns_dataframe(["VCB", "FPT"], lookback=5)

        assert not result.empty
        assert "VCB" in result.columns
        assert "FPT" in result.columns
        # Should have at least 20 returns (requirement in the code)
        assert len(result) >= 20

    @patch("src.data.loader.load_data")
    def test_load_returns_dataframe_insufficient_data(self, mock_load_data):
        """Test with insufficient data points"""
        # Only 10 data points, less than required 20
        mock_df = pd.DataFrame({"close": list(range(10))})
        mock_load_data.return_value = mock_df

        result = load_returns_dataframe(["VCB"], lookback=10)

        # Should not include symbol with insufficient data
        assert result.empty or "VCB" not in result.columns

    @patch("src.data.loader.load_data")
    def test_load_returns_dataframe_missing_close(self, mock_load_data):
        """Test with missing close column"""
        mock_df = pd.DataFrame({"open": [100, 102, 101]})
        mock_load_data.return_value = mock_df

        result = load_returns_dataframe(["VCB"], lookback=5)

        assert result.empty

    @patch("src.data.loader.load_data")
    def test_load_returns_dataframe_exception(self, mock_load_data):
        """Test handling of exceptions"""
        mock_load_data.side_effect = Exception("Data load failed")

        result = load_returns_dataframe(["VCB", "FPT"], lookback=5)

        assert result.empty


class TestCorrelationMatrix:
    """Tests for correlation matrix calculation"""

    @patch("src.risk.metrics.load_returns_dataframe")
    def test_calculate_correlation_matrix_success(self, mock_load_returns):
        """Test successful correlation calculation"""
        # Create mock returns data
        returns_df = pd.DataFrame(
            {
                "VCB": [0.01, 0.02, -0.01, 0.015],
                "FPT": [0.015, 0.01, -0.005, 0.02],
            }
        )
        mock_load_returns.return_value = returns_df

        result = calculate_correlation_matrix(["VCB", "FPT"], lookback=60)

        assert not result.empty
        assert "VCB" in result.columns
        assert "FPT" in result.columns
        # Diagonal should be 1.0
        assert result.loc["VCB", "VCB"] == pytest.approx(1.0)
        assert result.loc["FPT", "FPT"] == pytest.approx(1.0)
        # Correlation should be symmetric
        assert result.loc["VCB", "FPT"] == pytest.approx(result.loc["FPT", "VCB"])

    @patch("src.risk.metrics.load_returns_dataframe")
    def test_calculate_correlation_matrix_empty(self, mock_load_returns):
        """Test with empty returns data"""
        mock_load_returns.return_value = pd.DataFrame()

        result = calculate_correlation_matrix(["VCB", "FPT"], lookback=60)

        assert result.empty


class TestDistanceCorrelation:
    """Tests for distance correlation"""

    def test_distance_correlation_perfect(self):
        """Test distance correlation with perfectly correlated data"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x

        dcor = _distance_correlation(x, y)

        assert dcor > 0.9  # Should be very high

    def test_distance_correlation_independent(self):
        """Test distance correlation with independent data"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        dcor = _distance_correlation(x, y)

        assert 0.0 <= dcor <= 0.3  # Should be low for independent data

    def test_distance_correlation_empty(self):
        """Test distance correlation with empty arrays"""
        x = np.array([])
        y = np.array([])

        dcor = _distance_correlation(x, y)

        assert dcor == 0.0

    def test_distance_correlation_different_lengths(self):
        """Test distance correlation with different length arrays"""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        dcor = _distance_correlation(x, y)

        assert dcor == 0.0

    def test_distance_correlation_multidimensional(self):
        """Test distance correlation handles multidimensional input"""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([[2, 4], [6, 8], [10, 12]])

        # Should flatten and calculate
        dcor = _distance_correlation(x, y)

        assert dcor >= 0.0


class TestDistanceCorrelationMatrix:
    """Tests for distance correlation matrix"""

    def test_calculate_distance_correlation_matrix_success(self):
        """Test successful distance correlation matrix calculation"""
        returns_df = pd.DataFrame(
            {
                "VCB": [0.01, 0.02, -0.01, 0.015, 0.01],
                "FPT": [0.015, 0.01, -0.005, 0.02, 0.012],
                "VHM": [0.02, 0.015, -0.002, 0.018, 0.013],
            }
        )

        result = calculate_distance_correlation_matrix(returns_df)

        assert not result.empty
        assert result.shape == (3, 3)
        # Diagonal should be 1.0
        assert result.loc["VCB", "VCB"] == pytest.approx(1.0)
        assert result.loc["FPT", "FPT"] == pytest.approx(1.0)
        assert result.loc["VHM", "VHM"] == pytest.approx(1.0)
        # Should be symmetric
        assert result.loc["VCB", "FPT"] == pytest.approx(result.loc["FPT", "VCB"])

    def test_calculate_distance_correlation_matrix_empty(self):
        """Test with empty dataframe"""
        result = calculate_distance_correlation_matrix(pd.DataFrame())
        assert result.empty

    def test_calculate_distance_correlation_matrix_single_column(self):
        """Test with single column"""
        returns_df = pd.DataFrame({"VCB": [0.01, 0.02, -0.01]})
        result = calculate_distance_correlation_matrix(returns_df)
        assert result.empty


class TestCopulaCorrelationMatrix:
    """Tests for copula correlation matrix"""

    def test_calculate_copula_correlation_matrix_success(self):
        """Test successful copula correlation calculation"""
        returns_df = pd.DataFrame(
            {
                "VCB": [0.01, 0.02, -0.01, 0.015, 0.01],
                "FPT": [0.015, 0.01, -0.005, 0.02, 0.012],
            }
        )

        result = calculate_copula_correlation_matrix(returns_df)

        assert not result.empty
        assert "VCB" in result.columns
        assert "FPT" in result.columns
        # Diagonal should be 1.0
        assert result.loc["VCB", "VCB"] == pytest.approx(1.0)

    def test_calculate_copula_correlation_matrix_empty(self):
        """Test with empty dataframe"""
        result = calculate_copula_correlation_matrix(pd.DataFrame())
        assert result.empty


class TestCheckHighCorrelation:
    """Tests for checking high correlation pairs"""

    def test_check_high_correlation_found(self):
        """Test finding high correlation pairs"""
        corr_matrix = pd.DataFrame(
            {
                "VCB": [1.0, 0.85, 0.3],
                "TCB": [0.85, 1.0, 0.4],
                "FPT": [0.3, 0.4, 1.0],
            },
            index=["VCB", "TCB", "FPT"],
        )

        result = check_high_correlation(corr_matrix, threshold=0.7)

        assert len(result) == 1
        assert result[0][0] in ["VCB", "TCB"]
        assert result[0][1] in ["VCB", "TCB"]
        assert result[0][2] == pytest.approx(0.85)

    def test_check_high_correlation_with_holdings(self):
        """Test checking correlation only for current holdings"""
        corr_matrix = pd.DataFrame(
            {
                "VCB": [1.0, 0.85, 0.3],
                "TCB": [0.85, 1.0, 0.4],
                "FPT": [0.3, 0.4, 1.0],
            },
            index=["VCB", "TCB", "FPT"],
        )

        # Only check VCB and FPT (low correlation)
        result = check_high_correlation(
            corr_matrix, threshold=0.7, current_holdings=["VCB", "FPT"]
        )

        assert len(result) == 0

    def test_check_high_correlation_negative(self):
        """Test finding high negative correlation"""
        corr_matrix = pd.DataFrame(
            {
                "VCB": [1.0, -0.85, 0.3],
                "FPT": [-0.85, 1.0, 0.4],
                "VHM": [0.3, 0.4, 1.0],
            },
            index=["VCB", "FPT", "VHM"],
        )

        result = check_high_correlation(corr_matrix, threshold=0.7)

        # Should detect absolute value >= threshold
        assert len(result) == 1
        assert abs(result[0][2]) == pytest.approx(0.85)

    def test_check_high_correlation_empty(self):
        """Test with empty matrix"""
        result = check_high_correlation(pd.DataFrame(), threshold=0.7)
        assert len(result) == 0


class TestCheckHighDistanceCorrelation:
    """Tests for checking high distance correlation"""

    def test_check_high_distance_correlation_found(self):
        """Test finding high distance correlation pairs"""
        distance_matrix = pd.DataFrame(
            {
                "VCB": [1.0, 0.75, 0.3],
                "TCB": [0.75, 1.0, 0.4],
                "FPT": [0.3, 0.4, 1.0],
            },
            index=["VCB", "TCB", "FPT"],
        )

        result = check_high_distance_correlation(distance_matrix, threshold=0.6)

        assert len(result) == 1
        assert result[0][2] == pytest.approx(0.75)

    def test_check_high_distance_correlation_empty(self):
        """Test with empty matrix"""
        result = check_high_distance_correlation(pd.DataFrame(), threshold=0.6)
        assert len(result) == 0


class TestCheckHighCopulaCorrelation:
    """Tests for checking high copula correlation"""

    def test_check_high_copula_correlation_found(self):
        """Test finding high copula correlation pairs"""
        copula_matrix = pd.DataFrame(
            {
                "VCB": [1.0, 0.80, 0.3],
                "TCB": [0.80, 1.0, 0.4],
                "FPT": [0.3, 0.4, 1.0],
            },
            index=["VCB", "TCB", "FPT"],
        )

        result = check_high_copula_correlation(copula_matrix, threshold=0.7)

        assert len(result) == 1
        assert result[0][2] == pytest.approx(0.80)

    def test_check_high_copula_correlation_negative(self):
        """Test handling negative correlations"""
        copula_matrix = pd.DataFrame(
            {
                "VCB": [1.0, -0.80, 0.3],
                "FPT": [-0.80, 1.0, 0.4],
                "VHM": [0.3, 0.4, 1.0],
            },
            index=["VCB", "FPT", "VHM"],
        )

        result = check_high_copula_correlation(copula_matrix, threshold=0.7)

        assert len(result) == 1
        assert abs(result[0][2]) == pytest.approx(0.80)


class TestPortfolioCorrelationRisk:
    """Tests for calculate_portfolio_correlation_risk"""

    def test_portfolio_correlation_risk_insufficient_holdings(self):
        """Test with fewer than 2 holdings"""
        result = calculate_portfolio_correlation_risk(["VCB"], lookback=60)

        assert result["avg_correlation"] == 0.0
        assert result["risk_score"] == 0
        assert "thêm mã" in result["recommendation"].lower()

    @patch("src.risk.metrics.load_returns_dataframe")
    def test_portfolio_correlation_risk_no_data(self, mock_load_returns):
        """Test with no available data"""
        mock_load_returns.return_value = pd.DataFrame()

        result = calculate_portfolio_correlation_risk(["VCB", "FPT"], lookback=60)

        assert result["risk_score"] == 50
        assert "không đủ dữ liệu" in result["recommendation"].lower()

    @patch("src.risk.metrics.load_returns_dataframe")
    def test_portfolio_correlation_risk_high_correlation(self, mock_load_returns):
        """Test with high correlation portfolio"""
        # Create highly correlated returns
        returns_df = pd.DataFrame(
            {
                "VCB": [0.01, 0.02, -0.01, 0.015],
                "TCB": [0.011, 0.019, -0.009, 0.016],  # Very similar to VCB
            }
        )
        mock_load_returns.return_value = returns_df

        result = calculate_portfolio_correlation_risk(
            ["VCB", "TCB"], lookback=60, max_avg_correlation=0.5
        )

        assert result["avg_correlation"] > 0.7
        assert result["risk_score"] > 50
        assert len(result["high_corr_pairs"]) >= 1

    @patch("src.risk.metrics.load_returns_dataframe")
    def test_portfolio_correlation_risk_low_correlation(self, mock_load_returns):
        """Test with well-diversified portfolio"""
        # Create uncorrelated returns
        np.random.seed(42)
        returns_df = pd.DataFrame(
            {
                "VCB": np.random.randn(30),
                "FPT": np.random.randn(30),
                "VHM": np.random.randn(30),
            }
        )
        mock_load_returns.return_value = returns_df

        result = calculate_portfolio_correlation_risk(
            ["VCB", "FPT", "VHM"], lookback=60, max_avg_correlation=0.5
        )

        assert result["avg_correlation"] < 0.5
        assert "đa dạng hóa tốt" in result["recommendation"].lower()


class TestSummarizeExposure:
    """Tests for summarize_exposure"""

    def test_summarize_exposure_top_n(self):
        """Test summarizing top N sectors"""
        exposure = {
            "BANKING": 40.0,
            "TECHNOLOGY": 25.0,
            "REAL_ESTATE": 20.0,
            "RETAIL": 10.0,
            "FOOD_BEVERAGE": 5.0,
        }

        result = summarize_exposure(exposure, top_n=3)

        assert len(result) == 3
        assert "BANKING: 40.0%" in result[0]
        assert "TECHNOLOGY: 25.0%" in result[1]
        assert "REAL_ESTATE: 20.0%" in result[2]

    def test_summarize_exposure_empty(self):
        """Test with empty exposure"""
        result = summarize_exposure({}, top_n=5)
        assert len(result) == 0


class TestGetDiversificationRecommendation:
    """Tests for get_diversification_recommendation"""

    @patch("src.risk.metrics.calculate_portfolio_correlation_risk")
    def test_diversification_recommendation_overweight(self, mock_corr_risk):
        """Test recommendations for overweight sectors"""
        mock_corr_risk.return_value = {
            "avg_correlation": 0.3,
            "high_corr_pairs": [],
            "distance_correlation_avg": 0.2,
            "high_distance_pairs": [],
            "copula_correlation_avg": 0.25,
            "high_copula_pairs": [],
            "risk_score": 30,
            "recommendation": "✅ Portfolio đa dạng hóa tốt.",
        }

        holdings = {
            "VCB": {"current_value": 45000},
            "TCB": {"current_value": 15000},
            "FPT": {"current_value": 40000},
        }

        result = get_diversification_recommendation(
            holdings, max_sector_pct=40.0, min_sectors=3
        )

        assert len(result["warnings"]) > 0
        assert len(result["overweight_sectors"]) > 0
        # BANKING is 60% (45k + 15k = 60k / 100k)
        assert any("BANKING" in w for w in result["warnings"])

    @patch("src.risk.metrics.calculate_portfolio_correlation_risk")
    def test_diversification_recommendation_high_correlation(self, mock_corr_risk):
        """Test recommendations for high correlation"""
        mock_corr_risk.return_value = {
            "avg_correlation": 0.8,
            "high_corr_pairs": [("VCB", "TCB", 0.9)],
            "distance_correlation_avg": 0.7,
            "high_distance_pairs": [],
            "copula_correlation_avg": 0.75,
            "high_copula_pairs": [],
            "risk_score": 80,
            "recommendation": "⚠️ Portfolio có correlation cao (0.80). Nên đa dạng hóa.",
        }

        holdings = {
            "VCB": {"current_value": 50000},
            "TCB": {"current_value": 50000},
        }

        result = get_diversification_recommendation(holdings)

        assert result["correlation_risk"]["risk_score"] > 50
        assert len(result["warnings"]) > 0
        assert len(result["recommendations"]) > 0

    @patch("src.risk.metrics.calculate_portfolio_correlation_risk")
    def test_diversification_recommendation_few_sectors(self, mock_corr_risk):
        """Test recommendations for insufficient diversification"""
        mock_corr_risk.return_value = {
            "avg_correlation": 0.3,
            "high_corr_pairs": [],
            "distance_correlation_avg": 0.2,
            "high_distance_pairs": [],
            "copula_correlation_avg": 0.25,
            "high_copula_pairs": [],
            "risk_score": 30,
            "recommendation": "✅ Portfolio đa dạng hóa tốt.",
        }

        # Only 2 sectors (BANKING and TECHNOLOGY)
        holdings = {
            "VCB": {"current_value": 50000},
            "FPT": {"current_value": 50000},
        }

        result = get_diversification_recommendation(
            holdings, max_sector_pct=40.0, min_sectors=3
        )

        assert any("ngành" in w.lower() for w in result["warnings"])
        assert result["diversification_score"] < 100

    @patch("src.risk.metrics.calculate_portfolio_correlation_risk")
    def test_diversification_recommendation_perfect(self, mock_corr_risk):
        """Test with well-diversified portfolio"""
        mock_corr_risk.return_value = {
            "avg_correlation": 0.2,
            "high_corr_pairs": [],
            "distance_correlation_avg": 0.15,
            "high_distance_pairs": [],
            "copula_correlation_avg": 0.18,
            "high_copula_pairs": [],
            "risk_score": 20,
            "recommendation": "✅ Portfolio đa dạng hóa tốt.",
        }

        # Well diversified across sectors
        holdings = {
            "VCB": {"current_value": 25000},  # BANKING
            "FPT": {"current_value": 25000},  # TECHNOLOGY
            "VHM": {"current_value": 25000},  # REAL_ESTATE
            "MWG": {"current_value": 25000},  # RETAIL
        }

        result = get_diversification_recommendation(
            holdings, max_sector_pct=40.0, min_sectors=3
        )

        assert len(result["warnings"]) == 0
        assert result["diversification_score"] == 100


class TestSectorMap:
    """Tests for SECTOR_MAP integrity"""

    def test_sector_map_not_empty(self):
        """Test that sector map is populated"""
        assert len(SECTOR_MAP) > 0

    def test_sector_map_valid_values(self):
        """Test that all sector values are strings"""
        for symbol, sector in SECTOR_MAP.items():
            assert isinstance(symbol, str)
            assert isinstance(sector, str)
            assert symbol.isupper()  # Symbols should be uppercase

    def test_sector_map_major_banks(self):
        """Test that major banks are mapped"""
        major_banks = ["VCB", "TCB", "CTG", "BID", "MBB", "VPB", "ACB"]
        for bank in major_banks:
            assert bank in SECTOR_MAP
            assert SECTOR_MAP[bank] == "BANKING"

    def test_sector_map_major_tech(self):
        """Test that major tech stocks are mapped"""
        assert "FPT" in SECTOR_MAP
        assert SECTOR_MAP["FPT"] == "TECHNOLOGY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
