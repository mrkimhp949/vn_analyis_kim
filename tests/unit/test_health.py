"""
Unit tests for Health Monitoring
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests

from src.monitoring.health import HealthChecker


class TestHealthChecker:
    """Test HealthChecker class"""

    @pytest.fixture
    def health_checker(self):
        """Create health checker instance"""
        return HealthChecker(api_url="http://test-api:8080")

    def test_initialization(self, health_checker):
        """Test health checker initialization"""
        assert health_checker.api_url == "http://test-api:8080"
        assert health_checker.checks_passed == 0
        assert health_checker.checks_failed == 0
        assert health_checker.warnings == []

    def test_initialization_default_url(self):
        """Test initialization with default URL"""
        checker = HealthChecker()
        assert checker.api_url == "http://localhost:8080"

    @patch("requests.get")
    def test_check_api_server_healthy(self, mock_get, health_checker):
        """Test API server check when healthy"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        passed, message = health_checker.check_api_server()

        assert passed is True
        assert message == "API server is healthy"
        mock_get.assert_called_once_with("http://test-api:8080/health", timeout=5)

    @patch("requests.get")
    def test_check_api_server_unhealthy_status(self, mock_get, health_checker):
        """Test API server check with bad status code"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        passed, message = health_checker.check_api_server()

        assert passed is False
        assert "API server returned status 500" in message

    @patch("requests.get")
    def test_check_api_server_connection_error(self, mock_get, health_checker):
        """Test API server check with connection error"""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        passed, message = health_checker.check_api_server()

        assert passed is False
        assert message == "Cannot connect to API server"

    @patch("requests.get")
    def test_check_api_server_general_exception(self, mock_get, health_checker):
        """Test API server check with general exception"""
        mock_get.side_effect = Exception("Network error")

        passed, message = health_checker.check_api_server()

        assert passed is False
        assert message == "API check failed"

    @patch("src.data.database.get_db")
    def test_check_database_success(self, mock_get_db, health_checker):
        """Test database check success"""
        mock_db = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [5]  # 5 positions
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_db.get_connection.return_value = mock_conn
        mock_get_db.return_value = mock_db

        passed, message = health_checker.check_database()

        assert passed is True
        assert "Database OK (5 active positions)" in message

    @patch("src.data.database.get_db")
    def test_check_database_failure(self, mock_get_db, health_checker):
        """Test database check failure"""
        mock_get_db.side_effect = Exception("Database connection failed")

        passed, message = health_checker.check_database()

        assert passed is False
        assert message == "Database check failed"

    @patch("src.ml.models.predictor.MLPredictor")
    def test_check_models_success(self, mock_predictor_class, health_checker):
        """Test ML models check success"""
        mock_predictor = Mock()
        mock_predictor.load_models.return_value = True
        mock_predictor.rf_model = Mock()  # Not None
        mock_predictor_class.return_value = mock_predictor

        passed, message = health_checker.check_models()

        assert passed is True
        assert message == "ML models loaded successfully"

    @patch("src.ml.models.predictor.MLPredictor")
    def test_check_models_dummy_mode(self, mock_predictor_class, health_checker):
        """Test ML models check in dummy mode"""
        mock_predictor = Mock()
        mock_predictor.load_models.return_value = False
        mock_predictor.rf_model = None
        mock_predictor_class.return_value = mock_predictor

        passed, message = health_checker.check_models()

        assert passed is True
        assert message == "ML models available (dummy mode)"
        assert "ML models using dummy fallback" in health_checker.warnings

    @patch("src.ml.models.predictor.MLPredictor")
    def test_check_models_failure(self, mock_predictor_class, health_checker):
        """Test ML models check failure"""
        mock_predictor_class.side_effect = Exception("Model loading failed")

        passed, message = health_checker.check_models()

        assert passed is False
        assert message == "Model check failed"

    @patch("src.config.trading_config.get_config")
    def test_check_configuration_valid(self, mock_get_config, health_checker):
        """Test configuration check with valid config"""
        mock_get_config.return_value = {"some": "config"}

        passed, message = health_checker.check_configuration()

        assert passed is True
        assert message == "Configuration is valid"
        mock_get_config.assert_called_once_with(validate=True)

    @patch("src.config.trading_config.get_config")
    def test_check_configuration_invalid(self, mock_get_config, health_checker):
        """Test configuration check with invalid config"""
        from src.config.exceptions import ConfigurationError

        mock_get_config.side_effect = ConfigurationError("Invalid config")

        passed, message = health_checker.check_configuration()

        assert passed is False
        assert message == "Configuration invalid"

    @patch("src.config.trading_config.get_config")
    def test_check_configuration_general_error(self, mock_get_config, health_checker):
        """Test configuration check with general error"""
        mock_get_config.side_effect = Exception("Config error")

        passed, message = health_checker.check_configuration()

        assert passed is False
        assert message == "Configuration check failed"

    @patch("shutil.disk_usage")
    def test_check_disk_space_ok(self, mock_disk_usage, health_checker):
        """Test disk space check with sufficient space"""
        # 100GB total, 20GB free (20%)
        mock_disk_usage.return_value = (100 * 2**30, 80 * 2**30, 20 * 2**30)

        passed, message = health_checker.check_disk_space()

        assert passed is True
        assert "Disk space OK: 20GB (20.0%)" in message

    @patch("shutil.disk_usage")
    def test_check_disk_space_low_warning(self, mock_disk_usage, health_checker):
        """Test disk space check with low space warning"""
        # 100GB total, 8GB free (8%)
        mock_disk_usage.return_value = (100 * 2**30, 92 * 2**30, 8 * 2**30)

        passed, message = health_checker.check_disk_space()

        assert passed is True
        assert "Disk space OK: 8GB (8.0%)" in message
        assert "Disk space getting low: 8GB" in health_checker.warnings

    @patch("shutil.disk_usage")
    def test_check_disk_space_critical(self, mock_disk_usage, health_checker):
        """Test disk space check with critical low space"""
        # 100GB total, 3GB free (3%)
        mock_disk_usage.return_value = (100 * 2**30, 97 * 2**30, 3 * 2**30)

        passed, message = health_checker.check_disk_space()

        assert passed is False
        assert "Low disk space: 3GB (3.0%)" in message

    @patch("shutil.disk_usage")
    def test_check_disk_space_error(self, mock_disk_usage, health_checker):
        """Test disk space check with error"""
        mock_disk_usage.side_effect = Exception("Disk check failed")

        passed, message = health_checker.check_disk_space()

        assert passed is False
        assert message == "Disk space check failed"

    @patch("os.path.exists")
    def test_check_data_freshness_no_cache_dir(self, mock_exists, health_checker):
        """Test data freshness check with no cache directory"""
        mock_exists.return_value = False

        passed, message = health_checker.check_data_freshness()

        assert passed is True
        assert message == "No cache directory (will fetch fresh data)"
        assert "No cached data found" in health_checker.warnings

    @patch("os.listdir")
    @patch("os.path.exists")
    def test_check_data_freshness_empty_cache(self, mock_exists, mock_listdir, health_checker):
        """Test data freshness check with empty cache"""
        mock_exists.return_value = True
        mock_listdir.return_value = []

        passed, message = health_checker.check_data_freshness()

        assert passed is True
        assert message == "Cache empty (will fetch fresh data)"
        assert "Cache directory empty" in health_checker.warnings

    @patch("os.path.getmtime")
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_check_data_freshness_fresh_cache(
        self, mock_exists, mock_listdir, mock_getmtime, health_checker
    ):
        """Test data freshness check with fresh cache"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["data1.pkl", "data2.pkl"]

        # Mock file modification times - 2 hours ago
        recent_time = datetime.now() - timedelta(hours=2)
        mock_getmtime.return_value = recent_time.timestamp()

        passed, message = health_checker.check_data_freshness()

        assert passed is True
        assert "Cache is fresh (2.0h old)" in message

    @patch("os.path.getmtime")
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_check_data_freshness_old_cache(
        self, mock_exists, mock_listdir, mock_getmtime, health_checker
    ):
        """Test data freshness check with old cache"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["data1.pkl", "data2.pkl"]

        # Mock file modification times - 30 hours ago
        old_time = datetime.now() - timedelta(hours=30)
        mock_getmtime.return_value = old_time.timestamp()

        passed, message = health_checker.check_data_freshness()

        assert passed is True
        assert "Cache exists but old (30.0h)" in message
        assert "Cached data is 30.0 hours old" in health_checker.warnings

    @patch("src.portfolio.manager.get_portfolio_manager")
    def test_check_portfolio_risk_no_positions(self, mock_get_manager, health_checker):
        """Test portfolio risk check with no positions"""
        mock_manager = Mock()
        mock_manager.get_positions.return_value = []
        mock_get_manager.return_value = mock_manager

        passed, message = health_checker.check_portfolio_risk()

        assert passed is True
        assert message == "No positions (no risk)"

    @patch("src.portfolio.manager.get_portfolio_manager")
    def test_check_portfolio_risk_with_positions(self, mock_get_manager, health_checker):
        """Test portfolio risk check with positions"""
        mock_manager = Mock()
        mock_manager.get_positions.return_value = [{"symbol": "VCB"}, {"symbol": "VIC"}]
        mock_manager.get_portfolio_value.return_value = {
            "num_positions": 2,
            "pnl_percent": 5.5,
        }
        mock_get_manager.return_value = mock_manager

        passed, message = health_checker.check_portfolio_risk()

        assert passed is True
        assert "2 positions, P&L: +5.5%" in message

    @patch("src.portfolio.manager.get_portfolio_manager")
    def test_check_portfolio_risk_large_drawdown(self, mock_get_manager, health_checker):
        """Test portfolio risk check with large drawdown"""
        mock_manager = Mock()
        mock_manager.get_positions.return_value = [{"symbol": "VCB"}]
        mock_manager.get_portfolio_value.return_value = {
            "num_positions": 1,
            "pnl_percent": -20.0,
        }
        mock_get_manager.return_value = mock_manager

        passed, message = health_checker.check_portfolio_risk()

        assert passed is True
        assert "1 positions, P&L: -20.0%" in message
        assert "Large drawdown: -20.0%" in health_checker.warnings

    @patch("src.portfolio.manager.get_portfolio_manager")
    def test_check_portfolio_risk_error(self, mock_get_manager, health_checker):
        """Test portfolio risk check with error"""
        mock_get_manager.side_effect = Exception("Portfolio error")

        passed, message = health_checker.check_portfolio_risk()

        assert passed is True  # Portfolio check is optional
        assert message == "Portfolio check skipped"
        assert "Could not check portfolio" in health_checker.warnings

    @patch("builtins.print")
    def test_run_all_checks_all_pass(self, mock_print, health_checker):
        """Test running all checks when all pass"""
        # Mock all check methods to return success
        health_checker.check_api_server = Mock(return_value=(True, "API OK"))
        health_checker.check_database = Mock(return_value=(True, "DB OK"))
        health_checker.check_models = Mock(return_value=(True, "Models OK"))
        health_checker.check_configuration = Mock(return_value=(True, "Config OK"))
        health_checker.check_disk_space = Mock(return_value=(True, "Disk OK"))
        health_checker.check_data_freshness = Mock(return_value=(True, "Cache OK"))
        health_checker.check_portfolio_risk = Mock(return_value=(True, "Portfolio OK"))

        results = health_checker.run_all_checks()

        assert results["overall_status"] == "healthy"
        assert health_checker.checks_passed == 7
        assert health_checker.checks_failed == 0

        # Verify all checks were called
        health_checker.check_api_server.assert_called_once()
        health_checker.check_database.assert_called_once()
        health_checker.check_models.assert_called_once()

    @patch("builtins.print")
    def test_run_all_checks_with_failures(self, mock_print, health_checker):
        """Test running all checks with some failures"""
        # Mock some checks to fail
        health_checker.check_api_server = Mock(return_value=(False, "API Failed"))
        health_checker.check_database = Mock(return_value=(True, "DB OK"))
        health_checker.check_models = Mock(return_value=(False, "Models Failed"))
        health_checker.check_configuration = Mock(return_value=(True, "Config OK"))
        health_checker.check_disk_space = Mock(return_value=(True, "Disk OK"))
        health_checker.check_data_freshness = Mock(return_value=(True, "Cache OK"))
        health_checker.check_portfolio_risk = Mock(return_value=(True, "Portfolio OK"))

        results = health_checker.run_all_checks()

        assert results["overall_status"] == "unhealthy"
        assert health_checker.checks_passed == 5
        assert health_checker.checks_failed == 2

    @patch("builtins.print")
    def test_run_all_checks_with_warnings(self, mock_print, health_checker):
        """Test running all checks with warnings but no failures"""
        # Mock all checks to pass but add warnings
        health_checker.check_api_server = Mock(return_value=(True, "API OK"))
        health_checker.check_database = Mock(return_value=(True, "DB OK"))
        health_checker.check_models = Mock(return_value=(True, "Models OK"))
        health_checker.check_configuration = Mock(return_value=(True, "Config OK"))
        health_checker.check_disk_space = Mock(return_value=(True, "Disk OK"))
        health_checker.check_data_freshness = Mock(return_value=(True, "Cache OK"))
        health_checker.check_portfolio_risk = Mock(return_value=(True, "Portfolio OK"))

        # Add a warning
        health_checker.warnings.append("Test warning")

        results = health_checker.run_all_checks()

        assert results["overall_status"] == "warning"
        assert health_checker.checks_passed == 7
        assert health_checker.checks_failed == 0

    @patch("builtins.print")
    def test_run_all_checks_with_exception(self, mock_print, health_checker):
        """Test running all checks when one throws exception"""
        # Mock one check to throw exception
        health_checker.check_api_server = Mock(side_effect=Exception("Unexpected error"))
        health_checker.check_database = Mock(return_value=(True, "DB OK"))
        health_checker.check_models = Mock(return_value=(True, "Models OK"))
        health_checker.check_configuration = Mock(return_value=(True, "Config OK"))
        health_checker.check_disk_space = Mock(return_value=(True, "Disk OK"))
        health_checker.check_data_freshness = Mock(return_value=(True, "Cache OK"))
        health_checker.check_portfolio_risk = Mock(return_value=(True, "Portfolio OK"))

        results = health_checker.run_all_checks()

        assert results["API Server"]["passed"] is False
        assert results["API Server"]["message"] == "Error"
        assert health_checker.checks_failed == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
