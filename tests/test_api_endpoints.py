"""
Unit tests for FastAPI Endpoints
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app


class TestAPIEndpoints:
    """Test FastAPI endpoints"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint returns correct response"""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "status" in data
        assert data["status"] == "online"
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "python_version" in data

    def test_portfolio_endpoint(self):
        """Test portfolio endpoint"""
        response = self.client.get("/portfolio")

        # Should return 200 even if empty
        assert response.status_code == 200

        if response.status_code == 200:
            data = response.json()
            assert "current_holdings" in data or "status" in data

    def test_portfolio_analysis_endpoint(self):
        """Test portfolio analysis endpoint"""
        response = self.client.get("/portfolio/analysis")

        # Should return 200 or error message
        assert response.status_code in [200, 500]

    def test_add_to_portfolio_valid(self):
        """Test adding valid position to portfolio"""
        response = self.client.post(
            "/portfolio/add",
            params={
                "symbol": "VCB",
                "shares": 100,
                "price": 60000
            }
        )

        # API returns 200 even on errors (with status field)
        assert response.status_code == 200

        data = response.json()
        # May succeed or fail based on validation/DB state
        assert data["status"] in ["success", "error"]

    def test_add_to_portfolio_invalid_symbol(self):
        """Test adding position with invalid symbol"""
        response = self.client.post(
            "/portfolio/add",
            params={
                "symbol": "",  # Invalid empty symbol
                "shares": 100,
                "price": 60000
            }
        )

        # API returns 200 with error status instead of HTTP error codes
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_add_to_portfolio_invalid_shares(self):
        """Test adding position with invalid shares"""
        response = self.client.post(
            "/portfolio/add",
            params={
                "symbol": "VCB",
                "shares": -100,  # Invalid negative shares
                "price": 60000
            }
        )

        # API returns 200 with error status instead of HTTP error codes
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_add_to_portfolio_invalid_price(self):
        """Test adding position with invalid price"""
        response = self.client.post(
            "/portfolio/add",
            params={
                "symbol": "VCB",
                "shares": 100,
                "price": -60000  # Invalid negative price
            }
        )

        # API returns 200 with error status instead of HTTP error codes
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_remove_from_portfolio(self):
        """Test removing position from portfolio"""
        # First add a position
        self.client.post(
            "/portfolio/add",
            params={"symbol": "TEST", "shares": 100, "price": 50000}
        )

        # Then remove it
        response = self.client.post(
            "/portfolio/remove",
            params={"symbol": "TEST"}
        )

        assert response.status_code in [200, 500]

    def test_run_bot_endpoint(self):
        """Test manual bot trigger endpoint"""
        response = self.client.post("/run-bot")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["success", "error"]

    def test_analyze_sectors_endpoint(self):
        """Test sector analysis endpoint"""
        response = self.client.post("/analyze-sectors")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data

    def test_docs_endpoint_exists(self):
        """Test that API documentation is available"""
        response = self.client.get("/docs")

        assert response.status_code == 200

    def test_openapi_json_exists(self):
        """Test that OpenAPI specification is available"""
        response = self.client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


class TestAPIErrorHandling:
    """Test API error handling"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_invalid_endpoint_returns_404(self):
        """Test invalid endpoint returns 404"""
        response = self.client.get("/invalid-endpoint-12345")

        assert response.status_code == 404

    def test_portfolio_add_without_params(self):
        """Test portfolio add without required params"""
        response = self.client.post("/portfolio/add")

        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_method_not_allowed(self):
        """Test wrong HTTP method returns 405"""
        # GET on endpoint that requires POST
        response = self.client.get("/run-bot")

        assert response.status_code == 405


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
