# -*- coding: utf-8 -*-
"""Tests for NLP module (Phase 2/3)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "open": np.random.uniform(20, 25, 50),
            "high": np.random.uniform(25, 30, 50),
            "low": np.random.uniform(18, 22, 50),
            "close": np.random.uniform(20, 28, 50),
            "volume": np.random.uniform(1000000, 5000000, 50),
        },
        index=dates,
    )


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    def test_import(self):
        """Test module import."""
        from src.nlp.sentiment_analyzer import SentimentAnalyzer, get_sentiment_analyzer

        assert SentimentAnalyzer is not None
        assert get_sentiment_analyzer is not None

    def test_singleton(self):
        """Test singleton pattern."""
        from src.nlp.sentiment_analyzer import get_sentiment_analyzer

        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()
        assert analyzer1 is analyzer2

    def test_preprocessor(self):
        """Test Vietnamese text preprocessor."""
        from src.nlp.sentiment_analyzer import VietnameseTextPreprocessor

        preprocessor = VietnameseTextPreprocessor()

        # Test basic preprocessing
        text = "  Cổ phiếu VNM tăng mạnh!  "
        result = preprocessor.preprocess(text)
        assert "VNM" in result
        assert result == result.strip()

    def test_keyword_extraction(self):
        """Test keyword extraction."""
        from src.nlp.sentiment_analyzer import VietnameseTextPreprocessor

        preprocessor = VietnameseTextPreprocessor()

        text = "Lợi nhuận tăng trưởng mạnh, triển vọng tích cực"
        pos_kw, neg_kw = preprocessor.extract_keywords(text)

        assert len(pos_kw) > 0
        assert "lợi nhuận" in pos_kw or "tăng trưởng" in pos_kw

    def test_analyze_vietnamese(self):
        """Test Vietnamese sentiment analysis."""
        from src.nlp.sentiment_analyzer import get_sentiment_analyzer

        analyzer = get_sentiment_analyzer()

        # Positive text
        result = analyzer.analyze("Cổ phiếu tăng trưởng mạnh, lợi nhuận kỷ lục", language="vi")
        assert result.sentiment in ["VERY_POSITIVE", "POSITIVE", "NEUTRAL"]
        assert -1 <= result.score <= 1

    def test_analyze_empty(self):
        """Test empty text handling."""
        from src.nlp.sentiment_analyzer import get_sentiment_analyzer

        analyzer = get_sentiment_analyzer()

        result = analyzer.analyze("")
        assert result.sentiment == "NEUTRAL"
        assert result.confidence == 0.0


class TestNewsScraper:
    """Tests for NewsScraper."""

    def test_import(self):
        """Test module import."""
        from src.nlp.news_scraper import NewsScraperManager, get_news_scraper

        assert NewsScraperManager is not None
        assert get_news_scraper is not None

    def test_singleton(self):
        """Test singleton pattern."""
        from src.nlp.news_scraper import get_news_scraper

        scraper1 = get_news_scraper()
        scraper2 = get_news_scraper()
        assert scraper1 is scraper2

    def test_generate_id(self):
        """Test article ID generation."""
        from src.nlp.news_scraper import CafeFScraper

        scraper = CafeFScraper()

        id1 = scraper._generate_id("https://example.com/article1")
        id2 = scraper._generate_id("https://example.com/article2")

        assert id1 != id2
        assert len(id1) == 12

    def test_extract_symbols(self):
        """Test symbol extraction from text."""
        from src.nlp.news_scraper import CafeFScraper

        scraper = CafeFScraper()

        text = "Cổ phiếu VNM và HPG tăng mạnh trong phiên"
        symbols = scraper._extract_symbols(text)

        assert "VNM" in symbols
        assert "HPG" in symbols


class TestMultimodalFusion:
    """Tests for MultimodalFusion."""

    def test_import(self):
        """Test module import."""
        from src.nlp.multimodal_fusion import (
            MultimodalPredictor,
            get_multimodal_predictor,
            get_sentiment_adjustment,
        )

        assert MultimodalPredictor is not None
        assert get_multimodal_predictor is not None
        assert get_sentiment_adjustment is not None

    def test_singleton(self):
        """Test singleton pattern."""
        from src.nlp.multimodal_fusion import get_multimodal_predictor

        predictor1 = get_multimodal_predictor()
        predictor2 = get_multimodal_predictor()
        assert predictor1 is predictor2

    def test_multimodal_input(self):
        """Test MultimodalInput dataclass."""
        from src.nlp.multimodal_fusion import MultimodalInput

        input_data = MultimodalInput(
            symbol="VNM",
            price_sequence=np.zeros((20, 5)),
            technical_features=np.zeros(41),
            sentiment_score=0.5,
            sentiment_confidence=0.8,
        )

        assert input_data.symbol == "VNM"
        assert input_data.sentiment_score == 0.5

    def test_multimodal_prediction(self):
        """Test MultimodalPrediction dataclass."""
        from src.nlp.multimodal_fusion import MultimodalPrediction

        prediction = MultimodalPrediction(
            symbol="VNM",
            probability=0.65,
            signal="BUY",
            confidence=70,
            price_contribution=0.5,
            sentiment_contribution=0.2,
            technical_contribution=0.3,
            sentiment_score=0.4,
            news_count=5,
        )

        assert prediction.signal == "BUY"
        assert prediction.probability == 0.65

        # Test to_dict
        d = prediction.to_dict()
        assert d["symbol"] == "VNM"
        assert d["signal"] == "BUY"

    def test_predict_fallback(self, sample_df):
        """Test prediction with fallback mode (no PyTorch)."""
        from src.nlp.multimodal_fusion import MultimodalPredictor

        predictor = MultimodalPredictor()
        prediction = predictor.predict("VNM", sample_df)

        assert prediction.symbol == "VNM"
        assert prediction.signal in ["BUY", "SELL", "HOLD"]
        assert 0 <= prediction.probability <= 1
        assert 0 <= prediction.confidence <= 100

    def test_get_sentiment_adjustment(self, sample_df):
        """Test sentiment adjustment function."""
        from src.nlp.multimodal_fusion import get_sentiment_adjustment

        result = get_sentiment_adjustment("HPG", sample_df)

        assert "adjustment" in result
        assert "sentiment" in result
        assert "score" in result
        assert "confidence" in result
        assert "news_count" in result

        assert result["adjustment"] in [-20, -10, 0, 5, 10]
        assert result["sentiment"] in [
            "VERY_POSITIVE",
            "POSITIVE",
            "NEUTRAL",
            "NEGATIVE",
            "VERY_NEGATIVE",
        ]

    def test_extract_technical_features(self, sample_df):
        """Test technical feature extraction."""
        from src.nlp.multimodal_fusion import MultimodalPredictor, MultimodalConstants

        predictor = MultimodalPredictor()
        features = predictor._extract_technical_features(sample_df)

        assert len(features) == MultimodalConstants.TECHNICAL_DIM
        assert features.dtype == np.float32

    def test_normalize_price_sequence(self, sample_df):
        """Test price sequence normalization."""
        from src.nlp.multimodal_fusion import MultimodalPredictor

        predictor = MultimodalPredictor()
        price_seq = sample_df[["open", "high", "low", "close", "volume"]].values[-20:]

        normalized = predictor._normalize_price_sequence(price_seq)

        assert normalized.shape == price_seq.shape
        # First close should be normalized to 1.0
        assert abs(normalized[0, 3] - 1.0) < 0.01


class TestNLPModuleInit:
    """Tests for NLP module __init__.py."""

    def test_lazy_import(self):
        """Test lazy import functionality."""
        from src.nlp import get_sentiment_analyzer

        analyzer = get_sentiment_analyzer()
        assert analyzer is not None

    def test_all_exports(self):
        """Test __all__ exports."""
        from src import nlp

        assert hasattr(nlp, "__all__")
        assert "get_sentiment_analyzer" in nlp.__all__
        assert "get_news_scraper" in nlp.__all__
        assert "get_multimodal_predictor" in nlp.__all__
        assert "get_sentiment_adjustment" in nlp.__all__
