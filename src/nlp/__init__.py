# -*- coding: utf-8 -*-
"""
NLP Module for Vietnam Stock Market Sentiment Analysis

Components:
- sentiment_analyzer: FinBERT + PhoBERT sentiment analysis
- news_scraper: Vietnamese financial news scraping
- multimodal_fusion: LSTM + NLP fusion model

Usage:
    from src.nlp import get_sentiment_analyzer, get_sentiment_adjustment
    
    # Sentiment analysis
    analyzer = get_sentiment_analyzer()
    result = analyzer.analyze("Cổ phiếu VNM tăng mạnh")
    
    # Entry logic integration
    adjustment = get_sentiment_adjustment("VNM", df)
    confidence += adjustment["adjustment"]
"""

from typing import TYPE_CHECKING

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    from .sentiment_analyzer import SentimentAnalyzer, get_sentiment_analyzer
    from .news_scraper import NewsScraperManager, get_news_scraper
    from .multimodal_fusion import (
        MultimodalPredictor,
        get_multimodal_predictor,
        get_sentiment_adjustment,
    )

__all__ = [
    # Sentiment Analyzer
    "SentimentAnalyzer",
    "get_sentiment_analyzer",
    # News Scraper
    "NewsScraperManager",
    "get_news_scraper",
    # Multimodal Fusion
    "MultimodalPredictor",
    "get_multimodal_predictor",
    # Entry Logic Integration
    "get_sentiment_adjustment",
]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name in ("SentimentAnalyzer", "get_sentiment_analyzer"):
        from .sentiment_analyzer import SentimentAnalyzer, get_sentiment_analyzer

        return get_sentiment_analyzer if name == "get_sentiment_analyzer" else SentimentAnalyzer
    elif name in ("NewsScraperManager", "get_news_scraper"):
        from .news_scraper import NewsScraperManager, get_news_scraper

        return get_news_scraper if name == "get_news_scraper" else NewsScraperManager
    elif name in ("MultimodalPredictor", "get_multimodal_predictor", "get_sentiment_adjustment"):
        from .multimodal_fusion import (
            MultimodalPredictor,
            get_multimodal_predictor,
            get_sentiment_adjustment,
        )

        if name == "get_multimodal_predictor":
            return get_multimodal_predictor
        elif name == "get_sentiment_adjustment":
            return get_sentiment_adjustment
        return MultimodalPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
