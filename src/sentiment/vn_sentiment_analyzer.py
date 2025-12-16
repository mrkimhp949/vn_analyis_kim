# -*- coding: utf-8 -*-
"""
Vietnam Market Sentiment Analysis

Sentiment analysis for Vietnam stock market using:
- Vietnamese news analysis
- Social media sentiment
- Forum/community analysis
- Market sentiment indicators

Usage:
    from src.sentiment.vn_sentiment_analyzer import (
        VNSentimentAnalyzer,
        get_sentiment_analyzer,
    )
    
    analyzer = get_sentiment_analyzer()
    
    # Analyze sentiment for a stock
    result = analyzer.analyze_symbol("VNM")
    
    # Get market-wide sentiment
    market_sentiment = analyzer.get_market_sentiment()
"""

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class SentimentLevel(Enum):
    """Sentiment classification levels."""

    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class SourceType(Enum):
    """Source types for sentiment analysis."""

    NEWS = "news"
    SOCIAL = "social"
    FORUM = "forum"
    ANALYST = "analyst"
    MARKET_DATA = "market_data"


# Vietnamese sentiment keywords
VN_SENTIMENT_KEYWORDS = {
    "positive": [
        # Vietnamese positive terms
        "tăng",
        "tăng mạnh",
        "tăng trần",
        "khởi sắc",
        "bứt phá",
        "tích cực",
        "lạc quan",
        "kỳ vọng",
        "triển vọng tốt",
        "hấp dẫn",
        "mua vào",
        "tích lũy",
        "đầu tư",
        "cơ hội",
        "tiềm năng",
        "lợi nhuận cao",
        "cổ tức hấp dẫn",
        "tăng trưởng",
        "phục hồi",
        "vượt đỉnh",
        "đột phá",
        "mạnh mẽ",
        "sôi động",
        "thanh khoản tốt",
        "nước ngoài mua ròng",
        "khối ngoại mua",
        "dòng tiền vào",
        "nâng hạng",
        "thị trường bull",
        "xu hướng tăng",
        # English terms commonly used
        "buy",
        "bullish",
        "outperform",
        "upgrade",
        "positive",
    ],
    "negative": [
        # Vietnamese negative terms
        "giảm",
        "giảm mạnh",
        "giảm sàn",
        "sụt giảm",
        "lao dốc",
        "tiêu cực",
        "bi quan",
        "lo ngại",
        "rủi ro",
        "cảnh báo",
        "bán ra",
        "thoát hàng",
        "cắt lỗ",
        "rút vốn",
        "tháo chạy",
        "thua lỗ",
        "suy yếu",
        "giảm tốc",
        "sụp đổ",
        "phá sản",
        "thủng đáy",
        "bán tháo",
        "hoảng loạn",
        "thanh khoản kém",
        "nước ngoài bán ròng",
        "khối ngoại bán",
        "dòng tiền rút",
        "thị trường bear",
        "xu hướng giảm",
        "downgrade",
        # English terms commonly used
        "sell",
        "bearish",
        "underperform",
        "downgrade",
        "negative",
    ],
    "neutral": [
        # Neutral terms
        "đi ngang",
        "sideways",
        "tích lũy",
        "cân bằng",
        "chờ đợi",
        "quan sát",
        "trung lập",
        "ổn định",
        "giữ",
        "hold",
    ],
}


# News sources configuration
VN_NEWS_SOURCES = {
    "cafef": {
        "name": "CafeF",
        "base_url": "https://cafef.vn",
        "search_url": "https://cafef.vn/tim-kiem.chn?keysearch=",
        "weight": 1.0,
        "type": SourceType.NEWS,
    },
    "vietstock": {
        "name": "VietStock",
        "base_url": "https://vietstock.vn",
        "search_url": "https://vietstock.vn/tim-kiem.htm?q=",
        "weight": 1.0,
        "type": SourceType.NEWS,
    },
    "stockbiz": {
        "name": "StockBiz",
        "base_url": "https://stockbiz.vn",
        "weight": 0.8,
        "type": SourceType.NEWS,
    },
    "f319": {
        "name": "F319",
        "base_url": "https://f319.com",
        "weight": 0.6,
        "type": SourceType.FORUM,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SentimentScore:
    """Sentiment score for a single source."""

    source: str
    source_type: SourceType

    # Scores (-1 to 1)
    score: float = 0.0

    # Counts
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    # Confidence (0 to 1)
    confidence: float = 0.5

    # Timestamp
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SymbolSentiment:
    """Aggregated sentiment for a symbol."""

    symbol: str

    # Overall sentiment
    overall_score: float = 0.0  # -1 to 1
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL
    confidence: float = 0.5

    # Source breakdown
    news_score: float = 0.0
    social_score: float = 0.0
    forum_score: float = 0.0
    market_score: float = 0.0

    # Trend
    trend: str = "stable"  # improving, stable, deteriorating
    change_24h: float = 0.0
    change_7d: float = 0.0

    # Volume of mentions
    mention_count: int = 0
    mention_change_pct: float = 0.0

    # Individual source scores
    source_scores: List[SentimentScore] = field(default_factory=list)

    # Timestamps
    analyzed_at: datetime = None

    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "overall_score": self.overall_score,
            "sentiment_level": self.sentiment_level.value,
            "confidence": self.confidence,
            "trend": self.trend,
            "mention_count": self.mention_count,
            "breakdown": {
                "news": self.news_score,
                "social": self.social_score,
                "forum": self.forum_score,
                "market": self.market_score,
            },
            "analyzed_at": self.analyzed_at.isoformat(),
        }


@dataclass
class MarketSentiment:
    """Overall market sentiment."""

    # Overall
    overall_score: float = 0.0
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL

    # Index sentiments
    vnindex_sentiment: float = 0.0
    hn30_sentiment: float = 0.0

    # Sector sentiments
    sector_sentiments: Dict[str, float] = field(default_factory=dict)

    # Market indicators
    advance_decline_ratio: float = 1.0
    foreign_flow_sentiment: float = 0.0
    volume_sentiment: float = 0.0

    # Fear/Greed
    fear_greed_index: float = 50.0  # 0 = extreme fear, 100 = extreme greed

    # Trending topics
    hot_topics: List[str] = field(default_factory=list)

    # Timestamp
    analyzed_at: datetime = None

    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()


# =============================================================================
# VIETNAMESE TEXT PROCESSOR
# =============================================================================


class VietnameseTextProcessor:
    """Process Vietnamese text for sentiment analysis."""

    # Vietnamese diacritics normalization
    DIACRITICS_MAP = {
        "à": "a",
        "á": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ằ": "a",
        "ắ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ầ": "a",
        "ấ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "è": "e",
        "é": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ề": "e",
        "ế": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "ì": "i",
        "í": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ò": "o",
        "ó": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ồ": "o",
        "ố": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ờ": "o",
        "ớ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ù": "u",
        "ú": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ừ": "u",
        "ứ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ỳ": "y",
        "ý": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize Vietnamese text."""
        text = text.lower()
        # Keep diacritics for better matching
        return text.strip()

    @classmethod
    def remove_diacritics(cls, text: str) -> str:
        """Remove Vietnamese diacritics."""
        text = text.lower()
        for vn_char, latin_char in cls.DIACRITICS_MAP.items():
            text = text.replace(vn_char, latin_char)
        return text

    @classmethod
    def extract_symbols(cls, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Vietnamese stock symbols are 3 uppercase letters
        pattern = r"\b([A-Z]{3})\b"
        return list(set(re.findall(pattern, text.upper())))

    @classmethod
    def count_sentiment_keywords(cls, text: str) -> Dict[str, int]:
        """Count sentiment keywords in text."""
        text_lower = cls.normalize(text)
        text_no_diacritics = cls.remove_diacritics(text)

        counts = {"positive": 0, "negative": 0, "neutral": 0}

        for sentiment, keywords in VN_SENTIMENT_KEYWORDS.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check both with and without diacritics
                count = text_lower.count(keyword_lower)
                if count == 0:
                    count = text_no_diacritics.count(cls.remove_diacritics(keyword_lower))
                counts[sentiment] += count

        return counts


# =============================================================================
# NEWS CRAWLER INTEGRATION
# =============================================================================

# Try to import advanced news crawler
NEWS_CRAWLER_AVAILABLE = False
try:
    from src.data.vn_news_crawler import get_news_crawler, VNNewsCrawler

    NEWS_CRAWLER_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# NEWS SCRAPER
# =============================================================================


class VNNewsScraper:
    """Scrape Vietnamese financial news with advanced crawler integration."""

    def __init__(self):
        # Use advanced crawler if available
        self._advanced_crawler = None
        if NEWS_CRAWLER_AVAILABLE:
            try:
                self._advanced_crawler = get_news_crawler()
                logger.info("📰 Using advanced VN News Crawler")
            except Exception as e:
                logger.debug(f"Advanced crawler init failed: {e}")

        # Fallback to basic session
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            }
        )
        self._cache: Dict[str, Tuple[datetime, List[Dict]]] = {}
        self._cache_ttl = timedelta(minutes=30)

    def search_news(
        self,
        symbol: str,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Search news for a stock symbol.

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of news articles with title, content, date
        """
        cache_key = f"{symbol}_{days}"

        # Check cache
        if cache_key in self._cache:
            cache_time, data = self._cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                return data

        articles = []

        # Try advanced crawler first (preferred)
        if self._advanced_crawler is not None:
            try:
                crawled_articles = self._advanced_crawler.get_news_for_symbol(
                    symbol,
                    max_articles=20,
                    max_age_hours=days * 24,
                )
                for article in crawled_articles:
                    articles.append(
                        {
                            "source": article.source,
                            "title": article.title,
                            "content": article.summary or article.title,
                            "date": article.published_at,
                            "url": article.url,
                            "symbols": article.symbols,
                        }
                    )

                if articles:
                    logger.info(
                        f"📰 Got {len(articles)} articles for {symbol} via advanced crawler"
                    )
                    self._cache[cache_key] = (datetime.now(), articles)
                    return articles

            except Exception as e:
                logger.debug(f"Advanced crawler failed: {e}")

        # Fallback: Try CafeF
        try:
            cafef_articles = self._scrape_cafef(symbol)
            articles.extend(cafef_articles)
        except Exception as e:
            logger.debug(f"CafeF scrape failed: {e}")

        # Try VietStock
        try:
            vietstock_articles = self._scrape_vietstock(symbol)
            articles.extend(vietstock_articles)
        except Exception as e:
            logger.debug(f"VietStock scrape failed: {e}")

        # Cache results
        self._cache[cache_key] = (datetime.now(), articles)

        return articles

    def _scrape_cafef(self, symbol: str) -> List[Dict]:
        """Scrape CafeF for symbol news."""
        articles = []

        try:
            # Note: This is a simplified implementation
            # Real implementation would need proper HTML parsing
            url = f"https://cafef.vn/tim-kiem.chn?keysearch={symbol}"

            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                # Extract article titles and summaries from HTML
                # This is simplified - would use BeautifulSoup in production
                content = response.text

                # Look for article patterns
                title_pattern = r"<h3[^>]*>(.*?)</h3>"
                titles = re.findall(title_pattern, content)[:10]

                for title in titles:
                    # Clean HTML tags
                    title = re.sub(r"<[^>]+>", "", title).strip()
                    if title and symbol.upper() in title.upper():
                        articles.append(
                            {
                                "source": "cafef",
                                "title": title,
                                "content": title,  # Use title as content for now
                                "date": datetime.now(),
                            }
                        )

        except Exception as e:
            logger.debug(f"CafeF error: {e}")

        return articles

    def _scrape_vietstock(self, symbol: str) -> List[Dict]:
        """Scrape VietStock for symbol news."""
        articles = []

        try:
            url = f"https://vietstock.vn/tim-kiem.htm?q={symbol}"

            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                content = response.text

                # Similar pattern matching
                title_pattern = r'title="([^"]+)"'
                titles = re.findall(title_pattern, content)[:10]

                for title in titles:
                    if symbol.upper() in title.upper():
                        articles.append(
                            {
                                "source": "vietstock",
                                "title": title,
                                "content": title,
                                "date": datetime.now(),
                            }
                        )

        except Exception as e:
            logger.debug(f"VietStock error: {e}")

        return articles

    def get_market_headlines(self, limit: int = 20) -> List[Dict]:
        """Get general market headlines."""
        headlines = []

        # Try advanced crawler first
        if self._advanced_crawler is not None:
            try:
                latest_news = self._advanced_crawler.get_latest_news(max_articles=limit)
                for article in latest_news:
                    headlines.append(
                        {
                            "source": article.source,
                            "title": article.title,
                            "date": article.published_at,
                            "url": article.url,
                        }
                    )

                if headlines:
                    logger.info(f"📰 Got {len(headlines)} market headlines via advanced crawler")
                    return headlines

            except Exception as e:
                logger.debug(f"Advanced crawler headlines failed: {e}")

        # Fallback to basic scraping
        try:
            url = "https://cafef.vn/thi-truong-chung-khoan.chn"
            response = self._session.get(url, timeout=10)

            if response.status_code == 200:
                content = response.text

                # Extract headlines
                title_pattern = r'<h3[^>]*>.*?<a[^>]*title="([^"]+)"'
                titles = re.findall(title_pattern, content)[:limit]

                for title in titles:
                    headlines.append(
                        {
                            "source": "cafef",
                            "title": title,
                            "date": datetime.now(),
                        }
                    )

        except Exception as e:
            logger.debug(f"Headlines fetch error: {e}")

        return headlines

    def get_trending_symbols(self, top_n: int = 10) -> List[tuple]:
        """Get trending stock symbols from news."""
        if self._advanced_crawler is not None:
            try:
                return self._advanced_crawler.get_trending_symbols(top_n)
            except Exception as e:
                logger.debug(f"Trending symbols failed: {e}")
        return []


# =============================================================================
# MARKET DATA SENTIMENT
# =============================================================================


class MarketDataSentiment:
    """Calculate sentiment from market data."""

    @staticmethod
    def calculate_from_price_action(
        df: pd.DataFrame,
        period: int = 20,
    ) -> float:
        """
        Calculate sentiment from price action.

        Returns score from -1 (bearish) to 1 (bullish).
        """
        if len(df) < period:
            return 0.0

        close = df["close"].iloc[-period:]

        # Calculate momentum
        returns = close.pct_change()

        # Recent trend
        trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

        # Volatility-adjusted return
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Win rate (up days)
        win_rate = (returns > 0).sum() / len(returns)

        # Combine factors
        score = np.tanh(trend * 10) * 0.4 + np.tanh(sharpe) * 0.3 + (win_rate - 0.5) * 2 * 0.3

        return max(-1, min(1, score))

    @staticmethod
    def calculate_from_volume(
        df: pd.DataFrame,
        period: int = 20,
    ) -> float:
        """
        Calculate sentiment from volume patterns.

        Returns score from -1 (bearish) to 1 (bullish).
        """
        if len(df) < period:
            return 0.0

        recent = df.iloc[-period:]

        # Up days vs down days volume
        up_volume = recent[recent["close"] > recent["open"]]["volume"].sum()
        down_volume = recent[recent["close"] < recent["open"]]["volume"].sum()

        total = up_volume + down_volume
        if total == 0:
            return 0.0

        # Volume sentiment
        score = (up_volume - down_volume) / total

        return max(-1, min(1, score))

    @staticmethod
    def calculate_advance_decline(
        advances: int,
        declines: int,
        unchanged: int = 0,
    ) -> float:
        """Calculate advance/decline sentiment."""
        total = advances + declines + unchanged
        if total == 0:
            return 0.0

        return (advances - declines) / total


# =============================================================================
# MAIN ANALYZER
# =============================================================================


class VNSentimentAnalyzer:
    """
    Vietnam Market Sentiment Analyzer

    Combines multiple sources:
    - News articles
    - Social media
    - Forum discussions
    - Market data

    Usage:
        analyzer = VNSentimentAnalyzer()

        # Analyze a stock
        result = analyzer.analyze_symbol("VNM")

        # Get market sentiment
        market = analyzer.get_market_sentiment()
    """

    # Source weights
    SOURCE_WEIGHTS = {
        SourceType.NEWS: 0.30,
        SourceType.ANALYST: 0.25,
        SourceType.MARKET_DATA: 0.25,
        SourceType.FORUM: 0.10,
        SourceType.SOCIAL: 0.10,
    }

    def __init__(self):
        self._lock = RLock()

        # Initialize components
        self._text_processor = VietnameseTextProcessor()
        self._news_scraper = VNNewsScraper()
        self._market_sentiment = MarketDataSentiment()

        # Cache
        self._symbol_cache: Dict[str, SymbolSentiment] = {}
        self._cache_ttl = timedelta(minutes=15)

        # History for trend calculation
        self._sentiment_history: Dict[str, List[Tuple[datetime, float]]] = {}

        logger.info("🎭 VN Sentiment Analyzer initialized")

    def analyze_symbol(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        use_cache: bool = True,
    ) -> SymbolSentiment:
        """
        Analyze sentiment for a stock symbol.

        Args:
            symbol: Stock symbol
            df: Optional OHLCV DataFrame for market data sentiment
            use_cache: Use cached results if available

        Returns:
            SymbolSentiment with comprehensive analysis
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache and symbol in self._symbol_cache:
            cached = self._symbol_cache[symbol]
            if datetime.now() - cached.analyzed_at < self._cache_ttl:
                return cached

        with self._lock:
            source_scores = []

            # 1. News sentiment
            try:
                news_articles = self._news_scraper.search_news(symbol)
                news_score = self._analyze_text_sentiment(news_articles)
                source_scores.append(
                    SentimentScore(
                        source="news",
                        source_type=SourceType.NEWS,
                        score=news_score["score"],
                        positive_count=news_score["positive"],
                        negative_count=news_score["negative"],
                        confidence=news_score["confidence"],
                    )
                )
            except Exception as e:
                logger.debug(f"News analysis error: {e}")

            # 2. Market data sentiment
            if df is not None and len(df) >= 20:
                try:
                    price_sentiment = self._market_sentiment.calculate_from_price_action(df)
                    volume_sentiment = self._market_sentiment.calculate_from_volume(df)

                    market_score = (price_sentiment + volume_sentiment) / 2

                    source_scores.append(
                        SentimentScore(
                            source="market_data",
                            source_type=SourceType.MARKET_DATA,
                            score=market_score,
                            confidence=0.8,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Market data analysis error: {e}")

            # 3. Calculate overall sentiment
            result = self._aggregate_sentiment(symbol, source_scores)

            # 4. Calculate trend
            result = self._calculate_trend(symbol, result)

            # Cache result
            self._symbol_cache[symbol] = result

            return result

    def analyze_batch(
        self,
        symbols: List[str],
        data_provider=None,
    ) -> Dict[str, SymbolSentiment]:
        """Analyze sentiment for multiple symbols."""
        results = {}

        for symbol in symbols:
            try:
                df = None
                if data_provider:
                    try:
                        df = data_provider.get_historical_data(symbol, days=30)
                    except:
                        pass

                results[symbol] = self.analyze_symbol(symbol, df)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                results[symbol] = SymbolSentiment(symbol=symbol)

        return results

    def get_market_sentiment(self) -> MarketSentiment:
        """
        Get overall market sentiment.

        Analyzes general market headlines and indicators.
        """
        result = MarketSentiment()

        try:
            # Get market headlines
            headlines = self._news_scraper.get_market_headlines()

            if headlines:
                text_sentiment = self._analyze_text_sentiment(headlines)
                result.overall_score = text_sentiment["score"]

            # Determine sentiment level
            if result.overall_score > 0.3:
                result.sentiment_level = SentimentLevel.BULLISH
            elif result.overall_score > 0.6:
                result.sentiment_level = SentimentLevel.VERY_BULLISH
            elif result.overall_score < -0.3:
                result.sentiment_level = SentimentLevel.BEARISH
            elif result.overall_score < -0.6:
                result.sentiment_level = SentimentLevel.VERY_BEARISH
            else:
                result.sentiment_level = SentimentLevel.NEUTRAL

            # Calculate Fear/Greed Index (simplified)
            result.fear_greed_index = (result.overall_score + 1) * 50

            # Extract hot topics from headlines
            all_text = " ".join([h.get("title", "") for h in headlines])
            symbols = self._text_processor.extract_symbols(all_text)
            result.hot_topics = symbols[:10]

        except Exception as e:
            logger.warning(f"Market sentiment error: {e}")

        return result

    def get_sector_sentiment(
        self,
        sector: str,
        symbols: List[str],
    ) -> float:
        """Get sentiment for a sector."""
        scores = []

        for symbol in symbols[:10]:  # Limit to top 10
            sentiment = self.analyze_symbol(symbol)
            if sentiment.confidence > 0.3:
                scores.append(sentiment.overall_score)

        if scores:
            return np.mean(scores)
        return 0.0

    def check_sentiment_for_entry(
        self,
        symbol: str,
        side: str = "buy",
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Check if sentiment supports entry.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            df: Optional price data

        Returns:
            Dict with is_favorable, confidence_adjustment, reasons
        """
        sentiment = self.analyze_symbol(symbol, df)

        result = {
            "is_favorable": True,
            "confidence_adjustment": 0.0,
            "reasons": [],
        }

        if side == "buy":
            # For buy, want bullish sentiment
            if sentiment.sentiment_level in (SentimentLevel.BULLISH, SentimentLevel.VERY_BULLISH):
                result["confidence_adjustment"] = 0.05
                result["reasons"].append(f"Bullish sentiment ({sentiment.overall_score:.2f})")
            elif sentiment.sentiment_level in (SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH):
                result["confidence_adjustment"] = -0.10
                result["reasons"].append(f"Bearish sentiment ({sentiment.overall_score:.2f})")
                if sentiment.sentiment_level == SentimentLevel.VERY_BEARISH:
                    result["is_favorable"] = False
        else:
            # For sell/short, want bearish sentiment
            if sentiment.sentiment_level in (SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH):
                result["confidence_adjustment"] = 0.05
                result["reasons"].append(f"Bearish sentiment supports exit")
            elif sentiment.sentiment_level in (SentimentLevel.BULLISH, SentimentLevel.VERY_BULLISH):
                result["confidence_adjustment"] = -0.05
                result["reasons"].append(f"Bullish sentiment - consider holding")

        # Trend consideration
        if sentiment.trend == "improving" and side == "buy":
            result["confidence_adjustment"] += 0.02
            result["reasons"].append("Sentiment improving")
        elif sentiment.trend == "deteriorating" and side == "buy":
            result["confidence_adjustment"] -= 0.02
            result["reasons"].append("Sentiment deteriorating")

        return result

    def _analyze_text_sentiment(
        self,
        articles: List[Dict],
    ) -> Dict[str, Any]:
        """Analyze sentiment from text content."""
        total_positive = 0
        total_negative = 0
        total_neutral = 0

        for article in articles:
            text = article.get("title", "") + " " + article.get("content", "")
            counts = self._text_processor.count_sentiment_keywords(text)

            total_positive += counts["positive"]
            total_negative += counts["negative"]
            total_neutral += counts["neutral"]

        total = total_positive + total_negative + total_neutral

        if total == 0:
            return {
                "score": 0.0,
                "positive": 0,
                "negative": 0,
                "confidence": 0.3,
            }

        # Calculate score
        score = (total_positive - total_negative) / total

        # Confidence based on sample size
        confidence = min(0.9, 0.3 + (len(articles) * 0.03))

        return {
            "score": max(-1, min(1, score)),
            "positive": total_positive,
            "negative": total_negative,
            "confidence": confidence,
        }

    def _aggregate_sentiment(
        self,
        symbol: str,
        source_scores: List[SentimentScore],
    ) -> SymbolSentiment:
        """Aggregate sentiment from multiple sources."""
        result = SymbolSentiment(symbol=symbol, source_scores=source_scores)

        if not source_scores:
            return result

        # Weighted average
        total_weight = 0
        weighted_score = 0

        news_scores = []
        market_scores = []

        for score in source_scores:
            weight = self.SOURCE_WEIGHTS.get(score.source_type, 0.1) * score.confidence
            weighted_score += score.score * weight
            total_weight += weight

            if score.source_type == SourceType.NEWS:
                news_scores.append(score.score)
            elif score.source_type == SourceType.MARKET_DATA:
                market_scores.append(score.score)

        if total_weight > 0:
            result.overall_score = weighted_score / total_weight

        # Source averages
        if news_scores:
            result.news_score = np.mean(news_scores)
        if market_scores:
            result.market_score = np.mean(market_scores)

        # Determine sentiment level
        score = result.overall_score
        if score > 0.6:
            result.sentiment_level = SentimentLevel.VERY_BULLISH
        elif score > 0.2:
            result.sentiment_level = SentimentLevel.BULLISH
        elif score < -0.6:
            result.sentiment_level = SentimentLevel.VERY_BEARISH
        elif score < -0.2:
            result.sentiment_level = SentimentLevel.BEARISH
        else:
            result.sentiment_level = SentimentLevel.NEUTRAL

        # Calculate confidence
        result.confidence = np.mean([s.confidence for s in source_scores])

        # Mention count
        result.mention_count = sum(
            s.positive_count + s.negative_count + s.neutral_count for s in source_scores
        )

        return result

    def _calculate_trend(
        self,
        symbol: str,
        sentiment: SymbolSentiment,
    ) -> SymbolSentiment:
        """Calculate sentiment trend over time."""
        # Update history
        if symbol not in self._sentiment_history:
            self._sentiment_history[symbol] = []

        self._sentiment_history[symbol].append((datetime.now(), sentiment.overall_score))

        # Keep last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        self._sentiment_history[symbol] = [
            (t, s) for t, s in self._sentiment_history[symbol] if t > cutoff
        ]

        history = self._sentiment_history[symbol]

        if len(history) >= 2:
            # 24h change
            recent = [s for t, s in history if t > datetime.now() - timedelta(hours=24)]
            if recent:
                sentiment.change_24h = sentiment.overall_score - recent[0]

            # 7d change
            if len(history) > 0:
                sentiment.change_7d = sentiment.overall_score - history[0][1]

            # Trend
            if sentiment.change_24h > 0.1:
                sentiment.trend = "improving"
            elif sentiment.change_24h < -0.1:
                sentiment.trend = "deteriorating"
            else:
                sentiment.trend = "stable"

        return sentiment

    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            "cached_symbols": list(self._symbol_cache.keys()),
            "history_symbols": list(self._sentiment_history.keys()),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_analyzer_instance: Optional[VNSentimentAnalyzer] = None
_lock = RLock()


def get_sentiment_analyzer() -> VNSentimentAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer_instance
    with _lock:
        if _analyzer_instance is None:
            _analyzer_instance = VNSentimentAnalyzer()
        return _analyzer_instance


def reset_sentiment_analyzer():
    """Reset singleton."""
    global _analyzer_instance
    with _lock:
        _analyzer_instance = None


def analyze_sentiment(symbol: str) -> SymbolSentiment:
    """Convenience function to analyze sentiment."""
    return get_sentiment_analyzer().analyze_symbol(symbol)


def get_market_sentiment() -> MarketSentiment:
    """Convenience function to get market sentiment."""
    return get_sentiment_analyzer().get_market_sentiment()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING VN SENTIMENT ANALYZER")
    print("=" * 60)

    analyzer = get_sentiment_analyzer()

    # Test symbol sentiment
    print("\n📊 Symbol Sentiment Analysis:")
    print("-" * 60)

    for symbol in ["VNM", "VCB", "HPG"]:
        sentiment = analyzer.analyze_symbol(symbol)
        print(f"\n{symbol}:")
        print(f"  Overall Score: {sentiment.overall_score:+.3f}")
        print(f"  Level: {sentiment.sentiment_level.value}")
        print(f"  Confidence: {sentiment.confidence:.2%}")
        print(f"  Trend: {sentiment.trend}")
        print(f"  Mentions: {sentiment.mention_count}")

    # Test market sentiment
    print("\n🌍 Market Sentiment:")
    print("-" * 60)

    market = analyzer.get_market_sentiment()
    print(f"\n  Overall: {market.overall_score:+.3f}")
    print(f"  Level: {market.sentiment_level.value}")
    print(f"  Fear/Greed: {market.fear_greed_index:.0f}/100")
    if market.hot_topics:
        print(f"  Hot Topics: {', '.join(market.hot_topics[:5])}")

    # Test entry check
    print("\n📈 Entry Sentiment Check:")
    print("-" * 60)

    check = analyzer.check_sentiment_for_entry("VNM", "buy")
    print(f"\n  Favorable: {check['is_favorable']}")
    print(f"  Confidence Adj: {check['confidence_adjustment']:+.3f}")
    print(f"  Reasons: {check['reasons']}")

    print("\n" + "=" * 60)
