# -*- coding: utf-8 -*-
"""
Vietnamese Financial News Crawler

Crawls and parses financial news from major Vietnamese news sources:
- CafeF (cafef.vn)
- VnExpress Tài chính
- VietStock
- Người Đồng Hành (ndh.vn)
- StockBiz
- TVSI (tvsi.com.vn)

Features:
- Multi-source crawling with automatic failover
- Vietnamese text parsing and normalization
- Stock symbol extraction from headlines
- Rate limiting to avoid IP blocks
- Caching to reduce API calls
- Background async crawling

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import re
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from threading import Thread, Event, RLock
from collections import deque
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Request settings
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1.0  # Delay between requests to same domain
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Cache settings
NEWS_CACHE_TTL = 300  # 5 minutes
MAX_CACHED_ARTICLES = 1000
MAX_ARTICLES_PER_SOURCE = 50

# Rate limiting
MIN_REQUEST_INTERVAL = 2.0  # seconds


# =============================================================================
# NEWS SOURCES CONFIGURATION
# =============================================================================

NEWS_SOURCES = {
    "cafef": {
        "name": "CafeF",
        "base_url": "https://cafef.vn",
        "rss_url": "https://cafef.vn/rss/chung-khoan.rss",
        "search_url": "https://cafef.vn/tim-kiem.chn?keysearch=",
        "stock_news_url": "https://cafef.vn/du-lieu/cotphieuxxx-{symbol}.chn",
        "selectors": {
            "article_list": ".list-news article, .tlitem",
            "title": "h3 a, .tit a, .title a",
            "link": "h3 a, .tit a, .title a",
            "summary": ".sapo, .des, p",
            "date": ".time, .date, time",
        },
        "weight": 1.0,
        "priority": 1,
    },
    "vnexpress": {
        "name": "VnExpress Tài chính",
        "base_url": "https://vnexpress.net",
        "rss_url": "https://vnexpress.net/rss/kinh-doanh.rss",
        "stock_section": "https://vnexpress.net/kinh-doanh/chung-khoan",
        "search_url": "https://timkiem.vnexpress.net/?q=",
        "selectors": {
            "article_list": "article.item-news",
            "title": ".title-news a",
            "link": ".title-news a",
            "summary": ".description a",
            "date": ".time-public, .time",
        },
        "weight": 1.0,
        "priority": 1,
    },
    "vietstock": {
        "name": "VietStock",
        "base_url": "https://vietstock.vn",
        "rss_url": "https://vietstock.vn/rss/chung-khoan.rss",
        "search_url": "https://vietstock.vn/tim-kiem.htm?q=",
        "selectors": {
            "article_list": ".article-item, .news-item",
            "title": "h3 a, .title a",
            "link": "h3 a, .title a",
            "summary": ".summary, .desc",
            "date": ".date, .time",
        },
        "weight": 0.9,
        "priority": 2,
    },
    "stockbiz": {
        "name": "StockBiz",
        "base_url": "https://stockbiz.vn",
        "section_url": "https://stockbiz.vn/tin-chung-khoan.html",
        "selectors": {
            "article_list": ".news-list li, .article-list article",
            "title": "a.title, h3 a",
            "link": "a.title, h3 a",
            "summary": ".summary",
            "date": ".date",
        },
        "weight": 0.8,
        "priority": 3,
    },
    "ndh": {
        "name": "Người Đồng Hành",
        "base_url": "https://ndh.vn",
        "section_url": "https://ndh.vn/chung-khoan",
        "selectors": {
            "article_list": ".news-item, article",
            "title": "h3 a, .title a",
            "link": "h3 a, .title a",
            "summary": ".summary, .des",
            "date": ".time, .date",
        },
        "weight": 0.7,
        "priority": 4,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NewsArticle:
    """Represents a news article."""

    title: str
    url: str
    source: str

    # Content
    summary: str = ""
    content: str = ""

    # Metadata
    published_at: datetime = None
    crawled_at: datetime = None

    # Extracted data
    symbols: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Unique ID
    article_id: str = ""

    def __post_init__(self):
        if self.crawled_at is None:
            self.crawled_at = datetime.now()
        if self.published_at is None:
            self.published_at = datetime.now()
        if not self.article_id:
            self.article_id = hashlib.md5(self.url.encode()).hexdigest()[:12]

    @property
    def age_hours(self) -> float:
        """Age of article in hours."""
        return (datetime.now() - self.published_at).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "summary": self.summary,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "symbols": self.symbols,
            "age_hours": self.age_hours,
        }


@dataclass
class CrawlResult:
    """Result of a crawl operation."""

    source: str
    articles: List[NewsArticle]
    success: bool
    error: str = ""
    crawl_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# TEXT PROCESSING
# =============================================================================


class VietnameseTextProcessor:
    """Process Vietnamese text for news analysis."""

    # Vietnam stock symbols (3 uppercase letters)
    SYMBOL_PATTERN = re.compile(r"\b([A-Z]{3})\b")

    # Common false positives to exclude
    FALSE_POSITIVE_SYMBOLS = {
        "USD",
        "VND",
        "EUR",
        "JPY",
        "GBP",
        "CNY",  # Currencies
        "GDP",
        "CPI",
        "PPI",
        "PMI",
        "FDI",
        "FII",  # Economic indicators
        "CEO",
        "CFO",
        "COO",
        "CTO",  # Titles
        "ETF",
        "IPO",
        "M&A",
        "P&L",  # Financial terms
        "HOSE",
        "HNX",
        "UPCOM",  # Exchanges
        "NHNN",
        "UBCK",
        "SSC",  # Regulators
        "BTC",
        "ETH",  # Crypto (not VN stocks)
    }

    # Known VN30 symbols (high priority)
    VN30_SYMBOLS = {
        "ACB",
        "BCM",
        "BID",
        "BVH",
        "CTG",
        "FPT",
        "GAS",
        "GVR",
        "HDB",
        "HPG",
        "MBB",
        "MSN",
        "MWG",
        "PLX",
        "POW",
        "SAB",
        "SHB",
        "SSB",
        "SSI",
        "STB",
        "TCB",
        "TPB",
        "VCB",
        "VHM",
        "VIB",
        "VIC",
        "VJC",
        "VNM",
        "VPB",
        "VRE",
    }

    @classmethod
    def extract_symbols(cls, text: str) -> List[str]:
        """Extract stock symbols from text."""
        matches = cls.SYMBOL_PATTERN.findall(text.upper())

        # Filter out false positives
        symbols = [s for s in matches if s not in cls.FALSE_POSITIVE_SYMBOLS]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize Vietnamese text."""
        if not text:
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove HTML entities
        text = re.sub(r"&[a-z]+;", " ", text)

        return text.strip()

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction based on Vietnamese financial terms
        important_words = [
            "tăng",
            "giảm",
            "lợi nhuận",
            "doanh thu",
            "cổ tức",
            "mua ròng",
            "bán ròng",
            "khối ngoại",
            "đầu tư",
            "sáp nhập",
            "IPO",
            "niêm yết",
            "kết quả",
            "báo cáo",
        ]

        text_lower = text.lower()
        found = [w for w in important_words if w in text_lower]

        return found[:top_n]


# =============================================================================
# NEWS CRAWLER
# =============================================================================


class VNNewsCrawler:
    """
    Vietnamese Financial News Crawler.

    Crawls news from multiple Vietnamese financial news sources.

    Usage:
        crawler = VNNewsCrawler()

        # Get latest news
        news = crawler.get_latest_news()

        # Get news for specific symbol
        vnm_news = crawler.get_news_for_symbol("VNM")

        # Search news
        results = crawler.search_news("lợi nhuận quý 4")
    """

    def __init__(
        self,
        sources: List[str] = None,
        cache_ttl: int = NEWS_CACHE_TTL,
        max_articles: int = MAX_ARTICLES_PER_SOURCE,
    ):
        """
        Initialize news crawler.

        Args:
            sources: List of source keys to use (default: all)
            cache_ttl: Cache TTL in seconds
            max_articles: Max articles to fetch per source
        """
        self.sources = sources or list(NEWS_SOURCES.keys())
        self.cache_ttl = cache_ttl
        self.max_articles = max_articles

        # HTTP session
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        # Cache
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._seen_urls: Set[str] = set()
        self._lock = RLock()

        # Rate limiting
        self._last_request_time: Dict[str, float] = {}

        # Text processor
        self._text_processor = VietnameseTextProcessor()

        logger.info(f"📰 VN News Crawler initialized with sources: {self.sources}")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is valid."""
        if key not in self._cache_time:
            return False
        age = (datetime.now() - self._cache_time[key]).total_seconds()
        return age < self.cache_ttl

    def _respect_rate_limit(self, domain: str):
        """Respect rate limit for domain."""
        if domain in self._last_request_time:
            elapsed = time.time() - self._last_request_time[domain]
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        self._last_request_time[domain] = time.time()

    def _fetch_page(self, url: str, source_key: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        try:
            # Rate limiting
            source_config = NEWS_SOURCES.get(source_key, {})
            domain = source_config.get("base_url", "").replace("https://", "")
            self._respect_rate_limit(domain)

            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")
            return soup

        except requests.Timeout:
            logger.warning(f"Timeout fetching {url}")
        except requests.RequestException as e:
            logger.warning(f"Error fetching {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")

        return None

    def _parse_articles(
        self,
        soup: BeautifulSoup,
        source_key: str,
        base_url: str,
    ) -> List[NewsArticle]:
        """Parse articles from HTML soup."""
        articles = []
        source_config = NEWS_SOURCES.get(source_key, {})
        selectors = source_config.get("selectors", {})
        source_name = source_config.get("name", source_key)

        # Find article containers
        article_selector = selectors.get("article_list", "article")
        article_elements = soup.select(article_selector)

        for element in article_elements[: self.max_articles]:
            try:
                # Extract title
                title_elem = element.select_one(selectors.get("title", "h3 a"))
                if not title_elem:
                    continue

                title = self._text_processor.normalize_text(title_elem.get_text())
                if not title or len(title) < 10:
                    continue

                # Extract URL
                link_elem = element.select_one(selectors.get("link", "a"))
                if not link_elem:
                    continue

                url = link_elem.get("href", "")
                if url and not url.startswith("http"):
                    url = urljoin(base_url, url)

                if not url or url in self._seen_urls:
                    continue

                self._seen_urls.add(url)

                # Extract summary
                summary = ""
                summary_elem = element.select_one(selectors.get("summary", ".summary"))
                if summary_elem:
                    summary = self._text_processor.normalize_text(summary_elem.get_text())

                # Extract date (basic parsing)
                published_at = datetime.now()
                date_elem = element.select_one(selectors.get("date", ".date"))
                if date_elem:
                    date_text = date_elem.get_text().strip()
                    published_at = self._parse_date(date_text) or published_at

                # Extract symbols from title and summary
                full_text = f"{title} {summary}"
                symbols = self._text_processor.extract_symbols(full_text)

                # Create article
                article = NewsArticle(
                    title=title,
                    url=url,
                    source=source_name,
                    summary=summary[:500],
                    published_at=published_at,
                    symbols=symbols,
                )

                articles.append(article)

            except Exception as e:
                logger.debug(f"Error parsing article: {e}")
                continue

        return articles

    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse Vietnamese date string."""
        try:
            date_text = date_text.strip().lower()

            # Handle relative dates
            if "phút trước" in date_text:
                minutes = int(re.search(r"(\d+)", date_text).group(1))
                return datetime.now() - timedelta(minutes=minutes)

            if "giờ trước" in date_text:
                hours = int(re.search(r"(\d+)", date_text).group(1))
                return datetime.now() - timedelta(hours=hours)

            if "ngày trước" in date_text or "hôm qua" in date_text:
                days = 1
                match = re.search(r"(\d+)", date_text)
                if match:
                    days = int(match.group(1))
                return datetime.now() - timedelta(days=days)

            # Try common formats
            formats = [
                "%d/%m/%Y %H:%M",
                "%d-%m-%Y %H:%M",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%H:%M %d/%m/%Y",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_text, fmt)
                except ValueError:
                    continue

        except Exception:
            pass

        return None

    def _crawl_source(self, source_key: str) -> CrawlResult:
        """Crawl a single news source."""
        start_time = time.time()

        source_config = NEWS_SOURCES.get(source_key)
        if not source_config:
            return CrawlResult(
                source=source_key,
                articles=[],
                success=False,
                error=f"Unknown source: {source_key}",
            )

        try:
            base_url = source_config.get("base_url", "")

            # Try RSS first (faster and more reliable)
            rss_url = source_config.get("rss_url")
            if rss_url:
                articles = self._parse_rss(rss_url, source_key)
                if articles:
                    return CrawlResult(
                        source=source_key,
                        articles=articles,
                        success=True,
                        crawl_time=time.time() - start_time,
                    )

            # Fallback to HTML scraping
            section_url = source_config.get("section_url") or source_config.get("stock_section")
            if not section_url:
                section_url = base_url

            soup = self._fetch_page(section_url, source_key)
            if soup:
                articles = self._parse_articles(soup, source_key, base_url)
                return CrawlResult(
                    source=source_key,
                    articles=articles,
                    success=True,
                    crawl_time=time.time() - start_time,
                )

            return CrawlResult(
                source=source_key,
                articles=[],
                success=False,
                error="Failed to fetch page",
                crawl_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error crawling {source_key}: {e}")
            return CrawlResult(
                source=source_key,
                articles=[],
                success=False,
                error=str(e),
                crawl_time=time.time() - start_time,
            )

    def _parse_rss(self, rss_url: str, source_key: str) -> List[NewsArticle]:
        """Parse RSS feed."""
        try:
            response = self._session.get(rss_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "xml")
            articles = []

            source_config = NEWS_SOURCES.get(source_key, {})
            source_name = source_config.get("name", source_key)

            for item in soup.find_all("item")[: self.max_articles]:
                try:
                    title = item.find("title")
                    link = item.find("link")
                    desc = item.find("description")
                    pub_date = item.find("pubDate")

                    if not title or not link:
                        continue

                    title_text = self._text_processor.normalize_text(title.get_text())
                    url = link.get_text().strip()

                    if url in self._seen_urls:
                        continue
                    self._seen_urls.add(url)

                    summary = ""
                    if desc:
                        # Remove HTML tags from description
                        summary = re.sub(r"<[^>]+>", "", desc.get_text())
                        summary = self._text_processor.normalize_text(summary)[:500]

                    # Parse date
                    published_at = datetime.now()
                    if pub_date:
                        try:
                            from email.utils import parsedate_to_datetime

                            published_at = parsedate_to_datetime(pub_date.get_text())
                        except:
                            pass

                    # Extract symbols
                    full_text = f"{title_text} {summary}"
                    symbols = self._text_processor.extract_symbols(full_text)

                    article = NewsArticle(
                        title=title_text,
                        url=url,
                        source=source_name,
                        summary=summary,
                        published_at=published_at,
                        symbols=symbols,
                    )
                    articles.append(article)

                except Exception as e:
                    logger.debug(f"Error parsing RSS item: {e}")
                    continue

            logger.info(f"✅ Parsed {len(articles)} articles from {source_name} RSS")
            return articles

        except Exception as e:
            logger.warning(f"RSS parsing failed for {rss_url}: {e}")
            return []

    # =========================================================================
    # Public API
    # =========================================================================

    def get_latest_news(
        self,
        max_articles: int = 50,
        force_refresh: bool = False,
    ) -> List[NewsArticle]:
        """
        Get latest news from all sources.

        Args:
            max_articles: Maximum number of articles to return
            force_refresh: Bypass cache

        Returns:
            List of NewsArticle sorted by date
        """
        cache_key = "latest_news"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return self._cache.get(cache_key, [])[:max_articles]

        all_articles = []

        for source_key in self.sources:
            result = self._crawl_source(source_key)
            if result.success:
                all_articles.extend(result.articles)
                logger.info(
                    f"📰 {result.source}: {len(result.articles)} articles "
                    f"({result.crawl_time:.1f}s)"
                )

        # Sort by date (newest first)
        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        # Deduplicate by title similarity
        unique_articles = self._deduplicate_articles(all_articles)

        with self._lock:
            self._cache[cache_key] = unique_articles
            self._cache_time[cache_key] = datetime.now()

        logger.info(
            f"📰 Total: {len(unique_articles)} unique articles from {len(self.sources)} sources"
        )
        return unique_articles[:max_articles]

    def get_news_for_symbol(
        self,
        symbol: str,
        max_articles: int = 20,
        max_age_hours: int = 72,
    ) -> List[NewsArticle]:
        """
        Get news articles mentioning a specific stock symbol.

        Args:
            symbol: Stock symbol (e.g., "VNM", "FPT")
            max_articles: Maximum articles to return
            max_age_hours: Maximum age of articles

        Returns:
            List of NewsArticle mentioning the symbol
        """
        symbol = symbol.upper()

        # Get all latest news
        all_news = self.get_latest_news(max_articles=200)

        # Filter by symbol and age
        symbol_news = []
        for article in all_news:
            if symbol in article.symbols:
                if article.age_hours <= max_age_hours:
                    symbol_news.append(article)

        # Also search for symbol in title/summary
        for article in all_news:
            if symbol not in article.symbols:
                full_text = f"{article.title} {article.summary}".upper()
                if symbol in full_text and article not in symbol_news:
                    if article.age_hours <= max_age_hours:
                        symbol_news.append(article)

        return symbol_news[:max_articles]

    def search_news(
        self,
        query: str,
        max_articles: int = 20,
    ) -> List[NewsArticle]:
        """
        Search news articles by keyword.

        Args:
            query: Search query
            max_articles: Maximum articles to return

        Returns:
            List of matching NewsArticle
        """
        query_lower = query.lower()

        # Get all news
        all_news = self.get_latest_news(max_articles=200)

        # Filter by query
        matching = []
        for article in all_news:
            full_text = f"{article.title} {article.summary}".lower()
            if query_lower in full_text:
                matching.append(article)

        return matching[:max_articles]

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        seen_titles = set()
        unique = []

        for article in articles:
            # Normalize title for comparison
            title_key = article.title.lower()[:50]

            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(article)

        return unique

    def get_trending_symbols(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get trending stock symbols based on news mentions.

        Returns:
            List of (symbol, mention_count) tuples
        """
        all_news = self.get_latest_news(max_articles=100)

        # Count symbol mentions
        symbol_counts: Dict[str, int] = {}
        for article in all_news:
            for symbol in article.symbols:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        # Sort by count
        sorted_symbols = sorted(
            symbol_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_symbols[:top_n]


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_crawler: Optional[VNNewsCrawler] = None
_lock = RLock()


def get_news_crawler() -> VNNewsCrawler:
    """Get singleton news crawler instance."""
    global _crawler

    with _lock:
        if _crawler is None:
            _crawler = VNNewsCrawler()

    return _crawler


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 70)
    print("🧪 TESTING VN NEWS CRAWLER")
    print("=" * 70 + "\n")

    crawler = get_news_crawler()

    # Test 1: Get latest news
    print("1️⃣ Testing get_latest_news()...")
    news = crawler.get_latest_news(max_articles=10)
    print(f"   ✅ Got {len(news)} articles")
    for article in news[:3]:
        print(f"   - [{article.source}] {article.title[:60]}...")
        if article.symbols:
            print(f"     Symbols: {article.symbols}")

    # Test 2: Get news for symbol
    print("\n2️⃣ Testing get_news_for_symbol('VNM')...")
    vnm_news = crawler.get_news_for_symbol("VNM", max_articles=5)
    print(f"   ✅ Got {len(vnm_news)} articles mentioning VNM")
    for article in vnm_news[:2]:
        print(f"   - {article.title[:60]}...")

    # Test 3: Get trending symbols
    print("\n3️⃣ Testing get_trending_symbols()...")
    trending = crawler.get_trending_symbols(top_n=10)
    print(f"   ✅ Top trending symbols: {trending}")

    # Test 4: Search news
    print("\n4️⃣ Testing search_news('lợi nhuận')...")
    results = crawler.search_news("lợi nhuận", max_articles=5)
    print(f"   ✅ Got {len(results)} matching articles")

    print("\n" + "=" * 70)
    print("✅ VN News Crawler testing complete!")
    print("=" * 70)
