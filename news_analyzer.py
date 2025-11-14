import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

try:
    from config import TICKERS
except ImportError:
    TICKERS = []

try:
    from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

    SENTIMENT_ANALYZER = VietnameseSentimentAnalyzer()
except Exception:
    SENTIMENT_ANALYZER = None

NEWS_CACHE_FILE = "news_cache.json"
DEFAULT_LOOKBACK_HOURS = 48

RSS_SOURCES = [
    {
        "name": "VnExpress Chứng khoán",
        "url": "https://vnexpress.net/rss/kinh-doanh/chung-khoan.rss",
        "type": "domestic",
    },
    {
        "name": "CafeF",
        "url": "https://cafef.vn/trang-chu.rss",
        "type": "domestic",
    },
    {
        "name": "Vietstock",
        "url": "https://vietstock.vn/feed/chung-khoan.rss",
        "type": "domestic",
    },
    # International sources with Vietnamese keyword filtering
    {
        "name": "Bloomberg",
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "type": "international",
        "vietnam_keywords": [
            "vietnam",
            "viet nam",
            "ho chi minh",
            "hanoi",
            "vietnamese",
            "vnd",
            "dong",
        ],
    },
    {
        "name": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "type": "international",
        "vietnam_keywords": [
            "vietnam",
            "viet nam",
            "ho chi minh",
            "hanoi",
            "vietnamese",
            "vnd",
            "dong",
            "southeast asia",
        ],
    },
]

POSITIVE_KEYWORDS = [
    "tăng trưởng",
    "vượt",
    "lợi nhuận",
    "kỷ lục",
    "mua vào",
    "khuyến nghị mua",
    "tích cực",
    "bứt phá",
    "cao nhất",
    "thuận lợi",
    "tăng mạnh",
]

NEGATIVE_KEYWORDS = [
    "giảm",
    "lỗ",
    "bán ra",
    "khuyến nghị bán",
    "tiêu cực",
    "điều tra",
    "phạt",
    "rủi ro",
    "tụt",
    "thua lỗ",
    "suy giảm",
]

# Topic classification keywords
TOPIC_KEYWORDS = {
    "dividend": [
        "cổ tức",
        "chia cổ tức",
        "dividend",
        "payout",
        "cổ đông",
        "shareholder",
        "thưởng cổ phiếu",
        "stock dividend",
        "tiền mặt",
        "cash dividend",
    ],
    "litigation": [
        "kiện",
        "tố tụng",
        "litigation",
        "lawsuit",
        "tòa án",
        "court",
        "pháp lý",
        "tranh chấp",
        "dispute",
        "vi phạm",
        "violation",
        "phạt",
        "fine",
        "penalty",
    ],
    "macro": [
        "vĩ mô",
        "macro",
        "GDP",
        "lạm phát",
        "inflation",
        "lãi suất",
        "interest rate",
        "ngân hàng trung ương",
        "central bank",
        "chính sách",
        "policy",
        "kinh tế",
        "economy",
        "tăng trưởng",
        "growth",
        "thuế",
        "tax",
        "xuất khẩu",
        "export",
    ],
    "earnings": [
        "lợi nhuận",
        "earnings",
        "doanh thu",
        "revenue",
        "kết quả kinh doanh",
        "báo cáo tài chính",
        "financial report",
        "quý",
        "quarter",
        "năm",
        "year",
    ],
}


def _load_cache() -> Dict:
    if not os.path.exists(NEWS_CACHE_FILE):
        return {"last_updated": None, "articles": {}}
    try:
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_updated": None, "articles": {}}


def _save_cache(cache: Dict) -> None:
    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _classify_topic(text: str) -> List[str]:
    """Phân loại chủ đề của bài viết"""
    text_lower = text.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    return topics if topics else ["general"]


def _parse_rss(xml_text: str, source: Dict) -> List[Dict]:
    items = []
    source_name = source.get("name", "Unknown")
    source_type = source.get("type", "domestic")
    vietnam_keywords = source.get("vietnam_keywords", [])

    try:
        from xml.etree import ElementTree as ET

        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            description = item.findtext("description", default="").strip()
            pub_date = item.findtext("pubDate", default="")
            link = item.findtext("link", default="")

            try:
                published = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
            except Exception:
                published = datetime.utcnow()

            # Filter international sources by Vietnamese keywords
            if source_type == "international" and vietnam_keywords:
                full_text = f"{title} {description}".lower()
                if not any(
                    keyword.lower() in full_text for keyword in vietnam_keywords
                ):
                    continue  # Skip if no Vietnam-related keywords

            # Classify topic
            full_text = f"{title} {description}"
            topics = _classify_topic(full_text)

            items.append(
                {
                    "source": source_name,
                    "source_type": source_type,
                    "title": title,
                    "summary": re.sub("<[^<]+?>", "", description),
                    "published_at": published.isoformat(),
                    "url": link,
                    "topics": topics,
                }
            )
    except Exception:
        pass
    return items


def _fetch_source(source: Dict) -> List[Dict]:
    try:
        resp = requests.get(
            source["url"],
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        resp.raise_for_status()
        return _parse_rss(resp.text, source)
    except Exception:
        return []


def _extract_symbols(text: str, whitelist: Optional[List[str]] = None) -> List[str]:
    if not text:
        return []
    whitelist_set = {sym.upper() for sym in (whitelist or TICKERS)}
    pattern = r"\b[A-Z]{3,5}\b"
    matches = set(re.findall(pattern, text.upper()))
    return sorted(list(matches & whitelist_set))


def _score_sentiment(texts: List[str]) -> float:
    if SENTIMENT_ANALYZER:
        return SENTIMENT_ANALYZER.score(texts)

    score = 0.0
    for text in texts:
        text_lower = text.lower()
        for word in POSITIVE_KEYWORDS:
            if word in text_lower:
                score += 1.0
        for word in NEGATIVE_KEYWORDS:
            if word in text_lower:
                score -= 1.0
    return score / max(len(texts), 1)


def update_news_cache(
    symbols: Optional[List[str]] = None, lookback_hours: int = DEFAULT_LOOKBACK_HOURS
) -> Dict:
    """
    Fetch latest news, update cache and return it.
    Tracks source frequency and identifies hot news.
    """
    symbols = symbols or TICKERS
    cache = _load_cache()
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    if "articles" not in cache:
        cache["articles"] = {}
    if "source_frequency" not in cache:
        cache["source_frequency"] = {}
    if "hot_news" not in cache:
        cache["hot_news"] = []

    fetched_articles = []
    source_counts = {}

    for source in RSS_SOURCES:
        articles = _fetch_source(source)
        fetched_articles.extend(articles)
        source_name = source.get("name", "Unknown")
        source_counts[source_name] = len(articles)
        # Update source frequency
        cache["source_frequency"][source_name] = {
            "count": source_counts[source_name],
            "last_updated": datetime.utcnow().isoformat(),
            "update_frequency": cache["source_frequency"]
            .get(source_name, {})
            .get("update_frequency", 0)
            + 1,
        }

    # Track article frequency for hot news detection
    article_titles = {}
    for article in fetched_articles:
        published = datetime.fromisoformat(article["published_at"])
        if published < cutoff:
            continue

        summary_text = f"{article['title']} {article['summary']}"
        mentioned_symbols = _extract_symbols(summary_text, symbols)
        sentiment = _score_sentiment([summary_text])

        # Track title frequency (hot news indicator)
        title_key = article["title"].lower()[:100]  # First 100 chars
        article_titles[title_key] = article_titles.get(title_key, 0) + 1

        article_data = {
            "source": article.get("source", "Unknown"),
            "source_type": article.get("source_type", "domestic"),
            "title": article["title"],
            "summary": article["summary"],
            "url": article["url"],
            "published_at": article["published_at"],
            "sentiment": sentiment,
            "topics": article.get("topics", ["general"]),
            "frequency": article_titles[
                title_key
            ],  # How many times similar title appears
        }

        # Mark as hot news if frequency > 2 or high sentiment
        is_hot = article_titles[title_key] >= 2 or abs(sentiment) > 0.7
        if is_hot:
            article_data["is_hot"] = True
            cache["hot_news"].append(
                {
                    "symbol": mentioned_symbols[0] if mentioned_symbols else "_market",
                    "title": article["title"],
                    "sentiment": sentiment,
                    "published_at": article["published_at"],
                    "url": article["url"],
                }
            )

        if not mentioned_symbols:
            cache["articles"].setdefault("_market", []).append(article_data)
        else:
            for sym in mentioned_symbols:
                cache["articles"].setdefault(sym, []).append(article_data)

    # Clean old entries
    for sym, articles in list(cache["articles"].items()):
        filtered = []
        for article in articles:
            try:
                published = datetime.fromisoformat(article["published_at"])
            except Exception:
                published = cutoff
            if published >= cutoff:
                filtered.append(article)
        cache["articles"][sym] = filtered
        if not filtered:
            del cache["articles"][sym]

    # Keep only recent hot news (last 24h)
    hot_news_cutoff = datetime.utcnow() - timedelta(hours=24)
    cache["hot_news"] = [
        news
        for news in cache["hot_news"]
        if datetime.fromisoformat(news["published_at"]) >= hot_news_cutoff
    ][
        :20
    ]  # Keep top 20 hot news

    cache["last_updated"] = datetime.utcnow().isoformat()
    _save_cache(cache)
    return cache


def _ensure_cache(symbols: Optional[List[str]] = None) -> Dict:
    cache = _load_cache()
    last_updated = cache.get("last_updated")
    needs_refresh = True
    if last_updated:
        try:
            last_time = datetime.fromisoformat(last_updated)
            needs_refresh = datetime.utcnow() - last_time > timedelta(hours=2)
        except Exception:
            needs_refresh = True
    if needs_refresh:
        cache = update_news_cache(symbols=symbols)
    return cache


def analyze_news_trend(
    symbol: str, lookback_hours: int = DEFAULT_LOOKBACK_HOURS
) -> Dict:
    """Return sentiment summary for a symbol."""
    cache = _ensure_cache()
    articles = cache.get("articles", {}).get(symbol.upper(), [])

    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    filtered = []
    for article in articles:
        try:
            published = datetime.fromisoformat(article["published_at"])
        except Exception:
            published = cutoff
        if published >= cutoff:
            filtered.append(article)

    if not filtered:
        return {
            "symbol": symbol.upper(),
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "articles": [],
            "top_headlines": [],
        }

    sentiment_total = sum(article.get("sentiment", 0.0) for article in filtered)
    sentiment_avg = sentiment_total / max(len(filtered), 1)

    if sentiment_avg >= 1.0:
        label = "STRONGLY_POSITIVE"
    elif sentiment_avg >= 0.3:
        label = "POSITIVE"
    elif sentiment_avg <= -1.0:
        label = "STRONGLY_NEGATIVE"
    elif sentiment_avg <= -0.3:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    sorted_articles = sorted(
        filtered,
        key=lambda a: a.get("published_at", ""),
        reverse=True,
    )

    return {
        "symbol": symbol.upper(),
        "sentiment_score": round(sentiment_avg, 2),
        "sentiment_label": label,
        "articles": sorted_articles,
        "top_headlines": sorted_articles[:5],
    }


def get_top_news(symbol: str, limit: int = 3) -> List[Dict]:
    trend = analyze_news_trend(symbol)
    return trend.get("top_headlines", [])[:limit]


def format_news_brief(symbol: str, limit: int = 3) -> str:
    trend = analyze_news_trend(symbol)
    headlines = trend.get("top_headlines", [])[:limit]
    if not headlines:
        return f"Không tìm thấy tin tức đáng chú ý cho {symbol.upper()} trong {DEFAULT_LOOKBACK_HOURS}h qua."

    msg_lines = [
        f"📰 Tin tức {symbol.upper()} ({trend['sentiment_label']}, score={trend['sentiment_score']:+.2f})",
    ]
    for article in headlines:
        published = article.get("published_at", "")[:16].replace("T", " ")
        source = article.get("source", "Unknown")
        topics = article.get("topics", [])
        is_hot = article.get("is_hot", False)
        hot_indicator = "🔥 " if is_hot else ""
        topic_tags = " ".join([f"[{t}]" for t in topics if t != "general"])
        msg_lines.append(
            f"{hot_indicator}• {article['title']} ({source}, {published}) {topic_tags}"
        )
        if article.get("url"):
            msg_lines.append(f"  {article['url']}")
    return "\n".join(msg_lines)


def get_hot_news(limit: int = 10) -> List[Dict]:
    """Lấy danh sách tin nóng"""
    cache = _ensure_cache()
    return cache.get("hot_news", [])[:limit]


def get_source_statistics() -> Dict:
    """Lấy thống kê về tần suất cập nhật của các nguồn tin"""
    cache = _ensure_cache()
    return cache.get("source_frequency", {})


if __name__ == "__main__":
    print("🧪 Refreshing news cache...")
    update_news_cache()
    sample_symbol = (TICKERS or ["VNM"])[0]
    summary = analyze_news_trend(sample_symbol)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
