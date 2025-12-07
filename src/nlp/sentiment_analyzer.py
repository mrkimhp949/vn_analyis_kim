# -*- coding: utf-8 -*-
"""
Sentiment Analyzer for Vietnam Stock Market

Phase 2: PhoBERT for Vietnamese financial news
- FinBERT for English financial text
- PhoBERT for Vietnamese native text
- Ensemble sentiment scoring

Dependencies:
    pip install transformers torch underthesea
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with availability flags
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False
UNDERTHESEA_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Install: pip install torch")

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not installed. Install: pip install transformers")

try:
    from underthesea import word_tokenize, text_normalize

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    logger.warning("Underthesea not installed. Install: pip install underthesea")


# =============================================================================
# CONSTANTS
# =============================================================================


class SentimentConstants:
    """Constants for sentiment analysis."""

    # Model names
    FINBERT_MODEL = "ProsusAI/finbert"
    PHOBERT_MODEL = "vinai/phobert-base"
    PHOBERT_SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"

    # Sentiment thresholds
    VERY_POSITIVE_THRESHOLD = 0.7
    POSITIVE_THRESHOLD = 0.3
    NEUTRAL_THRESHOLD = -0.3
    NEGATIVE_THRESHOLD = -0.7

    # Confidence adjustments for trading
    SENTIMENT_ADJUSTMENTS = {
        "VERY_POSITIVE": 10,
        "POSITIVE": 5,
        "NEUTRAL": 0,
        "NEGATIVE": -10,
        "VERY_NEGATIVE": -20,
    }

    # Vietnamese financial keywords (positive)
    VN_POSITIVE_KEYWORDS = [
        "tăng trưởng",
        "lợi nhuận",
        "doanh thu",
        "kỷ lục",
        "đột phá",
        "khởi sắc",
        "tích cực",
        "triển vọng",
        "cơ hội",
        "hấp dẫn",
        "mua vào",
        "khuyến nghị mua",
        "outperform",
        "mục tiêu",
        "cổ tức",
        "chia thưởng",
        "thoái vốn",
        "M&A",
        "hợp tác",
        "xuất khẩu",
        "đơn hàng",
        "công suất",
        "mở rộng",
        "đầu tư",
    ]

    # Vietnamese financial keywords (negative)
    VN_NEGATIVE_KEYWORDS = [
        "giảm",
        "lỗ",
        "sụt giảm",
        "thua lỗ",
        "khó khăn",
        "rủi ro",
        "cảnh báo",
        "bán ra",
        "underperform",
        "hạ mục tiêu",
        "nợ xấu",
        "thanh tra",
        "vi phạm",
        "phạt",
        "kiện",
        "phá sản",
        "tái cơ cấu",
        "cắt giảm",
        "sa thải",
        "đình công",
        "margin call",
        "bán tháo",
        "hoảng loạn",
        "sàn",
        "trần",
    ]

    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 1000


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SentimentResult:
    """Container for sentiment analysis result."""

    text: str
    sentiment: str  # VERY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, VERY_NEGATIVE
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    model_used: str
    keywords_found: List[str] = field(default_factory=list)
    adjustment: int = 0  # Confidence adjustment for trading
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "sentiment": self.sentiment,
            "score": self.score,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "keywords_found": self.keywords_found,
            "adjustment": self.adjustment,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple news articles."""

    symbol: str
    overall_sentiment: str
    overall_score: float
    num_articles: int
    positive_count: int
    negative_count: int
    neutral_count: int
    confidence_adjustment: int
    top_keywords: List[str]
    articles: List[SentimentResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# VIETNAMESE TEXT PREPROCESSOR
# =============================================================================


class VietnameseTextPreprocessor:
    """Preprocessor for Vietnamese financial text."""

    def __init__(self):
        self.use_underthesea = UNDERTHESEA_AVAILABLE

    def preprocess(self, text: str) -> str:
        """
        Preprocess Vietnamese text for sentiment analysis.

        Args:
            text: Raw Vietnamese text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Normalize unicode
        text = self._normalize_unicode(text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove special characters but keep Vietnamese
        text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Use underthesea for text normalization if available
        if self.use_underthesea:
            try:
                text = text_normalize(text)
            except Exception:
                pass

        return text.strip()

    def tokenize(self, text: str) -> str:
        """
        Tokenize Vietnamese text (word segmentation).

        Args:
            text: Preprocessed text

        Returns:
            Tokenized text with word boundaries
        """
        if not text:
            return ""

        if self.use_underthesea:
            try:
                tokens = word_tokenize(text, format="text")
                return tokens
            except Exception:
                pass

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Vietnamese unicode characters."""
        import unicodedata

        return unicodedata.normalize("NFC", text)

    def extract_keywords(
        self,
        text: str,
        positive_keywords: List[str] = None,
        negative_keywords: List[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract financial keywords from text.

        Returns:
            Tuple of (positive_keywords_found, negative_keywords_found)
        """
        if positive_keywords is None:
            positive_keywords = SentimentConstants.VN_POSITIVE_KEYWORDS
        if negative_keywords is None:
            negative_keywords = SentimentConstants.VN_NEGATIVE_KEYWORDS

        text_lower = text.lower()

        pos_found = [kw for kw in positive_keywords if kw in text_lower]
        neg_found = [kw for kw in negative_keywords if kw in text_lower]

        return pos_found, neg_found


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================


class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer for Vietnam stock market.

    Supports:
    - FinBERT for English financial text
    - PhoBERT for Vietnamese text
    - Keyword-based fallback
    - Ensemble scoring
    """

    def __init__(
        self,
        use_finbert: bool = True,
        use_phobert: bool = True,
        device: str = "auto",
    ):
        """
        Initialize sentiment analyzer.

        Args:
            use_finbert: Use FinBERT for English text
            use_phobert: Use PhoBERT for Vietnamese text
            device: Device for inference ("auto", "cpu", "cuda")
        """
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE
        self.use_phobert = use_phobert and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE

        # Determine device
        if device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Models (lazy loaded)
        self._finbert_pipeline = None
        self._phobert_pipeline = None
        self._phobert_tokenizer = None
        self._phobert_model = None

        # Preprocessor
        self.preprocessor = VietnameseTextPreprocessor()

        # Cache
        self._cache: Dict[str, SentimentResult] = {}

        logger.info(
            f"SentimentAnalyzer initialized: "
            f"FinBERT={self.use_finbert}, PhoBERT={self.use_phobert}, "
            f"device={self.device}"
        )

    # =========================================================================
    # MODEL LOADING (Lazy)
    # =========================================================================

    def _load_finbert(self):
        """Load FinBERT model."""
        if self._finbert_pipeline is not None:
            return

        if not self.use_finbert:
            return

        try:
            logger.info("Loading FinBERT model...")
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=SentimentConstants.FINBERT_MODEL,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("✅ FinBERT loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.use_finbert = False

    def _load_phobert(self):
        """Load PhoBERT sentiment model."""
        if self._phobert_pipeline is not None:
            return

        if not self.use_phobert:
            return

        try:
            logger.info("Loading PhoBERT sentiment model...")
            # Use pre-trained Vietnamese sentiment model
            self._phobert_pipeline = pipeline(
                "sentiment-analysis",
                model=SentimentConstants.PHOBERT_SENTIMENT_MODEL,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("✅ PhoBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load PhoBERT sentiment: {e}")
            # Fallback to base PhoBERT
            try:
                logger.info("Trying base PhoBERT...")
                self._phobert_tokenizer = AutoTokenizer.from_pretrained(
                    SentimentConstants.PHOBERT_MODEL
                )
                self._phobert_model = AutoModelForSequenceClassification.from_pretrained(
                    SentimentConstants.PHOBERT_MODEL,
                    num_labels=3,  # pos, neg, neu
                )
                if self.device == "cuda":
                    self._phobert_model = self._phobert_model.cuda()
                logger.info("✅ Base PhoBERT loaded")
            except Exception as e2:
                logger.error(f"Failed to load PhoBERT: {e2}")
                self.use_phobert = False

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze(
        self,
        text: str,
        language: str = "auto",
        use_cache: bool = True,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            language: "vi", "en", or "auto" (detect)
            use_cache: Use cached results

        Returns:
            SentimentResult with sentiment score and classification
        """
        if not text or not text.strip():
            return self._empty_result(text)

        # Check cache
        cache_key = hash(text[:500])
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Preprocess
        processed_text = self.preprocessor.preprocess(text)

        # Detect language if auto
        if language == "auto":
            language = self._detect_language(processed_text)

        # Analyze based on language
        if language == "vi":
            result = self._analyze_vietnamese(processed_text, text)
        else:
            result = self._analyze_english(processed_text, text)

        # Cache result
        if use_cache and len(self._cache) < SentimentConstants.MAX_CACHE_SIZE:
            self._cache[cache_key] = result

        return result

    def analyze_batch(
        self,
        texts: List[str],
        language: str = "auto",
    ) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze
            language: Language setting

        Returns:
            List of SentimentResult
        """
        return [self.analyze(text, language) for text in texts]

    def aggregate_sentiment(
        self,
        symbol: str,
        articles: List[Dict],
        text_field: str = "content",
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment from multiple news articles for a symbol.

        Args:
            symbol: Stock symbol
            articles: List of article dicts with text content
            text_field: Field name containing text

        Returns:
            AggregatedSentiment with overall sentiment
        """
        results = []
        all_keywords = []

        for article in articles:
            text = article.get(text_field, "") or article.get("title", "")
            if text:
                result = self.analyze(text)
                results.append(result)
                all_keywords.extend(result.keywords_found)

        if not results:
            return AggregatedSentiment(
                symbol=symbol,
                overall_sentiment="NEUTRAL",
                overall_score=0.0,
                num_articles=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                confidence_adjustment=0,
                top_keywords=[],
            )

        # Count sentiments
        pos_count = sum(1 for r in results if r.sentiment in ["VERY_POSITIVE", "POSITIVE"])
        neg_count = sum(1 for r in results if r.sentiment in ["VERY_NEGATIVE", "NEGATIVE"])
        neu_count = sum(1 for r in results if r.sentiment == "NEUTRAL")

        # Calculate weighted average score
        total_weight = sum(r.confidence for r in results)
        if total_weight > 0:
            overall_score = sum(r.score * r.confidence for r in results) / total_weight
        else:
            overall_score = np.mean([r.score for r in results])

        # Determine overall sentiment
        overall_sentiment = self._score_to_sentiment(overall_score)

        # Calculate confidence adjustment
        adjustment = SentimentConstants.SENTIMENT_ADJUSTMENTS.get(overall_sentiment, 0)

        # Weight by article count
        if len(results) >= 5:
            adjustment = int(adjustment * 1.2)  # More articles = more confidence
        elif len(results) <= 2:
            adjustment = int(adjustment * 0.7)  # Few articles = less confidence

        # Get top keywords
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        top_keywords = sorted(keyword_counts.keys(), key=lambda k: keyword_counts[k], reverse=True)[
            :10
        ]

        return AggregatedSentiment(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            overall_score=overall_score,
            num_articles=len(results),
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            confidence_adjustment=adjustment,
            top_keywords=top_keywords,
            articles=results,
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _analyze_vietnamese(self, processed_text: str, original_text: str) -> SentimentResult:
        """Analyze Vietnamese text sentiment."""

        # Try PhoBERT first
        if self.use_phobert:
            self._load_phobert()

            if self._phobert_pipeline is not None:
                try:
                    # Tokenize for PhoBERT
                    tokenized = self.preprocessor.tokenize(processed_text)

                    # Truncate to max length
                    tokenized = tokenized[:256]

                    result = self._phobert_pipeline(tokenized)[0]

                    # Map labels to score
                    label = result["label"].upper()
                    confidence = result["score"]

                    if "POS" in label or "POSITIVE" in label:
                        score = confidence
                    elif "NEG" in label or "NEGATIVE" in label:
                        score = -confidence
                    else:
                        score = 0.0

                    sentiment = self._score_to_sentiment(score)

                    # Extract keywords
                    pos_kw, neg_kw = self.preprocessor.extract_keywords(original_text)
                    keywords = pos_kw + neg_kw

                    return SentimentResult(
                        text=original_text,
                        sentiment=sentiment,
                        score=score,
                        confidence=confidence,
                        model_used="PhoBERT",
                        keywords_found=keywords,
                        adjustment=SentimentConstants.SENTIMENT_ADJUSTMENTS.get(sentiment, 0),
                    )
                except Exception as e:
                    logger.warning(f"PhoBERT analysis failed: {e}")

        # Fallback to keyword-based
        return self._keyword_based_analysis(original_text, "Vietnamese-Keywords")

    def _analyze_english(self, processed_text: str, original_text: str) -> SentimentResult:
        """Analyze English text sentiment."""

        # Try FinBERT
        if self.use_finbert:
            self._load_finbert()

            if self._finbert_pipeline is not None:
                try:
                    # Truncate to max length
                    truncated = processed_text[:512]

                    result = self._finbert_pipeline(truncated)[0]

                    label = result["label"].lower()
                    confidence = result["score"]

                    if label == "positive":
                        score = confidence
                    elif label == "negative":
                        score = -confidence
                    else:
                        score = 0.0

                    sentiment = self._score_to_sentiment(score)

                    return SentimentResult(
                        text=original_text,
                        sentiment=sentiment,
                        score=score,
                        confidence=confidence,
                        model_used="FinBERT",
                        keywords_found=[],
                        adjustment=SentimentConstants.SENTIMENT_ADJUSTMENTS.get(sentiment, 0),
                    )
                except Exception as e:
                    logger.warning(f"FinBERT analysis failed: {e}")

        # Fallback to keyword-based
        return self._keyword_based_analysis(original_text, "English-Keywords")

    def _keyword_based_analysis(self, text: str, model_name: str) -> SentimentResult:
        """Fallback keyword-based sentiment analysis."""

        pos_kw, neg_kw = self.preprocessor.extract_keywords(text)

        pos_count = len(pos_kw)
        neg_count = len(neg_kw)
        total = pos_count + neg_count

        if total == 0:
            score = 0.0
            confidence = 0.3
        else:
            score = (pos_count - neg_count) / total
            confidence = min(0.5 + (total * 0.05), 0.8)

        sentiment = self._score_to_sentiment(score)

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            model_used=model_name,
            keywords_found=pos_kw + neg_kw,
            adjustment=SentimentConstants.SENTIMENT_ADJUSTMENTS.get(sentiment, 0),
        )

    def _detect_language(self, text: str) -> str:
        """Detect if text is Vietnamese or English."""
        # Vietnamese-specific characters
        vn_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")

        text_lower = text.lower()
        vn_char_count = sum(1 for c in text_lower if c in vn_chars)

        # If more than 2% Vietnamese characters, consider it Vietnamese
        if len(text) > 0 and vn_char_count / len(text) > 0.02:
            return "vi"

        # Check for common Vietnamese words
        vn_words = ["của", "và", "là", "trong", "được", "có", "này", "cho", "với", "các"]
        words = text_lower.split()
        vn_word_count = sum(1 for w in words if w in vn_words)

        if len(words) > 0 and vn_word_count / len(words) > 0.1:
            return "vi"

        return "en"

    def _score_to_sentiment(self, score: float) -> str:
        """Convert score to sentiment label."""
        if score >= SentimentConstants.VERY_POSITIVE_THRESHOLD:
            return "VERY_POSITIVE"
        elif score >= SentimentConstants.POSITIVE_THRESHOLD:
            return "POSITIVE"
        elif score >= SentimentConstants.NEUTRAL_THRESHOLD:
            return "NEUTRAL"
        elif score >= SentimentConstants.NEGATIVE_THRESHOLD:
            return "NEGATIVE"
        else:
            return "VERY_NEGATIVE"

    def _empty_result(self, text: str) -> SentimentResult:
        """Return empty result for invalid input."""
        return SentimentResult(
            text=text or "",
            sentiment="NEUTRAL",
            score=0.0,
            confidence=0.0,
            model_used="None",
            keywords_found=[],
            adjustment=0,
        )

    def clear_cache(self):
        """Clear sentiment cache."""
        self._cache.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get singleton sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING SENTIMENT ANALYZER")
    print("=" * 70 + "\n")

    analyzer = SentimentAnalyzer()

    # Test Vietnamese text
    vn_texts = [
        "Cổ phiếu VNM tăng trưởng mạnh, lợi nhuận quý 3 vượt kỳ vọng",
        "HPG gặp khó khăn do giá thép giảm, lỗ ròng 500 tỷ đồng",
        "Thị trường chứng khoán Việt Nam giao dịch ổn định trong phiên sáng",
        "FPT ký hợp đồng xuất khẩu phần mềm trị giá 100 triệu USD",
        "Ngân hàng cảnh báo rủi ro nợ xấu tăng cao trong quý tới",
    ]

    print("📊 Vietnamese Text Analysis:")
    print("-" * 50)
    for text in vn_texts:
        result = analyzer.analyze(text, language="vi")
        print(f"Text: {text[:50]}...")
        print(f"  Sentiment: {result.sentiment} (score: {result.score:.2f})")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Adjustment: {result.adjustment:+d}")
        print(f"  Keywords: {result.keywords_found}")
        print()

    # Test English text
    en_texts = [
        "Strong earnings growth expected for Vietnam tech sector",
        "Market crash fears as foreign investors sell off",
        "Neutral outlook for banking stocks in Q4",
    ]

    print("\n📊 English Text Analysis:")
    print("-" * 50)
    for text in en_texts:
        result = analyzer.analyze(text, language="en")
        print(f"Text: {text}")
        print(f"  Sentiment: {result.sentiment} (score: {result.score:.2f})")
        print(f"  Model: {result.model_used}")
        print()

    print("✅ Testing complete!")
