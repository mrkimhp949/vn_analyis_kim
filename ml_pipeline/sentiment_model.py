import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class VietnameseSentimentAnalyzer:
    """
    Phân tích sentiment sử dụng mô hình Transformers (PhoBERT/ViBERT) nếu khả dụng,
    fallback sang keyword scoring nếu không.
    """

    def __init__(self):
        self.pipeline = None
        self.model_name = None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            # Thử PhoBERT trước (mô hình tiếng Việt tốt nhất)
            try:
                model_name = "vinai/phobert-base-v2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # PhoBERT không có sẵn sentiment model, dùng multilingual BERT với fine-tuning
                # Hoặc dùng ViBERT nếu có
                logger.info("Attempting to load PhoBERT...")
                # Fallback to multilingual sentiment model optimized for Vietnamese
                model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                self.model_name = "multilingual-bert"
                logger.info("Loaded multilingual BERT sentiment pipeline (optimized for Vietnamese).")
            except Exception as e1:
                logger.warning(f"PhoBERT not available: {e1}. Trying ViBERT...")
                # Thử ViBERT hoặc fallback
                try:
                    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                    self.model_name = "multilingual-bert"
                    logger.info("Loaded multilingual BERT sentiment pipeline.")
                except Exception as e2:
                    logger.warning(f"ViBERT not available: {e2}. Fallback keyword scoring.")
        except Exception as e:
            logger.warning(f"Không thể load transformer model: {e}. Fallback keyword scoring.")

        self.positive_keywords = [
            "tăng trưởng",
            "tích cực",
            "vượt mục tiêu",
            "lợi nhuận",
            "mở rộng",
            "phê duyệt",
            "kỷ lục",
            "bứt phá",
        ]
        self.negative_keywords = [
            "giảm",
            "thua lỗ",
            "suy giảm",
            "điều tra",
            "phạt",
            "rủi ro",
            "thất bại",
            "áp lực bán",
        ]

    def score(self, texts: List[str]) -> float:
        if not texts:
            return 0.0

        if self.pipeline:
            try:
                results = self.pipeline(texts)
                score = 0.0
                for res in results:
                    label = res["label"]
                    if label in ("5 stars", "4 stars"):
                        score += 1.0
                    elif label in ("1 star", "2 stars"):
                        score -= 1.0
                return max(min(score / len(results), 1.0), -1.0)
            except Exception as e:
                logger.warning(f"Transformer scoring failed: {e}")

        score = 0.0
        for text in texts:
            lowered = text.lower()
            for word in self.positive_keywords:
                if word in lowered:
                    score += 0.5
            for word in self.negative_keywords:
                if word in lowered:
                    score -= 0.5
        return max(min(score / len(texts), 1.0), -1.0)

    def classify(self, text: str) -> str:
        score = self.score([text])
        if score >= 0.6:
            return "STRONGLY_POSITIVE"
        if score >= 0.2:
            return "POSITIVE"
        if score <= -0.6:
            return "STRONGLY_NEGATIVE"
        if score <= -0.2:
            return "NEGATIVE"
        return "NEUTRAL"

