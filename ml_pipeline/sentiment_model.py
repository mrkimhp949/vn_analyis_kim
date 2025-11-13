import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


class VietnameseSentimentAnalyzer:
    """
    Phân tích sentiment sử dụng mô hình Transformers (PhoBERT/ViBERT) nếu khả dụng,
    fallback sang keyword scoring nếu không.
    """

    def __init__(self):
        self.pipeline = None
        self.model_name = None
        self.id2label: Dict[int, str] = {}

        custom_model_path = os.environ.get("PHOBERT_FINE_TUNED_PATH", "models/phobert_vi_financial")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            if os.path.isdir(custom_model_path):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(custom_model_path)
                    self.pipeline = pipeline(
                        "text-classification", model=model, tokenizer=tokenizer, truncation=True
                    )
                    self.model_name = "phobert-finetuned"
                    self.id2label = getattr(model.config, "id2label", {})
                    logger.info("Loaded fine-tuned PhoBERT sentiment model from %s", custom_model_path)
                except Exception as custom_exc:
                    logger.warning("Không thể load PhoBERT fine-tuned: %s", custom_exc)
                    self.pipeline = None
                    self.model_name = None
                    self.id2label = {}

            if self.pipeline is None:
                # Try PhoBERT base with fallback to multilingual sentiment
                try:
                    logger.info("Attempting to load PhoBERT base model...")
                    base_model_name = "vinai/phobert-base-v2"
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        "nlptown/bert-base-multilingual-uncased-sentiment"
                    )
                    self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                    self.model_name = "multilingual-bert"
                    self.id2label = getattr(model.config, "id2label", {})
                    logger.info("Loaded multilingual BERT sentiment pipeline (optimized for Vietnamese).")
                except Exception as e1:
                    logger.warning(f"PhoBERT not available: {e1}. Trying multilingual directly...")
                    try:
                        fallback_model = "nlptown/bert-base-multilingual-uncased-sentiment"
                        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                        model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
                        self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                        self.model_name = "multilingual-bert"
                        self.id2label = getattr(model.config, "id2label", {})
                        logger.info("Loaded multilingual BERT sentiment pipeline.")
                    except Exception as e2:
                        logger.warning(f"ViBERT/Multilingual model not available: {e2}. Fallback keyword scoring.")
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

    def _normalize_label(self, raw_label: str) -> str:
        label = raw_label
        if raw_label.upper().startswith("LABEL_"):
            try:
                idx = int(raw_label.split("_")[-1])
                label = self.id2label.get(idx, raw_label)
            except ValueError:
                pass
        return label.lower()

    def score(self, texts: List[str]) -> float:
        if not texts:
            return 0.0

        if self.pipeline:
            try:
                results = self.pipeline(texts)
                score = 0.0
                for res in results:
                    label = self._normalize_label(res.get("label", ""))
                    prob = float(res.get("score", 0.0))
                    if any(key in label for key in ["5 star", "4 star", "positive", "pos"]):
                        score += prob
                    elif any(key in label for key in ["1 star", "2 star", "negative", "neg"]):
                        score -= prob
                    elif "neutral" in label or "neu" in label or "middle" in label:
                        score += 0.0
                    else:
                        # Unknown label - treat as neutral
                        score += 0.0
                return max(min(score / max(len(results), 1), 1.0), -1.0)
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

