"""
Unit tests for ml_pipeline/sentiment_model.py
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# Mock suppress_warnings before importing sentiment_model
sys.modules["suppress_warnings"] = MagicMock()

from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer


class TestVietnameseSentimentAnalyzerInitialization:
    """Tests for VietnameseSentimentAnalyzer initialization"""

    @patch("ml_pipeline.sentiment_model.os.path.isdir")
    @patch("ml_pipeline.sentiment_model.os.environ.get")
    def test_init_without_transformers(self, mock_env_get, mock_isdir):
        """Test initialization when transformers is not available"""
        mock_env_get.return_value = "models/phobert_vi_financial"
        mock_isdir.return_value = False

        with patch.dict("sys.modules", {"transformers": None}):
            analyzer = VietnameseSentimentAnalyzer()

        assert analyzer.pipeline is None
        assert analyzer.model_name is None
        assert analyzer.id2label == {}
        assert len(analyzer.positive_keywords) > 0
        assert len(analyzer.negative_keywords) > 0

    @pytest.mark.skip(reason="Transformers import conflicts with environment variables")
    @patch("ml_pipeline.sentiment_model.os.path.isdir")
    @patch("ml_pipeline.sentiment_model.os.environ.get")
    def test_init_with_custom_model_path(self, mock_env_get, mock_isdir):
        """Test initialization with custom fine-tuned model"""
        custom_path = "custom/model/path"
        mock_env_get.return_value = custom_path
        mock_isdir.return_value = True

        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_config = Mock()
        mock_config.id2label = {0: "negative", 1: "positive"}
        mock_model.config = mock_config
        mock_pipeline = Mock()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer, patch(
            "transformers.AutoModelForSequenceClassification"
        ) as mock_auto_model, patch("transformers.pipeline") as mock_pipeline_func:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_pipeline_func.return_value = mock_pipeline

            analyzer = VietnameseSentimentAnalyzer()

            assert analyzer.pipeline == mock_pipeline
            assert analyzer.model_name == "phobert-finetuned"
            assert analyzer.id2label == {0: "negative", 1: "positive"}
            mock_auto_tokenizer.from_pretrained.assert_called_with(custom_path)
            mock_auto_model.from_pretrained.assert_called_with(custom_path)

    @pytest.mark.skip(reason="Transformers import conflicts with environment variables")
    @patch("ml_pipeline.sentiment_model.os.path.isdir")
    @patch("ml_pipeline.sentiment_model.os.environ.get")
    def test_init_custom_model_fails_fallback_to_phobert(
        self, mock_env_get, mock_isdir
    ):
        """Test fallback to PhoBERT base when custom model fails"""
        mock_env_get.return_value = "models/phobert_vi_financial"
        mock_isdir.return_value = True

        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_config = Mock()
        mock_config.id2label = {
            0: "1 star",
            1: "2 stars",
            2: "3 stars",
            3: "4 stars",
            4: "5 stars",
        }
        mock_model.config = mock_config
        mock_pipeline = Mock()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer, patch(
            "transformers.AutoModelForSequenceClassification"
        ) as mock_auto_model, patch("transformers.pipeline") as mock_pipeline_func:
            # First call (custom model) fails
            mock_auto_tokenizer.from_pretrained.side_effect = [
                Exception("Custom model not found"),
                mock_tokenizer,  # Second call (PhoBERT) succeeds
            ]
            mock_auto_model.from_pretrained.side_effect = [
                Exception("Custom model not found"),
                mock_model,  # Multilingual model
            ]
            mock_pipeline_func.return_value = mock_pipeline

            analyzer = VietnameseSentimentAnalyzer()

            assert analyzer.pipeline == mock_pipeline
            assert analyzer.model_name == "multilingual-bert"

    @pytest.mark.skip(reason="Transformers import conflicts with environment variables")
    @patch("ml_pipeline.sentiment_model.os.path.isdir")
    @patch("ml_pipeline.sentiment_model.os.environ.get")
    def test_init_all_models_fail(self, mock_env_get, mock_isdir):
        """Test when all model loading attempts fail"""
        mock_env_get.return_value = "models/phobert_vi_financial"
        mock_isdir.return_value = False

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.side_effect = Exception(
                "All models failed"
            )

            analyzer = VietnameseSentimentAnalyzer()

            # Should fall back to keyword scoring
            assert analyzer.pipeline is None
            assert analyzer.model_name is None
            assert len(analyzer.positive_keywords) > 0
            assert len(analyzer.negative_keywords) > 0


class TestNormalizeLabel:
    """Tests for _normalize_label method"""

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_normalize_label_with_id(self, mock_env, mock_isdir):
        """Test normalizing label with LABEL_X format"""
        analyzer = VietnameseSentimentAnalyzer()
        analyzer.id2label = {0: "negative", 1: "positive", 2: "neutral"}

        assert analyzer._normalize_label("LABEL_0") == "negative"
        assert analyzer._normalize_label("LABEL_1") == "positive"
        assert analyzer._normalize_label("LABEL_2") == "neutral"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_normalize_label_without_mapping(self, mock_env, mock_isdir):
        """Test normalizing label when no mapping exists"""
        analyzer = VietnameseSentimentAnalyzer()
        analyzer.id2label = {}

        result = analyzer._normalize_label("LABEL_5")
        assert result == "label_5"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_normalize_label_plain_text(self, mock_env, mock_isdir):
        """Test normalizing plain text label"""
        analyzer = VietnameseSentimentAnalyzer()

        assert analyzer._normalize_label("positive") == "positive"
        assert analyzer._normalize_label("NEGATIVE") == "negative"
        assert analyzer._normalize_label("Neutral") == "neutral"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_normalize_label_star_rating(self, mock_env, mock_isdir):
        """Test normalizing star rating labels"""
        analyzer = VietnameseSentimentAnalyzer()
        analyzer.id2label = {0: "1 star", 4: "5 stars"}

        assert analyzer._normalize_label("LABEL_0") == "1 star"
        assert analyzer._normalize_label("LABEL_4") == "5 stars"


class TestScoreMethod:
    """Tests for score method"""

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_empty_texts(self, mock_env, mock_isdir):
        """Test scoring with empty text list"""
        analyzer = VietnameseSentimentAnalyzer()

        assert analyzer.score([]) == 0.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_positive_keywords(self, mock_env, mock_isdir):
        """Test scoring with positive keywords"""
        analyzer = VietnameseSentimentAnalyzer()

        texts = [
            "Công ty có tăng trưởng mạnh",
            "Lợi nhuận vượt mục tiêu",
            "Kết quả tích cực",
        ]

        score = analyzer.score(texts)
        assert score > 0.0
        assert score <= 1.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_negative_keywords(self, mock_env, mock_isdir):
        """Test scoring with negative keywords"""
        analyzer = VietnameseSentimentAnalyzer()

        texts = ["Doanh thu giảm mạnh", "Công ty thua lỗ", "Rủi ro cao"]

        score = analyzer.score(texts)
        assert score < 0.0
        assert score >= -1.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_neutral_text(self, mock_env, mock_isdir):
        """Test scoring with neutral text"""
        analyzer = VietnameseSentimentAnalyzer()

        texts = ["Công ty công bố báo cáo tài chính"]

        score = analyzer.score(texts)
        assert score == 0.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_mixed_sentiment(self, mock_env, mock_isdir):
        """Test scoring with mixed sentiment"""
        analyzer = VietnameseSentimentAnalyzer()

        texts = ["Lợi nhuận tăng trưởng", "Nhưng rủi ro cao"]  # positive  # negative

        score = analyzer.score(texts)
        assert -1.0 <= score <= 1.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_with_pipeline_positive(self, mock_env, mock_isdir):
        """Test scoring with transformer pipeline returning positive"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "5 stars", "score": 0.95},
            {"label": "4 stars", "score": 0.85},
        ]
        analyzer.pipeline = mock_pipeline

        texts = ["Great news!", "Excellent performance!"]
        score = analyzer.score(texts)

        assert score > 0.0
        assert score <= 1.0
        mock_pipeline.assert_called_once_with(texts)

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_with_pipeline_negative(self, mock_env, mock_isdir):
        """Test scoring with transformer pipeline returning negative"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "1 star", "score": 0.90},
            {"label": "2 star", "score": 0.80},
        ]
        analyzer.pipeline = mock_pipeline

        texts = ["Bad news", "Poor performance"]
        score = analyzer.score(texts)

        assert score < 0.0
        assert score >= -1.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_with_pipeline_neutral(self, mock_env, mock_isdir):
        """Test scoring with transformer pipeline returning neutral"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.75}]
        analyzer.pipeline = mock_pipeline

        texts = ["Company announces report"]
        score = analyzer.score(texts)

        assert score == 0.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_with_pipeline_label_ids(self, mock_env, mock_isdir):
        """Test scoring with transformer pipeline returning LABEL_X format"""
        analyzer = VietnameseSentimentAnalyzer()
        analyzer.id2label = {0: "1 star", 4: "5 stars"}

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "LABEL_4", "score": 0.92}]
        analyzer.pipeline = mock_pipeline

        texts = ["Great!"]
        score = analyzer.score(texts)

        assert score > 0.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_with_pipeline_exception_fallback(self, mock_env, mock_isdir):
        """Test fallback to keyword scoring when pipeline fails"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline error")
        analyzer.pipeline = mock_pipeline

        texts = ["Lợi nhuận tăng trưởng tích cực"]
        score = analyzer.score(texts)

        # Should fall back to keyword scoring
        assert score > 0.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_with_pipeline_unknown_label(self, mock_env, mock_isdir):
        """Test scoring with unknown label treated as neutral"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "unknown_label", "score": 0.85}]
        analyzer.pipeline = mock_pipeline

        texts = ["Some text"]
        score = analyzer.score(texts)

        assert score == 0.0

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_score_clamping(self, mock_env, mock_isdir):
        """Test that score is clamped between -1.0 and 1.0"""
        analyzer = VietnameseSentimentAnalyzer()

        # Many positive keywords to test upper bound
        texts = [
            "tăng trưởng tích cực lợi nhuận vượt mục tiêu mở rộng bứt phá kỷ lục"
        ] * 10

        score = analyzer.score(texts)
        assert score <= 1.0
        assert score >= -1.0


class TestClassifyMethod:
    """Tests for classify method"""

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_classify_strongly_positive(self, mock_env, mock_isdir):
        """Test classify for strongly positive text"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "5 stars", "score": 0.95}]
        analyzer.pipeline = mock_pipeline

        result = analyzer.classify("Excellent performance")
        assert result == "STRONGLY_POSITIVE"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_classify_positive(self, mock_env, mock_isdir):
        """Test classify for positive text"""
        analyzer = VietnameseSentimentAnalyzer()

        # Mock score to return between 0.2 and 0.6
        with patch.object(analyzer, "score", return_value=0.4):
            result = analyzer.classify("Good news")
            assert result == "POSITIVE"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_classify_neutral(self, mock_env, mock_isdir):
        """Test classify for neutral text"""
        analyzer = VietnameseSentimentAnalyzer()

        with patch.object(analyzer, "score", return_value=0.0):
            result = analyzer.classify("Company announces report")
            assert result == "NEUTRAL"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_classify_negative(self, mock_env, mock_isdir):
        """Test classify for negative text"""
        analyzer = VietnameseSentimentAnalyzer()

        with patch.object(analyzer, "score", return_value=-0.4):
            result = analyzer.classify("Bad performance")
            assert result == "NEGATIVE"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_classify_strongly_negative(self, mock_env, mock_isdir):
        """Test classify for strongly negative text"""
        analyzer = VietnameseSentimentAnalyzer()

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "1 star", "score": 0.95}]
        analyzer.pipeline = mock_pipeline

        result = analyzer.classify("Terrible news")
        assert result == "STRONGLY_NEGATIVE"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_classify_boundary_values(self, mock_env, mock_isdir):
        """Test classify with boundary score values"""
        analyzer = VietnameseSentimentAnalyzer()

        # Test exact boundary at 0.6
        with patch.object(analyzer, "score", return_value=0.6):
            assert analyzer.classify("text") == "STRONGLY_POSITIVE"

        # Test just below 0.6
        with patch.object(analyzer, "score", return_value=0.59):
            assert analyzer.classify("text") == "POSITIVE"

        # Test exact boundary at 0.2
        with patch.object(analyzer, "score", return_value=0.2):
            assert analyzer.classify("text") == "POSITIVE"

        # Test just below 0.2
        with patch.object(analyzer, "score", return_value=0.19):
            assert analyzer.classify("text") == "NEUTRAL"

        # Test exact boundary at -0.2
        with patch.object(analyzer, "score", return_value=-0.2):
            assert analyzer.classify("text") == "NEGATIVE"

        # Test just above -0.2
        with patch.object(analyzer, "score", return_value=-0.19):
            assert analyzer.classify("text") == "NEUTRAL"

        # Test exact boundary at -0.6
        with patch.object(analyzer, "score", return_value=-0.6):
            assert analyzer.classify("text") == "STRONGLY_NEGATIVE"

        # Test just above -0.6
        with patch.object(analyzer, "score", return_value=-0.59):
            assert analyzer.classify("text") == "NEGATIVE"


class TestKeywords:
    """Tests for keyword lists"""

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_positive_keywords_exist(self, mock_env, mock_isdir):
        """Test that positive keywords are defined"""
        analyzer = VietnameseSentimentAnalyzer()

        assert len(analyzer.positive_keywords) > 0
        assert "tăng trưởng" in analyzer.positive_keywords
        assert "lợi nhuận" in analyzer.positive_keywords
        assert "tích cực" in analyzer.positive_keywords

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_negative_keywords_exist(self, mock_env, mock_isdir):
        """Test that negative keywords are defined"""
        analyzer = VietnameseSentimentAnalyzer()

        assert len(analyzer.negative_keywords) > 0
        assert "giảm" in analyzer.negative_keywords
        assert "thua lỗ" in analyzer.negative_keywords
        assert "rủi ro" in analyzer.negative_keywords

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_keywords_are_lowercase(self, mock_env, mock_isdir):
        """Test that all keywords are lowercase"""
        analyzer = VietnameseSentimentAnalyzer()

        for keyword in analyzer.positive_keywords:
            assert keyword == keyword.lower()

        for keyword in analyzer.negative_keywords:
            assert keyword == keyword.lower()


class TestIntegration:
    """Integration tests for full workflow"""

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_full_workflow_keyword_scoring(self, mock_env, mock_isdir):
        """Test complete workflow with keyword-based scoring"""
        analyzer = VietnameseSentimentAnalyzer()

        # Test positive sentiment
        positive_texts = [
            "Công ty đạt lợi nhuận kỷ lục với tăng trưởng mạnh",
            "Chiến lược mở rộng thành công vượt mục tiêu",
        ]
        positive_score = analyzer.score(positive_texts)
        positive_class = analyzer.classify(positive_texts[0])

        assert positive_score > 0
        assert positive_class in ["POSITIVE", "STRONGLY_POSITIVE"]

        # Test negative sentiment
        negative_texts = [
            "Doanh thu giảm mạnh do rủi ro thị trường",
            "Công ty phải chịu thua lỗ lớn",
        ]
        negative_score = analyzer.score(negative_texts)
        negative_class = analyzer.classify(negative_texts[0])

        assert negative_score < 0
        assert negative_class in ["NEGATIVE", "STRONGLY_NEGATIVE"]

        # Test neutral sentiment
        neutral_text = "Công ty công bố báo cáo tài chính quý 3"
        neutral_score = analyzer.score([neutral_text])
        neutral_class = analyzer.classify(neutral_text)

        assert neutral_score == 0.0
        assert neutral_class == "NEUTRAL"

    @patch("ml_pipeline.sentiment_model.os.path.isdir", return_value=False)
    @patch(
        "ml_pipeline.sentiment_model.os.environ.get",
        return_value="models/phobert_vi_financial",
    )
    def test_case_insensitive_scoring(self, mock_env, mock_isdir):
        """Test that keyword scoring is case-insensitive"""
        analyzer = VietnameseSentimentAnalyzer()

        lower_case = analyzer.score(["tăng trưởng lợi nhuận"])
        upper_case = analyzer.score(["TĂNG TRƯỞNG LỢI NHUẬN"])
        mixed_case = analyzer.score(["Tăng Trưởng Lợi Nhuận"])

        assert lower_case == upper_case == mixed_case
        assert lower_case > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
