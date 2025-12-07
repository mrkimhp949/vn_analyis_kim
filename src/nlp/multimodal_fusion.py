# -*- coding: utf-8 -*-
"""
Multimodal Fusion Model for Vietnam Stock Market

Phase 3: Combines LSTM (price data) + NLP (news sentiment) + XGBoost (technical)

Architecture:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Price Data     │     │  News/Text      │     │  Technical      │
    │  (OHLCV)        │     │  (Vietnamese)   │     │  Indicators     │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
        ┌────▼────┐            ┌─────▼─────┐           ┌─────▼─────┐
        │  LSTM   │            │  PhoBERT  │           │  XGBoost  │
        │ Encoder │            │  Encoder  │           │  Features │
        └────┬────┘            └─────┬─────┘           └─────┬─────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  Attention  │
                              │   Fusion    │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   Dense     │
                              │  Classifier │
                              └─────────────┘

Dependencies:
    pip install torch transformers numpy pandas scikit-learn
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Install: pip install torch")

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not installed")


# =============================================================================
# CONSTANTS
# =============================================================================


class MultimodalConstants:
    """Constants for multimodal model."""

    # Model dimensions
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    PHOBERT_DIM = 768  # PhoBERT base output dimension
    TECHNICAL_DIM = 41  # Number of technical features

    FUSION_DIM = 256
    OUTPUT_DIM = 2  # Binary classification (up/down)

    # Training
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Sentiment weights
    SENTIMENT_WEIGHT = 0.2  # Weight for sentiment in final prediction
    PRICE_WEIGHT = 0.5  # Weight for price LSTM
    TECHNICAL_WEIGHT = 0.3  # Weight for technical features

    # Model paths
    MODEL_DIR = "models/multimodal"
    FUSION_MODEL_PATH = "models/multimodal/fusion_model.pt"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MultimodalInput:
    """Container for multimodal input data."""

    symbol: str
    price_sequence: np.ndarray  # (seq_len, price_features)
    technical_features: np.ndarray  # (technical_dim,)
    sentiment_score: float  # -1 to 1
    sentiment_confidence: float  # 0 to 1
    news_embedding: Optional[np.ndarray] = None  # (phobert_dim,)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultimodalPrediction:
    """Container for multimodal prediction result."""

    symbol: str
    probability: float  # Probability of price going up
    signal: str  # BUY, SELL, HOLD
    confidence: int  # 0-100

    # Component contributions
    price_contribution: float
    sentiment_contribution: float
    technical_contribution: float

    # Metadata
    sentiment_score: float
    news_count: int
    model_version: str = "v1.0"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "probability": self.probability,
            "signal": self.signal,
            "confidence": self.confidence,
            "price_contribution": self.price_contribution,
            "sentiment_contribution": self.sentiment_contribution,
            "technical_contribution": self.technical_contribution,
            "sentiment_score": self.sentiment_score,
            "news_count": self.news_count,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# PYTORCH MODELS (if available)
# =============================================================================

if TORCH_AVAILABLE:

    class PriceLSTM(nn.Module):
        """LSTM encoder for price sequence data."""

        def __init__(
            self,
            input_size: int = 5,  # OHLCV
            hidden_size: int = MultimodalConstants.LSTM_HIDDEN_SIZE,
            num_layers: int = MultimodalConstants.LSTM_NUM_LAYERS,
            dropout: float = MultimodalConstants.LSTM_DROPOUT,
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )

            self.output_size = hidden_size * 2  # Bidirectional

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: (batch, seq_len, input_size)

            Returns:
                (batch, hidden_size * 2)
            """
            # LSTM output
            output, (hidden, cell) = self.lstm(x)

            # Concatenate final hidden states from both directions
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]

            return torch.cat([hidden_forward, hidden_backward], dim=1)

    class AttentionFusion(nn.Module):
        """Attention-based fusion of multiple modalities."""

        def __init__(
            self,
            price_dim: int,
            sentiment_dim: int,
            technical_dim: int,
            fusion_dim: int = MultimodalConstants.FUSION_DIM,
        ):
            super().__init__()

            # Project each modality to same dimension
            self.price_proj = nn.Linear(price_dim, fusion_dim)
            self.sentiment_proj = nn.Linear(sentiment_dim, fusion_dim)
            self.technical_proj = nn.Linear(technical_dim, fusion_dim)

            # Attention weights
            self.attention = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.Tanh(),
                nn.Linear(fusion_dim, 3),
                nn.Softmax(dim=1),
            )

            self.output_dim = fusion_dim

        def forward(
            self,
            price_features: torch.Tensor,
            sentiment_features: torch.Tensor,
            technical_features: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass with attention fusion.

            Returns:
                fused_features: (batch, fusion_dim)
                attention_weights: (batch, 3)
            """
            # Project to common dimension
            price_proj = self.price_proj(price_features)
            sentiment_proj = self.sentiment_proj(sentiment_features)
            technical_proj = self.technical_proj(technical_features)

            # Concatenate for attention
            concat = torch.cat([price_proj, sentiment_proj, technical_proj], dim=1)

            # Calculate attention weights
            attention_weights = self.attention(concat)

            # Weighted sum
            stacked = torch.stack([price_proj, sentiment_proj, technical_proj], dim=1)
            attention_weights_expanded = attention_weights.unsqueeze(2)
            fused = (stacked * attention_weights_expanded).sum(dim=1)

            return fused, attention_weights

    class MultimodalFusionModel(nn.Module):
        """
        Complete multimodal fusion model.

        Combines:
        - LSTM for price sequences
        - Sentiment features (from PhoBERT)
        - Technical indicators
        """

        def __init__(
            self,
            price_input_size: int = 5,
            sentiment_dim: int = 3,  # score, confidence, news_count
            technical_dim: int = MultimodalConstants.TECHNICAL_DIM,
            fusion_dim: int = MultimodalConstants.FUSION_DIM,
            output_dim: int = MultimodalConstants.OUTPUT_DIM,
        ):
            super().__init__()

            # Price LSTM encoder
            self.price_encoder = PriceLSTM(input_size=price_input_size)

            # Sentiment encoder (simple MLP)
            self.sentiment_encoder = nn.Sequential(
                nn.Linear(sentiment_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 64),
            )

            # Technical encoder
            self.technical_encoder = nn.Sequential(
                nn.Linear(technical_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
            )

            # Attention fusion
            self.fusion = AttentionFusion(
                price_dim=self.price_encoder.output_size,
                sentiment_dim=64,
                technical_dim=64,
                fusion_dim=fusion_dim,
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim),
            )

        def forward(
            self,
            price_seq: torch.Tensor,
            sentiment_features: torch.Tensor,
            technical_features: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                price_seq: (batch, seq_len, price_features)
                sentiment_features: (batch, sentiment_dim)
                technical_features: (batch, technical_dim)

            Returns:
                logits: (batch, output_dim)
                attention_weights: (batch, 3)
            """
            # Encode each modality
            price_encoded = self.price_encoder(price_seq)
            sentiment_encoded = self.sentiment_encoder(sentiment_features)
            technical_encoded = self.technical_encoder(technical_features)

            # Fuse with attention
            fused, attention_weights = self.fusion(
                price_encoded,
                sentiment_encoded,
                technical_encoded,
            )

            # Classify
            logits = self.classifier(fused)

            return logits, attention_weights

        def predict_proba(
            self,
            price_seq: torch.Tensor,
            sentiment_features: torch.Tensor,
            technical_features: torch.Tensor,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Predict probabilities.

            Returns:
                probabilities: (batch, output_dim)
                attention_weights: (batch, 3)
            """
            self.eval()
            with torch.no_grad():
                logits, attention = self.forward(price_seq, sentiment_features, technical_features)
                probs = F.softmax(logits, dim=1)

            return probs.cpu().numpy(), attention.cpu().numpy()


# =============================================================================
# MULTIMODAL DATASET (if PyTorch available)
# =============================================================================

if TORCH_AVAILABLE:

    class MultimodalDataset(Dataset):
        """PyTorch Dataset for multimodal training."""

        def __init__(
            self,
            price_sequences: np.ndarray,
            sentiment_features: np.ndarray,
            technical_features: np.ndarray,
            labels: np.ndarray,
        ):
            self.price_sequences = torch.FloatTensor(price_sequences)
            self.sentiment_features = torch.FloatTensor(sentiment_features)
            self.technical_features = torch.FloatTensor(technical_features)
            self.labels = torch.LongTensor(labels)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
            return (
                self.price_sequences[idx],
                self.sentiment_features[idx],
                self.technical_features[idx],
                self.labels[idx],
            )


# =============================================================================
# MULTIMODAL PREDICTOR (Main Interface)
# =============================================================================


class MultimodalPredictor:
    """
    Main predictor class for multimodal stock prediction.

    Combines:
    - Price LSTM for temporal patterns
    - PhoBERT sentiment for Vietnamese news
    - Technical indicators from XGBoost features
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize multimodal predictor.

        Args:
            model_path: Path to saved model weights
            device: Device for inference ("auto", "cpu", "cuda")
        """
        self.model_path = model_path or MultimodalConstants.FUSION_MODEL_PATH

        # Determine device
        if device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model (lazy loaded)
        self._model: Optional["MultimodalFusionModel"] = None
        self._is_trained = False

        # Sentiment analyzer
        self._sentiment_analyzer = None
        self._news_scraper = None

        logger.info(f"MultimodalPredictor initialized (device={self.device})")

    def _load_model(self) -> bool:
        """Load or initialize the fusion model."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using fallback mode")
            return False

        if self._model is not None:
            return True

        try:
            self._model = MultimodalFusionModel()

            # Load weights if available
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self._model.load_state_dict(state_dict)
                self._is_trained = True
                logger.info(f"✅ Loaded model from {self.model_path}")
            else:
                logger.info("No saved model found - using untrained model")

            if self.device == "cuda":
                self._model = self._model.cuda()

            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _get_sentiment_analyzer(self):
        """Get or create sentiment analyzer."""
        if self._sentiment_analyzer is None:
            from src.nlp.sentiment_analyzer import get_sentiment_analyzer

            self._sentiment_analyzer = get_sentiment_analyzer()
        return self._sentiment_analyzer

    def _get_news_scraper(self):
        """Get or create news scraper."""
        if self._news_scraper is None:
            from src.nlp.news_scraper import get_news_scraper

            self._news_scraper = get_news_scraper()
        return self._news_scraper

    def prepare_input(
        self,
        symbol: str,
        df: pd.DataFrame,
        technical_features: Optional[np.ndarray] = None,
    ) -> MultimodalInput:
        """
        Prepare multimodal input from raw data.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            technical_features: Pre-computed technical features (optional)

        Returns:
            MultimodalInput ready for prediction
        """
        # Extract price sequence (last N candles)
        seq_len = MultimodalConstants.SEQUENCE_LENGTH
        price_cols = ["open", "high", "low", "close", "volume"]

        if len(df) < seq_len:
            # Pad with zeros if not enough data
            price_data = df[price_cols].values
            padding = np.zeros((seq_len - len(df), len(price_cols)))
            price_sequence = np.vstack([padding, price_data])
        else:
            price_sequence = df[price_cols].tail(seq_len).values

        # Normalize price sequence
        price_sequence = self._normalize_price_sequence(price_sequence)

        # Get sentiment from news
        sentiment_score, sentiment_confidence, news_count = self._get_symbol_sentiment(symbol)

        # Technical features
        if technical_features is None:
            technical_features = self._extract_technical_features(df)

        return MultimodalInput(
            symbol=symbol,
            price_sequence=price_sequence,
            technical_features=technical_features,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
        )

    def _normalize_price_sequence(self, price_seq: np.ndarray) -> np.ndarray:
        """Normalize price sequence for LSTM input."""
        # Normalize OHLC by first close price
        if price_seq.shape[0] == 0:
            return price_seq

        # Find first non-zero close
        close_col = 3  # close is 4th column
        first_close = price_seq[0, close_col]
        if first_close == 0:
            first_close = 1.0

        normalized = price_seq.copy()
        # Normalize OHLC (columns 0-3) by first close
        normalized[:, :4] = normalized[:, :4] / first_close
        # Normalize volume by mean
        vol_mean = normalized[:, 4].mean()
        if vol_mean > 0:
            normalized[:, 4] = normalized[:, 4] / vol_mean

        return normalized

    def _get_symbol_sentiment(self, symbol: str) -> Tuple[float, float, int]:
        """
        Get sentiment for a symbol from news.

        Returns:
            Tuple of (sentiment_score, confidence, news_count)
        """
        try:
            scraper = self._get_news_scraper()
            analyzer = self._get_sentiment_analyzer()

            # Get recent news
            articles = scraper.get_news_for_symbol(symbol, limit_per_source=5)

            if not articles:
                return (0.0, 0.3, 0)  # Neutral with low confidence

            # Aggregate sentiment
            aggregated = analyzer.aggregate_sentiment(
                symbol=symbol,
                articles=[{"content": a.content, "title": a.title} for a in articles],
            )

            return (
                aggregated.overall_score,
                min(0.5 + len(articles) * 0.1, 0.9),  # Confidence based on article count
                aggregated.num_articles,
            )
        except Exception as e:
            logger.warning(f"Failed to get sentiment for {symbol}: {e}")
            return (0.0, 0.3, 0)

    def _extract_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract technical features from DataFrame."""
        features = []

        try:
            # Basic price features
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            volume = df["volume"].values

            # Returns
            returns = np.diff(close) / close[:-1] if len(close) > 1 else [0]
            features.extend(
                [
                    returns[-1] if len(returns) > 0 else 0,  # Last return
                    np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5-day avg return
                    np.std(returns[-20:]) if len(returns) >= 20 else 0,  # 20-day volatility
                ]
            )

            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(close) >= period:
                    ma = np.mean(close[-period:])
                    features.append(close[-1] / ma - 1)  # Distance from MA
                else:
                    features.append(0)

            # RSI approximation
            if len(returns) >= 14:
                gains = np.maximum(returns[-14:], 0)
                losses = np.abs(np.minimum(returns[-14:], 0))
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                features.append(rsi / 100)  # Normalize to 0-1
            else:
                features.append(0.5)

            # Volume features
            if len(volume) >= 20:
                vol_ma = np.mean(volume[-20:])
                features.append(volume[-1] / vol_ma if vol_ma > 0 else 1)
            else:
                features.append(1)

            # ATR approximation
            if len(high) >= 14:
                tr = np.maximum(
                    high[-14:] - low[-14:], np.abs(high[-14:] - np.roll(close[-14:], 1))
                )
                atr = np.mean(tr)
                features.append(atr / close[-1] if close[-1] > 0 else 0)
            else:
                features.append(0)

        except Exception as e:
            logger.warning(f"Error extracting technical features: {e}")

        # Pad to expected dimension
        while len(features) < MultimodalConstants.TECHNICAL_DIM:
            features.append(0)

        return np.array(features[: MultimodalConstants.TECHNICAL_DIM], dtype=np.float32)

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame,
        technical_features: Optional[np.ndarray] = None,
    ) -> MultimodalPrediction:
        """
        Make prediction for a symbol.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            technical_features: Pre-computed technical features

        Returns:
            MultimodalPrediction with signal and confidence
        """
        # Prepare input
        input_data = self.prepare_input(symbol, df, technical_features)

        # Use PyTorch model if available and trained
        if TORCH_AVAILABLE and self._load_model() and self._is_trained:
            return self._predict_with_model(input_data)
        else:
            # Fallback to weighted ensemble
            return self._predict_fallback(input_data)

    def _predict_with_model(self, input_data: MultimodalInput) -> MultimodalPrediction:
        """Predict using trained PyTorch model."""
        self._model.eval()

        with torch.no_grad():
            # Prepare tensors
            price_seq = torch.FloatTensor(input_data.price_sequence).unsqueeze(0)
            sentiment = torch.FloatTensor(
                [
                    input_data.sentiment_score,
                    input_data.sentiment_confidence,
                    1.0,  # news_count normalized
                ]
            ).unsqueeze(0)
            technical = torch.FloatTensor(input_data.technical_features).unsqueeze(0)

            if self.device == "cuda":
                price_seq = price_seq.cuda()
                sentiment = sentiment.cuda()
                technical = technical.cuda()

            # Forward pass
            logits, attention = self._model(price_seq, sentiment, technical)
            probs = F.softmax(logits, dim=1)

            # Extract results
            prob_up = probs[0, 1].item()
            attention_weights = attention[0].cpu().numpy()

        # Determine signal
        if prob_up >= 0.6:
            signal = "BUY"
        elif prob_up <= 0.4:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = int(abs(prob_up - 0.5) * 200)  # 0-100 scale

        return MultimodalPrediction(
            symbol=input_data.symbol,
            probability=prob_up,
            signal=signal,
            confidence=confidence,
            price_contribution=attention_weights[0],
            sentiment_contribution=attention_weights[1],
            technical_contribution=attention_weights[2],
            sentiment_score=input_data.sentiment_score,
            news_count=0,
        )

    def _predict_fallback(self, input_data: MultimodalInput) -> MultimodalPrediction:
        """
        Fallback prediction using weighted ensemble (no PyTorch required).

        Uses simple weighted combination of:
        - Price momentum (from price sequence)
        - Sentiment score
        - Technical features
        """
        # Price momentum score
        price_seq = input_data.price_sequence
        if len(price_seq) >= 5:
            # Simple momentum: compare last close to 5-period ago
            momentum = (price_seq[-1, 3] - price_seq[-5, 3]) / max(price_seq[-5, 3], 0.001)
            price_score = np.clip(momentum * 5 + 0.5, 0, 1)  # Scale to 0-1
        else:
            price_score = 0.5

        # Sentiment score (already -1 to 1, convert to 0-1)
        sentiment_score = (input_data.sentiment_score + 1) / 2

        # Technical score (use RSI-like feature if available)
        tech_features = input_data.technical_features
        if len(tech_features) > 7:
            rsi_normalized = tech_features[7]  # RSI feature
            # Inverse RSI for buy signal (low RSI = buy opportunity)
            technical_score = 1 - rsi_normalized
        else:
            technical_score = 0.5

        # Weighted combination
        weights = [
            MultimodalConstants.PRICE_WEIGHT,
            MultimodalConstants.SENTIMENT_WEIGHT,
            MultimodalConstants.TECHNICAL_WEIGHT,
        ]

        prob_up = (
            price_score * weights[0] + sentiment_score * weights[1] + technical_score * weights[2]
        )

        # Determine signal
        if prob_up >= 0.55:
            signal = "BUY"
        elif prob_up <= 0.45:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = int(abs(prob_up - 0.5) * 200)

        return MultimodalPrediction(
            symbol=input_data.symbol,
            probability=prob_up,
            signal=signal,
            confidence=confidence,
            price_contribution=weights[0],
            sentiment_contribution=weights[1],
            technical_contribution=weights[2],
            sentiment_score=input_data.sentiment_score,
            news_count=0,
            model_version="fallback_v1.0",
        )

    def train(
        self,
        train_data: List[Tuple[pd.DataFrame, int]],
        val_data: Optional[List[Tuple[pd.DataFrame, int]]] = None,
        epochs: int = MultimodalConstants.MAX_EPOCHS,
        batch_size: int = MultimodalConstants.BATCH_SIZE,
    ) -> Dict:
        """
        Train the multimodal model.

        Args:
            train_data: List of (DataFrame, label) tuples
            val_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history dict
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for training")
            return {"error": "PyTorch not available"}

        self._load_model()

        # Prepare datasets
        train_dataset = self._prepare_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data:
            val_dataset = self._prepare_dataset(val_data)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training setup
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=MultimodalConstants.LEARNING_RATE,
        )
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            self._model.train()
            train_loss = 0.0

            for batch in train_loader:
                price_seq, sentiment, technical, labels = batch

                if self.device == "cuda":
                    price_seq = price_seq.cuda()
                    sentiment = sentiment.cuda()
                    technical = technical.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                logits, _ = self._model(price_seq, sentiment, technical)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1

                if patience_counter >= MultimodalConstants.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f}")

        self._is_trained = True
        return history

    def _prepare_dataset(
        self,
        data: List[Tuple[pd.DataFrame, int]],
    ) -> "MultimodalDataset":
        """Prepare PyTorch dataset from raw data."""
        price_sequences = []
        sentiment_features = []
        technical_features = []
        labels = []

        for df, label in data:
            # Prepare input
            input_data = self.prepare_input("TRAIN", df)

            price_sequences.append(input_data.price_sequence)
            sentiment_features.append(
                [
                    input_data.sentiment_score,
                    input_data.sentiment_confidence,
                    1.0,
                ]
            )
            technical_features.append(input_data.technical_features)
            labels.append(label)

        return MultimodalDataset(
            np.array(price_sequences),
            np.array(sentiment_features),
            np.array(technical_features),
            np.array(labels),
        )

    def _validate(
        self,
        val_loader: "DataLoader",
        criterion: "nn.Module",
    ) -> Tuple[float, float]:
        """Validate model on validation set."""
        self._model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                price_seq, sentiment, technical, labels = batch

                if self.device == "cuda":
                    price_seq = price_seq.cuda()
                    sentiment = sentiment.cuda()
                    technical = technical.cuda()
                    labels = labels.cuda()

                logits, _ = self._model(price_seq, sentiment, technical)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def save_model(self, path: Optional[str] = None) -> bool:
        """Save model weights to file."""
        if not TORCH_AVAILABLE or self._model is None:
            return False

        save_path = path or self.model_path

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self._model.state_dict(), save_path)
            logger.info(f"✅ Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        """Load model weights from file."""
        load_path = path or self.model_path

        if not os.path.exists(load_path):
            logger.warning(f"Model file not found: {load_path}")
            return False

        return self._load_model()


# =============================================================================
# SINGLETON
# =============================================================================

_multimodal_predictor: Optional[MultimodalPredictor] = None


def get_multimodal_predictor() -> MultimodalPredictor:
    """Get singleton multimodal predictor instance."""
    global _multimodal_predictor
    if _multimodal_predictor is None:
        _multimodal_predictor = MultimodalPredictor()
    return _multimodal_predictor


# =============================================================================
# SENTIMENT INTEGRATION FOR ENTRY LOGIC
# =============================================================================


def get_sentiment_adjustment(symbol: str, df: pd.DataFrame) -> Dict:
    """
    Get sentiment-based confidence adjustment for entry logic.

    This function is designed to be called from entry_logic.py
    to incorporate NLP sentiment into trading decisions.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data

    Returns:
        Dict with:
        - adjustment: Confidence adjustment (-20 to +10)
        - sentiment: Sentiment label
        - score: Raw sentiment score (-1 to 1)
        - confidence: Sentiment confidence (0 to 1)
        - news_count: Number of news articles analyzed
    """
    try:
        predictor = get_multimodal_predictor()
        prediction = predictor.predict(symbol, df)

        # Map sentiment score to adjustment
        # Based on SentimentConstants.SENTIMENT_ADJUSTMENTS
        score = prediction.sentiment_score

        if score >= 0.7:
            adjustment = 10
            sentiment = "VERY_POSITIVE"
        elif score >= 0.3:
            adjustment = 5
            sentiment = "POSITIVE"
        elif score >= -0.3:
            adjustment = 0
            sentiment = "NEUTRAL"
        elif score >= -0.7:
            adjustment = -10
            sentiment = "NEGATIVE"
        else:
            adjustment = -20
            sentiment = "VERY_NEGATIVE"

        return {
            "adjustment": adjustment,
            "sentiment": sentiment,
            "score": score,
            "confidence": prediction.confidence / 100,
            "news_count": prediction.news_count,
            "signal": prediction.signal,
            "probability": prediction.probability,
        }
    except Exception as e:
        logger.warning(f"Failed to get sentiment adjustment for {symbol}: {e}")
        return {
            "adjustment": 0,
            "sentiment": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.0,
            "news_count": 0,
            "signal": "HOLD",
            "probability": 0.5,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING MULTIMODAL FUSION")
    print("=" * 70 + "\n")

    # Create sample data
    import numpy as np

    # Sample DataFrame
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    sample_df = pd.DataFrame(
        {
            "open": np.random.uniform(20, 25, 50),
            "high": np.random.uniform(25, 30, 50),
            "low": np.random.uniform(18, 22, 50),
            "close": np.random.uniform(20, 28, 50),
            "volume": np.random.uniform(1000000, 5000000, 50),
        },
        index=dates,
    )

    print("📊 Testing MultimodalPredictor...")
    predictor = MultimodalPredictor()

    # Test prediction
    prediction = predictor.predict("VNM", sample_df)
    print(f"\nPrediction for VNM:")
    print(f"  Signal: {prediction.signal}")
    print(f"  Probability: {prediction.probability:.2%}")
    print(f"  Confidence: {prediction.confidence}")
    print(f"  Sentiment Score: {prediction.sentiment_score:.2f}")
    print(f"  Contributions:")
    print(f"    - Price: {prediction.price_contribution:.2%}")
    print(f"    - Sentiment: {prediction.sentiment_contribution:.2%}")
    print(f"    - Technical: {prediction.technical_contribution:.2%}")

    # Test sentiment adjustment
    print("\n📊 Testing get_sentiment_adjustment...")
    adjustment = get_sentiment_adjustment("HPG", sample_df)
    print(f"\nSentiment adjustment for HPG:")
    print(f"  Adjustment: {adjustment['adjustment']:+d}")
    print(f"  Sentiment: {adjustment['sentiment']}")
    print(f"  Score: {adjustment['score']:.2f}")

    print("\n✅ Testing complete!")
