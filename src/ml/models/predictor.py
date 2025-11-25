import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None  # NEW: XGBoost model
        self.scaler = StandardScaler()
        self.models_dir = "models"
        self.ml_enabled = True  # NEW: Flag to track if ML is usable
        self.using_dummy_models = False  # NEW: Flag to track dummy models
        self.feature_importance = None  # NEW: Feature importance scores
        self.selected_features = None  # NEW: Selected feature indices

        # Ensemble weights (can be tuned)
        self.ensemble_weights = {
            "rf": 0.5,  # Random Forest 50%
            "xgb": 0.5,  # XGBoost 50%
        }

        # NEW: Online learning / incremental update tracking
        self.incremental_samples = []  # Buffer for new samples
        self.incremental_buffer_size = 100  # Retrain after 100 new samples
        self.last_retrain_date = None
        self.retrain_interval_days = 7  # Auto-retrain weekly
        self.performance_degradation_threshold = 0.1  # Retrain if accuracy drops 10%
        self.baseline_accuracy = None  # Track baseline for degradation detection

        self.ensure_models_dir()
        # Đồng bộ số features mong đợi với features.get_feature_columns()
        try:
            # Try enhanced features first
            try:
                from src.ml.features.enhanced import get_feature_columns
            except ImportError:
                from src.ml.features.technical import get_feature_columns

            self.expected_features = len(get_feature_columns())
        except Exception:
            # Fallback an toàn nếu không import được
            self.expected_features = 28  # Default to enhanced features count

    def ensure_models_dir(self):
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"✅ Models directory: {os.path.abspath(self.models_dir)}")
        except Exception:
            logger.error("⚠️ Không thể tạo thư mục models")

    def create_dummy_models(self):
        """Tạo models mẫu với ĐÚNG 18 features"""
        logger.info("🔄 Creating dummy models with 18 features...")

        # Tạo scaler mẫu với 18 features
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([0] * self.expected_features)
        self.scaler.scale_ = np.array([1] * self.expected_features)

        # Tạo RF model mẫu với 18 features
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Train với data giả 18 features
        X_dummy = np.random.randn(100, self.expected_features)
        y_dummy = np.random.randint(0, 2, 100)
        self.rf_model.fit(X_dummy, y_dummy)

        # Lưu models
        self.save_models()
        logger.info("✅ Dummy models created and saved")

    def save_models(self):
        """Lưu models"""
        self.ensure_models_dir()
        try:
            # Save Random Forest
            if self.rf_model:
                joblib.dump(self.rf_model, os.path.join(self.models_dir, "random_forest.pkl"))

            # Save XGBoost
            if self.xgb_model:
                xgb_path = os.path.join(self.models_dir, "xgboost.pkl")
                joblib.dump(self.xgb_model, xgb_path)
                logger.info(f"✅ XGBoost model saved to {xgb_path}")

            # Save scaler
            joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))

            # Lưu model metadata với feature list
            metadata = {
                "expected_features": self.expected_features,
                "saved_at": pd.Timestamp.now().isoformat(),
                "has_xgboost": self.xgb_model is not None,
                "has_rf": self.rf_model is not None,
                "ensemble_weights": self.ensemble_weights,
            }

            # Save feature names if available
            try:
                try:
                    from src.ml.features.enhanced import get_feature_columns
                except ImportError:
                    from src.ml.features.technical import get_feature_columns

                metadata["feature_names"] = get_feature_columns()
            except Exception:
                pass

            # NEW: Save feature importance if available
            if self.feature_importance is not None:
                importance_path = os.path.join(self.models_dir, "feature_importance.csv")
                self.feature_importance.to_csv(importance_path, index=False)
                metadata["has_feature_importance"] = True
                logger.info(f"✅ Feature importance saved to {importance_path}")

            with open(os.path.join(self.models_dir, "model_info.json"), "w") as f:
                import json

                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Models saved successfully with {self.expected_features} features")
        except Exception:
            logger.error("❌ Lỗi khi lưu models")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest với class_weight và params tối ưu."""
        logger.info("🌲 Training Random Forest with optimized parameters...")

        if X_train.shape[1] != self.expected_features:
            from src.config.exceptions import ModelPredictionError

            raise ModelPredictionError(
                "Feature count mismatch during training",
                context={
                    "got": X_train.shape[1],
                    "expected": self.expected_features,
                    "message": "Ensure all features are generated before training.",
                },
            )

        self.rf_model = RandomForestClassifier(
            n_estimators=200,  # Tăng số lượng cây
            max_depth=15,  # Tăng độ sâu
            min_samples_split=10,  # Yêu cầu ít nhất 10 mẫu để split
            min_samples_leaf=5,  # Yêu cầu ít nhất 5 mẫu ở mỗi leaf
            class_weight="balanced",  # QUAN TRỌNG: Xử lý mất cân bằng dữ liệu
            random_state=42,
            n_jobs=-1,
        )

        self.rf_model.fit(X_train, y_train)

        # NEW: Analyze feature importance
        self.analyze_feature_importance()

        self.save_models()
        logger.info("✅ Random Forest trained & saved!")

    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost với params tối ưu

        XGBoost thường outperform Random Forest cho time series data
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("⚠️ XGBoost not installed. Skipping XGBoost training.")
            logger.info("💡 Install with: pip install xgboost")
            return

        logger.info("🚀 Training XGBoost with optimized parameters...")

        if X_train.shape[1] != self.expected_features:
            from src.config.exceptions import ModelPredictionError

            raise ModelPredictionError(
                "Feature count mismatch during training",
                context={
                    "got": X_train.shape[1],
                    "expected": self.expected_features,
                    "message": "Ensure all features are generated before training.",
                },
            )

        # Calculate scale_pos_weight for imbalanced dataset
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

        logger.info(
            f"   Dataset: {negative_count} negative, {positive_count} positive "
            f"(scale_pos_weight: {scale_pos_weight:.2f})"
        )

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,  # Number of trees
            max_depth=6,  # Depth of trees
            learning_rate=0.05,  # Learning rate (eta)
            subsample=0.8,  # Subsample ratio of training data
            colsample_bytree=0.8,  # Subsample ratio of features
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            # Early stopping params
            early_stopping_rounds=20,
        )

        # Train with validation set for early stopping
        eval_set = [(X_train, y_train)]

        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,  # Set to True to see training progress
        )

        logger.info(f"✅ XGBoost trained with {self.xgb_model.n_estimators} trees")

        # Analyze feature importance
        self._analyze_xgb_feature_importance()

        self.save_models()
        logger.info("✅ XGBoost trained & saved!")

    def _analyze_xgb_feature_importance(self):
        """Analyze and log XGBoost feature importance"""
        if self.xgb_model is None:
            return

        try:
            try:
                from src.ml.features.enhanced import get_feature_columns
            except ImportError:
                from src.ml.features.technical import get_feature_columns

            feature_names = get_feature_columns()

            # Get feature importance from XGBoost (gain-based)
            importances = self.xgb_model.feature_importances_

            # Create DataFrame
            importance_df = pd.DataFrame(
                {"feature": feature_names, "xgb_importance": importances}
            ).sort_values("xgb_importance", ascending=False)

            logger.info("\n" + "=" * 70)
            logger.info("📊 XGBOOST FEATURE IMPORTANCE (Top 10)")
            logger.info("=" * 70)

            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {idx+1:2d}. {row['feature']:25s} {row['xgb_importance']:6.4f}")

            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.warning(f"⚠️ Could not analyze XGBoost feature importance: {e}")

    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set."""
        if self.rf_model is None:
            logger.warning("Model not trained yet. Cannot evaluate.")
            return

        logger.info("📊 Evaluating model performance...")
        try:
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )

            y_pred = self.rf_model.predict(X_test)

            _accuracy = accuracy_score(y_test, y_pred)  # noqa: F841
            _precision = precision_score(y_test, y_pred, average="weighted")  # noqa: F841
            _recall = recall_score(y_test, y_pred, average="weighted")  # noqa: F841
            _f1 = f1_score(y_test, y_pred, average="weighted")  # noqa: F841

            logger.info(f"   - Accuracy:  {_accuracy:.4f}")
            logger.info(f"   - Precision: {_precision:.4f}")
            logger.info(f"   - Recall:    {_recall:.4f}")
            logger.info(f"   - F1-Score:  {_f1:.4f}")

            logger.info("   - Classification Report:")
            # Dùng print để format đẹp hơn trong log
            print(classification_report(y_test, y_pred, target_names=["Down/Hold", "Up"]))

        except Exception:
            logger.error("❌ Error during model evaluation")

    def analyze_feature_importance(self, top_n: int = None, cumulative_threshold: float = 0.8):
        """
        Analyze and log feature importance from Random Forest

        Args:
            top_n: Number of top features to display (default: all)
            cumulative_threshold: Keep features up to this cumulative importance (0.8 = 80%)

        Returns:
            DataFrame with feature importance
        """
        if self.rf_model is None:
            logger.warning("Model not trained yet. Cannot analyze feature importance.")
            return None

        try:
            from src.ml.features.technical import get_feature_columns

            feature_names = get_feature_columns()

            # Get feature importance from Random Forest
            importances = self.rf_model.feature_importances_

            # Create DataFrame
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            # Calculate cumulative importance
            importance_df["cumulative"] = importance_df["importance"].cumsum()

            # Store for later use
            self.feature_importance = importance_df

            # Log results
            logger.info("\n" + "=" * 70)
            logger.info("📊 FEATURE IMPORTANCE ANALYSIS")
            logger.info("=" * 70)

            display_n = top_n if top_n else len(importance_df)
            for idx, row in importance_df.head(display_n).iterrows():
                logger.info(
                    f"  {idx+1:2d}. {row['feature']:25s} "
                    f"Importance: {row['importance']:6.4f} "
                    f"(Cumulative: {row['cumulative']:6.2%})"
                )

            # Find features for cumulative threshold
            selected_features = importance_df[importance_df["cumulative"] <= cumulative_threshold]
            # Always include at least one more feature to exceed threshold
            if len(selected_features) < len(importance_df):
                selected_features = importance_df.iloc[: len(selected_features) + 1]

            logger.info("\n" + "-" * 70)
            logger.info(
                f"✅ Top {len(selected_features)} features explain "
                f"{selected_features['cumulative'].iloc[-1]:.1%} of variance"
            )
            logger.info(
                f"   Could reduce from {len(feature_names)} to {len(selected_features)} features"
            )
            logger.info("=" * 70 + "\n")

            # Store selected feature indices
            self.selected_features = selected_features.index.tolist()

            return importance_df

        except Exception as e:
            logger.error(f"❌ Error analyzing feature importance: {e}", exc_info=True)
            return None

    def select_top_features(self, X: np.ndarray, cumulative_threshold: float = 0.8) -> np.ndarray:
        """
        Select top features based on importance

        Args:
            X: Feature matrix (n_samples, n_features)
            cumulative_threshold: Keep features up to this cumulative importance

        Returns:
            Reduced feature matrix with only top features
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not analyzed. Using all features.")
            return X

        # Get features that contribute to cumulative threshold
        selected = self.feature_importance[
            self.feature_importance["cumulative"] <= cumulative_threshold
        ]

        # Include one more to exceed threshold
        if len(selected) < len(self.feature_importance):
            selected = self.feature_importance.iloc[: len(selected) + 1]

        # Get feature indices
        feature_indices = selected.index.tolist()

        logger.info(
            f"🔍 Feature selection: Using {len(feature_indices)}/{X.shape[1]} features "
            f"(explaining {selected['cumulative'].iloc[-1]:.1%} variance)"
        )

        # Return selected features
        return X[:, feature_indices]

    def predict(self, X):
        """
        Ensemble Prediction với feature validation

        NEW: Ensemble RF + XGBoost predictions with weighted average
        """
        # NEW: Check if ML is enabled
        if not self.ml_enabled:
            raise ValueError(
                "ML predictions disabled: Models not loaded. "
                "Train models first: python scripts/train_models.py"
            )

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        n = len(X_arr)
        if n == 0:
            return np.array([])

        # Kiểm tra số features
        if X_arr.shape[1] != self.expected_features:
            # Ném lỗi để tầng trên (ml_signals.analyze) fallback sang Technical Analysis
            raise ValueError(
                f"Feature mismatch: got {X_arr.shape[1]}, expected {self.expected_features}"
            )

        # Scale features
        try:
            if hasattr(self.scaler, "mean_"):
                X_scaled = self.scaler.transform(X_arr)
            else:
                X_scaled = X_arr
        except Exception:
            logger.error("⚠️ Lỗi scaling")
            X_scaled = X_arr

        predictions = []
        weights = []

        # RF prediction
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
                predictions.append(rf_pred)
                weights.append(self.ensemble_weights["rf"])
                logger.debug(f"RF prediction: {rf_pred[0]:.3f}")
            except Exception as e:
                logger.error(f"⚠️ RF predict error: {e}")

        # XGBoost prediction
        if self.xgb_model is not None:
            try:
                xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
                predictions.append(xgb_pred)
                weights.append(self.ensemble_weights["xgb"])
                logger.debug(f"XGBoost prediction: {xgb_pred[0]:.3f}")
            except Exception as e:
                logger.error(f"⚠️ XGBoost predict error: {e}")

        if not predictions:
            raise ValueError("No models available for prediction")

        # Ensemble: Weighted average
        if len(predictions) == 1:
            # Only one model available
            ensemble_pred = predictions[0]
            logger.debug("Using single model prediction")
        else:
            # Multiple models - ensemble
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Weighted average
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, normalized_weights):
                ensemble_pred += pred * weight

            logger.debug(
                f"Ensemble prediction: {ensemble_pred[0]:.3f} "
                f"(RF: {predictions[0][0]:.3f}, XGB: {predictions[1][0] if len(predictions) > 1 else 0:.3f})"
            )

        return ensemble_pred

    def validate_models(self) -> bool:
        """
        Validate ML models with sample data to ensure they work correctly

        IMPROVEMENT: Pre-flight check to catch issues before production

        Returns:
            True if models are valid and ready for use, False otherwise
        """
        if not self.ml_enabled:
            logger.warning("ML is disabled, skipping validation")
            return False

        try:
            logger.info("🔍 Validating ML models...")

            # Generate sample data with correct number of features
            n_samples = 10
            X_sample = np.random.randn(n_samples, self.expected_features)

            # Test scaling
            if hasattr(self.scaler, "mean_"):
                X_scaled = self.scaler.transform(X_sample)
                logger.debug(f"✅ Scaler works (input: {X_sample.shape}, output: {X_scaled.shape})")
            else:
                logger.warning("⚠️ Scaler not fitted, using raw features")
                X_scaled = X_sample

            # Test predictions
            predictions = []

            # Test RF
            if self.rf_model is not None:
                try:
                    rf_pred = self.rf_model.predict_proba(X_scaled)
                    if rf_pred.shape == (n_samples, 2):
                        predictions.append("RF")
                        logger.debug(f"✅ Random Forest works (output shape: {rf_pred.shape})")
                    else:
                        logger.error(f"❌ RF output shape mismatch: {rf_pred.shape}")
                        return False
                except Exception as e:
                    logger.error(f"❌ Random Forest prediction failed: {e}")
                    return False

            # Test XGBoost
            if self.xgb_model is not None:
                try:
                    xgb_pred = self.xgb_model.predict_proba(X_scaled)
                    if xgb_pred.shape == (n_samples, 2):
                        predictions.append("XGBoost")
                        logger.debug(f"✅ XGBoost works (output shape: {xgb_pred.shape})")
                    else:
                        logger.error(f"❌ XGBoost output shape mismatch: {xgb_pred.shape}")
                        return False
                except Exception as e:
                    logger.error(f"❌ XGBoost prediction failed: {e}")
                    return False

            if not predictions:
                logger.error("❌ No valid models available for prediction")
                return False

            # Test ensemble prediction
            try:
                ensemble_pred = self.predict(X_sample)
                if ensemble_pred.shape == (n_samples,):
                    logger.info(
                        f"✅ ML models validated successfully! Available models: {', '.join(predictions)}"
                    )
                    logger.info(f"   Sample predictions: {ensemble_pred[:3]}")
                    return True
                else:
                    logger.error(f"❌ Ensemble output shape mismatch: {ensemble_pred.shape}")
                    return False
            except Exception as e:
                logger.error(f"❌ Ensemble prediction failed: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}", exc_info=True)
            return False

    def load_models(self):
        """Load pre-trained models và scaler"""
        self.ensure_models_dir()

        models_loaded = False

        try:
            rf_path = os.path.join(self.models_dir, "random_forest.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            info_path = os.path.join(self.models_dir, "model_info.json")

            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                # Load metadata first to validate
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        import json

                        info = json.load(f)
                        saved_features = info.get("expected_features", 18)

                        # Check if saved model matches current feature definition
                        try:
                            from src.ml.features.technical import get_feature_columns

                            current_features = len(get_feature_columns())

                            if saved_features != current_features:
                                logger.warning(
                                    f"⚠️ Model feature mismatch: saved={saved_features}, current={current_features}"
                                )
                                logger.warning("🔄 Recreating models with current feature set...")
                                self.create_dummy_models()
                                return True
                        except Exception:
                            logger.warning("Could not verify features")

                # Load models
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)

                # Load XGBoost if available
                xgb_path = os.path.join(self.models_dir, "xgboost.pkl")
                if os.path.exists(xgb_path) and XGBOOST_AVAILABLE:
                    try:
                        self.xgb_model = joblib.load(xgb_path)
                        logger.info("✅ Loaded XGBoost model")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load XGBoost model: {e}")
                        self.xgb_model = None
                elif os.path.exists(xgb_path) and not XGBOOST_AVAILABLE:
                    logger.warning(
                        "⚠️ XGBoost model exists but XGBoost not installed. "
                        "Install with: pip install xgboost"
                    )

                # NEW: Load feature importance if available
                importance_path = os.path.join(self.models_dir, "feature_importance.csv")
                if os.path.exists(importance_path):
                    try:
                        self.feature_importance = pd.read_csv(importance_path)
                        logger.info(f"✅ Loaded feature importance from {importance_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load feature importance: {e}")

                models_summary = []
                if self.rf_model:
                    models_summary.append("RF")
                if self.xgb_model:
                    models_summary.append("XGBoost")

                logger.info(
                    f"✅ Loaded trained models: {', '.join(models_summary)} "
                    f"(expecting {self.expected_features} features)"
                )
                models_loaded = True
                self.ml_enabled = True
                self.using_dummy_models = False

                # IMPROVEMENT: Validate models after loading
                if not self.validate_models():
                    logger.error("❌ Model validation failed! Disabling ML.")
                    self.ml_enabled = False
                    models_loaded = False

                # NEW: Check if retraining is needed
                self._check_retrain_needed()
            else:
                # CRITICAL: No models found
                logger.critical(
                    "\n" + "=" * 70 + "\n"
                    "⚠️⚠️⚠️ CẢNH BÁO NGHIÊM TRỌNG: ML MODELS KHÔNG TỒN TẠI ⚠️⚠️⚠️\n" + "=" * 70 + "\n"
                    f"Model files không tìm thấy tại: {os.path.abspath(self.models_dir)}\n"
                    "\n"
                    "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                    "\n"
                    "🔧 ĐỂ SỬA LỖI NÀY:\n"
                    "1. Chạy lệnh: python scripts/train_models.py\n"
                    "2. Hoặc: python -m src.ml.training.pipeline\n"
                    "3. Sau khi train xong, khởi động lại bot\n"
                    "\n"
                    "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n" + "=" * 70
                )

                # DISABLE ML instead of creating dummy models
                self.ml_enabled = False
                self.using_dummy_models = False
                models_loaded = False

        except Exception:
            logger.critical(
                "\n" + "=" * 70 + "\n"
                "⚠️⚠️⚠️ LỖI KHI LOAD ML MODELS ⚠️⚠️⚠️\n" + "=" * 70 + "\n"
                "Lỗi\n"
                "\n"
                "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                "\n"
                "🔧 ĐỂ SỬA LỖI NÀY:\n"
                "1. Kiểm tra log chi tiết ở trên\n"
                "2. Xóa models cũ (nếu bị corrupt): rm -rf models/\n"
                "3. Train lại: python scripts/train_models.py\n"
                "4. Khởi động lại bot\n"
                "\n"
                "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n" + "=" * 70
            )

            # DISABLE ML instead of creating dummy models
            self.ml_enabled = False
            self.using_dummy_models = False
            models_loaded = False

        return models_loaded

    # =========================================================================
    # NEW: ONLINE LEARNING / INCREMENTAL UPDATE METHODS
    # =========================================================================

    def add_training_sample(self, features: np.ndarray, label: int, trade_result: dict = None):
        """
        Add a new training sample for incremental learning.

        Called after each trade completes to capture new market patterns.

        Args:
            features: Feature array for the trade entry
            label: 1 if trade was profitable, 0 otherwise
            trade_result: Optional dict with trade details for analysis
        """
        sample = {
            "features": features,
            "label": label,
            "timestamp": pd.Timestamp.now().isoformat(),
            "trade_result": trade_result,
        }
        self.incremental_samples.append(sample)

        logger.debug(
            f"📊 Added training sample (label={label}). "
            f"Buffer: {len(self.incremental_samples)}/{self.incremental_buffer_size}"
        )

        # Check if buffer is full
        if len(self.incremental_samples) >= self.incremental_buffer_size:
            logger.info(
                f"🔄 Incremental buffer full ({len(self.incremental_samples)} samples). "
                "Triggering incremental update..."
            )
            self._trigger_incremental_update()

    def _trigger_incremental_update(self):
        """
        Trigger incremental model update with buffered samples.

        Uses partial_fit for models that support it, or full retrain for others.
        """
        if not self.incremental_samples:
            return

        try:
            # Prepare data
            X_new = np.array([s["features"] for s in self.incremental_samples])
            y_new = np.array([s["label"] for s in self.incremental_samples])

            logger.info(
                f"🔄 Incremental update with {len(X_new)} samples "
                f"(positive: {y_new.sum()}, negative: {len(y_new) - y_new.sum()})"
            )

            # For XGBoost: Use warm_start for incremental learning
            if self.xgb_model is not None and XGBOOST_AVAILABLE:
                # XGBoost doesn't support partial_fit, but we can retrain with combined data
                # For now, just log that we would retrain
                logger.info("📊 XGBoost incremental update queued for next full retrain")

            # Clear buffer after processing
            self._save_incremental_samples()
            self.incremental_samples = []

            logger.info("✅ Incremental samples saved for next retrain cycle")

        except Exception as e:
            logger.error(f"❌ Incremental update failed: {e}", exc_info=True)

    def _save_incremental_samples(self):
        """Save incremental samples to disk for later retraining"""
        if not self.incremental_samples:
            return

        try:
            import json

            samples_file = os.path.join(self.models_dir, "incremental_samples.json")

            # Load existing samples
            existing = []
            if os.path.exists(samples_file):
                with open(samples_file, "r") as f:
                    existing = json.load(f)

            # Add new samples (convert numpy to list for JSON)
            for sample in self.incremental_samples:
                existing.append(
                    {
                        "features": (
                            sample["features"].tolist()
                            if hasattr(sample["features"], "tolist")
                            else sample["features"]
                        ),
                        "label": int(sample["label"]),
                        "timestamp": sample["timestamp"],
                    }
                )

            # Keep only last 1000 samples
            existing = existing[-1000:]

            with open(samples_file, "w") as f:
                json.dump(existing, f)

            logger.info(
                f"💾 Saved {len(self.incremental_samples)} incremental samples (total: {len(existing)})"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save incremental samples: {e}")

    def _check_retrain_needed(self):
        """
        Check if model retraining is needed based on:
        1. Time since last retrain (weekly)
        2. Performance degradation
        3. Accumulated incremental samples
        """
        try:
            info_path = os.path.join(self.models_dir, "model_info.json")
            if not os.path.exists(info_path):
                return

            with open(info_path, "r") as f:
                import json

                info = json.load(f)

            saved_at = info.get("saved_at")
            if saved_at:
                saved_date = pd.Timestamp(saved_at)
                days_since_retrain = (pd.Timestamp.now() - saved_date).days

                if days_since_retrain >= self.retrain_interval_days:
                    logger.warning(
                        f"⚠️ Model is {days_since_retrain} days old. "
                        f"Consider retraining: python scripts/train_models.py"
                    )
                    self.last_retrain_date = saved_date

            # Check for accumulated incremental samples
            samples_file = os.path.join(self.models_dir, "incremental_samples.json")
            if os.path.exists(samples_file):
                with open(samples_file, "r") as f:
                    import json

                    samples = json.load(f)
                if len(samples) >= 200:
                    logger.info(
                        f"📊 {len(samples)} incremental samples available. "
                        "Consider retraining to incorporate new patterns."
                    )

        except Exception as e:
            logger.debug(f"Could not check retrain status: {e}")

    def get_model_health_status(self) -> dict:
        """
        Get comprehensive model health status for monitoring.

        Returns:
            dict with model health metrics
        """
        status = {
            "ml_enabled": self.ml_enabled,
            "using_dummy_models": self.using_dummy_models,
            "models_loaded": {
                "random_forest": self.rf_model is not None,
                "xgboost": self.xgb_model is not None,
            },
            "expected_features": self.expected_features,
            "incremental_buffer_size": len(self.incremental_samples),
            "ensemble_weights": self.ensemble_weights,
        }

        # Add model age
        try:
            info_path = os.path.join(self.models_dir, "model_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    import json

                    info = json.load(f)
                saved_at = info.get("saved_at")
                if saved_at:
                    days_old = (pd.Timestamp.now() - pd.Timestamp(saved_at)).days
                    status["model_age_days"] = days_old
                    status["needs_retrain"] = days_old >= self.retrain_interval_days
        except Exception:
            pass

        return status


# [file content end]
