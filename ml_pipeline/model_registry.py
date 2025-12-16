"""
Model Registry - Quản lý và versioning ML models.

Hỗ trợ:
- Model versioning với semantic versioning (v1.0.0)
- Model metadata tracking
- Model comparison và selection
- Automatic model backup và rollback
- Model lifecycle management
- Production model selection
"""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Các giai đoạn của model trong lifecycle."""

    DEVELOPMENT = "development"  # Đang phát triển
    STAGING = "staging"  # Đang test
    PRODUCTION = "production"  # Đang sử dụng
    ARCHIVED = "archived"  # Lưu trữ
    DEPRECATED = "deprecated"  # Không còn sử dụng


class ModelType(Enum):
    """Các loại model."""

    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    STACKING_ENSEMBLE = "stacking_ensemble"
    VOLATILITY_FORECASTER = "volatility_forecaster"
    SENTIMENT = "sentiment"


@dataclass
class ModelMetrics:
    """Metrics của model."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    sharpe_ratio: Optional[float] = None  # Backtesting metric
    max_drawdown: Optional[float] = None  # Backtesting metric
    win_rate: Optional[float] = None  # Trading metric
    profit_factor: Optional[float] = None  # Trading metric

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ModelVersion:
    """Thông tin một version của model."""

    version: str
    model_type: str
    created_at: str
    stage: str = ModelStage.DEVELOPMENT.value
    description: str = ""
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    feature_columns: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    file_hash: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        metrics_data = data.pop("metrics", {})
        metrics = ModelMetrics(**metrics_data) if metrics_data else ModelMetrics()
        return cls(metrics=metrics, **data)


class ModelRegistry:
    """
    Registry để quản lý và versioning ML models.

    Features:
    - Đăng ký model với metadata
    - Track model versions
    - Model comparison
    - Production model selection
    - Automatic backup và rollback
    """

    def __init__(self, registry_path: str = "models"):
        """
        Khởi tạo Model Registry.

        Args:
            registry_path: Đường dẫn thư mục chứa models
        """
        self.registry_path = Path(registry_path)
        self.registry_file = self.registry_path / "registry.json"
        self.models: Dict[str, Dict[str, ModelVersion]] = (
            {}
        )  # {model_type: {version: ModelVersion}}
        self.production_models: Dict[str, str] = {}  # {model_type: version}

        self._ensure_directories()
        self._load_registry()

    def _ensure_directories(self):
        """Tạo các thư mục cần thiết."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        (self.registry_path / "backups").mkdir(exist_ok=True)
        (self.registry_path / "staging").mkdir(exist_ok=True)

    def _load_registry(self):
        """Load registry từ file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load models
                for model_type, versions in data.get("models", {}).items():
                    self.models[model_type] = {}
                    for version, version_data in versions.items():
                        self.models[model_type][version] = ModelVersion.from_dict(version_data)

                # Load production models
                self.production_models = data.get("production_models", {})

                logger.info(f"Loaded registry với {len(self.models)} model types")
            except Exception as e:
                logger.error(f"Lỗi load registry: {e}")

    def _save_registry(self):
        """Lưu registry vào file."""
        data = {
            "models": {
                model_type: {version: mv.to_dict() for version, mv in versions.items()}
                for model_type, versions in self.models.items()
            },
            "production_models": self.production_models,
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("Đã lưu registry")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Tính hash của file để detect thay đổi."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse version string (v1.2.3) thành tuple."""
        version = version.lstrip("v")
        parts = version.split(".")
        return tuple(int(p) for p in parts[:3])

    def _get_next_version(self, model_type: str, bump: str = "patch") -> str:
        """
        Tính version tiếp theo.

        Args:
            model_type: Loại model
            bump: "major", "minor", hoặc "patch"
        """
        if model_type not in self.models or not self.models[model_type]:
            return "v1.0.0"

        # Lấy version cao nhất
        versions = list(self.models[model_type].keys())
        latest = max(versions, key=lambda v: self._parse_version(v))
        major, minor, patch = self._parse_version(latest)

        if bump == "major":
            return f"v{major + 1}.0.0"
        elif bump == "minor":
            return f"v{major}.{minor + 1}.0"
        else:
            return f"v{major}.{minor}.{patch + 1}"

    def register_model(
        self,
        model: Any,
        model_type: str,
        metrics: Optional[ModelMetrics] = None,
        version: Optional[str] = None,
        version_bump: str = "patch",
        description: str = "",
        feature_columns: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
    ) -> str:
        """
        Đăng ký model mới vào registry.

        Args:
            model: Model object (sklearn, xgboost, etc.)
            model_type: Loại model (từ ModelType enum)
            metrics: Metrics của model
            version: Version cụ thể (nếu None, auto-increment)
            version_bump: "major", "minor", "patch"
            description: Mô tả
            feature_columns: Danh sách features
            hyperparameters: Hyperparameters đã sử dụng
            training_data_info: Thông tin training data
            tags: Tags để phân loại
            stage: Stage của model

        Returns:
            Version string đã đăng ký
        """
        # Xác định version
        if version is None:
            version = self._get_next_version(model_type, version_bump)

        # Tạo file path
        model_filename = f"{model_type}_{version}.pkl"
        model_path = self.registry_path / model_filename

        # Lưu model
        joblib.dump(model, model_path)
        file_hash = self._compute_file_hash(model_path)

        # Tạo model version
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            stage=stage.value,
            description=description,
            metrics=metrics or ModelMetrics(),
            feature_columns=feature_columns or [],
            hyperparameters=hyperparameters or {},
            training_data_info=training_data_info or {},
            file_path=str(model_path),
            file_hash=file_hash,
            tags=tags or [],
        )

        # Đăng ký vào registry
        if model_type not in self.models:
            self.models[model_type] = {}
        self.models[model_type][version] = model_version

        self._save_registry()
        logger.info(f"Đã đăng ký model {model_type} version {version}")

        return version

    def load_model(
        self,
        model_type: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> Tuple[Any, ModelVersion]:
        """
        Load model từ registry.

        Args:
            model_type: Loại model
            version: Version cụ thể (nếu None, load production hoặc latest)
            stage: Stage của model (nếu cung cấp, override version)

        Returns:
            Tuple (model, ModelVersion)
        """
        if model_type not in self.models:
            raise ValueError(f"Không tìm thấy model type: {model_type}")

        # Xác định version để load
        if version is None:
            if stage:
                # Tìm model với stage cụ thể
                matching = [
                    v for v, mv in self.models[model_type].items() if mv.stage == stage.value
                ]
                if not matching:
                    raise ValueError(f"Không có model {model_type} ở stage {stage.value}")
                version = max(matching, key=lambda v: self._parse_version(v))
            elif model_type in self.production_models:
                version = self.production_models[model_type]
            else:
                # Load latest
                version = max(self.models[model_type].keys(), key=lambda v: self._parse_version(v))

        if version not in self.models[model_type]:
            raise ValueError(f"Không tìm thấy version {version} cho {model_type}")

        model_version = self.models[model_type][version]
        model = joblib.load(model_version.file_path)

        logger.info(f"Đã load model {model_type} version {version}")
        return model, model_version

    def promote_model(
        self,
        model_type: str,
        version: str,
        to_stage: ModelStage,
    ):
        """
        Promote model lên stage cao hơn.

        Args:
            model_type: Loại model
            version: Version cần promote
            to_stage: Stage đích
        """
        if model_type not in self.models:
            raise ValueError(f"Không tìm thấy model type: {model_type}")
        if version not in self.models[model_type]:
            raise ValueError(f"Không tìm thấy version {version}")

        # Backup trước khi promote
        if to_stage == ModelStage.PRODUCTION:
            self._backup_production_model(model_type)

            # Demote model production hiện tại
            if model_type in self.production_models:
                old_version = self.production_models[model_type]
                if old_version in self.models[model_type]:
                    self.models[model_type][old_version].stage = ModelStage.ARCHIVED.value

            self.production_models[model_type] = version

        self.models[model_type][version].stage = to_stage.value
        self._save_registry()

        logger.info(f"Đã promote {model_type} {version} lên {to_stage.value}")

    def _backup_production_model(self, model_type: str):
        """Backup model production hiện tại."""
        if model_type not in self.production_models:
            return

        version = self.production_models[model_type]
        model_version = self.models[model_type].get(version)
        if not model_version:
            return

        source = Path(model_version.file_path)
        if source.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_type}_{version}_backup_{timestamp}.pkl"
            dest = self.registry_path / "backups" / backup_name
            shutil.copy(source, dest)
            logger.info(f"Đã backup {model_type} {version} vào {dest}")

    def rollback_model(self, model_type: str, to_version: str):
        """
        Rollback production model về version trước.

        Args:
            model_type: Loại model
            to_version: Version cần rollback về
        """
        if model_type not in self.models:
            raise ValueError(f"Không tìm thấy model type: {model_type}")
        if to_version not in self.models[model_type]:
            raise ValueError(f"Không tìm thấy version {to_version}")

        # Demote model hiện tại
        if model_type in self.production_models:
            current = self.production_models[model_type]
            self.models[model_type][current].stage = ModelStage.ARCHIVED.value

        # Promote version cũ
        self.models[model_type][to_version].stage = ModelStage.PRODUCTION.value
        self.production_models[model_type] = to_version

        self._save_registry()
        logger.info(f"Đã rollback {model_type} về version {to_version}")

    def compare_models(
        self,
        model_type: str,
        versions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        So sánh các versions của model.

        Args:
            model_type: Loại model
            versions: Danh sách versions cần so sánh (None = tất cả)

        Returns:
            List các model info để so sánh
        """
        if model_type not in self.models:
            return []

        if versions is None:
            versions = list(self.models[model_type].keys())

        results = []
        for version in versions:
            if version in self.models[model_type]:
                mv = self.models[model_type][version]
                results.append(
                    {
                        "version": version,
                        "stage": mv.stage,
                        "created_at": mv.created_at,
                        "metrics": mv.metrics.to_dict(),
                        "description": mv.description,
                    }
                )

        # Sort by version
        results.sort(key=lambda x: self._parse_version(x["version"]), reverse=True)
        return results

    def list_models(
        self,
        model_type: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> Dict[str, List[str]]:
        """
        Liệt kê models trong registry.

        Args:
            model_type: Lọc theo loại model
            stage: Lọc theo stage

        Returns:
            Dict {model_type: [versions]}
        """
        results = {}

        for mt, versions in self.models.items():
            if model_type and mt != model_type:
                continue

            filtered_versions = []
            for version, mv in versions.items():
                if stage and mv.stage != stage.value:
                    continue
                filtered_versions.append(version)

            if filtered_versions:
                results[mt] = sorted(
                    filtered_versions, key=lambda v: self._parse_version(v), reverse=True
                )

        return results

    def get_production_models(self) -> Dict[str, ModelVersion]:
        """Lấy tất cả production models."""
        result = {}
        for model_type, version in self.production_models.items():
            if model_type in self.models and version in self.models[model_type]:
                result[model_type] = self.models[model_type][version]
        return result

    def delete_model(
        self,
        model_type: str,
        version: str,
        force: bool = False,
    ):
        """
        Xóa model khỏi registry.

        Args:
            model_type: Loại model
            version: Version cần xóa
            force: Nếu True, có thể xóa production model
        """
        if model_type not in self.models:
            raise ValueError(f"Không tìm thấy model type: {model_type}")
        if version not in self.models[model_type]:
            raise ValueError(f"Không tìm thấy version {version}")

        mv = self.models[model_type][version]

        # Không cho xóa production model trừ khi force
        if mv.stage == ModelStage.PRODUCTION.value and not force:
            raise ValueError("Không thể xóa production model. Sử dụng force=True nếu chắc chắn.")

        # Xóa file
        if os.path.exists(mv.file_path):
            os.remove(mv.file_path)

        # Xóa khỏi registry
        del self.models[model_type][version]

        # Cập nhật production models nếu cần
        if model_type in self.production_models and self.production_models[model_type] == version:
            del self.production_models[model_type]

        self._save_registry()
        logger.info(f"Đã xóa model {model_type} version {version}")

    def get_model_info(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """Lấy thông tin chi tiết của model."""
        if model_type in self.models and version in self.models[model_type]:
            return self.models[model_type][version]
        return None

    def update_model_metrics(
        self,
        model_type: str,
        version: str,
        metrics: ModelMetrics,
    ):
        """Cập nhật metrics của model sau khi evaluate."""
        if model_type not in self.models or version not in self.models[model_type]:
            raise ValueError(f"Không tìm thấy model {model_type} {version}")

        self.models[model_type][version].metrics = metrics
        self._save_registry()
        logger.info(f"Đã cập nhật metrics cho {model_type} {version}")

    def add_tags(self, model_type: str, version: str, tags: List[str]):
        """Thêm tags cho model."""
        if model_type not in self.models or version not in self.models[model_type]:
            raise ValueError(f"Không tìm thấy model {model_type} {version}")

        existing_tags = set(self.models[model_type][version].tags)
        existing_tags.update(tags)
        self.models[model_type][version].tags = list(existing_tags)
        self._save_registry()

    def search_by_tags(self, tags: List[str]) -> List[Tuple[str, str, ModelVersion]]:
        """Tìm models theo tags."""
        results = []
        for model_type, versions in self.models.items():
            for version, mv in versions.items():
                if any(tag in mv.tags for tag in tags):
                    results.append((model_type, version, mv))
        return results

    def get_registry_stats(self) -> Dict[str, Any]:
        """Lấy thống kê của registry."""
        total_models = sum(len(v) for v in self.models.values())

        stage_counts = {}
        for versions in self.models.values():
            for mv in versions.values():
                stage_counts[mv.stage] = stage_counts.get(mv.stage, 0) + 1

        return {
            "total_model_types": len(self.models),
            "total_versions": total_models,
            "production_models": len(self.production_models),
            "stage_distribution": stage_counts,
            "model_types": list(self.models.keys()),
        }


# Singleton instance
_registry: Optional[ModelRegistry] = None


def get_registry(registry_path: str = "models") -> ModelRegistry:
    """Lấy singleton instance của ModelRegistry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(registry_path)
    return _registry
