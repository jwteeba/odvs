"""
ODVS Configuration using environment-aware settings.
Follows 12-factor app principles. All values can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class S3Config:
    endpoint_url: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"))
    access_key: str = field(default_factory=lambda: os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    region: str = field(default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    bucket: str = field(default_factory=lambda: os.environ.get("ODVS_S3_BUCKET", "odvs-warehouse"))
    warehouse_prefix: str = "warehouse"

    @property
    def warehouse_path(self) -> str:
        return f"s3a://{self.bucket}/{self.warehouse_prefix}"


@dataclass(frozen=True)
class SparkConfig:
    app_name: str = "ODVS"
    master: str = field(default_factory=lambda: os.environ.get("SPARK_MASTER", "local[*]"))
    driver_memory: str = field(default_factory=lambda: os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
    executor_memory: str = field(default_factory=lambda: os.environ.get("SPARK_EXECUTOR_MEMORY", "4g"))
    iceberg_catalog_name: str = "odvs_catalog"
    iceberg_catalog_type: str = "hadoop"
    # Iceberg + Spark package versions — pin for reproducibility
    spark_version: str = "4.0"
    scala_version: str = "2.13"
    iceberg_version: str = "1.10.1"
    aws_bundle_version: str = "1.10.1"


@dataclass(frozen=True)
class IcebergConfig:
    catalog_name: str = "odvs_catalog"
    database: str = field(default_factory=lambda: os.environ.get("ODVS_ICEBERG_DB", "odvs"))
    default_write_format: str = "parquet"
    default_compression: str = "zstd"
    target_file_size_bytes: int = 128 * 1024 * 1024  # 128 MB
    sort_order_enabled: bool = True


@dataclass(frozen=True)
class RegistryConfig:
    registry_path: str = field(
        default_factory=lambda: os.environ.get(
            "ODVS_REGISTRY_PATH",
            str(Path.home() / ".odvs" / "registry"),
        )
    )
    registry_file: str = "datasets.json"

    @property
    def registry_filepath(self) -> Path:
        return Path(self.registry_path) / self.registry_file


@dataclass(frozen=True)
class LoggingConfig:
    level: str = field(default_factory=lambda: os.environ.get("ODVS_LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%dT%H:%M:%S"
    structured: bool = field(default_factory=lambda: os.environ.get("ODVS_STRUCTURED_LOGS", "false").lower() == "true")


@dataclass(frozen=True)
class ODVSConfig:
    s3: S3Config = field(default_factory=S3Config)
    spark: SparkConfig = field(default_factory=SparkConfig)
    iceberg: IcebergConfig = field(default_factory=IcebergConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    environment: str = field(default_factory=lambda: os.environ.get("ODVS_ENV", "development"))

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_config() -> ODVSConfig:
    """
    Singleton config accessor. Cached after first call.
    Use invalidate_config() in tests to reset.
    """
    return ODVSConfig()


def invalidate_config() -> None:
    """Clear cached config — useful in tests."""
    get_config.cache_clear()