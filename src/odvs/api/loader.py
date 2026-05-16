"""
ODVS API Loader — The primary Python API for downstream dataset consumers.

This is the module users import when consuming ODVS-managed datasets.
It provides a stable, version-aware interface for:
- Loading datasets as pandas or Spark DataFrames
- Time-travel reads by version tag or snapshot ID
- Dataset metadata access
- Schema inspection
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from odvs.config import get_config
from odvs.iceberg.versioning import IcebergVersionManager
from odvs.logger import get_logger
from odvs.registry.dataset_registry import DatasetRegistry, DatasetNotFoundError

logger = get_logger(__name__)


class DatasetLoader:
    """
    High-level dataset loading API.

    Usage:
        loader = DatasetLoader()

        # Load latest version as pandas
        df = loader.load("my_dataset")

        # Load specific version
        df = loader.load("my_dataset", version="v1.2.0")

        # Load at a specific point in time
        df = loader.load("my_dataset", as_of=datetime(2024, 6, 1))

        # Load as Spark DataFrame
        spark_df = loader.load_spark("my_dataset")
    """

    def __init__(
        self,
        registry: Optional[DatasetRegistry] = None,
    ) -> None:
        self._registry = registry or DatasetRegistry()

    def load(
        self,
        dataset_name: str,
        version: Optional[str] = None,
        as_of: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load a dataset as a pandas DataFrame.

        Args:
            dataset_name: Name of the dataset in the ODVS registry.
            version: Version tag (e.g. "v1.0.0"). Defaults to latest.
            as_of: Load table state as of this timestamp (time-travel).
            columns: Subset of columns to return.
            limit: Maximum number of rows to return.

        Returns:
            pandas DataFrame.
        """
        ds = self._registry.get_dataset(dataset_name)
        logger.info(f"Loading dataset: {dataset_name} (version={version or 'latest'})")

        spark_df = self.load_spark(
            dataset_name=dataset_name,
            version=version,
            as_of=as_of,
            columns=columns,
        )

        if limit is not None:
            spark_df = spark_df.limit(limit)

        return spark_df.toPandas()

    def load_spark(
        self,
        dataset_name: str,
        version: Optional[str] = None,
        as_of: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> "pyspark.sql.DataFrame":  # type: ignore[name-defined]
        """
        Load a dataset as a Spark DataFrame backed by Iceberg.

        Returns a lazy Spark DataFrame — no data is read until action is called.
        """
        from odvs.iceberg.spark_session import get_or_create_session
        from odvs.iceberg.versioning import IcebergVersionManager

        cfg = get_config()
        spark = get_or_create_session()
        version_mgr = IcebergVersionManager(spark=spark)

        ds = self._registry.get_dataset(dataset_name)
        table_name = dataset_name.replace("-", "_").replace("/", "__")

        if as_of is not None:
            df = version_mgr.read_at_timestamp(table_name, as_of=as_of)
        elif version is not None:
            version_record = self._registry.get_version(dataset_name, version)
            snapshot_id = version_record.get("snapshot_id")
            if snapshot_id:
                df = version_mgr.read_at_snapshot(table_name, snapshot_id=snapshot_id)
            else:
                logger.warning(
                    f"Version '{version}' has no snapshot_id, falling back to current table state"
                )
                df = spark.table(f"{cfg.iceberg.catalog_name}.{cfg.iceberg.database}.{table_name}")
        else:
            df = spark.table(f"{cfg.iceberg.catalog_name}.{cfg.iceberg.database}.{table_name}")

        if columns:
            df = df.select(columns)

        return df

    def load_version_as_of_snapshot(
        self,
        dataset_name: str,
        snapshot_id: int,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load a dataset at a specific Iceberg snapshot ID."""
        from odvs.iceberg.spark_session import get_or_create_session
        from odvs.iceberg.versioning import IcebergVersionManager

        spark = get_or_create_session()
        version_mgr = IcebergVersionManager(spark=spark)
        table_name = dataset_name.replace("-", "_").replace("/", "__")
        df = version_mgr.read_at_snapshot(table_name, snapshot_id=snapshot_id)

        if columns:
            df = df.select(columns)

        return df.toPandas()


    def info(self, dataset_name: str) -> Dict[str, Any]:
        """Return full registry metadata for a dataset."""
        return self._registry.get_dataset(dataset_name)

    def versions(self, dataset_name: str) -> List[str]:
        """Return all version tags for a dataset."""
        return self._registry.list_versions(dataset_name)

    def schema(self, dataset_name: str) -> Dict[str, str]:
        """Return the latest schema for a dataset."""
        return self._registry.get_dataset(dataset_name).get("schema", {})

    def exists(self, dataset_name: str) -> bool:
        try:
            self._registry.get_dataset(dataset_name)
            return True
        except DatasetNotFoundError:
            return False

    def list_datasets(self) -> List[str]:
        return self._registry.list_datasets()



def load_dataset(
    dataset_name: str,
    version: Optional[str] = None,
    as_of: Optional[datetime] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Top-level convenience function.

    Usage:
        from odvs.api.loader import load_dataset
        df = load_dataset("my_dataset", version="v1.0.0")
    """
    return DatasetLoader().load(
        dataset_name=dataset_name,
        version=version,
        as_of=as_of,
        columns=columns,
        limit=limit,
    )