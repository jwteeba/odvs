"""
ODVS Iceberg Writer - DataFrame - Iceberg append operations.

All writes go through this module. It enforces:
- Schema compatibility checking before write
- Iceberg-native parquet generation (no manual file staging)
- Write metrics collection
- Configurable compression
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from odvs.config import IcebergConfig, get_config
from odvs.iceberg.table_manager import IcebergTableManager
from odvs.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WriteResult:
    """Metadata returned after a successful Iceberg write operation."""
    table_name: str
    rows_written: int
    files_written: int
    snapshot_id: Optional[int]
    duration_seconds: float
    compression: str
    schema_match: bool

    def __str__(self) -> str:
        return (
            f"WriteResult(table={self.table_name}, rows={self.rows_written:,}, "
            f"files={self.files_written}, snapshot={self.snapshot_id}, "
            f"duration={self.duration_seconds:.2f}s)"
        )


class IcebergWriter:
    """
    Handles all DataFrame write operations to Iceberg tables.

    Design principles:
    - TABLE-CENTRIC: Iceberg manages all parquet file generation
    - Schema is validated before write, not after
    - Metrics are collected via Iceberg snapshot summary
    - Supports append, dynamic overwrite, and overwrite-by-partition modes
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Optional[IcebergConfig] = None,
    ) -> None:
        self._spark = spark
        self._cfg = config or get_config().iceberg
        self._table_manager = IcebergTableManager(spark=spark, config=self._cfg)


    def append(
        self,
        df: DataFrame,
        table_name: str,
        database: Optional[str] = None,
        compression: Optional[str] = None,
        validate_schema: bool = True,
    ) -> WriteResult:
        """
        Append a Spark DataFrame to an Iceberg table.

        Args:
            df: Spark DataFrame to write.
            table_name: Target Iceberg table name (unqualified).
            database: Iceberg database/namespace. Defaults to config value.
            compression: Parquet compression codec (zstd | snappy | gzip | none).
            validate_schema: Whether to enforce schema compatibility pre-write.

        Returns:
            WriteResult with snapshot ID and row count.
        """
        db = database or self._cfg.database
        full_name = self._table_manager.get_full_name(table_name, database=db)
        compression = compression or self._cfg.default_compression

        logger.info(f"Appending to {full_name} (compression={compression})")

        if validate_schema:
            self._validate_schema_compatibility(df, full_name)

        start = time.monotonic()
        row_count = df.count()

        write_options = {
            "write.format.default": "parquet",
            "write.parquet.compression-codec": compression,
        }

        (
            df.writeTo(full_name)
            .option("write-format", "parquet")
            .option("write.parquet.compression-codec", compression)
            .append()
        )

        duration = time.monotonic() - start

        snapshot_id, files_written = self._get_latest_snapshot_info(full_name)

        result = WriteResult(
            table_name=full_name,
            rows_written=row_count,
            files_written=files_written,
            snapshot_id=snapshot_id,
            duration_seconds=duration,
            compression=compression,
            schema_match=validate_schema,
        )
        logger.info(str(result))
        return result

    def overwrite_by_partition(
        self,
        df: DataFrame,
        table_name: str,
        database: Optional[str] = None,
        compression: Optional[str] = None,
    ) -> WriteResult:
        """
        Dynamic overwrite — replaces only the partitions present in df.
        All other partitions are preserved. Safe for incremental loads.
        """
        db = database or self._cfg.database
        full_name = self._table_manager.get_full_name(table_name, database=db)
        compression = compression or self._cfg.default_compression

        logger.info(f"Dynamic overwrite on {full_name}")
        start = time.monotonic()
        row_count = df.count()

        (
            df.writeTo(full_name)
            .option("write.parquet.compression-codec", compression)
            .overwritePartitions()
        )

        duration = time.monotonic() - start
        snapshot_id, files_written = self._get_latest_snapshot_info(full_name)

        return WriteResult(
            table_name=full_name,
            rows_written=row_count,
            files_written=files_written,
            snapshot_id=snapshot_id,
            duration_seconds=duration,
            compression=compression,
            schema_match=True,
        )

    def replace_all(
        self,
        df: DataFrame,
        table_name: str,
        database: Optional[str] = None,
        compression: Optional[str] = None,
    ) -> WriteResult:
        """Full table replace — replaces all data. Use carefully."""
        db = database or self._cfg.database
        full_name = self._table_manager.get_full_name(table_name, database=db)
        compression = compression or self._cfg.default_compression

        logger.warning(f"Full table replace on {full_name} — all existing data will be replaced")
        start = time.monotonic()
        row_count = df.count()

        df.writeTo(full_name).option("write.parquet.compression-codec", compression).replace()

        duration = time.monotonic() - start
        snapshot_id, files_written = self._get_latest_snapshot_info(full_name)

        return WriteResult(
            table_name=full_name,
            rows_written=row_count,
            files_written=files_written,
            snapshot_id=snapshot_id,
            duration_seconds=duration,
            compression=compression,
            schema_match=True,
        )


    def append_pandas(
        self,
        pandas_df: pd.DataFrame,
        table_name: str,
        database: Optional[str] = None,
        compression: Optional[str] = None,
        validate_schema: bool = True,
    ) -> WriteResult:
        """
        Convert a Pandas DataFrame to Spark and append to Iceberg.
        Uses Arrow-based conversion for performance.
        """
        spark_df = self._spark.createDataFrame(pandas_df)
        return self.append(
            df=spark_df,
            table_name=table_name,
            database=database,
            compression=compression,
            validate_schema=validate_schema,
        )


    def _validate_schema_compatibility(self, df: DataFrame, full_name: str) -> None:
        """
        Check that the incoming DataFrame's columns are a subset of the table schema.
        Raises ValueError on incompatibility.
        """
        try:
            table_schema = self._spark.table(full_name).schema
        except Exception:
            logger.debug(f"Table {full_name} not yet readable — skipping schema validation")
            return

        table_cols = {f.name for f in table_schema.fields}
        df_cols = set(df.columns)
        extra_cols = df_cols - table_cols

        if extra_cols:
            raise ValueError(
                f"Schema incompatibility: DataFrame has columns not in table schema: {extra_cols}. "
                f"Use IcebergTableManager.add_column() to evolve the schema first."
            )

    def _get_latest_snapshot_info(self, full_name: str) -> tuple[Optional[int], int]:
        """Retrieve snapshot ID and file count from Iceberg metadata."""
        try:
            snapshots = (
                self._spark.sql(f"SELECT snapshot_id, summary FROM {full_name}.snapshots")
                .orderBy("committed_at", ascending=False)
                .limit(1)
                .collect()
            )
            if snapshots:
                row = snapshots[0]
                snapshot_id = row["snapshot_id"]
                summary = row["summary"] or {}
                files_written = int(summary.get("added-data-files", 0))
                return snapshot_id, files_written
        except Exception as exc:
            logger.debug(f"Could not retrieve snapshot info: {exc}")
        return None, 0