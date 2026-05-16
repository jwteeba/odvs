"""
ODVS Iceberg Versioning — Snapshot management, time travel, and version querying.

This module provides the developer-facing API for working with Iceberg's
snapshot history, enabling rollback, diff, and audit workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession

from odvs.config import IcebergConfig, get_config
from odvs.iceberg.table_manager import IcebergTableManager
from odvs.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Snapshot:
    """Represents a single Iceberg snapshot."""
    snapshot_id: int
    committed_at: datetime
    operation: str  # append | overwrite | replace | delete
    summary: Dict[str, str]
    manifest_list: Optional[str]

    @property
    def added_files(self) -> int:
        return int(self.summary.get("added-data-files", 0))

    @property
    def added_records(self) -> int:
        return int(self.summary.get("added-records", 0))

    @property
    def deleted_files(self) -> int:
        return int(self.summary.get("deleted-data-files", 0))

    @property
    def total_records(self) -> int:
        return int(self.summary.get("total-records", 0))

    @property
    def total_data_files(self) -> int:
        return int(self.summary.get("total-data-files", 0))

    def __str__(self) -> str:
        return (
            f"Snapshot(id={self.snapshot_id}, op={self.operation}, "
            f"ts={self.committed_at.isoformat()}, "
            f"records={self.added_records:,}/{self.total_records:,})"
        )


class IcebergVersionManager:
    """
    Manages Iceberg snapshot lifecycle: list, rollback, time-travel read, expiry.

    This module is READ-HEAVY — it wraps Iceberg metadata tables.
    Write operations (appends, replaces) live in IcebergWriter.
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Optional[IcebergConfig] = None,
    ) -> None:
        self._spark = spark
        self._cfg = config or get_config().iceberg
        self._table_manager = IcebergTableManager(spark=spark, config=self._cfg)


    def list_snapshots(
        self,
        table_name: str,
        database: Optional[str] = None,
        limit: int = 50,
    ) -> List[Snapshot]:
        """Return all snapshots for the given table, newest first."""
        full_name = self._table_manager.get_full_name(table_name, database=database)

        rows = (
            self._spark.sql(
                f"""
                SELECT
                    snapshot_id,
                    committed_at,
                    operation,
                    summary,
                    manifest_list
                FROM {full_name}.snapshots
                ORDER BY committed_at DESC
                LIMIT {limit}
                """
            )
            .collect()
        )

        snapshots = []
        for row in rows:
            snapshots.append(
                Snapshot(
                    snapshot_id=row["snapshot_id"],
                    committed_at=row["committed_at"].astimezone(timezone.utc)
                    if hasattr(row["committed_at"], "astimezone")
                    else datetime.fromtimestamp(row["committed_at"] / 1000, tz=timezone.utc),
                    operation=row["operation"],
                    summary=dict(row["summary"]) if row["summary"] else {},
                    manifest_list=row["manifest_list"],
                )
            )
        return snapshots

    def get_current_snapshot(
        self,
        table_name: str,
        database: Optional[str] = None,
    ) -> Optional[Snapshot]:
        snapshots = self.list_snapshots(table_name, database=database, limit=1)
        return snapshots[0] if snapshots else None

    def get_snapshot_by_id(
        self,
        table_name: str,
        snapshot_id: int,
        database: Optional[str] = None,
    ) -> Optional[Snapshot]:
        snapshots = self.list_snapshots(table_name, database=database, limit=500)
        for s in snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        return None


    def read_at_snapshot(
        self,
        table_name: str,
        snapshot_id: int,
        database: Optional[str] = None,
    ) -> DataFrame:
        """Return a DataFrame representing table state at the given snapshot."""
        full_name = self._table_manager.get_full_name(table_name, database=database)
        logger.info(f"Time-travel read: {full_name} @ snapshot {snapshot_id}")
        return self._spark.read.option("snapshot-id", snapshot_id).table(full_name)

    def read_at_timestamp(
        self,
        table_name: str,
        as_of: datetime,
        database: Optional[str] = None,
    ) -> DataFrame:
        """Return a DataFrame representing table state at or before the given timestamp."""
        full_name = self._table_manager.get_full_name(table_name, database=database)
        ts_ms = int(as_of.timestamp() * 1000)
        logger.info(f"Time-travel read: {full_name} @ {as_of.isoformat()}")
        return self._spark.read.option("as-of-timestamp", ts_ms).table(full_name)

    def read_between_snapshots(
        self,
        table_name: str,
        start_snapshot_id: int,
        end_snapshot_id: int,
        database: Optional[str] = None,
    ) -> DataFrame:
        """
        Return incremental data added between two snapshots.
        Uses Iceberg's incremental read API.
        """
        full_name = self._table_manager.get_full_name(table_name, database=database)
        return (
            self._spark.read
            .format("iceberg")
            .option("start-snapshot-id", start_snapshot_id)
            .option("end-snapshot-id", end_snapshot_id)
            .load(full_name)
        )


    def rollback_to_snapshot(
        self,
        table_name: str,
        snapshot_id: int,
        database: Optional[str] = None,
    ) -> None:
        """
        Roll back the table to a previous snapshot.
        This creates a new snapshot with the same state — it does NOT delete history.
        """
        full_name = self._table_manager.get_full_name(table_name, database=database)
        logger.warning(f"Rolling back {full_name} to snapshot {snapshot_id}")
        self._spark.sql(
            f"CALL {self._cfg.catalog_name}.system.rollback_to_snapshot('{full_name}', {snapshot_id})"
        )
        logger.info(f"Rollback complete: {full_name} → snapshot {snapshot_id}")

    def rollback_to_timestamp(
        self,
        table_name: str,
        as_of: datetime,
        database: Optional[str] = None,
    ) -> None:
        full_name = self._table_manager.get_full_name(table_name, database=database)
        ts_ms = int(as_of.timestamp() * 1000)
        logger.warning(f"Rolling back {full_name} to {as_of.isoformat()}")
        self._spark.sql(
            f"CALL {self._cfg.catalog_name}.system.rollback_to_timestamp('{full_name}', {ts_ms})"
        )


    def expire_snapshots(
        self,
        table_name: str,
        older_than: datetime,
        database: Optional[str] = None,
        retain_last: int = 5,
    ) -> Dict[str, int]:
        """
        Expire snapshots older than the given timestamp, keeping at least retain_last.
        Returns dict with counts of deleted snapshots and data files.
        """
        full_name = self._table_manager.get_full_name(table_name, database=database)
        ts_ms = int(older_than.timestamp() * 1000)
        logger.info(f"Expiring snapshots older than {older_than.isoformat()} on {full_name}")

        result = self._spark.sql(
            f"""
            CALL {self._cfg.catalog_name}.system.expire_snapshots(
                table => '{full_name}',
                older_than => TIMESTAMP '{older_than.strftime("%Y-%m-%d %H:%M:%S")}',
                retain_last => {retain_last}
            )
            """
        ).collect()

        if result:
            row = result[0]
            return {
                "deleted_data_files": row.get("deleted_data_files_count", 0),
                "deleted_manifest_files": row.get("deleted_manifest_files_count", 0),
                "deleted_manifest_lists": row.get("deleted_manifest_lists_count", 0),
            }
        return {}

    def rewrite_data_files(
        self,
        table_name: str,
        database: Optional[str] = None,
        strategy: str = "binpack",
    ) -> Dict[str, int]:
        """
        Compact small data files using Iceberg's rewrite_data_files procedure.
        Reduces file count and improves read performance.
        """
        full_name = self._table_manager.get_full_name(table_name, database=database)
        logger.info(f"Compacting {full_name} with strategy={strategy}")

        result = self._spark.sql(
            f"""
            CALL {self._cfg.catalog_name}.system.rewrite_data_files(
                table => '{full_name}',
                strategy => '{strategy}'
            )
            """
        ).collect()

        if result:
            row = result[0]
            return {
                "rewritten_files": row.get("rewritten_files_count", 0),
                "added_files": row.get("added_files_count", 0),
                "rewritten_bytes": row.get("rewritten_bytes_count", 0),
            }
        return {}


    def get_files(self, table_name: str, database: Optional[str] = None) -> DataFrame:
        """Return a DataFrame of current data files with size and row count metadata."""
        full_name = self._table_manager.get_full_name(table_name, database=database)
        return self._spark.sql(f"SELECT * FROM {full_name}.files")

    def get_manifests(self, table_name: str, database: Optional[str] = None) -> DataFrame:
        full_name = self._table_manager.get_full_name(table_name, database=database)
        return self._spark.sql(f"SELECT * FROM {full_name}.manifests")

    def get_history(self, table_name: str, database: Optional[str] = None) -> DataFrame:
        full_name = self._table_manager.get_full_name(table_name, database=database)
        return self._spark.sql(f"SELECT * FROM {full_name}.history")

    def get_partitions(self, table_name: str, database: Optional[str] = None) -> DataFrame:
        full_name = self._table_manager.get_full_name(table_name, database=database)
        return self._spark.sql(f"SELECT * FROM {full_name}.partitions")