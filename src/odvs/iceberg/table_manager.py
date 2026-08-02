"""
ODVS Iceberg Table Manager — DDL and catalog management for Iceberg tables.

Handles:
- Table creation with proper partitioning and sort orders
- Schema evolution
- Table property management
- Namespace/database lifecycle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from odvs.config import IcebergConfig, get_config
from odvs.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PartitionSpec:
    """Iceberg partition spec definition."""
    column: str
    transform: str = "identity"  # identity | bucket[N] | truncate[N] | year | month | day | hour

    def to_sql(self) -> str:
        if self.transform == "identity":
            return self.column
        return f"{self.transform}({self.column})"


@dataclass
class SortOrderSpec:
    column: str
    direction: str = "ASC"
    nulls: str = "NULLS LAST"

    def to_sql(self) -> str:
        return f"{self.column} {self.direction} {self.nulls}"


class IcebergTableManager:
    """
    Manages Iceberg table DDL operations via SparkSQL.

    This class is intentionally decoupled from write operations —
    it handles schema/catalog concerns only. Write operations live in IcebergWriter.
    """

    SYSTEM_PROPERTIES: Dict[str, str] = {
        "write.format.default": "parquet",
        "write.parquet.compression-codec": "zstd",
        "write.target-file-size-bytes": str(128 * 1024 * 1024),
        "write.metadata.delete-after-commit.enabled": "true",
        "write.metadata.previous-versions-max": "10",
        "history.expire.min-snapshots-to-keep": "5",
        "history.expire.max-snapshot-age-ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
    }

    def __init__(
        self,
        spark: SparkSession,
        config: Optional[IcebergConfig] = None,
    ) -> None:
        self._spark = spark
        self._cfg = config or get_config().iceberg


    def ensure_namespace(self, database: Optional[str] = None) -> None:
        db = database or self._cfg.database
        self._spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {self._cfg.catalog_name}.{db}")
        logger.info(f"Namespace ensured: {self._cfg.catalog_name}.{db}")


    def table_exists(self, table_name: str, database: Optional[str] = None) -> bool:
        db = database or self._cfg.database
        self.ensure_namespace(database=db)
        rows = self._spark.sql(f"SHOW TABLES IN {self._cfg.catalog_name}.{db}").collect()
        return any(row["tableName"] == table_name for row in rows)

    def create_table(
        self,
        table_name: str,
        schema: StructType,
        partition_specs: Optional[List[PartitionSpec]] = None,
        sort_orders: Optional[List[SortOrderSpec]] = None,
        properties: Optional[Dict[str, str]] = None,
        database: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> str:
        """
        Create an Iceberg table with full DDL including partitioning, sort orders, and properties.

        Returns the fully qualified table name.
        """
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"

        if self.table_exists(table_name, database=db):
            logger.info(f"Table already exists, skipping creation: {full_name}")
            return full_name

        # Build schema DDL from StructType
        schema_ddl = self._struct_to_ddl(schema)

        # Build PARTITIONED BY clause
        partition_clause = ""
        if partition_specs:
            parts = ", ".join(p.to_sql() for p in partition_specs)
            partition_clause = f"PARTITIONED BY ({parts})"

        # Build TBLPROPERTIES
        merged_props = {**self.SYSTEM_PROPERTIES, **(properties or {})}
        if comment:
            merged_props["comment"] = comment
        props_ddl = self._props_to_ddl(merged_props)

        ddl = f"""
        CREATE TABLE {full_name}
        ({schema_ddl})
        USING iceberg
        {partition_clause}
        TBLPROPERTIES ({props_ddl})
        """

        self._spark.sql(ddl)
        logger.info(f"Created Iceberg table: {full_name}")

        # Apply sort order separately (Iceberg DDL extension)
        if sort_orders and self._cfg.sort_order_enabled:
            self._apply_sort_order(full_name, sort_orders)

        return full_name

    def drop_table(self, table_name: str, database: Optional[str] = None, purge: bool = False) -> None:
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"
        purge_clause = "PURGE" if purge else ""
        self._spark.sql(f"DROP TABLE IF EXISTS {full_name} {purge_clause}")
        logger.info(f"Dropped table: {full_name} (purge={purge})")

    def get_table_properties(self, table_name: str, database: Optional[str] = None) -> Dict[str, str]:
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"
        rows = self._spark.sql(f"SHOW TBLPROPERTIES {full_name}").collect()
        return {row["key"]: row["value"] for row in rows}

    def alter_table_property(
        self,
        table_name: str,
        key: str,
        value: str,
        database: Optional[str] = None,
    ) -> None:
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"
        self._spark.sql(f"ALTER TABLE {full_name} SET TBLPROPERTIES ('{key}'='{value}')")

    def list_tables(self, database: Optional[str] = None) -> List[str]:
        db = database or self._cfg.database
        rows = self._spark.sql(f"SHOW TABLES IN {self._cfg.catalog_name}.{db}").collect()
        return [row["tableName"] for row in rows]

    def get_schema(self, table_name: str, database: Optional[str] = None) -> StructType:
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"
        return self._spark.table(full_name).schema


    def add_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
        after: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"
        after_clause = f"AFTER {after}" if after else ""
        self._spark.sql(
            f"ALTER TABLE {full_name} ADD COLUMN {column_name} {column_type} {after_clause}"
        )
        logger.info(f"Added column {column_name}:{column_type} to {full_name}")

    def rename_column(
        self,
        table_name: str,
        old_name: str,
        new_name: str,
        database: Optional[str] = None,
    ) -> None:
        db = database or self._cfg.database
        full_name = f"{self._cfg.catalog_name}.{db}.{table_name}"
        self._spark.sql(f"ALTER TABLE {full_name} RENAME COLUMN {old_name} TO {new_name}")


    def _struct_to_ddl(self, schema: StructType) -> str:
        parts = []
        for field in schema.fields:
            nullable = "" if field.nullable else " NOT NULL"
            parts.append(f"`{field.name}` {field.dataType.simpleString()}{nullable}")
        return ",\n  ".join(parts)

    def _props_to_ddl(self, props: Dict[str, str]) -> str:
        return ", ".join(f"'{k}'='{v}'" for k, v in props.items())

    def _apply_sort_order(self, full_name: str, sort_orders: List[SortOrderSpec]) -> None:
        order_ddl = ", ".join(s.to_sql() for s in sort_orders)
        self._spark.sql(f"ALTER TABLE {full_name} WRITE ORDERED BY {order_ddl}")
        logger.info(f"Applied sort order to {full_name}: {order_ddl}")

    def get_full_name(self, table_name: str, database: Optional[str] = None) -> str:
        db = database or self._cfg.database
        return f"{self._cfg.catalog_name}.{db}.{table_name}"