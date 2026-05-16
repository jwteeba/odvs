#!/usr/bin/env python3
"""
ODVS create_table.py — Create or evolve an Iceberg table from a schema definition.

Supports:
- Creating a new table from a JSON schema file
- Adding columns to an existing table (schema evolution)
- Displaying table metadata and history
- Dropping tables (with --purge to remove data files)

Usage:
    # Create a table from a schema file:
    python scripts/create_table.py \\
        --table-name ecommerce_events \\
        --schema-file examples/schema_config.json \\
        --partition-by event_date \\
        --sort-by user_id

    # Add a column to an existing table:
    python scripts/create_table.py \\
        --table-name ecommerce_events \\
        --add-column session_id string

    # Show table info:
    python scripts/create_table.py --table-name ecommerce_events --info

    # Drop table:
    python scripts/create_table.py --table-name ecommerce_events --drop --purge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from odvs.iceberg.spark_session import get_or_create_session
from odvs.iceberg.table_manager import IcebergTableManager, PartitionSpec, SortOrderSpec
from odvs.iceberg.versioning import IcebergVersionManager
from odvs.logger import configure_root_logger, get_logger
from odvs.storage.s3_client import S3Client

configure_root_logger()
logger = get_logger(__name__)


_TYPE_MAP = {
    "string": StringType(),
    "str": StringType(),
    "varchar": StringType(),
    "text": StringType(),
    "int": IntegerType(),
    "integer": IntegerType(),
    "int32": IntegerType(),
    "long": LongType(),
    "int64": LongType(),
    "bigint": LongType(),
    "float": FloatType(),
    "float32": FloatType(),
    "double": DoubleType(),
    "float64": DoubleType(),
    "boolean": BooleanType(),
    "bool": BooleanType(),
    "date": DateType(),
    "timestamp": TimestampType(),
}


def parse_type(type_str: str):
    """Convert a string type name to a Spark DataType."""
    t = type_str.strip().lower()
    if t not in _TYPE_MAP:
        raise ValueError(
            f"Unsupported type: '{type_str}'. Supported: {sorted(_TYPE_MAP.keys())}"
        )
    return _TYPE_MAP[t]


def schema_from_file(path: str) -> StructType:
    """
    Build a Spark StructType from a JSON schema file.

    Expected format:
    {
      "columns": [
        {"name": "user_id", "type": "string", "nullable": false},
        {"name": "event_ts", "type": "timestamp", "nullable": true},
        {"name": "amount",   "type": "double",    "nullable": true}
      ]
    }
    """
    with open(path) as f:
        raw = json.load(f)

    columns_raw = raw.get("columns", raw)  # Support top-level list or {"columns": [...]}
    if isinstance(columns_raw, list):
        cols_list = columns_raw
    elif isinstance(columns_raw, dict) and "columns" in columns_raw:
        cols_list = columns_raw["columns"]
    else:
        raise ValueError(f"Cannot parse schema from {path}: expected 'columns' list")

    fields = []
    for col in cols_list:
        name = col["name"]
        dtype = parse_type(col.get("type", col.get("dtype", "string")))
        nullable = col.get("nullable", True)
        fields.append(StructField(name, dtype, nullable))

    return StructType(fields)


def schema_from_dict(columns: Dict[str, str]) -> StructType:
    """Build a StructType from a {name: type_str} dict."""
    return StructType([
        StructField(name, parse_type(type_str), True)
        for name, type_str in columns.items()
    ])


def action_create(args: argparse.Namespace) -> int:
    if not args.schema_file and not args.columns:
        print("ERROR: Provide --schema-file or --columns to create a table", file=sys.stderr)
        return 1

    s3 = S3Client()
    s3.ensure_bucket()

    spark = get_or_create_session(app_name_suffix="create_table")
    mgr = IcebergTableManager(spark=spark)

    if args.schema_file:
        schema = schema_from_file(args.schema_file)
    else:
        col_pairs = {}
        for col_def in args.columns:
            parts = col_def.split(":")
            if len(parts) != 2:
                print(f"ERROR: Column definition must be name:type, got: {col_def}", file=sys.stderr)
                return 1
            col_pairs[parts[0].strip()] = parts[1].strip()
        schema = schema_from_dict(col_pairs)

    partition_specs = [PartitionSpec(col) for col in args.partition_by] if args.partition_by else None
    sort_specs = [SortOrderSpec(col) for col in args.sort_by] if args.sort_by else None

    full_name = mgr.create_table(
        table_name=args.table_name,
        schema=schema,
        partition_specs=partition_specs,
        sort_orders=sort_specs,
        comment=args.comment or f"ODVS managed table: {args.table_name}",
    )
    print(f"\n✓ Created Iceberg table: {full_name}")
    _print_schema(schema)
    return 0


def action_add_column(args: argparse.Namespace) -> int:
    parts = args.add_column
    if len(parts) < 2:
        print("ERROR: --add-column requires NAME TYPE [after:COLUMN]", file=sys.stderr)
        return 1

    col_name = parts[0]
    col_type = parts[1]
    after = parts[2] if len(parts) > 2 else None

    spark = get_or_create_session(app_name_suffix="schema_evolution")
    mgr = IcebergTableManager(spark=spark)

    mgr.add_column(
        table_name=args.table_name,
        column_name=col_name,
        column_type=col_type,
        after=after,
    )
    print(f"✓ Added column {col_name}:{col_type} to {args.table_name}")
    return 0


def action_info(args: argparse.Namespace) -> int:
    spark = get_or_create_session(app_name_suffix="table_info")
    mgr = IcebergTableManager(spark=spark)
    version_mgr = IcebergVersionManager(spark=spark)

    if not mgr.table_exists(args.table_name):
        print(f"Table does not exist: {args.table_name}", file=sys.stderr)
        return 1

    schema = mgr.get_schema(args.table_name)
    props = mgr.get_table_properties(args.table_name)
    snapshots = version_mgr.list_snapshots(args.table_name, limit=5)

    print(f"\n{'═' * 55}")
    print(f"  Table: {args.table_name}")
    print(f"{'═' * 55}")
    print(f"\n  Schema ({len(schema.fields)} columns):")
    for field in schema.fields:
        nullable = "nullable" if field.nullable else "NOT NULL"
        print(f"    {field.name:<30} {field.dataType.simpleString():<15} {nullable}")

    print(f"\n  Table Properties:")
    for k, v in sorted(props.items()):
        if not k.startswith("spark."):
            print(f"    {k:<45} {v}")

    print(f"\n  Recent Snapshots ({len(snapshots)}):")
    for snap in snapshots:
        print(f"    {snap.snapshot_id} | {snap.committed_at.isoformat()[:19]} | "
              f"{snap.operation:<10} | +{snap.added_records:,} rows")

    print(f"{'═' * 55}\n")
    return 0


def action_drop(args: argparse.Namespace) -> int:
    spark = get_or_create_session(app_name_suffix="drop_table")
    mgr = IcebergTableManager(spark=spark)

    if not args.yes:
        confirm = input(
            f"Drop table '{args.table_name}'? "
            f"{'(data files will be DELETED)' if args.purge else '(metadata only)'} [y/N]: "
        )
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return 0

    mgr.drop_table(args.table_name, purge=args.purge)
    print(f"✓ Dropped table: {args.table_name}")
    return 0


def action_list(args: argparse.Namespace) -> int:
    spark = get_or_create_session(app_name_suffix="list_tables")
    mgr = IcebergTableManager(spark=spark)
    tables = mgr.list_tables()

    if not tables:
        print("No tables found in the ODVS catalog.")
        return 0

    print(f"\n  {'Table Name':<40} {'Exists'}")
    print(f"  {'─'*40} {'─'*6}")
    for t in sorted(tables):
        print(f"  {t:<40} ✓")
    print(f"\n  {len(tables)} table(s) found.\n")
    return 0


def _print_schema(schema: StructType) -> None:
    print(f"\n  Schema ({len(schema.fields)} columns):")
    for field in schema.fields:
        nullable = "nullable" if field.nullable else "NOT NULL"
        print(f"    {field.name:<30} {field.dataType.simpleString():<15} {nullable}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ODVS — Create and manage Iceberg tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--table-name", required=True, help="Iceberg table name (unqualified)")

    # Mutually exclusive actions
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--info", action="store_true", help="Show table info and snapshot history")
    action_group.add_argument("--drop", action="store_true", help="Drop the table")
    action_group.add_argument("--list", action="store_true", help="List all tables")
    action_group.add_argument(
        "--add-column",
        nargs="+",
        metavar=("NAME", "TYPE"),
        help="Add a column: NAME TYPE [after:COLUMN]",
    )

    # Create options
    parser.add_argument("--schema-file", help="JSON file describing the table schema")
    parser.add_argument(
        "--columns",
        nargs="+",
        metavar="NAME:TYPE",
        help="Column definitions as name:type pairs (e.g. user_id:string amount:double)",
    )
    parser.add_argument(
        "--partition-by",
        nargs="+",
        help="Partition column names",
    )
    parser.add_argument(
        "--sort-by",
        nargs="+",
        help="Sort order column names",
    )
    parser.add_argument("--comment", help="Table comment/description")

    # Drop options
    parser.add_argument("--purge", action="store_true", help="Remove data files when dropping")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        return action_list(args)
    if args.info:
        return action_info(args)
    if args.drop:
        return action_drop(args)
    if args.add_column:
        return action_add_column(args)
    return action_create(args)


if __name__ == "__main__":
    sys.exit(main())