#!/usr/bin/env python3
"""
ODVS run_pipeline.py — End-to-end dataset versioning pipeline.

Orchestrates the full ODVS data flow:
  Source → Ingest → Transform → Dedup → Validate → Spark → Iceberg → Registry → HF Sim

Usage:
    python scripts/run_pipeline.py \\
        --source examples/sample_dataset.csv \\
        --dataset-name ecommerce_events \\
        --version-tag v1.0.0 \\
        --description "E-commerce click events dataset" \\
        --tags ecommerce,events,clickstream \\
        --compression zstd

    # Incremental append (new version from same or different source):
    python scripts/run_pipeline.py \\
        --source s3://my-bucket/new_events.csv \\
        --dataset-name ecommerce_events \\
        --version-tag v1.1.0 \\
        --incremental

    # Run with schema validation config:
    python scripts/run_pipeline.py \\
        --source examples/sample_dataset.csv \\
        --dataset-name ecommerce_events \\
        --version-tag v1.0.0 \\
        --schema-config examples/schema_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from odvs.analytics.benchmarks import PipelineBenchmarker
from odvs.analytics.lineage import LineageTracker
from odvs.config import get_config
from odvs.hf.simulator import HFSimulator
from odvs.iceberg.spark_session import get_or_create_session
from odvs.iceberg.table_manager import IcebergTableManager, PartitionSpec, SortOrderSpec
from odvs.iceberg.versioning import IcebergVersionManager
from odvs.iceberg.writer import IcebergWriter
from odvs.logger import configure_root_logger, get_logger
from odvs.processing.deduplication.hash_dedup import HashDeduplicator, HashAlgorithm
from odvs.processing.ingestion.ingest import DatasetIngester, SourceType
from odvs.processing.optimization.compression import CompressionBenchmarker
from odvs.processing.transforms.normalize import TransformPipeline
from odvs.processing.validation.schema_validator import (
    ColumnSpec,
    SchemaSpec,
    SchemaValidator,
    ValidationSeverity,
)
from odvs.registry.dataset_registry import DatasetRegistry
from odvs.storage.s3_client import S3Client

configure_root_logger()
logger = get_logger(__name__)


class PipelineResult:
    """Collects all outputs from a completed pipeline run."""

    def __init__(self) -> None:
        self.success: bool = False
        self.dataset_name: str = ""
        self.version_tag: str = ""
        self.rows_ingested: int = 0
        self.rows_after_dedup: int = 0
        self.rows_written: int = 0
        self.snapshot_id: Optional[int] = None
        self.lineage_node_id: Optional[str] = None
        self.compression_recommendation: Optional[str] = None
        self.duration_seconds: float = 0.0
        self.validation_passed: bool = True
        self.errors: List[str] = []

    def print_summary(self) -> None:
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        print(f"\n{'═' * 62}")
        print(f"  ODVS Pipeline Run — {status}")
        print(f"{'═' * 62}")
        print(f"  Dataset      : {self.dataset_name}")
        print(f"  Version      : {self.version_tag}")
        print(f"  Rows Ingested: {self.rows_ingested:,}")
        print(f"  After Dedup  : {self.rows_after_dedup:,}")
        print(f"  Written      : {self.rows_written:,}")
        print(f"  Snapshot ID  : {self.snapshot_id}")
        print(f"  Duration     : {self.duration_seconds:.2f}s")
        print(f"  Validation   : {'PASSED' if self.validation_passed else 'FAILED'}")
        if self.compression_recommendation:
            print(f"  Compression  : {self.compression_recommendation} (recommended)")
        if self.errors:
            print(f"  Errors:")
            for err in self.errors:
                print(f"    - {err}")
        print(f"{'═' * 62}\n")


def load_schema_config(path: str) -> SchemaSpec:
    """
    Load a SchemaSpec from a JSON config file.

    Expected format:
    {
      "allow_extra_columns": true,
      "min_rows": 1,
      "columns": [
        {"name": "user_id", "dtype": "object", "nullable": false, "required": true},
        {"name": "amount", "dtype": "float64", "min_value": 0}
      ]
    }
    """
    with open(path) as f:
        raw = json.load(f)

    cols = [ColumnSpec(**c) for c in raw.get("columns", [])]
    return SchemaSpec(
        columns=cols,
        allow_extra_columns=raw.get("allow_extra_columns", True),
        min_rows=raw.get("min_rows"),
        max_rows=raw.get("max_rows"),
        require_unique=raw.get("require_unique"),
    )


class ODVSPipeline:
    """
    End-to-end ODVS dataset versioning pipeline.

    Instantiate once per pipeline run. Not thread-safe across concurrent runs
    on the same Spark session (Spark is single-session per JVM).
    """

    def __init__(
        self,
        dataset_name: str,
        version_tag: str,
        source: str,
        source_type: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        compression: str = "zstd",
        schema_spec: Optional[SchemaSpec] = None,
        incremental: bool = False,
        run_compression_benchmark: bool = True,
        partition_columns: Optional[List[str]] = None,
        sort_columns: Optional[List[str]] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.version_tag = version_tag
        self.source = source
        self.source_type = SourceType(source_type) if source_type else None
        self.description = description
        self.tags = tags or []
        self.compression = compression
        self.schema_spec = schema_spec
        self.incremental = incremental
        self.run_compression_benchmark = run_compression_benchmark
        self.partition_columns = partition_columns or []
        self.sort_columns = sort_columns or []

        self._cfg = get_config()
        self._result = PipelineResult()
        self._result.dataset_name = dataset_name
        self._result.version_tag = version_tag

        # Derive safe Iceberg table name from dataset name
        self._table_name = dataset_name.replace("-", "_").replace("/", "__").replace(" ", "_").lower()

    def run(self) -> PipelineResult:
        """Execute the full pipeline. Returns PipelineResult."""
        import time
        start = time.monotonic()

        bench = PipelineBenchmarker(pipeline_name=f"{self.dataset_name}@{self.version_tag}")

        try:
            # Step 1: Ensure S3 bucket exists 
            with bench.step("s3_bootstrap"):
                s3 = S3Client()
                s3.ensure_bucket()
                logger.info(f"S3 bucket ready: {self._cfg.s3.bucket}")

            # Step 2: Ingest 
            with bench.step("ingestion"):
                ingester = DatasetIngester()
                ingest_result = ingester.ingest(
                    source=self.source,
                    source_type=self.source_type,
                )
                df = ingest_result.df
                manifest = ingest_result.manifest
                self._result.rows_ingested = manifest.row_count
                logger.info(f"Ingested {manifest.row_count:,} rows from {self.source}")

            # Step 3: Transform / Normalize 
            with bench.step("transform"):
                pipeline = TransformPipeline.standard()
                df = pipeline.run(df)
                logger.info(f"Transforms applied. Shape: {df.shape}")

            # Step 4: Deduplication
            with bench.step("deduplication"):
                deduplicator = HashDeduplicator(algorithm=HashAlgorithm.SHA256)
                df, dedup_result = deduplicator.deduplicate(df)
                self._result.rows_after_dedup = dedup_result.deduplicated_rows
                logger.info(str(dedup_result))

            # Step 5: Schema Validation 
            with bench.step("validation"):
                if self.schema_spec:
                    validator = SchemaValidator()
                    report = validator.validate(df, self.schema_spec)
                    logger.info(report.summary())
                    if not report.passed:
                        self._result.validation_passed = False
                        error_msgs = [str(e) for e in report.errors]
                        self._result.errors.extend(error_msgs)
                        logger.error("Schema validation failed — aborting pipeline")
                        self._result.duration_seconds = time.monotonic() - start
                        return self._result
                    logger.info("Schema validation PASSED")
                else:
                    logger.info("No schema spec provided — skipping validation")

            # Step 6: Compression Benchmark (advisory, non-blocking)
            if self.run_compression_benchmark:
                with bench.step("compression_benchmark"):
                    benchmarker = CompressionBenchmarker()
                    comp_report = benchmarker.benchmark(df, dataset_name=self.dataset_name)
                    self._result.compression_recommendation = comp_report.recommendation
                    logger.info(
                        f"Compression recommendation: {comp_report.recommendation} "
                        f"(user chose: {self.compression})"
                    )

            # Step 7: Build SparkSession
            with bench.step("spark_init"):
                spark = get_or_create_session(app_name_suffix=f"{self.dataset_name}")
                table_mgr = IcebergTableManager(spark=spark)
                writer = IcebergWriter(spark=spark)
                logger.info("SparkSession ready")

            # Step 8: Create table (idempotent)
            with bench.step("table_bootstrap"):
                spark_df = spark.createDataFrame(df)
                schema = spark_df.schema

                partition_specs = [
                    PartitionSpec(column=col)
                    for col in self.partition_columns
                    if col in df.columns
                ]
                sort_specs = [
                    SortOrderSpec(column=col)
                    for col in self.sort_columns
                    if col in df.columns
                ]

                full_table_name = table_mgr.create_table(
                    table_name=self._table_name,
                    schema=schema,
                    partition_specs=partition_specs or None,
                    sort_orders=sort_specs or None,
                    comment=self.description or f"ODVS managed dataset: {self.dataset_name}",
                    properties={
                        "odvs.dataset_name": self.dataset_name,
                        "odvs.managed": "true",
                    },
                )
                logger.info(f"Iceberg table ready: {full_table_name}")

            # Step 9: Iceberg Write
            with bench.step("iceberg_write"):
                write_result = writer.append(
                    df=spark_df,
                    table_name=self._table_name,
                    compression=self.compression,
                    validate_schema=False,  # Already validated above
                )
                self._result.rows_written = write_result.rows_written
                self._result.snapshot_id = write_result.snapshot_id
                logger.info(str(write_result))

            # Step 10: Registry Update
            with bench.step("registry_update"):
                registry = DatasetRegistry()
                registry.register(
                    name=self.dataset_name,
                    description=self.description,
                    tags=self.tags,
                    source_uri=self.source,
                    table_path=full_table_name,
                    schema={col: str(dtype) for col, dtype in df.dtypes.items()},
                )
                registry.add_version(
                    dataset_name=self.dataset_name,
                    version_tag=self.version_tag,
                    row_count=write_result.rows_written,
                    snapshot_id=write_result.snapshot_id,
                    checksum=manifest.checksum_sha256[:16],
                    schema={col: str(dtype) for col, dtype in df.dtypes.items()},
                )
                logger.info(f"Registry updated: {self.dataset_name}@{self.version_tag}")

            # Step 11: Lineage Record
            with bench.step("lineage_record"):
                lineage = LineageTracker()
                node = lineage.record(
                    dataset_name=self.dataset_name,
                    version_tag=self.version_tag,
                    source_uris=[self.source],
                    transforms_applied=[
                        "lowercase_columns",
                        "strip_whitespace",
                        "drop_empty_rows",
                        "normalize_unicode",
                        "hash_dedup_sha256",
                    ],
                    schema_snapshot={col: str(dtype) for col, dtype in df.dtypes.items()},
                    row_count=write_result.rows_written,
                    snapshot_id=write_result.snapshot_id,
                )
                self._result.lineage_node_id = node.node_id
                logger.info(f"Lineage recorded: node_id={node.node_id}")

            # Step 12: HF Hub Simulation
            with bench.step("hf_simulation"):
                hf_sim = HFSimulator(registry=registry)
                hub_result = hf_sim.push_to_hub_simulation(
                    dataset_name=self.dataset_name,
                    df=df,
                    version_tag=self.version_tag,
                    commit_message=f"Add {self.version_tag} via ODVS pipeline",
                )
                logger.info(f"HF Hub simulation: {hub_result['hub_url']}")

            # Pipeline complete
            self._result.success = True
            self._result.duration_seconds = time.monotonic() - start

            bench_report = bench.report(row_count=write_result.rows_written)
            bench_report.print_report()

        except Exception as exc:
            self._result.success = False
            self._result.duration_seconds = time.monotonic() - start
            self._result.errors.append(str(exc))
            logger.error(f"Pipeline FAILED: {exc}")
            logger.debug(traceback.format_exc())

        return self._result



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ODVS — Open Dataset Versioning System pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path or URI to the source dataset (local path, S3 URI, or HTTP URL)",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Logical name for this dataset (e.g. 'ecommerce_events')",
    )
    parser.add_argument(
        "--version-tag",
        required=True,
        help="Version identifier for this pipeline run (e.g. 'v1.0.0')",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Human-readable description of this dataset",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags (e.g. 'ecommerce,events,clickstream')",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "none"],
        help="Parquet compression codec for Iceberg writes (default: zstd)",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        choices=[t.value for t in SourceType],
        help="Force a specific source type (auto-detected by default)",
    )
    parser.add_argument(
        "--schema-config",
        default=None,
        help="Path to a JSON schema validation config file",
    )
    parser.add_argument(
        "--partition-columns",
        default="",
        help="Comma-separated columns to use as Iceberg partition keys",
    )
    parser.add_argument(
        "--sort-columns",
        default="",
        help="Comma-separated columns to use as Iceberg sort order",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Append-only mode — do not recreate the table",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the compression benchmark step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ingest and transform only — do not write to Iceberg or update registry",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    partition_cols = [c.strip() for c in args.partition_columns.split(",") if c.strip()]
    sort_cols = [c.strip() for c in args.sort_columns.split(",") if c.strip()]

    schema_spec = None
    if args.schema_config:
        logger.info(f"Loading schema config from: {args.schema_config}")
        schema_spec = load_schema_config(args.schema_config)

    if args.dry_run:
        logger.info("DRY RUN mode — pipeline will stop before Iceberg write")
        ingester = DatasetIngester()
        ingest_result = ingester.ingest(source=args.source)
        pipeline = TransformPipeline.standard()
        df = pipeline.run(ingest_result.df)
        dedup = HashDeduplicator()
        df, dedup_result = dedup.deduplicate(df)
        print(f"\nDRY RUN complete:")
        print(f"  Rows ingested  : {ingest_result.manifest.row_count:,}")
        print(f"  After dedup    : {dedup_result.deduplicated_rows:,}")
        print(f"  Columns        : {list(df.columns)}")
        print(f"  Schema preview :")
        print(df.dtypes.to_string())
        return 0

    pipeline = ODVSPipeline(
        dataset_name=args.dataset_name,
        version_tag=args.version_tag,
        source=args.source,
        source_type=args.source_type,
        description=args.description,
        tags=tags,
        compression=args.compression,
        schema_spec=schema_spec,
        incremental=args.incremental,
        run_compression_benchmark=not args.skip_benchmark,
        partition_columns=partition_cols,
        sort_columns=sort_cols,
    )

    result = pipeline.run()
    result.print_summary()

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())