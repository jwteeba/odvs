#!/usr/bin/env python3
"""
ODVS benchmark_pipeline.py — Compression and read/write performance benchmarks.

Runs compression benchmarks across all supported Parquet codecs,
then optionally runs a full end-to-end pipeline timing to measure
Iceberg write throughput.

Usage:
    # Benchmark compression on a local CSV:
    python scripts/benchmark_pipeline.py \\
        --source examples/sample_dataset.csv \\
        --dataset-name ecommerce_events

    # Full pipeline benchmark (includes Spark + Iceberg write timing):
    python scripts/benchmark_pipeline.py \\
        --source examples/sample_dataset.csv \\
        --dataset-name ecommerce_events \\
        --full-pipeline

    # Benchmark with all codecs:
    python scripts/benchmark_pipeline.py \\
        --source examples/sample_dataset.csv \\
        --dataset-name ecommerce_events \\
        --codecs snappy gzip zstd none

    # Output results as JSON:
    python scripts/benchmark_pipeline.py \\
        --source examples/sample_dataset.csv \\
        --dataset-name ecommerce_events \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odvs.analytics.benchmarks import PipelineBenchmarker
from odvs.logger import configure_root_logger, get_logger
from odvs.processing.compression_bench_runner import run_compression_benchmark
from odvs.processing.ingestion.ingest import DatasetIngester, SourceType
from odvs.processing.optimization.compression import CompressionBenchmarker, SUPPORTED_CODECS
from odvs.processing.transforms.normalize import TransformPipeline
from odvs.processing.deduplication.hash_dedup import HashDeduplicator

configure_root_logger()
logger = get_logger(__name__)


def run_compression_benchmark_standalone(
    source: str,
    dataset_name: str,
    codecs: List[str],
    sample_n: int,
    output_path: Optional[str],
) -> int:
    """Ingest a dataset and run compression benchmarks across the given codecs."""
    logger.info(f"Starting compression benchmark for: {dataset_name}")

    bench = PipelineBenchmarker(f"compression_bench:{dataset_name}")

    with bench.step("ingest"):
        ingester = DatasetIngester()
        result = ingester.ingest(source=source)
        df = result.df
        logger.info(f"Ingested {len(df):,} rows")

    with bench.step("transform"):
        pipeline = TransformPipeline.standard()
        df = pipeline.run(df)

    with bench.step("dedup"):
        deduplicator = HashDeduplicator()
        df, dedup_result = deduplicator.deduplicate(df)
        logger.info(str(dedup_result))

    with bench.step("compression_benchmark"):
        benchmarker = CompressionBenchmarker(codecs=codecs)
        report = benchmarker.benchmark(df, dataset_name=dataset_name, sample_n=sample_n)

    report.print_report()

    pipeline_report = bench.report(row_count=len(df))
    pipeline_report.print_report()

    if output_path:
        output = {
            "dataset_name": dataset_name,
            "source": source,
            "row_count": len(df),
            "compression_results": [
                {
                    "codec": r.codec,
                    "original_bytes": r.original_bytes,
                    "compressed_bytes": r.compressed_bytes,
                    "compression_ratio": round(r.compression_ratio, 4),
                    "write_duration_ms": round(r.write_duration_ms, 2),
                    "read_duration_ms": round(r.read_duration_ms, 2),
                    "throughput_mb_per_sec": round(r.throughput_mb_per_sec, 2),
                }
                for r in report.results
            ],
            "recommendation": report.recommendation,
            "pipeline_steps": pipeline_report.to_dict()["operations"],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Benchmark results written to: {output_path}")

    return 0


def run_full_pipeline_benchmark(
    source: str,
    dataset_name: str,
    compression: str,
    output_path: Optional[str],
) -> int:
    """
    Run a complete end-to-end pipeline including Spark + Iceberg writes,
    and report per-step timing.
    """
    from odvs.iceberg.spark_session import get_or_create_session
    from odvs.iceberg.table_manager import IcebergTableManager
    from odvs.iceberg.writer import IcebergWriter
    from odvs.storage.s3_client import S3Client

    bench = PipelineBenchmarker(f"full_pipeline_bench:{dataset_name}")

    with bench.step("s3_bootstrap"):
        s3 = S3Client()
        s3.ensure_bucket()

    with bench.step("ingest"):
        ingester = DatasetIngester()
        result = ingester.ingest(source=source)
        df = result.df

    with bench.step("transform"):
        pipeline = TransformPipeline.standard()
        df = pipeline.run(df)

    with bench.step("dedup"):
        deduplicator = HashDeduplicator()
        df, _ = deduplicator.deduplicate(df)

    with bench.step("spark_init"):
        spark = get_or_create_session(app_name_suffix="benchmark")

    with bench.step("pandas_to_spark"):
        spark_df = spark.createDataFrame(df)
        schema = spark_df.schema

    with bench.step("table_bootstrap"):
        table_name = f"bench_{dataset_name}".replace("-", "_")
        mgr = IcebergTableManager(spark=spark)
        full_name = mgr.create_table(table_name=table_name, schema=schema)

    with bench.step("iceberg_write"):
        writer = IcebergWriter(spark=spark)
        write_result = writer.append(
            df=spark_df,
            table_name=table_name,
            compression=compression,
            validate_schema=False,
        )

    with bench.step("iceberg_read_full_scan"):
        from odvs.config import get_config
        cfg = get_config()
        full_table = f"{cfg.iceberg.catalog_name}.{cfg.iceberg.database}.{table_name}"
        read_count = spark.table(full_table).count()
        logger.info(f"Full scan verified: {read_count:,} rows")

    report = bench.report(row_count=write_result.rows_written)
    report.print_report()

    if output_path:
        output = report.to_dict()
        output["write_snapshot_id"] = write_result.snapshot_id
        output["compression"] = compression
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Benchmark results written to: {output_path}")

    # Cleanup benchmark table
    mgr.drop_table(table_name, purge=True)
    logger.info(f"Cleaned up benchmark table: {table_name}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ODVS — Compression and pipeline benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Source dataset path or URI")
    parser.add_argument("--dataset-name", required=True, help="Dataset name label for the report")
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=["snappy", "gzip", "zstd", "none"],
        choices=SUPPORTED_CODECS,
        help="Codecs to benchmark (default: snappy gzip zstd none)",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=10_000,
        help="Number of rows to use for compression benchmark (default: 10000)",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline benchmark including Spark + Iceberg (requires running Spark)",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Codec to use for full pipeline benchmark write (default: zstd)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write benchmark results as JSON to FILE",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.full_pipeline:
        return run_full_pipeline_benchmark(
            source=args.source,
            dataset_name=args.dataset_name,
            compression=args.compression,
            output_path=args.output,
        )

    return run_compression_benchmark_standalone(
        source=args.source,
        dataset_name=args.dataset_name,
        codecs=args.codecs,
        sample_n=args.sample_n,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())