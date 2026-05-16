"""
ODVS Compression Benchmarks — Measure compression ratio and I/O performance
across codecs supported by Iceberg/Parquet.

This module is used both as a standalone benchmarking tool and as a
configuration advisor for the write pipeline.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from odvs.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_CODECS = ["snappy", "gzip", "brotli", "zstd", "lz4", "none"]


@dataclass
class CompressionBenchmarkResult:
    codec: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float  # original / compressed — higher is better
    write_duration_ms: float
    read_duration_ms: float
    throughput_mb_per_sec: float

    def __str__(self) -> str:
        return (
            f"{self.codec:<10} | "
            f"ratio={self.compression_ratio:.2f}x | "
            f"size={self.compressed_bytes / 1024:.1f}KB | "
            f"write={self.write_duration_ms:.1f}ms | "
            f"read={self.read_duration_ms:.1f}ms | "
            f"throughput={self.throughput_mb_per_sec:.1f}MB/s"
        )


@dataclass
class CompressionReport:
    dataset_name: str
    row_count: int
    column_count: int
    results: List[CompressionBenchmarkResult]

    @property
    def best_ratio(self) -> CompressionBenchmarkResult:
        return max(self.results, key=lambda r: r.compression_ratio)

    @property
    def best_throughput(self) -> CompressionBenchmarkResult:
        return max(self.results, key=lambda r: r.throughput_mb_per_sec)

    @property
    def recommendation(self) -> str:
        """
        Pragmatic recommendation following Iceberg community guidelines:
        - zstd for cold storage / long-term archival
        - snappy for hot paths / frequent reads
        - none for already-compressed data or latency-sensitive paths
        """
        codec_map = {r.codec: r for r in self.results}

        if "zstd" in codec_map and "snappy" in codec_map:
            zstd = codec_map["zstd"]
            snappy = codec_map["snappy"]
            if zstd.compression_ratio > snappy.compression_ratio * 1.2:
                return "zstd"  # Significantly better compression, use it
            if snappy.throughput_mb_per_sec > zstd.throughput_mb_per_sec * 1.5:
                return "snappy"  # Significantly faster, prefer for hot data
        return "zstd"  # Sensible default for most Iceberg workloads

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "codec": r.codec,
                "compressed_bytes": r.compressed_bytes,
                "compression_ratio": round(r.compression_ratio, 3),
                "write_ms": round(r.write_duration_ms, 1),
                "read_ms": round(r.read_duration_ms, 1),
                "throughput_mb_s": round(r.throughput_mb_per_sec, 1),
            }
            for r in self.results
        ])

    def print_report(self) -> None:
        print(f"\n{'='*70}")
        print(f"  Compression Report: {self.dataset_name}")
        print(f"  Rows: {self.row_count:,} | Columns: {self.column_count}")
        print(f"{'='*70}")
        print(f"{'Codec':<10} | {'Ratio':>7} | {'Size':>8} | {'Write':>8} | {'Read':>8} | {'MB/s':>8}")
        print(f"{'-'*70}")
        for r in sorted(self.results, key=lambda x: x.compression_ratio, reverse=True):
            print(str(r))
        print(f"{'-'*70}")
        print(f"  Recommended codec: {self.recommendation}")
        print(f"{'='*70}\n")


class CompressionBenchmarker:
    """
    Benchmarks Parquet compression codecs against a sample DataFrame.

    Uses PyArrow directly (not Spark) to avoid Spark overhead in benchmarking.
    Results are valid proxies for Iceberg's internal parquet write behavior.
    """

    def __init__(self, codecs: Optional[List[str]] = None) -> None:
        self._codecs = codecs or ["snappy", "gzip", "zstd", "none"]

    def benchmark(
        self,
        df: pd.DataFrame,
        dataset_name: str = "unnamed",
        sample_n: Optional[int] = 10_000,
    ) -> CompressionReport:
        """
        Run compression benchmark across configured codecs.

        Args:
            df: Pandas DataFrame to benchmark.
            dataset_name: Label used in the report.
            sample_n: Limit to N rows for benchmarking (None = use full df).

        Returns:
            CompressionReport with per-codec metrics.
        """
        if sample_n and len(df) > sample_n:
            bench_df = df.sample(n=sample_n, random_state=42)
            logger.info(f"Benchmarking on {sample_n:,}-row sample of {len(df):,}-row dataset")
        else:
            bench_df = df

        table = pa.Table.from_pandas(bench_df, preserve_index=False)
        original_bytes = self._estimate_uncompressed_size(table)

        results = []
        for codec in self._codecs:
            try:
                result = self._benchmark_codec(table, codec, original_bytes)
                results.append(result)
                logger.debug(str(result))
            except Exception as exc:
                logger.warning(f"Codec '{codec}' benchmark failed: {exc}")

        report = CompressionReport(
            dataset_name=dataset_name,
            row_count=len(bench_df),
            column_count=len(bench_df.columns),
            results=results,
        )
        return report

    def _benchmark_codec(
        self,
        table: pa.Table,
        codec: str,
        original_bytes: int,
    ) -> CompressionBenchmarkResult:
        compression = None if codec == "none" else codec

        # Write benchmark
        buf = io.BytesIO()
        write_start = time.monotonic()
        pq.write_table(table, buf, compression=compression, use_dictionary=True)
        write_duration_ms = (time.monotonic() - write_start) * 1000

        compressed_bytes = buf.tell()
        compression_ratio = original_bytes / max(compressed_bytes, 1)

        # Read benchmark
        buf.seek(0)
        read_start = time.monotonic()
        pq.read_table(buf)
        read_duration_ms = (time.monotonic() - read_start) * 1000

        # Throughput: MB/s based on original data size
        original_mb = original_bytes / (1024 * 1024)
        total_seconds = (write_duration_ms + read_duration_ms) / 1000
        throughput = original_mb / max(total_seconds, 1e-9)

        return CompressionBenchmarkResult(
            codec=codec,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            write_duration_ms=write_duration_ms,
            read_duration_ms=read_duration_ms,
            throughput_mb_per_sec=throughput,
        )

    def _estimate_uncompressed_size(self, table: pa.Table) -> int:
        """Estimate uncompressed size by writing with no compression."""
        buf = io.BytesIO()
        pq.write_table(table, buf, compression=None)
        return buf.tell()