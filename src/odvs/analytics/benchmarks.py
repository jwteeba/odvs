"""
ODVS Benchmarks — Pipeline performance benchmarking and profiling.

Measures end-to-end pipeline metrics:
- Ingestion throughput
- Transform latency
- Deduplication overhead
- Iceberg write performance
- Read/scan performance
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from odvs.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimedOperation:
    name: str
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.duration_seconds * 1000

    def __str__(self) -> str:
        return f"{self.name}: {self.duration_ms:.1f}ms"


@dataclass
class PipelineBenchmarkReport:
    pipeline_name: str
    total_rows: int
    operations: List[TimedOperation]

    @property
    def total_duration_seconds(self) -> float:
        return sum(op.duration_seconds for op in self.operations)

    @property
    def rows_per_second(self) -> float:
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_rows / self.total_duration_seconds

    def slowest_operations(self, n: int = 3) -> List[TimedOperation]:
        return sorted(self.operations, key=lambda x: x.duration_seconds, reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "total_rows": self.total_rows,
            "total_duration_seconds": round(self.total_duration_seconds, 4),
            "rows_per_second": round(self.rows_per_second, 1),
            "operations": [
                {
                    "name": op.name,
                    "duration_ms": round(op.duration_ms, 2),
                    **op.metadata,
                }
                for op in self.operations
            ],
        }

    def print_report(self) -> None:
        print(f"\n{'='*55}")
        print(f"  Pipeline Benchmark: {self.pipeline_name}")
        print(f"  Rows: {self.total_rows:,} | Total: {self.total_duration_seconds:.2f}s")
        print(f"  Throughput: {self.rows_per_second:,.0f} rows/sec")
        print(f"{'='*55}")
        for op in self.operations:
            bar_len = int((op.duration_seconds / max(self.total_duration_seconds, 1e-9)) * 30)
            bar = "█" * bar_len
            print(f"  {op.name:<30} {op.duration_ms:>8.1f}ms  {bar}")
        print(f"{'='*55}")
        print(f"  Slowest: {self.slowest_operations(1)[0].name}")
        print(f"{'='*55}\n")


class PipelineBenchmarker:
    """
    Context-manager based pipeline benchmarker.

    Usage:
        bench = PipelineBenchmarker("my_pipeline")
        with bench.step("ingestion"):
            df = ingester.ingest(...)
        with bench.step("dedup"):
            df, _ = deduplicator.deduplicate(df)
        report = bench.report(row_count=len(df))
        report.print_report()
    """

    def __init__(self, pipeline_name: str) -> None:
        self._name = pipeline_name
        self._operations: List[TimedOperation] = []

    @contextmanager
    def step(self, name: str, **metadata: Any) -> Generator[None, None, None]:
        start = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start
            op = TimedOperation(name=name, duration_seconds=duration, metadata=metadata)
            self._operations.append(op)
            logger.debug(str(op))

    def record(self, name: str, duration_seconds: float, **metadata: Any) -> None:
        """Manually record a timed operation (for async or external measurements)."""
        self._operations.append(
            TimedOperation(name=name, duration_seconds=duration_seconds, metadata=metadata)
        )

    def report(self, row_count: int = 0) -> PipelineBenchmarkReport:
        return PipelineBenchmarkReport(
            pipeline_name=self._name,
            total_rows=row_count,
            operations=list(self._operations),
        )

    def reset(self) -> None:
        self._operations.clear()