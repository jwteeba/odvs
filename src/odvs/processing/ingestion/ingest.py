"""
ODVS Ingestion — Multi-source dataset ingestion with pluggable readers.

Supports CSV, Parquet, JSON, Delta, and Hugging Face Hub as source formats.
Returns a standardized pandas DataFrame with provenance metadata attached.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd

from odvs.logger import get_logger

logger = get_logger(__name__)


class SourceType(str, Enum):
    CSV = "csv"
    PARQUET = "parquet" 
    JSON = "json"
    JSONL = "jsonl"
    HF_HUB = "hf_hub"
    S3 = "s3"
    HTTP = "http"


@dataclass
class IngestionManifest:
    """
    Immutable record of a completed ingestion operation.
    Attached to a dataset as lineage metadata.
    """
    source_uri: str
    source_type: SourceType
    ingested_at: datetime
    row_count: int
    column_count: int
    size_bytes: int
    checksum_sha256: str
    duration_seconds: float
    schema_snapshot: Dict[str, str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "source_type": self.source_type.value,
            "ingested_at": self.ingested_at.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "duration_seconds": round(self.duration_seconds, 4),
            "schema_snapshot": self.schema_snapshot,
            **self.extra,
        }


@dataclass
class IngestionResult:
    df: pd.DataFrame
    manifest: IngestionManifest


def _compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 checksum over DataFrame contents."""
    # Sort columns for determinism, then hash the CSV bytes
    normalized = df.reindex(sorted(df.columns), axis=1)
    content = normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _schema_snapshot(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


class DatasetIngester:
    """
    Multi-source ingestion engine.

    Handles format detection, reading, and manifest creation.
    Always returns pandas DataFrames — the Spark conversion happens downstream.
    """

    def ingest(
        self,
        source: Union[str, Path],
        source_type: Optional[SourceType] = None,
        read_options: Optional[Dict[str, Any]] = None,
        sample_n: Optional[int] = None,
    ) -> IngestionResult:
        """
        Ingest a dataset from any supported source.

        Args:
            source: File path, S3 URI, HTTP URL, or HF dataset name.
            source_type: Override auto-detected source type.
            read_options: Passed directly to the underlying reader (e.g. pandas.read_csv kwargs).
            sample_n: If set, return at most N rows (for development/testing).

        Returns:
            IngestionResult with DataFrame and manifest.
        """
        source_str = str(source)
        detected_type = source_type or self._detect_type(source_str)
        read_options = read_options or {}

        logger.info(f"Ingesting from {source_str} (type={detected_type.value})")
        start = time.monotonic()

        df = self._read(source_str, detected_type, read_options)

        if sample_n is not None and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled to {sample_n} rows")

        duration = time.monotonic() - start

        manifest = IngestionManifest(
            source_uri=source_str,
            source_type=detected_type,
            ingested_at=datetime.now(timezone.utc),
            row_count=len(df),
            column_count=len(df.columns),
            size_bytes=df.memory_usage(deep=True).sum(),
            checksum_sha256=_compute_dataframe_checksum(df),
            duration_seconds=duration,
            schema_snapshot=_schema_snapshot(df),
        )

        logger.info(
            f"Ingestion complete: {manifest.row_count:,} rows, "
            f"{manifest.column_count} columns, "
            f"{manifest.duration_seconds:.2f}s"
        )

        return IngestionResult(df=df, manifest=manifest)

    def ingest_many(
        self,
        sources: List[Union[str, Path]],
        source_type: Optional[SourceType] = None,
    ) -> IngestionResult:
        """
        Ingest multiple sources and concatenate into a single DataFrame.
        All sources must have compatible schemas.
        """
        results = [self.ingest(src, source_type=source_type) for src in sources]
        combined_df = pd.concat([r.df for r in results], ignore_index=True)

        first_manifest = results[0].manifest
        total_duration = sum(r.manifest.duration_seconds for r in results)

        merged_manifest = IngestionManifest(
            source_uri=f"multi:{len(sources)} sources",
            source_type=first_manifest.source_type,
            ingested_at=datetime.now(timezone.utc),
            row_count=len(combined_df),
            column_count=len(combined_df.columns),
            size_bytes=combined_df.memory_usage(deep=True).sum(),
            checksum_sha256=_compute_dataframe_checksum(combined_df),
            duration_seconds=total_duration,
            schema_snapshot=_schema_snapshot(combined_df),
            extra={"source_count": len(sources)},
        )

        return IngestionResult(df=combined_df, manifest=merged_manifest)


    def _detect_type(self, source: str) -> SourceType:
        parsed = urlparse(source)
        suffix = Path(parsed.path).suffix.lower()

        if parsed.scheme == "s3" or parsed.scheme == "s3a":
            if suffix == ".parquet":
                return SourceType.PARQUET
            if suffix == ".csv":
                return SourceType.CSV
            if suffix in (".json", ".jsonl"):
                return SourceType.JSON
            return SourceType.S3

        if parsed.scheme in ("http", "https"):
            return SourceType.HTTP

        # Local file detection by extension
        type_map = {
            ".csv": SourceType.CSV,
            ".tsv": SourceType.CSV,
            ".parquet": SourceType.PARQUET,
            ".json": SourceType.JSON,
            ".jsonl": SourceType.JSONL,
            ".ndjson": SourceType.JSONL,
        }
        if suffix in type_map:
            return type_map[suffix]

        raise ValueError(
            f"Cannot detect source type from: {source}. "
            f"Provide source_type explicitly."
        )


    def _read(
        self,
        source: str,
        source_type: SourceType,
        options: Dict[str, Any],
    ) -> pd.DataFrame:
        reader_map = {
            SourceType.CSV: self._read_csv,
            SourceType.PARQUET: self._read_parquet,
            SourceType.JSON: self._read_json,
            SourceType.JSONL: self._read_jsonl,
            SourceType.HF_HUB: self._read_hf_hub,
            SourceType.HTTP: self._read_http,
            SourceType.S3: self._read_s3,
        }
        reader = reader_map.get(source_type)
        if reader is None:
            raise ValueError(f"No reader registered for source type: {source_type}")
        return reader(source, options)

    def _read_csv(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        defaults = {"low_memory": False, "on_bad_lines": "warn"}
        return pd.read_csv(source, **{**defaults, **options})

    def _read_parquet(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        return pd.read_parquet(source, **options)

    def _read_json(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        return pd.read_json(source, **options)

    def _read_jsonl(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        defaults = {"lines": True}
        return pd.read_json(source, **{**defaults, **options})

    def _read_hf_hub(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        """
        Load a Hugging Face dataset by name.
        source format: 'hf_hub://dataset_name' or just 'dataset_name'
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Hugging Face `datasets` library required for HF Hub ingestion. "
                "Install with: pip install datasets"
            ) from exc

        dataset_name = source.replace("hf_hub://", "")
        split = options.pop("split", "train")
        ds = load_dataset(dataset_name, split=split, **options)
        return ds.to_pandas()

    def _read_http(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        suffix = Path(urlparse(source).path).suffix.lower()
        if suffix == ".csv":
            return self._read_csv(source, options)
        if suffix == ".parquet":
            return self._read_parquet(source, options)
        if suffix in (".json", ".jsonl"):
            return self._read_jsonl(source, options)
        raise ValueError(f"Cannot infer format from HTTP URL: {source}")

    def _read_s3(self, source: str, options: Dict[str, Any]) -> pd.DataFrame:
        """
        Read from S3 path. Falls back to extension detection.
        Requires s3fs to be installed and configured.
        """
        import s3fs  # type: ignore

        suffix = Path(urlparse(source).path).suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(source, **options)
        if suffix == ".csv":
            return pd.read_csv(source, **options)
        raise ValueError(f"Cannot infer format from S3 URI: {source}")