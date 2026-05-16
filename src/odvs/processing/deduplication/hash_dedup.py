"""
ODVS Hash Deduplication — Row-level deduplication using configurable hash strategies.

Supports:
- Full-row hashing (MD5, SHA256, xxhash)
- Column-subset hashing
- Cross-batch deduplication via bloom filter or hash registry
- Spark-native deduplication for large-scale operation
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd

from odvs.logger import get_logger

logger = get_logger(__name__)


class HashAlgorithm(str, Enum):
    MD5 = "md5"
    SHA256 = "sha256"
    SHA1 = "sha1"


@dataclass
class DeduplicationResult:
    original_rows: int
    deduplicated_rows: int
    duplicates_removed: int
    duplicate_fraction: float
    duration_seconds: float
    hash_column: str
    strategy_columns: Optional[List[str]]

    @property
    def deduplication_rate(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return self.duplicates_removed / self.original_rows

    def __str__(self) -> str:
        return (
            f"Dedup: {self.original_rows:,} → {self.deduplicated_rows:,} rows "
            f"({self.duplicates_removed:,} removed, "
            f"{self.deduplication_rate:.1%} duplicate rate, "
            f"{self.duration_seconds:.2f}s)"
        )


def _hash_row_md5(row: pd.Series) -> str:
    content = "|".join(str(v) for v in row.values)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _hash_row_sha256(row: pd.Series) -> str:
    content = "|".join(str(v) for v in row.values)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _hash_row_sha1(row: pd.Series) -> str:
    content = "|".join(str(v) for v in row.values)
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


_HASHERS = {
    HashAlgorithm.MD5: _hash_row_md5,
    HashAlgorithm.SHA256: _hash_row_sha256,
    HashAlgorithm.SHA1: _hash_row_sha1,
}


class HashDeduplicator:
    """
    Row-level hash-based deduplication for pandas DataFrames.

    Design:
    - Hash is computed per-row (or per-subset of columns)
    - Hash column is appended to the DataFrame and kept after dedup
      so it can be stored in Iceberg and used for cross-batch dedup
    - Optionally excludes metadata columns (starting with _odvs_) from hash
    """

    HASH_COLUMN = "_odvs_row_hash"

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        hash_columns: Optional[List[str]] = None,
        exclude_system_columns: bool = True,
        known_hashes: Optional[Set[str]] = None,
    ) -> None:
        """
        Args:
            algorithm: Hash algorithm to use.
            hash_columns: Columns to include in hash. If None, all non-system columns are used.
            exclude_system_columns: Exclude _odvs_* columns from the hash computation.
            known_hashes: Set of hashes already seen (for cross-batch dedup).
        """
        self._algorithm = algorithm
        self._hash_columns = hash_columns
        self._exclude_system = exclude_system_columns
        self._known_hashes: Set[str] = known_hashes or set()
        self._hasher = _HASHERS[algorithm]


    def deduplicate(
        self,
        df: pd.DataFrame,
        keep: str = "first",
    ) -> tuple[pd.DataFrame, DeduplicationResult]:
        """
        Deduplicate a DataFrame by row hash.

        Args:
            df: Input DataFrame.
            keep: 'first' | 'last' — which duplicate to retain.

        Returns:
            Tuple of (deduplicated DataFrame with hash column, DeduplicationResult).
        """
        start = time.monotonic()
        original_rows = len(df)

        if original_rows == 0:
            return df, self._empty_result(original_rows, start)

        # Compute hashes
        hash_input_cols = self._resolve_hash_columns(df)
        df = df.copy()
        df[self.HASH_COLUMN] = df[hash_input_cols].apply(self._hasher, axis=1)

        # In-batch deduplication
        df = df.drop_duplicates(subset=[self.HASH_COLUMN], keep=keep)

        # Cross-batch deduplication (filter against known hashes)
        if self._known_hashes:
            before_cross = len(df)
            df = df[~df[self.HASH_COLUMN].isin(self._known_hashes)]
            cross_removed = before_cross - len(df)
            if cross_removed:
                logger.info(f"Cross-batch dedup removed {cross_removed:,} rows")

        # Update known hashes with new ones
        self._known_hashes.update(df[self.HASH_COLUMN].tolist())

        duration = time.monotonic() - start
        deduplicated_rows = len(df)
        duplicates_removed = original_rows - deduplicated_rows

        result = DeduplicationResult(
            original_rows=original_rows,
            deduplicated_rows=deduplicated_rows,
            duplicates_removed=duplicates_removed,
            duplicate_fraction=duplicates_removed / original_rows if original_rows else 0.0,
            duration_seconds=duration,
            hash_column=self.HASH_COLUMN,
            strategy_columns=hash_input_cols,
        )
        logger.info(str(result))

        return df.reset_index(drop=True), result

    def get_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the duplicate rows (excluding the first occurrence)."""
        hash_input_cols = self._resolve_hash_columns(df)
        df = df.copy()
        df[self.HASH_COLUMN] = df[hash_input_cols].apply(self._hasher, axis=1)
        duplicate_mask = df.duplicated(subset=[self.HASH_COLUMN], keep="first")
        return df[duplicate_mask].reset_index(drop=True)

    def compute_hashes(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of row hashes without modifying the DataFrame."""
        hash_input_cols = self._resolve_hash_columns(df)
        return df[hash_input_cols].apply(self._hasher, axis=1)

    @property
    def known_hash_count(self) -> int:
        return len(self._known_hashes)

    def clear_known_hashes(self) -> None:
        self._known_hashes.clear()


    def _resolve_hash_columns(self, df: pd.DataFrame) -> List[str]:
        """Determine which columns to include in the hash."""
        if self._hash_columns:
            missing = [c for c in self._hash_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Hash columns not found in DataFrame: {missing}")
            return self._hash_columns

        cols = list(df.columns)
        if self._exclude_system:
            cols = [c for c in cols if not c.startswith("_odvs_")]
        if not cols:
            raise ValueError("No columns available for hash computation after exclusion.")
        return cols

    def _empty_result(self, original_rows: int, start: float) -> DeduplicationResult:
        import time as _time
        return DeduplicationResult(
            original_rows=original_rows,
            deduplicated_rows=0,
            duplicates_removed=0,
            duplicate_fraction=0.0,
            duration_seconds=_time.monotonic() - start,
            hash_column=self.HASH_COLUMN,
            strategy_columns=None,
        )


def spark_deduplicate(
    spark_df: "pyspark.sql.DataFrame",  # type: ignore[name-defined]
    subset_columns: Optional[List[str]] = None,
) -> "pyspark.sql.DataFrame":  # type: ignore[name-defined]
    """
    Spark-native deduplication using dropDuplicates.
    Use for large-scale batch dedup where pandas is not practical.
    """
    if subset_columns:
        return spark_df.dropDuplicates(subset_columns)
    return spark_df.dropDuplicates()