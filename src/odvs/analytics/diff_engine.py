"""
ODVS Diff Engine — Compute structural and statistical diffs between two dataset versions.

Supports:
- Schema diffs (added/removed/changed columns)
- Row-level diffs (added/removed/changed rows via hash)
- Statistical diffs (distribution changes per column)
- Iceberg snapshot-to-snapshot incremental diffs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from odvs.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaDiff:
    added_columns: List[str]
    removed_columns: List[str]
    type_changed: Dict[str, Tuple[str, str]]  # col → (old_type, new_type)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_columns or self.removed_columns or self.type_changed)

    def summary(self) -> str:
        parts = []
        if self.added_columns:
            parts.append(f"+{len(self.added_columns)} cols added")
        if self.removed_columns:
            parts.append(f"-{len(self.removed_columns)} cols removed")
        if self.type_changed:
            parts.append(f"~{len(self.type_changed)} type changes")
        return ", ".join(parts) if parts else "No schema changes"


@dataclass
class RowDiff:
    rows_added: int
    rows_removed: int
    rows_unchanged: int
    rows_total_before: int
    rows_total_after: int

    @property
    def change_rate(self) -> float:
        if self.rows_total_before == 0:
            return 1.0
        return (self.rows_added + self.rows_removed) / self.rows_total_before

    def summary(self) -> str:
        return (
            f"+{self.rows_added:,} rows | -{self.rows_removed:,} rows | "
            f"={self.rows_unchanged:,} unchanged | "
            f"change_rate={self.change_rate:.1%}"
        )


@dataclass
class ColumnStats:
    column: str
    dtype: str
    null_count: int
    null_fraction: float
    unique_count: Optional[int]
    min_value: Optional[Any]
    max_value: Optional[Any]
    mean_value: Optional[float]
    std_value: Optional[float]
    top_values: Optional[Dict[Any, int]]  # value → count


@dataclass
class StatisticalDiff:
    column: str
    before: ColumnStats
    after: ColumnStats
    null_fraction_delta: float
    unique_count_delta: Optional[int]
    mean_delta: Optional[float]
    std_delta: Optional[float]

    @property
    def is_significant(self) -> bool:
        """Flag statistically significant changes."""
        if abs(self.null_fraction_delta) > 0.05:
            return True
        if self.mean_delta is not None and self.before.mean_value:
            relative_change = abs(self.mean_delta) / max(abs(self.before.mean_value), 1e-9)
            if relative_change > 0.1:
                return True
        return False


@dataclass
class DatasetDiff:
    dataset_name: str
    version_before: str
    version_after: str
    schema_diff: SchemaDiff
    row_diff: RowDiff
    statistical_diffs: List[StatisticalDiff]

    @property
    def has_breaking_changes(self) -> bool:
        return bool(self.schema_diff.removed_columns or self.schema_diff.type_changed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "schema": {
                "added_columns": self.schema_diff.added_columns,
                "removed_columns": self.schema_diff.removed_columns,
                "type_changed": {
                    k: {"from": v[0], "to": v[1]}
                    for k, v in self.schema_diff.type_changed.items()
                },
            },
            "rows": {
                "added": self.row_diff.rows_added,
                "removed": self.row_diff.rows_removed,
                "unchanged": self.row_diff.rows_unchanged,
                "total_before": self.row_diff.rows_total_before,
                "total_after": self.row_diff.rows_total_after,
                "change_rate": round(self.row_diff.change_rate, 4),
            },
            "significant_column_changes": [
                s.column for s in self.statistical_diffs if s.is_significant
            ],
            "has_breaking_changes": self.has_breaking_changes,
        }

    def print_report(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Diff: {self.dataset_name}")
        print(f"  {self.version_before} → {self.version_after}")
        print(f"{'='*60}")
        print(f"  Schema: {self.schema_diff.summary()}")
        print(f"  Rows:   {self.row_diff.summary()}")
        if self.has_breaking_changes:
            print(f"  ⚠️  BREAKING CHANGES DETECTED")
        sig_cols = [s.column for s in self.statistical_diffs if s.is_significant]
        if sig_cols:
            print(f"  Significant column changes: {sig_cols}")
        print(f"{'='*60}\n")


class DiffEngine:
    """
    Compute diffs between two versions of a dataset.

    Works with pandas DataFrames for in-memory comparison.
    For Iceberg-native incremental reads, use IcebergVersionManager.read_between_snapshots().
    """

    def __init__(self, hash_column: str = "_odvs_row_hash") -> None:
        self._hash_col = hash_column

    def diff(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        dataset_name: str = "unnamed",
        version_before: str = "v_before",
        version_after: str = "v_after",
        include_statistics: bool = True,
    ) -> DatasetDiff:
        """
        Compute a full diff between two DataFrames.

        Args:
            df_before: Older version of the dataset.
            df_after: Newer version of the dataset.
            dataset_name: Label for the dataset.
            version_before: Label for the older version.
            version_after: Label for the newer version.
            include_statistics: Whether to compute per-column statistics.

        Returns:
            DatasetDiff object with all diff information.
        """
        logger.info(f"Computing diff: {version_before} ({len(df_before):,} rows) → {version_after} ({len(df_after):,} rows)")

        schema_diff = self._compute_schema_diff(df_before, df_after)
        row_diff = self._compute_row_diff(df_before, df_after)

        statistical_diffs = []
        if include_statistics:
            common_cols = set(df_before.columns) & set(df_after.columns)
            common_cols -= {self._hash_col}
            for col in sorted(common_cols):
                stat_diff = self._compute_column_stat_diff(df_before[col], df_after[col], col)
                statistical_diffs.append(stat_diff)

        return DatasetDiff(
            dataset_name=dataset_name,
            version_before=version_before,
            version_after=version_after,
            schema_diff=schema_diff,
            row_diff=row_diff,
            statistical_diffs=statistical_diffs,
        )


    def _compute_schema_diff(self, before: pd.DataFrame, after: pd.DataFrame) -> SchemaDiff:
        before_schema = {c: str(t) for c, t in before.dtypes.items()}
        after_schema = {c: str(t) for c, t in after.dtypes.items()}

        added = [c for c in after_schema if c not in before_schema]
        removed = [c for c in before_schema if c not in after_schema]
        type_changed = {
            c: (before_schema[c], after_schema[c])
            for c in before_schema
            if c in after_schema and before_schema[c] != after_schema[c]
        }

        return SchemaDiff(
            added_columns=added,
            removed_columns=removed,
            type_changed=type_changed,
        )


    def _compute_row_diff(self, before: pd.DataFrame, after: pd.DataFrame) -> RowDiff:
        before_hashes = self._get_hashes(before)
        after_hashes = self._get_hashes(after)

        added = len(after_hashes - before_hashes)
        removed = len(before_hashes - after_hashes)
        unchanged = len(before_hashes & after_hashes)

        return RowDiff(
            rows_added=added,
            rows_removed=removed,
            rows_unchanged=unchanged,
            rows_total_before=len(before),
            rows_total_after=len(after),
        )

    def _get_hashes(self, df: pd.DataFrame) -> Set[str]:
        if self._hash_col in df.columns:
            return set(df[self._hash_col].dropna().tolist())

        # Compute on-the-fly if hash column not present
        import hashlib
        non_system_cols = [c for c in df.columns if not c.startswith("_odvs_")]
        return set(
            df[non_system_cols].apply(
                lambda row: hashlib.sha256(
                    "|".join(str(v) for v in row.values).encode()
                ).hexdigest(),
                axis=1,
            ).tolist()
        )


    def _compute_column_stat_diff(
        self, before: pd.Series, after: pd.Series, col_name: str
    ) -> StatisticalDiff:
        before_stats = self._column_stats(before, col_name)
        after_stats = self._column_stats(after, col_name)

        null_delta = after_stats.null_fraction - before_stats.null_fraction
        unique_delta = (
            (after_stats.unique_count - before_stats.unique_count)
            if before_stats.unique_count is not None and after_stats.unique_count is not None
            else None
        )
        mean_delta = (
            (after_stats.mean_value - before_stats.mean_value)
            if before_stats.mean_value is not None and after_stats.mean_value is not None
            else None
        )
        std_delta = (
            (after_stats.std_value - before_stats.std_value)
            if before_stats.std_value is not None and after_stats.std_value is not None
            else None
        )

        return StatisticalDiff(
            column=col_name,
            before=before_stats,
            after=after_stats,
            null_fraction_delta=null_delta,
            unique_count_delta=unique_delta,
            mean_delta=mean_delta,
            std_delta=std_delta,
        )

    def _column_stats(self, series: pd.Series, col_name: str) -> ColumnStats:
        null_count = series.isnull().sum()
        null_fraction = null_count / len(series) if len(series) else 0.0
        non_null = series.dropna()

        numeric = pd.to_numeric(non_null, errors="coerce")
        is_numeric = numeric.notna().mean() > 0.5

        return ColumnStats(
            column=col_name,
            dtype=str(series.dtype),
            null_count=int(null_count),
            null_fraction=float(null_fraction),
            unique_count=int(non_null.nunique()) if len(non_null) < 100_000 else None,
            min_value=series.min() if is_numeric else None,
            max_value=series.max() if is_numeric else None,
            mean_value=float(numeric.mean()) if is_numeric else None,
            std_value=float(numeric.std()) if is_numeric else None,
            top_values=non_null.value_counts().head(5).to_dict() if not is_numeric else None,
        )