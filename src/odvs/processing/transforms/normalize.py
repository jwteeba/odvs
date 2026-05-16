"""
ODVS Normalize — Column-level normalization transforms applied pre-ingestion.

Transforms are composable and order-dependent. Each transform is a pure function
operating on a pandas DataFrame. The TransformPipeline applies them in sequence.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from odvs.logger import get_logger

logger = get_logger(__name__)

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names to lowercase with underscores."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9_]", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]
    return df


def deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Append _N suffix to duplicate column names."""
    seen: Dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df = df.copy()
    df.columns = new_cols
    return df


def strip_whitespace(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string columns."""
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)
    return df


def drop_empty_columns(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Drop columns where the fraction of null/empty values >= threshold.
    threshold=1.0 drops only fully empty columns.
    threshold=0.9 drops columns with 90%+ missing data.
    """
    null_fractions = df.isnull().mean()
    cols_to_drop = null_fractions[null_fractions >= threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} empty column(s): {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ALL values are null."""
    original = len(df)
    df = df.dropna(how="all")
    dropped = original - len(df)
    if dropped:
        logger.info(f"Dropped {dropped:,} fully-empty rows")
    return df.reset_index(drop=True)


def cast_numeric_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Coerce object columns to numeric where possible.
    Non-convertible values become NaN.
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()
    for col in target_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0 and coerced.notna().mean() > 0.5:
            df[col] = coerced
    return df


def normalize_unicode(df: pd.DataFrame, form: str = "NFC") -> pd.DataFrame:
    """Normalize unicode in all string columns to the given normal form."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: unicodedata.normalize(form, x) if isinstance(x, str) else x
        )
    return df


def add_ingestion_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add a UTC ingestion timestamp column to the DataFrame."""
    import datetime
    df = df.copy()
    df["_odvs_ingested_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return df


def select_columns(columns: List[str]) -> Transform:
    """Return a transform that selects (and orders) specific columns."""
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return df[columns]
    return _transform


def rename_columns(mapping: Dict[str, str]) -> Transform:
    """Return a transform that renames columns by the given mapping."""
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=mapping)
    return _transform


def drop_columns(columns: List[str]) -> Transform:
    """Return a transform that drops specific columns."""
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in columns if c in df.columns]
        return df.drop(columns=existing)
    return _transform


def fill_nulls(fill_values: Dict[str, Any]) -> Transform:
    """Return a transform that fills nulls per column with given values."""
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(fill_values)
    return _transform


def coerce_dtypes(dtype_map: Dict[str, str]) -> Transform:
    """Return a transform that casts columns to specified dtypes."""
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as exc:
                    logger.warning(f"Could not cast {col} to {dtype}: {exc}")
        return df
    return _transform


class TransformPipeline:
    """
    Composable transformation pipeline.

    Usage:
        pipeline = (
            TransformPipeline()
            .add(lowercase_columns)
            .add(strip_whitespace)
            .add(drop_empty_rows)
            .add(rename_columns({"user_id": "uid"}))
        )
        df_clean = pipeline.run(df_raw)
    """

    def __init__(self) -> None:
        self._transforms: List[Transform] = []

    def add(self, transform: Transform) -> "TransformPipeline":
        self._transforms.append(transform)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Running TransformPipeline with {len(self._transforms)} step(s)")
        for i, transform in enumerate(self._transforms):
            name = getattr(transform, "__name__", repr(transform))
            logger.debug(f"  Step {i + 1}: {name}")
            df = transform(df)
        return df

    @classmethod
    def standard(cls) -> "TransformPipeline":
        """
        Return the standard ODVS normalization pipeline.
        Suitable for most structured datasets.
        """
        return (
            cls()
            .add(lowercase_columns)
            .add(deduplicate_column_names)
            .add(strip_whitespace)
            .add(drop_empty_rows)
            .add(drop_empty_columns)
            .add(normalize_unicode)
            .add(add_ingestion_timestamp)
        )