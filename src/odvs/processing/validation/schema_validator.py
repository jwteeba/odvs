"""
ODVS Schema Validator — Pre-write schema validation for DataFrames.

Validates column presence, type compatibility, nullability constraints,
and value range rules against a declarative schema spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from odvs.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    column: Optional[str]
    message: str
    severity: ValidationSeverity
    rule: str

    def __str__(self) -> str:
        location = f"[{self.column}]" if self.column else "[global]"
        return f"{self.severity.value.upper()} {location} ({self.rule}): {self.message}"


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    passed: bool = True

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.passed = False

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def raise_if_invalid(self) -> None:
        if not self.passed:
            error_msgs = "\n".join(str(e) for e in self.errors)
            raise SchemaValidationError(f"Schema validation failed:\n{error_msgs}")

    def summary(self) -> str:
        return (
            f"ValidationReport: passed={self.passed}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)}, "
            f"total_issues={len(self.issues)}"
        )


class SchemaValidationError(Exception):
    """Raised when a DataFrame fails schema validation with ERROR-level issues."""


@dataclass
class ColumnSpec:
    """Declarative spec for a single column."""
    name: str
    dtype: Optional[str] = None          # e.g. "int64", "float64", "object", "datetime64[ns]"
    nullable: bool = True
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    max_null_fraction: Optional[float] = None  # e.g. 0.1 = at most 10% nulls
    regex_pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass
class SchemaSpec:
    """Full schema specification for a dataset."""
    columns: List[ColumnSpec]
    allow_extra_columns: bool = True
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    require_unique: Optional[List[str]] = None  # Column names that must be unique

    @property
    def required_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.required]

    @property
    def column_map(self) -> Dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}


class SchemaValidator:
    """
    Validates a pandas DataFrame against a SchemaSpec.

    Returns a ValidationReport — callers decide whether to raise or warn.
    All rules are evaluated independently (no short-circuit on first error).
    """

    def validate(self, df: pd.DataFrame, spec: SchemaSpec) -> ValidationReport:
        report = ValidationReport()
        self._check_row_count(df, spec, report)
        self._check_required_columns(df, spec, report)
        self._check_extra_columns(df, spec, report)
        self._check_unique_columns(df, spec, report)

        for col_spec in spec.columns:
            if col_spec.name not in df.columns:
                continue  # Already handled in required column check
            self._validate_column(df, col_spec, report)

        if report.passed:
            logger.info(f"Schema validation passed: {len(df):,} rows, {len(df.columns)} columns")
        else:
            logger.warning(f"Schema validation failed: {report.summary()}")

        return report


    def _check_row_count(
        self, df: pd.DataFrame, spec: SchemaSpec, report: ValidationReport
    ) -> None:
        if spec.min_rows is not None and len(df) < spec.min_rows:
            report.add(ValidationIssue(
                column=None,
                message=f"DataFrame has {len(df):,} rows, minimum required is {spec.min_rows:,}",
                severity=ValidationSeverity.ERROR,
                rule="min_rows",
            ))
        if spec.max_rows is not None and len(df) > spec.max_rows:
            report.add(ValidationIssue(
                column=None,
                message=f"DataFrame has {len(df):,} rows, maximum allowed is {spec.max_rows:,}",
                severity=ValidationSeverity.WARNING,
                rule="max_rows",
            ))

    def _check_required_columns(
        self, df: pd.DataFrame, spec: SchemaSpec, report: ValidationReport
    ) -> None:
        for col_name in spec.required_columns:
            if col_name not in df.columns:
                report.add(ValidationIssue(
                    column=col_name,
                    message=f"Required column '{col_name}' is missing from DataFrame",
                    severity=ValidationSeverity.ERROR,
                    rule="required_column",
                ))

    def _check_extra_columns(
        self, df: pd.DataFrame, spec: SchemaSpec, report: ValidationReport
    ) -> None:
        if not spec.allow_extra_columns:
            defined = {c.name for c in spec.columns}
            extra = set(df.columns) - defined
            if extra:
                report.add(ValidationIssue(
                    column=None,
                    message=f"Unexpected columns present: {sorted(extra)}",
                    severity=ValidationSeverity.WARNING,
                    rule="extra_columns",
                ))

    def _check_unique_columns(
        self, df: pd.DataFrame, spec: SchemaSpec, report: ValidationReport
    ) -> None:
        if not spec.require_unique:
            return
        available = [c for c in spec.require_unique if c in df.columns]
        if not available:
            return
        duplicates = df[available].duplicated().sum()
        if duplicates > 0:
            report.add(ValidationIssue(
                column=", ".join(available),
                message=f"Found {duplicates:,} duplicate rows across unique key columns: {available}",
                severity=ValidationSeverity.ERROR,
                rule="unique_constraint",
            ))


    def _validate_column(
        self, df: pd.DataFrame, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        series = df[spec.name]

        self._check_nullable(series, spec, report)
        self._check_null_fraction(series, spec, report)
        self._check_dtype(series, spec, report)
        self._check_value_range(series, spec, report)
        self._check_allowed_values(series, spec, report)
        self._check_string_constraints(series, spec, report)

    def _check_nullable(
        self, series: pd.Series, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        if not spec.nullable and series.isnull().any():
            null_count = series.isnull().sum()
            report.add(ValidationIssue(
                column=spec.name,
                message=f"Column is non-nullable but has {null_count:,} null values",
                severity=ValidationSeverity.ERROR,
                rule="nullable",
            ))

    def _check_null_fraction(
        self, series: pd.Series, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        if spec.max_null_fraction is None:
            return
        null_frac = series.isnull().mean()
        if null_frac > spec.max_null_fraction:
            report.add(ValidationIssue(
                column=spec.name,
                message=(
                    f"Null fraction {null_frac:.1%} exceeds max allowed {spec.max_null_fraction:.1%}"
                ),
                severity=ValidationSeverity.ERROR,
                rule="max_null_fraction",
            ))

    def _check_dtype(
        self, series: pd.Series, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        if spec.dtype is None:
            return
        actual = str(series.dtype)
        if actual != spec.dtype:
            report.add(ValidationIssue(
                column=spec.name,
                message=f"Expected dtype '{spec.dtype}', got '{actual}'",
                severity=ValidationSeverity.WARNING,
                rule="dtype",
            ))

    def _check_value_range(
        self, series: pd.Series, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isnull().all():
            return  # Not a numeric column

        if spec.min_value is not None:
            violations = (numeric < spec.min_value).sum()
            if violations:
                report.add(ValidationIssue(
                    column=spec.name,
                    message=f"{violations:,} values below minimum {spec.min_value}",
                    severity=ValidationSeverity.ERROR,
                    rule="min_value",
                ))

        if spec.max_value is not None:
            violations = (numeric > spec.max_value).sum()
            if violations:
                report.add(ValidationIssue(
                    column=spec.name,
                    message=f"{violations:,} values above maximum {spec.max_value}",
                    severity=ValidationSeverity.ERROR,
                    rule="max_value",
                ))

    def _check_allowed_values(
        self, series: pd.Series, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        if not spec.allowed_values:
            return
        non_null = series.dropna()
        violations = (~non_null.isin(spec.allowed_values)).sum()
        if violations:
            report.add(ValidationIssue(
                column=spec.name,
                message=f"{violations:,} values not in allowed set {spec.allowed_values}",
                severity=ValidationSeverity.ERROR,
                rule="allowed_values",
            ))

    def _check_string_constraints(
        self, series: pd.Series, spec: ColumnSpec, report: ValidationReport
    ) -> None:
        import re as _re

        non_null_str = series.dropna().astype(str)

        if spec.regex_pattern:
            pattern = _re.compile(spec.regex_pattern)
            violations = (~non_null_str.str.match(pattern)).sum()
            if violations:
                report.add(ValidationIssue(
                    column=spec.name,
                    message=f"{violations:,} values do not match pattern '{spec.regex_pattern}'",
                    severity=ValidationSeverity.ERROR,
                    rule="regex_pattern",
                ))

        if spec.min_length is not None:
            violations = (non_null_str.str.len() < spec.min_length).sum()
            if violations:
                report.add(ValidationIssue(
                    column=spec.name,
                    message=f"{violations:,} values shorter than min_length={spec.min_length}",
                    severity=ValidationSeverity.ERROR,
                    rule="min_length",
                ))

        if spec.max_length is not None:
            violations = (non_null_str.str.len() > spec.max_length).sum()
            if violations:
                report.add(ValidationIssue(
                    column=spec.name,
                    message=f"{violations:,} values longer than max_length={spec.max_length}",
                    severity=ValidationSeverity.WARNING,
                    rule="max_length",
                ))