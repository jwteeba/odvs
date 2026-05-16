"""
ODVS Logger — Structured logging with optional JSON output.
Follows the pattern used in Airbnb / Stripe internal tooling.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from odvs.config import get_config


class _StructuredFormatter(logging.Formatter):
    """Emit log records as single-line JSON for log aggregation pipelines (Datadog, Splunk, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger configured for ODVS.

    Usage:
        logger = get_logger(__name__)
        logger.info("Dataset written", extra={"rows": 1000})
    """
    cfg = get_config().logging
    effective_level = level or cfg.level

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, effective_level.upper(), logging.INFO))

    if cfg.structured:
        handler.setFormatter(_StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(fmt=cfg.format, datefmt=cfg.datefmt)
        )

    logger.addHandler(handler)
    return logger


def configure_root_logger() -> None:
    """Configure the root logger to suppress noisy third-party logs."""
    cfg = get_config().logging
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format=cfg.format,
        datefmt=cfg.datefmt,
        stream=sys.stdout,
    )