"""
ODVS Spark Session — SparkSession factory with full Iceberg + S3 configuration.

This module is the single source of truth for all Spark session creation in ODVS.
All sessions are pre-configured for Iceberg catalog management and S3-compatible storage.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pyspark.sql import SparkSession

from odvs.config import ODVSConfig, SparkConfig, S3Config, IcebergConfig, get_config
from odvs.logger import get_logger

logger = get_logger(__name__)


def _iceberg_packages(spark_cfg: SparkConfig) -> str:
    """Construct the Maven package string for Iceberg + AWS bundle + Hadoop AWS."""
    return (
        f"org.apache.iceberg:iceberg-spark-runtime-{spark_cfg.spark_version}_{spark_cfg.scala_version}:{spark_cfg.iceberg_version},"
        f"org.apache.iceberg:iceberg-aws-bundle:{spark_cfg.aws_bundle_version},"
        f"org.apache.hadoop:hadoop-aws:3.4.2"
    )


def build_spark_session(
    config: Optional[ODVSConfig] = None,
    app_name_suffix: Optional[str] = None,
) -> SparkSession:
    """
    Build a fully configured SparkSession for ODVS pipelines.

    Configuration covers:
    - Iceberg catalog (Hadoop or REST) backed by S3
    - S3A connector with endpoint override for MinIO/S3-compatible stores
    - Iceberg SQL extensions for DDL/DML
    - Kryo serializer for performance
    - Dynamic allocation disabled for predictable local dev behavior

    Args:
        config: ODVSConfig instance (defaults to singleton).
        app_name_suffix: Optional suffix appended to the app name (useful for scripts).

    Returns:
        Configured SparkSession.
    """
    cfg = config or get_config()
    spark_cfg = cfg.spark
    s3_cfg = cfg.s3
    ice_cfg = cfg.iceberg

    app_name = spark_cfg.app_name
    if app_name_suffix:
        app_name = f"{app_name}::{app_name_suffix}"

    logger.info(f"Building SparkSession: app={app_name}, master={spark_cfg.master}")

    warehouse = s3_cfg.warehouse_path

    builder = (
        SparkSession.builder.appName(app_name)
        .master(spark_cfg.master)
        # Iceberg catalog
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(
            f"spark.sql.catalog.{ice_cfg.catalog_name}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(
            f"spark.sql.catalog.{ice_cfg.catalog_name}.type",
            spark_cfg.iceberg_catalog_type,
        )
        .config(
            f"spark.sql.catalog.{ice_cfg.catalog_name}.warehouse",
            warehouse,
        )
        # S3A connector (hadoop-aws provides S3AFileSystem)
        .config("spark.hadoop.fs.s3a.endpoint", s3_cfg.endpoint_url)
        .config("spark.hadoop.fs.s3a.access.key", s3_cfg.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", s3_cfg.secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # S3 via Iceberg S3FileIO (iceberg-aws-bundle provides the implementation)
        .config(f"spark.sql.catalog.{ice_cfg.catalog_name}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.endpoint", s3_cfg.endpoint_url)
        .config(f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.access-key-id", s3_cfg.access_key)
        .config(f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.secret-access-key", s3_cfg.secret_key)
        .config(f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.path-style-access", "true")
        .config(f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.region", s3_cfg.region)
        # Memory / performance 
        .config("spark.driver.memory", spark_cfg.driver_memory)
        .config("spark.executor.memory", spark_cfg.executor_memory)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Iceberg write defaults
        .config("spark.sql.defaultCatalog", ice_cfg.catalog_name)
        .config(
            f"spark.sql.catalog.{ice_cfg.catalog_name}.io-impl",
            "org.apache.iceberg.aws.s3.S3FileIO",
        )
        .config(
            f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.endpoint",
            s3_cfg.endpoint_url,
        )
        .config(
            f"spark.sql.catalog.{ice_cfg.catalog_name}.s3.path-style-access",
            "true",
        )
        # Packages (loaded via --packages at submit time in prod)
        .config("spark.jars.packages", _iceberg_packages(spark_cfg))
    )

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")

    logger.info(f"SparkSession active. Warehouse: {warehouse}")
    return session


def get_or_create_session(
    config: Optional[ODVSConfig] = None,
    app_name_suffix: Optional[str] = None,
) -> SparkSession:
    """
    Return the active SparkSession if one exists, otherwise build a new one.
    This is the primary entry point for pipeline modules.
    """
    existing = SparkSession.getActiveSession()
    if existing is not None:
        logger.debug("Reusing existing SparkSession")
        return existing
    return build_spark_session(config=config, app_name_suffix=app_name_suffix)