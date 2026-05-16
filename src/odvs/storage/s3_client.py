"""
ODVS S3 Client — Wrapper around boto3 for S3-compatible storage.
Supports MinIO (local dev), AWS S3 (prod), and any S3-compatible backend.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from odvs.config import ODVSConfig, get_config
from odvs.logger import get_logger

logger = get_logger(__name__)


class S3Client:
    """
    Opinionated S3 client for ODVS operations.

    Does NOT replace Iceberg's internal S3FileIO — this client is for
    registry metadata, audit logs, and direct object operations that live
    outside the Iceberg table lifecycle.
    """

    def __init__(self, config: Optional[ODVSConfig] = None) -> None:
        self._cfg = config or get_config()
        self._s3_cfg = self._cfg.s3
        self._client = self._build_client()

    def _build_client(self) -> boto3.client:
        return boto3.client(
            "s3",
            endpoint_url=self._s3_cfg.endpoint_url,
            aws_access_key_id=self._s3_cfg.access_key,
            aws_secret_access_key=self._s3_cfg.secret_key,
            region_name=self._s3_cfg.region,
            config=Config(
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            ),
        )


    def ensure_bucket(self, bucket: Optional[str] = None) -> None:
        """Create bucket if it does not exist. Idempotent."""
        bucket = bucket or self._s3_cfg.bucket
        try:
            self._client.head_bucket(Bucket=bucket)
            logger.debug(f"Bucket already exists: {bucket}")
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=bucket)
                logger.info(f"Created bucket: {bucket}")
            else:
                raise

    def put_object(self, key: str, data: bytes, bucket: Optional[str] = None) -> None:
        """Upload raw bytes to the given key."""
        bucket = bucket or self._s3_cfg.bucket
        self._client.put_object(Bucket=bucket, Key=key, Body=data)
        logger.debug(f"PUT s3://{bucket}/{key} ({len(data)} bytes)")

    def get_object(self, key: str, bucket: Optional[str] = None) -> bytes:
        """Download raw bytes from the given key."""
        bucket = bucket or self._s3_cfg.bucket
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def delete_object(self, key: str, bucket: Optional[str] = None) -> None:
        bucket = bucket or self._s3_cfg.bucket
        self._client.delete_object(Bucket=bucket, Key=key)
        logger.debug(f"DELETE s3://{bucket}/{key}")

    def object_exists(self, key: str, bucket: Optional[str] = None) -> bool:
        bucket = bucket or self._s3_cfg.bucket
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    def list_objects(self, prefix: str, bucket: Optional[str] = None) -> Iterator[str]:
        """Yield all object keys matching the given prefix (handles pagination)."""
        bucket = bucket or self._s3_cfg.bucket
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def upload_file(self, local_path: Path, key: str, bucket: Optional[str] = None) -> None:
        bucket = bucket or self._s3_cfg.bucket
        self._client.upload_file(str(local_path), bucket, key)
        logger.info(f"Uploaded {local_path} → s3://{bucket}/{key}")

    def download_file(self, key: str, local_path: Path, bucket: Optional[str] = None) -> None:
        bucket = bucket or self._s3_cfg.bucket
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(bucket, key, str(local_path))
        logger.info(f"Downloaded s3://{bucket}/{key} → {local_path}")

    def get_object_size(self, key: str, bucket: Optional[str] = None) -> int:
        bucket = bucket or self._s3_cfg.bucket
        response = self._client.head_object(Bucket=bucket, Key=key)
        return response["ContentLength"]

    def copy_object(self, src_key: str, dst_key: str, bucket: Optional[str] = None) -> None:
        bucket = bucket or self._s3_cfg.bucket
        self._client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": src_key},
            Key=dst_key,
        )
        logger.debug(f"COPY s3://{bucket}/{src_key} → s3://{bucket}/{dst_key}")


@lru_cache(maxsize=1)
def get_s3_client() -> S3Client:
    """Singleton S3 client accessor."""
    return S3Client()