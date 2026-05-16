"""S3/Minio client wrapper for object storage operations.

Usage:
    from shared.storage.s3_client import S3Client

    s3 = S3Client(endpoint="http://minio:9000", bucket="vietcv", ...)
    s3.upload_file(file_bytes, "uploads/batch_001/course.docx")
    content = s3.download_file("uploads/batch_001/course.docx")
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Client:
    """Wrapper for S3/Minio operations."""

    def __init__(
        self,
        endpoint: str = "http://minio:9000",
        bucket: str = "vietcv",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        region: str = "us-east-1",
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )
        self._ensure_bucket()
        logger.info("S3Client connected to %s bucket=%s", endpoint, bucket)

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            try:
                self._client.create_bucket(Bucket=self._bucket)
                logger.info("Created S3 bucket: %s", self._bucket)
            except ClientError as exc:
                logger.warning("Could not create bucket %s: %s", self._bucket, exc)

    def upload_file(self, content: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to S3. Returns the S3 key."""
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )
        logger.debug("Uploaded %d bytes to s3://%s/%s", len(content), self._bucket, key)
        return f"s3://{self._bucket}/{key}"

    def download_file(self, key: str) -> bytes:
        """Download file content from S3."""
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def list_objects(self, prefix: str = "") -> List[str]:
        """List object keys under a prefix."""
        keys: List[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def delete_object(self, key: str) -> None:
        """Delete a single object."""
        self._client.delete_object(Bucket=self._bucket, Key=key)
        logger.debug("Deleted s3://%s/%s", self._bucket, key)

    def object_exists(self, key: str) -> bool:
        """Check if an object exists."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def verify_connection(self) -> bool:
        """Check if S3/Minio is reachable."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return True
        except Exception:
            return False

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for downloading an object."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
