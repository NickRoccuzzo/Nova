# utils.py

import os
import json
import tempfile
import logging
from datetime import datetime
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from flask import current_app, session, flash

logger = logging.getLogger(__name__)

# Module‑level cache for the on‑disk JSON
SEEKER_DB = None
SEEKER_DB_LAST_LOAD = None  # timestamp of last local load


def download_s3_file_if_newer(
    bucket: str,
    key: str,
    local_path: str,
    aws_key: str,
    aws_secret: str,
    region: str = None,
    max_retries: int = 3,
) -> bool:
    """
    Atomically download s3://bucket/key → local_path if the remote
    object is newer than local_path (or if local doesn't exist).
    Returns True if a download occurred, False if skipped.
    Raises on fatal errors.
    """
    # Build a boto3 client with retry logic
    session = boto3.session.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=region
    )
    s3 = session.client(
        "s3",
        config=BotoConfig(retries={"max_attempts": max_retries, "mode": "standard"})
    )

    # 1) HEAD to get remote metadata
    head = s3.head_object(Bucket=bucket, Key=key)
    remote_mtime = head["LastModified"]

    # 2) Compare to local file
    if os.path.exists(local_path):
        local_mtime = datetime.fromtimestamp(
            os.path.getmtime(local_path),
            tz=remote_mtime.tzinfo
        )
        if remote_mtime <= local_mtime:
            logger.debug(f"Local file {local_path} is up-to-date; skipping download.")
            return False

    # 3) Download to a temporary file
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(local_path) or ".") as tmp:
        tmp_path = tmp.name

    try:
        s3.download_file(bucket, key, tmp_path)
        os.replace(tmp_path, local_path)
        # Preserve the remote timestamp
        os.utime(local_path, (remote_mtime.timestamp(), remote_mtime.timestamp()))
        logger.info(f"Downloaded s3://{bucket}/{key} → {local_path}")
        return True
    except Exception:
        logger.exception(f"Failed to download s3://{bucket}/{key}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_seeker_db(local_path: str = None) -> dict:
    """
    Load seeker_database.json from disk and cache it in memory.
    Automatically reloads if the file's mtime changes.
    """
    global SEEKER_DB, SEEKER_DB_LAST_LOAD
    local_path = local_path or current_app.config["LOCAL_DB_FILE"]
    try:
        mtime = os.path.getmtime(local_path)
    except FileNotFoundError:
        logger.error(f"{local_path} not found.")
        return {}

    # First load or file changed?
    if SEEKER_DB is None or SEEKER_DB_LAST_LOAD is None or mtime > SEEKER_DB_LAST_LOAD:
        with open(local_path, "r") as f:
            SEEKER_DB = json.load(f)
        SEEKER_DB_LAST_LOAD = mtime
        logger.info(f"Loaded seeker DB ({local_path}) into memory.")

    return SEEKER_DB


def load_seeker_db_s3(
    aws_key: str,
    aws_secret: str,
    local_path: str = None
) -> dict:
    """
    Ensure the local seeker_database.json is up-to-date by
    downloading from S3 if needed, then return the parsed JSON.
    """
    cfg = current_app.config
    bucket = cfg["S3_BUCKET"]
    key    = cfg["S3_DB_KEY"]
    local  = local_path or os.path.join(os.getcwd(), cfg["LOCAL_DB_FILE"])

    try:
        download_s3_file_if_newer(
            bucket=bucket,
            key=key,
            local_path=local,
            aws_key=aws_key,
            aws_secret=aws_secret,
            region=cfg.get("S3_REGION")
        )
    except ClientError as e:
        logger.warning(f"S3 HEAD/download failed: {e}")
        # If we have no local copy, that's fatal
        if not os.path.exists(local):
            raise

    return load_seeker_db(local)


def download_seeker_db_if_missing() -> bool:
    """
    Make sure the local seeker_database.json exists by downloading
    it from S3 if missing. Returns True if the file is now available.
    """
    cfg = current_app.config
    local = os.path.join(os.getcwd(), cfg["LOCAL_DB_FILE"])

    # If it's already present, nothing to do
    if os.path.exists(local):
        return True

    ak = session.get("aws_access_key_id")
    sk = session.get("aws_secret_access_key")
    if not (ak and sk):
        flash("AWS credentials missing. Please log in.", "danger")
        return False

    try:
        load_seeker_db_s3(ak, sk, local_path=local)
        return True
    except Exception as e:
        flash(f"Failed to download database: {e}", "danger")
        return False
