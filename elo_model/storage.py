"""Atomic local artifacts shared by collection, caching and export."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from .config import StorageConfig

logger = logging.getLogger(__name__)


def replace_checkpoint(source: Path, destination: Path) -> None:
    """Atomically replace a checkpoint despite short-lived Windows file locks.

    Args:
        source: Completed temporary file on the destination filesystem.
        destination: Checkpoint filename to replace.

    Returns:
        None; persistent permissions errors and all other failures propagate.
    """
    settings = StorageConfig()

    for attempt in range(settings.replace_attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if attempt + 1 == settings.replace_attempts:
                raise

            logger.warning(
                "Checkpoint temporarily locked: %s; retry %d/%d: %s",
                destination,
                attempt + 1,
                settings.replace_attempts - 1,
                error,
            )
            time.sleep(settings.retry_delay)


def write_json(path: str | Path, payload) -> None:
    """Atomically replace a strict JSON artifact on the same filesystem.

    Args:
        path: Destination filename; parent directories are created.
        payload: JSON-compatible data without NaN or infinity.

    Returns:
        None; serialization or filesystem errors propagate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".tmp", mode="w", encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, allow_nan=False)
        except (TypeError, ValueError):
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        replace_checkpoint(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a complete Parquet file before replacing the destination.

    Args:
        path: Destination filename; parent directories are created.
        frame: Rows to persist without a pandas index.

    Returns:
        None; temporary files are cleaned even on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".parquet"
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        replace_checkpoint(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
