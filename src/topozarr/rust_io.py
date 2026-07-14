"""Native write path: regions are encoded and stored by zarrs (Rust),
bypassing zarr-python's sync bridge. Metadata stays on zarr-python."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import zarr.storage


def store_to_url(store: Any) -> tuple[str, dict[str, str]]:
    """Derive a (url, object_store options) pair from a zarr store target.

    Supports local paths, ``s3://`` URL strings, ``zarr.storage.LocalStore``,
    and ``zarr.storage.ObjectStore`` wrapping an obstore ``S3Store`` (obstore
    config keys are object_store config keys, so they forward directly).
    """
    if isinstance(store, (str, Path)):
        return str(store), {}
    if isinstance(store, zarr.storage.LocalStore):
        return str(store.root), {}
    if isinstance(store, zarr.storage.ObjectStore):
        inner = store.store
        try:
            config = dict(getattr(inner, "config", None) or {})
        except BaseException as err:  # obstore config getter panics (pyo3)
            raise TypeError(
                f"cannot read config from {type(inner).__name__}: {err}"
            ) from None
        # client options (allow_http, timeouts) are valid object_store
        # config keys too
        config.update(getattr(inner, "client_options", None) or {})
        bucket = config.pop("bucket", None)
        if bucket is None:
            raise TypeError(
                f"cannot derive a bucket from {type(inner).__name__}; "
                "io='rust' supports obstore S3Store-backed ObjectStores"
            )
        prefix = getattr(inner, "prefix", None)
        url = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
        return url, {
            k: (str(v).lower() if isinstance(v, bool) else str(v))
            for k, v in config.items()
        }
    raise TypeError(
        f"io='rust' does not support store type {type(store).__name__}; "
        "pass a local path, s3:// url, LocalStore, or obstore-backed ObjectStore"
    )


def make_rust_writer(store: Any) -> Any:
    """Build a ``RustWriter`` for ``store``."""
    from topozarr_core import RustWriter

    url, options = store_to_url(store)
    return RustWriter(url, options or None)
