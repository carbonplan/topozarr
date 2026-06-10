import pytest
import numpy as np
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor

_TEST_BUCKET = "test-topozarr"
_BOTO3_CREDS = {
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "region_name": "us-east-1",
}
_OBSTORE_CREDS = {
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "aws_region": "us-east-1",
}


@pytest.fixture(scope="session")
def moto_s3_server():
    import socket

    pytest.importorskip("boto3")
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    server = ThreadedMotoServer(port=port)
    server.start()
    yield port
    server.stop()


@pytest.fixture
def s3_zarr_store(moto_s3_server):
    import boto3
    from obstore.store import S3Store
    from zarr.storage import ObjectStore

    endpoint_url = f"http://127.0.0.1:{moto_s3_server}"
    boto3.client("s3", endpoint_url=endpoint_url, **_BOTO3_CREDS).create_bucket(
        Bucket=_TEST_BUCKET
    )
    obs = S3Store(
        _TEST_BUCKET,
        config={"endpoint": endpoint_url, "allow_http": "true", **_OBSTORE_CREDS},
    )
    return ObjectStore(obs)


@pytest.fixture
def create_dataset():
    def _generate(nx=16, ny=16, x_dim="x", y_dim="y", epsg="EPSG:4326", add_crs=True):
        ds = xr.Dataset(
            {"elevation": ((y_dim, x_dim), np.random.rand(ny, nx).astype("f4"))},
            coords={
                x_dim: np.linspace(0, nx - 1, nx),
                y_dim: np.linspace(0, ny - 1, ny),
            },
        )
        if add_crs:
            return ds.proj.assign_crs(spatial_ref=epsg)
        return ds

    return _generate
