import numpy as np
import pytest
import xarray as xr
from topozarr.coarsen import create_pyramid
from topozarr.rust_io import store_to_url


def _dataset_with_nodata(create_dataset, nx=64, ny=64):
    ds = create_dataset(nx=nx, ny=ny)
    # all-fill corner exercises skip_empty alongside written regions
    ds.elevation.values[: ny // 2, : nx // 2] = np.nan
    return ds


def _assert_pyramids_equal(a, b):
    dt_a = xr.open_datatree(a, engine="zarr", consolidated=False)
    dt_b = xr.open_datatree(b, engine="zarr", consolidated=False)
    assert set(dt_a.children) == set(dt_b.children)
    for lvl in dt_a.children:
        np.testing.assert_array_equal(
            dt_a[lvl].ds.elevation.values, dt_b[lvl].ds.elevation.values
        )


def test_rust_io_local_matches_python(create_dataset, tmp_path):
    ds = _dataset_with_nodata(create_dataset)
    py_dst = tmp_path / "py.zarr"
    rs_dst = tmp_path / "rs.zarr"
    create_pyramid(ds, levels=3).write(str(py_dst))
    create_pyramid(ds, levels=3).write(str(rs_dst), io="rust")
    _assert_pyramids_equal(py_dst, rs_dst)


def test_rust_io_local_no_fusion(create_dataset, tmp_path):
    # without fusion, levels 1+ read back through zarr-python while writes
    # go through rust; exercises mixed read/write paths on one store
    ds = _dataset_with_nodata(create_dataset)
    dst = tmp_path / "rs.zarr"
    create_pyramid(ds, levels=3).write(str(dst), io="rust", keep_levels_in_memory=False)
    dt = xr.open_datatree(dst, engine="zarr", consolidated=False)
    expected = ds.coarsen(x=2, y=2, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values, expected.elevation.values, rtol=1e-6
    )


def test_rust_io_s3(create_dataset, s3_zarr_store, tmp_path):
    ds = _dataset_with_nodata(create_dataset)
    create_pyramid(ds, levels=3).write(s3_zarr_store, io="rust")
    py_dst = tmp_path / "py.zarr"
    create_pyramid(ds, levels=3).write(str(py_dst))
    _assert_pyramids_equal(s3_zarr_store, py_dst)


@pytest.mark.parametrize("order", ["C", "F"])
def test_rust_io_writes_noncontiguous_block(tmp_path, order):
    # A: write_region borrows the numpy buffer zero-copy for C-contiguous
    # blocks and falls back to a single owned copy otherwise. An F-ordered
    # block exercises that fallback; both orders must round-trip identically.
    import zarr
    from topozarr_core import RustWriter

    dst = tmp_path / "a.zarr"
    root = zarr.open_group(str(dst), mode="w", zarr_format=3)
    root.create_array(
        "elevation",
        shape=(16, 16),
        chunks=(8, 8),
        shards=(16, 16),
        dtype="float32",
        fill_value=np.nan,
    )
    data = np.arange(16 * 16, dtype="float32").reshape(16, 16)
    block = np.asarray(data, order=order)
    assert block.flags["C_CONTIGUOUS"] == (order == "C")

    writer = RustWriter(str(dst))
    writer.write_region("/elevation", [0, 0], block)
    writer.flush()

    out = zarr.open_array(str(dst), path="elevation", mode="r")
    np.testing.assert_array_equal(out[:], data)


def test_store_to_url_obstore(s3_zarr_store):
    url, options = store_to_url(s3_zarr_store)
    assert url == "s3://test-topozarr"
    assert options["endpoint"].startswith("http://127.0.0.1:")
    assert "bucket" not in options


def test_store_to_url_rejects_memory_store():
    import zarr

    with pytest.raises(TypeError, match="does not support store type"):
        store_to_url(zarr.storage.MemoryStore())
