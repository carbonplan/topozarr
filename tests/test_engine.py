"""Kernel and driver parity tests against xarray.coarsen(boundary="trim")."""

import numpy as np
import pytest
import xarray as xr
import zarr
from topozarr_core import block_reduce

from topozarr.coarsen import create_pyramid
from topozarr.engine import copy_array, copy_region_shape, downsample_level

METHODS = ["mean", "max", "min", "sum"]


def _xr_reference(a: np.ndarray, stride: tuple[int, ...], method: str) -> np.ndarray:
    dims = [f"d{i}" for i in range(a.ndim)]
    windows = {d: s for d, s in zip(dims, stride) if s > 1}
    da = xr.DataArray(a, dims=dims)
    return getattr(da.coarsen(windows, boundary="trim"), method)().values


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "shape", [(8, 8), (101, 100), (7, 5), (3,), (2, 16, 17), (2, 3, 8, 8)]
)
def test_block_reduce_matches_xarray_float(method, shape):
    rng = np.random.default_rng(42)
    a = rng.random(shape).astype("f8")
    stride = tuple(2 if i >= len(shape) - 2 else 1 for i in range(len(shape)))
    got = block_reduce(a, stride, method)
    want = _xr_reference(a, stride, method)
    np.testing.assert_allclose(got, want)
    assert got.dtype == a.dtype


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_block_reduce_skipna_matches_xarray(method, dtype):
    rng = np.random.default_rng(0)
    a = rng.random((33, 34)).astype(dtype)
    a[2:6, 3:9] = np.nan  # interior hole
    a[10:12, 10:12] = np.nan  # full window hole
    a[:, -1] = np.nan  # coastline edge
    got = block_reduce(a, (2, 2), method)
    want = _xr_reference(a, (2, 2), method).astype(dtype)
    np.testing.assert_allclose(got, want, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("method", ["max", "min"])
@pytest.mark.parametrize("dtype", ["u1", "u2", "i2", "i4", "i8"])
def test_block_reduce_int_exact(method, dtype):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(7)
    a = rng.integers(0, 100, (15, 22)).astype(dtype)
    got = block_reduce(a, (2, 2), method)
    want = _xr_reference(a, (2, 2), method)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == dtype


@pytest.mark.parametrize("dtype", ["u1", "u2", "i2", "i4", "i8"])
def test_block_reduce_int_sum_exact(dtype):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(11)
    a = rng.integers(0, 10, (15, 22)).astype(dtype)
    got = block_reduce(a, (2, 2), "sum")
    want = _xr_reference(a, (2, 2), "sum")
    np.testing.assert_array_equal(got, want)
    assert got.dtype == dtype


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4"])
def test_block_reduce_int_mean_truncates_toward_zero(dtype):
    dtype = np.dtype(dtype)
    # left window: 1,2,3,4 -> mean 2.5 -> truncates to 2
    # right window (signed only): -1,-1,-1,-2 -> mean -1.25 -> truncates
    # toward zero to -1 (flooring would give -2)
    if dtype.kind == "u":
        a = np.array([[1, 2, 5, 7], [3, 4, 5, 6]], dtype=dtype)
        want = np.array([[2, 5]], dtype=dtype)
    else:
        a = np.array([[1, 2, -1, -1], [3, 4, -1, -2]], dtype=dtype)
        want = np.array([[2, -1]], dtype=dtype)
    got = block_reduce(a, (2, 2), "mean")
    np.testing.assert_array_equal(got, want)
    assert got.dtype == dtype


def test_block_reduce_fill_value_int():
    a = np.array([[1, 3, -9, -9], [5, 7, -9, -9]], dtype="i2")
    got = block_reduce(a, (2, 2), "mean", fill_value=-9)
    # valid window averages 1,3,5,7 = 4; all-missing window returns fill
    np.testing.assert_array_equal(got, np.array([[4, -9]], dtype="i2"))

    got_max = block_reduce(a, (2, 2), "max", fill_value=-9)
    np.testing.assert_array_equal(got_max, np.array([[7, -9]], dtype="i2"))


def test_block_reduce_skipna_false_propagates_nan():
    a = np.ones((4, 4), dtype="f8")
    a[0, 0] = np.nan
    got = block_reduce(a, (2, 2), "mean", skipna=False)
    assert np.isnan(got[0, 0])
    assert got[1, 1] == 1.0


def test_block_reduce_smaller_than_stride():
    # axis smaller than stride collapses to size 1, matching max(n // s, 1)
    a = np.arange(6, dtype="f8").reshape(1, 6)
    got = block_reduce(a, (2, 2), "mean")
    assert got.shape == (1, 3)
    np.testing.assert_allclose(got[0], [0.5, 2.5, 4.5])


def test_block_reduce_validation():
    a = np.zeros((4, 4), dtype="f4")
    with pytest.raises(ValueError, match="method"):
        block_reduce(a, (2, 2), "median")
    with pytest.raises(ValueError, match="stride"):
        block_reduce(a, (2,), "mean")
    with pytest.raises(TypeError, match="dtype"):
        block_reduce(a.astype("c8"), (2, 2), "mean")


@pytest.mark.parametrize("shards", [None, (8, 8)])
def test_downsample_level_zarr_roundtrip(shards):
    rng = np.random.default_rng(3)
    data = rng.random((33, 31)).astype("f4")
    data[5:9, 5:9] = np.nan

    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    src = group.create_array("src", shape=data.shape, dtype=data.dtype, chunks=(8, 8))
    src[:] = data
    dst = group.create_array(
        "dst", shape=(16, 15), dtype=data.dtype, chunks=(4, 4), shards=shards
    )

    downsample_level(src, dst, stride=(2, 2), method="mean")
    want = _xr_reference(data, (2, 2), "mean").astype("f4")
    np.testing.assert_allclose(dst[:], want, rtol=1e-6, equal_nan=True)


def test_copy_array():
    rng = np.random.default_rng(4)
    data = rng.random((20, 17)).astype("f4")
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    dst = group.create_array("a", shape=data.shape, dtype=data.dtype, chunks=(6, 6))
    copy_array(data, dst)
    np.testing.assert_array_equal(dst[:], data)


class _RegionRecorder:
    """Indexable shim that records requested regions and forbids full reads."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.shape = data.shape
        self.regions: list[tuple[slice, ...]] = []

    def __getitem__(self, region):
        self.regions.append(region)
        return self.data[region]

    def __array__(self, *args, **kwargs):
        raise AssertionError("full-array materialization")


def _make_dst(group, shape, chunks, shards=None, dtype="f4"):
    return group.create_array(
        "dst", shape=shape, dtype=dtype, chunks=chunks, shards=shards
    )


def test_default_max_workers(monkeypatch):
    from topozarr import engine

    class FakeMem:
        available = 2 * 5 * 1024**2 * 3  # budget of 3 x 5MB-footprint regions

    monkeypatch.setattr(engine.psutil, "virtual_memory", lambda: FakeMem)
    monkeypatch.setattr(engine.os, "cpu_count", lambda: 4)

    # memory-bound: 3 regions fit the budget
    assert engine.default_max_workers(1024**2) == 3
    # cpu-bound: tiny regions cap at cpu_count * 2
    assert engine.default_max_workers(1) == 8
    # never below 1, even when a region exceeds the budget
    assert engine.default_max_workers(2**40) == 1


def test_copy_region_shape():
    shard, shape, itemsize = (8, 8), (64, 64), 4

    # no source chunk info → plain shard grid
    assert copy_region_shape(shard, shape, itemsize, None, 2**30) == (8, 8)
    # source chunks a multiple of the shard → region widens to source chunks
    assert copy_region_shape(shard, shape, itemsize, (16, 16), 2**30) == (16, 16)
    # misaligned grids → lcm, clipped at array bounds
    assert copy_region_shape(shard, shape, itemsize, (12, 12), 2**30) == (24, 24)
    assert copy_region_shape(shard, shape, itemsize, (48, 96), 2**30) == (48, 64)
    # lcm region over budget → fall back to shard grid
    assert copy_region_shape(shard, shape, itemsize, (16, 16), 4) == (8, 8)


def test_copy_array_streams_regions():
    rng = np.random.default_rng(5)
    data = rng.random((32, 32)).astype("f4")
    src = _RegionRecorder(data)

    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    dst = _make_dst(group, data.shape, chunks=(4, 4), shards=(8, 8))
    copy_array(src, dst, source_chunks=(16, 16))

    np.testing.assert_array_equal(dst[:], data)
    # regions widened to the 16x16 source chunk grid; each chunk read once
    assert len(src.regions) == 4
    starts = {(r[0].start, r[1].start) for r in src.regions}
    assert starts == {(0, 0), (0, 16), (16, 0), (16, 16)}


class _CountingStore(zarr.storage.MemoryStore):
    """MemoryStore that records set/delete keys."""

    def __init__(self) -> None:
        super().__init__()
        self.set_keys: list[str] = []
        self.delete_keys: list[str] = []

    async def set(self, key, value):
        self.set_keys.append(key)
        await super().set(key, value)

    async def delete(self, key):
        self.delete_keys.append(key)
        await super().delete(key)


def test_copy_array_skips_all_fill_regions():
    from topozarr.engine import RegionTimer

    store = _CountingStore()
    group = zarr.open_group(store, mode="w")
    data = np.full((16, 16), np.nan, dtype="f4")
    data[:8, :8] = 1.0
    dst = group.create_array(
        "a",
        shape=data.shape,
        dtype="f4",
        chunks=(4, 4),
        shards=(8, 8),
        fill_value=np.nan,
    )

    timer = RegionTimer()
    copy_array(data, dst, timer=timer)

    # only the non-fill shard is written; no deletes for the all-NaN shards
    assert [k for k in store.set_keys if "/c/" in k] == ["a/c/0/0"]
    assert store.delete_keys == []
    assert timer.skipped == 3
    np.testing.assert_array_equal(dst[:], data)


def test_copy_array_skip_empty_disabled_writes_all_regions():
    data = np.full((16, 16), np.nan, dtype="f4")
    data[:8, :8] = 1.0
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    dst = group.create_array(
        "a",
        shape=data.shape,
        dtype="f4",
        chunks=(4, 4),
        shards=(8, 8),
        fill_value=np.nan,
    )
    copy_array(data, dst, skip_empty=False)
    np.testing.assert_array_equal(dst[:], data)


def test_copy_array_lazy_zarr_source():
    rng = np.random.default_rng(6)
    data = rng.random((30, 20)).astype("f4")

    src_store = zarr.storage.MemoryStore()
    xr.Dataset({"v": (("y", "x"), data)}).to_zarr(
        src_store, consolidated=False, encoding={"v": {"chunks": (10, 10)}}
    )
    lazy = xr.open_dataset(src_store, engine="zarr", chunks=None, consolidated=False)
    var = lazy["v"].variable
    assert lazy["v"].encoding["chunks"] == (10, 10)

    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    dst = _make_dst(group, data.shape, chunks=(5, 5))
    copy_array(var, dst, source_chunks=(10, 10))
    np.testing.assert_array_equal(dst[:], data)


@pytest.mark.parametrize("method", METHODS)
def test_pyramid_parity_with_xarray(create_dataset, method):
    """Full-pipeline parity: written levels match coarsen(boundary='trim')."""
    ds = create_dataset(nx=37, ny=41)
    ds["elevation"].values[3:7, 3:7] = np.nan
    ds["elevation"].values[:, 0] = np.nan

    pyramid = create_pyramid(ds, levels=3, method=method)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)

    ref = ds
    for lvl in range(3):
        got = dt[str(lvl)].ds
        np.testing.assert_allclose(
            got.elevation.values,
            ref.elevation.values.astype("f4"),
            rtol=1e-5,
            equal_nan=True,
        )
        np.testing.assert_allclose(got.x.values, ref.x.values)
        np.testing.assert_allclose(got.y.values, ref.y.values)
        ref = getattr(ref.coarsen(x=2, y=2, boundary="trim"), method)()


def test_pyramid_extra_dim(create_dataset):
    ds = create_dataset(nx=16, ny=16).expand_dims(time=3).copy(deep=True)
    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)

    assert dt["1"].ds.elevation.shape == (3, 8, 8)
    want = ds.coarsen(x=2, y=2, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values, want.elevation.values, rtol=1e-6
    )


def test_pyramid_write_local_store(create_dataset, tmp_path):
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)
    pyramid.write(tmp_path / "pyramid.zarr")

    dt = xr.open_datatree(tmp_path / "pyramid.zarr", engine="zarr", consolidated=False)
    assert set(dt.children) == {"0", "1"}
    np.testing.assert_array_equal(dt["0"].ds.elevation.values, ds.elevation.values)


def test_pyramid_write_s3_obstore(create_dataset, s3_zarr_store):
    pytest.importorskip("moto")
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)
    pyramid.write(s3_zarr_store)

    dt = xr.open_datatree(s3_zarr_store, engine="zarr", consolidated=False)
    assert set(dt.children) == {"0", "1"}
    np.testing.assert_array_equal(dt["0"].ds.elevation.values, ds.elevation.values)
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values,
        ds.coarsen(x=2, y=2, boundary="trim").mean().elevation.values,
        rtol=1e-6,
    )


def test_pyramid_write_icechunk(create_dataset, tmp_path):
    icechunk = pytest.importorskip("icechunk")
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)

    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    session = repo.writable_session("main")
    pyramid.write(session.store)
    session.commit("write pyramid")

    dt = xr.open_datatree(
        repo.readonly_session("main").store, engine="zarr", consolidated=False
    )
    assert set(dt.children) == {"0", "1"}
    np.testing.assert_array_equal(dt["0"].ds.elevation.values, ds.elevation.values)
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values,
        ds.coarsen(x=2, y=2, boundary="trim").mean().elevation.values,
        rtol=1e-6,
    )
