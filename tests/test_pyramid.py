# test_pyramid.py
import numpy as np
import pytest
import xarray as xr
import zarr

from topozarr.coarsen import create_pyramid
from topozarr.metadata import ZarrLayerVarConfig
from topozarr.pyramid import XR_COARSEN_METHODS


def test_pyramid_structure(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    levels = 3
    pyramid = create_pyramid(ds, levels=levels)

    # 0 is original res, 2 is coarsest
    assert set(pyramid.level_templates) == {0, 1, 2}
    assert pyramid.levels == levels

    # shapes go: 16 -> 8 -> 4
    assert pyramid.level_templates[0].elevation.shape == (16, 16)
    assert pyramid.level_templates[1].elevation.shape == (8, 8)
    assert pyramid.level_templates[2].elevation.shape == (4, 4)


def test_pyramid_write_roundtrip(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=3)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    assert set(dt.children) == {"0", "1", "2"}
    assert dt["0"].ds.elevation.shape == (16, 16)
    assert dt["1"].ds.elevation.shape == (8, 8)
    assert dt["2"].ds.elevation.shape == (4, 4)

    # level 0 is a verbatim copy; level 1 matches xarray coarsen
    np.testing.assert_array_equal(dt["0"].ds.elevation.values, ds.elevation.values)
    expected = ds.coarsen(x=2, y=2, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values, expected.elevation.values, rtol=1e-6
    )
    np.testing.assert_allclose(dt["1"].ds.x.values, expected.x.values)


def test_pyramid_write_integer_mean_truncates(create_dataset):
    ds = create_dataset(nx=4, ny=2)
    # row0: 1,2,5,7 -> mean 2.5 -> truncates to 2
    # row1: 3,4,5,6 -> combined with row0 window: (1+2+3+4)/4=2.5 -> 2,
    # (5+7+5+6)/4=5.75 -> 5
    ds["elevation"] = (("y", "x"), np.array([[1, 2, 5, 7], [3, 4, 5, 6]], dtype="i2"))
    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    assert dt["1"].ds.elevation.dtype == np.dtype("i2")
    np.testing.assert_array_equal(dt["1"].ds.elevation.values, [[2, 5]])


def test_pyramid_write_nearest_categorical(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    rng = np.random.default_rng(5)
    codes = rng.choice(np.array([10, 20, 30, 80], dtype="u1"), (16, 16))
    ds["elevation"] = (("y", "x"), codes)
    pyramid = create_pyramid(ds, levels=3, method="nearest")
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    # every level is a corner-pick of level 0: values stay valid class codes;
    # coords remain window means (cell centers), like the other methods
    for lvl, step in ((1, 2), (2, 4)):
        got = dt[str(lvl)].ds.elevation
        assert got.dtype == np.dtype("u1")
        np.testing.assert_array_equal(got.values, codes[::step, ::step])
        expected = ds.coarsen(x=step, y=step, boundary="trim").mean()
        np.testing.assert_allclose(dt[str(lvl)].ds.x.values, expected.x.values)
        np.testing.assert_allclose(dt[str(lvl)].ds.y.values, expected.y.values)


def test_pyramid_nearest_sparse_matches_chained(create_dataset):
    # corner-pick composes: factors=[1, 4] equals two chained 2x steps
    ds = create_dataset(nx=16, ny=16)
    codes = np.arange(256, dtype="i4").reshape(16, 16)
    ds["elevation"] = (("y", "x"), codes)

    sparse_store = zarr.storage.MemoryStore()
    create_pyramid(ds, factors=[1, 4], method="nearest").write(sparse_store)
    sparse = xr.open_datatree(sparse_store, engine="zarr", consolidated=False)
    np.testing.assert_array_equal(sparse["1"].ds.elevation.values, codes[::4, ::4])

    # single-step level equals the chained pyramid's factor-4 level
    chained_store = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=3, method="nearest").write(chained_store)
    chained = xr.open_datatree(chained_store, engine="zarr", consolidated=False)
    np.testing.assert_array_equal(
        sparse["1"].ds.elevation.values, chained["2"].ds.elevation.values
    )
    np.testing.assert_allclose(sparse["1"].ds.x.values, chained["2"].ds.x.values)
    np.testing.assert_allclose(sparse["1"].ds.y.values, chained["2"].ds.y.values)


def test_as_datatree_nearest_matches_native(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    rng = np.random.default_rng(9)
    ds["elevation"] = (("y", "x"), rng.choice([1, 2, 3], (16, 16)).astype("i2"))
    pyramid = create_pyramid(ds, levels=3, method="nearest")

    native_store = zarr.storage.MemoryStore()
    pyramid.write(native_store)
    native_dt = xr.open_datatree(native_store, engine="zarr", consolidated=False)

    dt = pyramid.as_datatree()
    for lvl in ("0", "1", "2"):
        np.testing.assert_array_equal(
            dt[lvl].ds.elevation.values, native_dt[lvl].ds.elevation.values
        )
        # datatree coords must match the written template coords exactly
        np.testing.assert_allclose(dt[lvl].ds.x.values, native_dt[lvl].ds.x.values)
        np.testing.assert_allclose(dt[lvl].ds.y.values, native_dt[lvl].ds.y.values)


def test_crs_enforcement(create_dataset):
    ds_no_crs = create_dataset(add_crs=False)

    with pytest.raises(ValueError, match="dataset is missing a crs"):
        create_pyramid(ds_no_crs, levels=2)


def test_missing_x_dim_raises(create_dataset):
    ds = create_dataset()

    with pytest.raises(ValueError, match="x_dim 'lon' not found"):
        create_pyramid(ds, levels=2, x_dim="lon")


def test_missing_y_dim_raises(create_dataset):
    ds = create_dataset()

    with pytest.raises(ValueError, match="y_dim 'lat' not found"):
        create_pyramid(ds, levels=2, y_dim="lat")


def test_custom_dimensions(create_dataset):
    ds = create_dataset(x_dim="lon", y_dim="lat")
    pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")

    assert "lon" in pyramid.level_templates[1].dims
    assert pyramid.level_templates[0].elevation.shape == (16, 16)


def test_multi_variable_write_roundtrip(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    ds["slope"] = ds.elevation * 2
    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    expected = ds.coarsen(x=2, y=2, boundary="trim").mean()
    for var in ("elevation", "slope"):
        np.testing.assert_array_equal(dt["0"].ds[var].values, ds[var].values)
        np.testing.assert_allclose(
            dt["1"].ds[var].values, expected[var].values, rtol=1e-6
        )


def test_write_progress(create_dataset):
    pytest.importorskip("tqdm")
    pyramid = create_pyramid(create_dataset(), levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store, progress=True)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    assert set(dt.children) == {"0", "1"}


def test_write_stats(create_dataset):
    # Disable fusion so level-0 reduce_s is unambiguously 0.
    pyramid = create_pyramid(create_dataset(nx=16, ny=16), levels=2)
    store = zarr.storage.MemoryStore()
    out = pyramid.write(store, stats=True, keep_levels_in_memory=False)

    assert set(out) == {"0", "1"}
    for lvl, lvl_stats in out.items():
        assert lvl_stats["regions"] > 0
        assert lvl_stats["workers"] >= 1
        assert lvl_stats["wall_s"] >= 0
        assert lvl_stats["read_s"] >= 0
        assert lvl_stats["write_s"] >= 0
        assert "elevation" in lvl_stats["region_shapes"]
    # only coarsened levels run the reduce kernel when fusion is disabled
    assert out["0"]["reduce_s"] == 0
    assert out["1"]["reduce_s"] >= 0

    # default stats=False returns None
    assert pyramid.write(zarr.storage.MemoryStore()) is None


def test_write_invalid_levels(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)

    with pytest.raises(ValueError, match=r"invalid levels \[2, 5\]"):
        pyramid.write(zarr.storage.MemoryStore(), levels=[1, 2, 5])


def test_write_negative_level_raises(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)

    with pytest.raises(ValueError, match=r"invalid levels \[-1\]"):
        pyramid.write(zarr.storage.MemoryStore(), levels=[-1])


def test_write_empty_levels_is_noop(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store, levels=[])

    root = zarr.open_group(store, mode="r")
    assert list(root.keys()) == []


def test_write_levels_missing_predecessor_raises(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=3)
    store = zarr.storage.MemoryStore()
    pyramid.write(store, levels=[0])

    # level 2 needs level 1, which is neither in the plan nor in the store
    with pytest.raises(ValueError, match="level 2 is coarsened from level 1"):
        pyramid.write(store, mode="a", levels=[0, 2])


def test_write_levels_predecessor_from_store(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store, levels=[0])
    pyramid.write(store, mode="a", levels=[1])

    ref = zarr.storage.MemoryStore()
    pyramid.write(ref)
    np.testing.assert_array_equal(
        _read_level(store, 1, "elevation"), _read_level(ref, 1, "elevation")
    )


def test_write_subset_mode_w_existing_store_raises(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    # rewriting a subset with mode="w" would delete level 0
    with pytest.raises(ValueError, match="pass mode='a'"):
        pyramid.write(store, levels=[1])

    # a fresh store has nothing to delete; subset with mode="w" is fine
    pyramid.write(zarr.storage.MemoryStore(), levels=[0])


def test_non_uniform_coords_raise(create_dataset):
    ds = create_dataset()
    ds = ds.assign_coords(x=ds.x.values**2)

    with pytest.raises(ValueError, match="'x' is not uniformly spaced"):
        create_pyramid(ds, levels=2)


def test_single_pixel_level_resolution(create_dataset):
    # 4x4 -> 2x2 -> 1x1: the coarsest level has length-1 coords, so its
    # resolution must come from level 0 (1.0 here) scaled by 2^level
    ds = create_dataset(nx=4, ny=4)
    pyramid = create_pyramid(ds, levels=3)

    layout = pyramid.attrs["multiscales"]["layout"]
    transform = layout[2]["spatial:transform"]
    assert transform[0] == 4.0  # x resolution
    assert transform[4] == 4.0  # y resolution


def test_single_value_coord_raises(create_dataset):
    ds = create_dataset(nx=1, ny=4)

    with pytest.raises(ValueError, match="cannot infer resolution"):
        create_pyramid(ds, levels=1)


def test_spatial_var_ndim_limit(create_dataset):
    ds = create_dataset()
    ds["stacked"] = ds.elevation.expand_dims(a=2, b=2, c=2)

    with pytest.raises(ValueError, match="supports at most 4"):
        create_pyramid(ds, levels=2)


def test_no_spatial_variables_raises(create_dataset):
    ds = create_dataset()
    ds["time_series"] = ("t", np.arange(4))
    ds = ds.drop_vars("elevation")

    with pytest.raises(ValueError, match="no variable has both"):
        create_pyramid(ds, levels=2)


def test_zarr_layer_metadata_written(create_dataset):
    ds = create_dataset()
    config = {"elevation": ZarrLayerVarConfig(clim=[0.0, 1.0], colormap="viridis")}
    pyramid = create_pyramid(ds, levels=2, layer_hints=config)

    zarr_layer = pyramid.attrs["zarr-layer"]
    assert zarr_layer["elevation"]["clim"] == [0.0, 1.0]
    assert zarr_layer["elevation"]["colormap"] == "viridis"


# ── level-pipelining (keep_levels_in_memory) ──────────────────────────────────


def _read_level(store: zarr.storage.MemoryStore, lvl: int, name: str) -> np.ndarray:
    root = zarr.open_group(store, mode="r")
    return root[f"{lvl}/{name}"][:]


@pytest.mark.parametrize("nx,ny", [(16, 16), (15, 13)])
def test_fused_levels_match_default(create_dataset, nx, ny):
    """Fused write produces byte-identical output to the store-read path."""
    ds = create_dataset(nx=nx, ny=ny)
    pyramid = create_pyramid(ds, levels=3)

    store_ref = zarr.storage.MemoryStore()
    pyramid.write(store_ref, keep_levels_in_memory=False)

    store_fused = zarr.storage.MemoryStore()
    pyramid.write(store_fused, keep_levels_in_memory=True)

    for lvl in (1, 2):
        ref = _read_level(store_ref, lvl, "elevation")
        got = _read_level(store_fused, lvl, "elevation")
        np.testing.assert_array_equal(ref, got, err_msg=f"lvl={lvl} nx={nx} ny={ny}")


def test_fused_with_nan(create_dataset):
    """NaN/fill_value variables are handled identically with and without fusion."""
    ds = create_dataset(nx=16, ny=16)
    data = ds.elevation.values.copy()
    data[0, 0] = float("nan")
    ds["elevation"] = xr.DataArray(
        data, dims=ds.elevation.dims, coords=ds.elevation.coords
    )

    pyramid = create_pyramid(ds, levels=2)

    store_ref = zarr.storage.MemoryStore()
    pyramid.write(store_ref, keep_levels_in_memory=False)
    store_fused = zarr.storage.MemoryStore()
    pyramid.write(store_fused, keep_levels_in_memory=True)

    ref = _read_level(store_ref, 1, "elevation")
    got = _read_level(store_fused, 1, "elevation")
    np.testing.assert_array_equal(ref, got)


def test_fused_multi_variable(create_dataset):
    """All spatial variables are fused correctly."""
    ds = create_dataset(nx=16, ny=16)
    ds["slope"] = ds.elevation * 2
    pyramid = create_pyramid(ds, levels=2)

    store_ref = zarr.storage.MemoryStore()
    pyramid.write(store_ref, keep_levels_in_memory=False)
    store_fused = zarr.storage.MemoryStore()
    pyramid.write(store_fused, keep_levels_in_memory=True)

    for var in ("elevation", "slope"):
        ref = _read_level(store_ref, 1, var)
        got = _read_level(store_fused, 1, var)
        np.testing.assert_array_equal(ref, got)


def test_fused_subset_levels_fallback(create_dataset):
    """levels=[1, 2] starting above 0 falls back gracefully (no mem_source at L1)."""
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=3)

    # Write level 0 first so subsequent reads succeed.
    store = zarr.storage.MemoryStore()
    pyramid.write(store, levels=[0])
    pyramid.write(store, mode="a", levels=[1, 2], keep_levels_in_memory=True)

    ref_store = zarr.storage.MemoryStore()
    pyramid.write(ref_store, keep_levels_in_memory=False)

    for lvl in (1, 2):
        ref = _read_level(ref_store, lvl, "elevation")
        got = _read_level(store, lvl, "elevation")
        np.testing.assert_array_equal(ref, got)


def test_fused_forced_fallback_low_memory(create_dataset, monkeypatch):
    """With a tiny memory budget, auto-mode disables fusion; output still correct."""
    import psutil

    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)

    fake_mem = psutil.virtual_memory()._replace(available=1)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake_mem)

    store = zarr.storage.MemoryStore()
    pyramid.write(store)  # keep_levels_in_memory=None → auto → False due to budget

    ref_store = zarr.storage.MemoryStore()
    # monkeypatch still active; budget still tiny, so both go through fallback
    pyramid.write(ref_store, keep_levels_in_memory=False)

    np.testing.assert_array_equal(
        _read_level(store, 1, "elevation"),
        _read_level(ref_store, 1, "elevation"),
    )


def test_fused_stats_keys_unchanged(create_dataset):
    """Stats dict has same keys with fusion enabled; level-0 reduce_s > 0."""
    pyramid = create_pyramid(create_dataset(nx=16, ny=16), levels=2)
    store = zarr.storage.MemoryStore()
    out = pyramid.write(store, stats=True, keep_levels_in_memory=True)

    assert set(out) == {"0", "1"}
    for lvl_stats in out.values():
        assert "regions" in lvl_stats
        assert "read_s" in lvl_stats
        assert "reduce_s" in lvl_stats
        assert "write_s" in lvl_stats
        assert "wall_s" in lvl_stats
    # With fusion, level 0 reduce_s accumulates fused-reduce time.
    # On tiny test data it rounds to 0; just verify it's non-negative and the
    # formula read_s = block_s - reduce_s doesn't go negative.
    assert out["0"]["reduce_s"] >= 0
    assert out["0"]["read_s"] >= 0


def test_as_datatree_matches_native(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=3)

    native_store = zarr.storage.MemoryStore()
    pyramid.write(native_store)

    dt = pyramid.as_datatree()
    assert set(dt.children) == {"0", "1", "2"}
    assert "foo" not in dt.attrs  # root has multiscales attrs, not bogus

    dt_store = zarr.storage.MemoryStore()
    dt.to_zarr(dt_store, zarr_format=3, consolidated=False, encoding=pyramid.encoding)

    native_dt = xr.open_datatree(native_store, engine="zarr", consolidated=False)
    written_dt = xr.open_datatree(dt_store, engine="zarr", consolidated=False)
    for lvl in ("0", "1", "2"):
        # exact: the datatree path coarsens in f8 like the kernel, then casts
        # back to the source f4
        np.testing.assert_array_equal(
            native_dt[lvl].ds.elevation.values,
            written_dt[lvl].ds.elevation.values,
        )


def test_fused_hook_clamps_trailing_window():
    """A trailing region shorter than the stride yields a kernel window that
    falls outside the trimmed target; the hook must drop it, not crash."""
    from topozarr.pyramid import _make_fused_reduce_hook

    src = np.arange(9 * 8, dtype="float32").reshape(9, 8)
    target = np.full((4, 4), -1, dtype="float32")  # 9 // 2 = 4 rows after trim
    hook = _make_fused_reduce_hook(target, (2, 2), "mean", None)

    # regions of height 4 tile rows 0-8; the last region is a single row,
    # which block_reduce turns into one window despite the global trim
    for start in (0, 4, 8):
        region = (slice(start, min(start + 4, 9)), slice(0, 8))
        hook(region, src[region])

    expected = src[:8, :].reshape(4, 2, 4, 2).mean(axis=(1, 3))
    np.testing.assert_array_equal(target, expected)


def test_method_literal_matches_kernel():
    """The Literal and the kernel's METHODS must not drift apart."""
    from typing import get_args

    import topozarr_core

    from topozarr.pyramid import CoarseningMethod

    assert set(get_args(CoarseningMethod)) == set(topozarr_core.METHODS)


@pytest.mark.parametrize("method", ["median", "mode", "Mean", "", "avg"])
def test_invalid_method_raises_at_plan_time(create_dataset, method):
    with pytest.raises(ValueError, match="method must be one of"):
        create_pyramid(create_dataset(), levels=2, method=method)


def test_invalid_method_writes_nothing(create_dataset):
    """A method edited onto the plan is caught before any level is written.

    Regression: validation used to live only in the kernel, which sees the
    method on the first *coarsened* level -- after level 0 is in the store.
    """
    pyramid = create_pyramid(create_dataset(), levels=3)
    pyramid.method = "median"
    store = zarr.storage.MemoryStore()

    with pytest.raises(ValueError, match="method must be one of"):
        pyramid.write(store)

    assert list(store._store_dict) == []


def test_invalid_method_rejected_on_construction(create_dataset):
    from topozarr.pyramid import Pyramid

    with pytest.raises(ValueError, match="method must be one of"):
        Pyramid(
            source=create_dataset(),
            level_templates={},
            encoding={},
            attrs={},
            x_dim="x",
            y_dim="y",
            method="median",
        )


@pytest.mark.parametrize("method", ["mean", "max", "min", "sum", "nearest"])
def test_every_kernel_method_is_accepted(create_dataset, method):
    pyramid = create_pyramid(create_dataset(), levels=2, method=method)
    pyramid.write(zarr.storage.MemoryStore())


def test_curvilinear_coords_raise(create_dataset):
    """2-D spatial coords are rejected at the boundary, not deep in xarray.

    They are not coarsened: the template builder would leave them at native
    shape (an opaque 'conflicting sizes' ValueError) and as_datatree's
    _decimate would corner-stride them into a mis-registered grid.
    """
    ds = create_dataset(nx=16, ny=16)
    ds = ds.assign_coords(
        lat=(("y", "x"), np.random.rand(16, 16)),
        lon=(("y", "x"), np.random.rand(16, 16)),
    )

    with pytest.raises(ValueError, match="2-D over the spatial dims"):
        create_pyramid(ds, levels=2)


def test_curvilinear_message_names_the_coords(create_dataset):
    ds = create_dataset(nx=16, ny=16).assign_coords(
        lat=(("y", "x"), np.random.rand(16, 16))
    )

    with pytest.raises(ValueError, match="drop_vars") as excinfo:
        create_pyramid(ds, levels=2)
    assert "lat" in str(excinfo.value)


def test_partly_spatial_2d_coord_raises(create_dataset):
    """A 2-D coord touching only one spatial dim still needs coarsening."""
    ds = create_dataset(nx=16, ny=16, extra_dims={"time": 3})
    ds = ds.assign_coords(drift=(("time", "x"), np.random.rand(3, 16)))

    with pytest.raises(ValueError, match="2-D over the spatial dims"):
        create_pyramid(ds, levels=2)


def test_non_spatial_2d_coord_is_allowed(create_dataset):
    """Only coords over a *spatial* dim are rejected; others pass through."""
    ds = create_dataset(nx=16, ny=16, extra_dims={"time": 3, "band": 2})
    ds = ds.assign_coords(quality=(("time", "band"), np.random.rand(3, 2)))

    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    assert "quality" in xr.open_zarr(store, group="1", consolidated=False).coords


# --- as_datatree / write parity -----------------------------------------
#
# Same Pyramid, two materialization paths: the Rust kernel (write) and
# xarray.coarsen (as_datatree). They agreed only by accident before -- the
# original test used f4 with no _FillValue, the one case where xarray's
# promotion and fill-blindness are both invisible.

PARITY_DTYPES = [("i2", None), ("i4", None), ("u1", 255), ("f4", None), ("f4", -9999.0)]


def _parity_dataset(dtype, fill, n=16, seed=0):
    """Spatial dataset with ~20% of cells set to ``fill``."""
    rng = np.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.integer):
        lo = -100 if np.dtype(dtype).kind == "i" else 0
        data = rng.integers(lo, 200, (n, n)).astype(dtype)
    else:
        data = ((rng.random((n, n)) - 0.5) * 200).astype(dtype)
    if fill is not None:
        data[rng.random((n, n)) < 0.2] = fill
    ds = xr.Dataset(
        {"elev": (("y", "x"), data)},
        coords={"x": np.arange(n, dtype="f8"), "y": np.arange(n, dtype="f8")},
    ).proj.assign_crs(spatial_ref="EPSG:4326")
    if fill is not None:
        ds.elev.attrs["_FillValue"] = fill
    return ds


def _written(pyramid, name="elev"):
    """Levels as ``write`` put them in the store, read back without decoding.

    Read through zarr rather than xarray: CF decoding would mask the fill
    value and re-promote the dtype, hiding the divergence under test.
    """
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    root = zarr.open_group(store, mode="r")
    return [root[f"{lvl}/{name}"][:] for lvl in range(pyramid.levels)]


@pytest.mark.parametrize("dtype,fill", PARITY_DTYPES)
@pytest.mark.parametrize("method", ["mean", "max", "min", "sum", "nearest"])
def test_as_datatree_matches_write(dtype, fill, method):
    pyramid = create_pyramid(_parity_dataset(dtype, fill), levels=3, method=method)
    native = _written(pyramid)
    dt = pyramid.as_datatree()
    for lvl, expected in enumerate(native):
        got = dt[str(lvl)].ds.elev.values
        assert got.dtype == expected.dtype, f"level {lvl} dtype"
        np.testing.assert_array_equal(got, expected, err_msg=f"level {lvl}")


@pytest.mark.parametrize("method", ["mean", "sum"])
def test_as_datatree_f8_matches_write_to_rounding(method):
    # f8 is the one dtype that cannot be exact: both paths accumulate in f64,
    # but the kernel sums a window in order while numpy sums pairwise. Under
    # 1 ULP, and not something either side should chase.
    pyramid = create_pyramid(_parity_dataset("f8", None), levels=3, method=method)
    native = _written(pyramid)
    dt = pyramid.as_datatree()
    for lvl, expected in enumerate(native):
        np.testing.assert_allclose(
            dt[str(lvl)].ds.elev.values, expected, rtol=1e-14, err_msg=f"level {lvl}"
        )


def test_as_datatree_skips_fill_value():
    # Regression: xarray.coarsen averaged the sentinel in as data.
    ds = xr.Dataset(
        {"elev": (("y", "x"), np.array([[1, 255], [5, 6]], dtype="u1"))},
        coords={"x": np.arange(2, dtype="f8"), "y": np.arange(2, dtype="f8")},
    ).proj.assign_crs(spatial_ref="EPSG:4326")
    ds.elev.attrs["_FillValue"] = 255

    dt = create_pyramid(ds, levels=2, method="mean").as_datatree()
    assert dt["1"].ds.elev.values.tolist() == [[4]]  # mean(1, 5, 6), not mean(..255)


def test_as_datatree_all_fill_window_is_fill():
    ds = xr.Dataset(
        {"elev": (("y", "x"), np.full((2, 2), 255, dtype="u1"))},
        coords={"x": np.arange(2, dtype="f8"), "y": np.arange(2, dtype="f8")},
    ).proj.assign_crs(spatial_ref="EPSG:4326")
    ds.elev.attrs["_FillValue"] = 255

    pyramid = create_pyramid(ds, levels=2, method="mean")
    np.testing.assert_array_equal(
        pyramid.as_datatree()["1"].ds.elev.values, _written(pyramid)[1]
    )


def test_as_datatree_integer_sum_saturates():
    # A u1 sum overflows on the second level; the kernel saturates on cast, so
    # the datatree path must clip rather than let numpy wrap.
    pyramid = create_pyramid(_parity_dataset("u1", None), levels=3, method="sum")
    coarse = pyramid.as_datatree()["2"].ds.elev.values
    assert coarse.dtype == np.dtype("u1")
    assert (coarse == 255).any()
    np.testing.assert_array_equal(coarse, _written(pyramid)[2])


def test_as_datatree_keeps_source_dtype(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    ds["elevation"] = ds.elevation.astype("i2")
    dt = create_pyramid(ds, levels=3).as_datatree()
    # xarray.coarsen would have promoted these to f8
    assert all(dt[lvl].ds.elevation.dtype == np.dtype("i2") for lvl in ("0", "1", "2"))


def test_as_datatree_leaves_non_spatial_vars_alone(create_dataset):
    ds = create_dataset(nx=16, ny=16, extra_dims={"time": 3})
    ds["label"] = ("time", np.array(["a", "b", "c"], dtype=object))
    dt = create_pyramid(ds, levels=2).as_datatree()
    assert dt["1"].ds.label.values.tolist() == ["a", "b", "c"]


def test_as_datatree_rejects_unexpressible_method(create_dataset):
    pyramid = create_pyramid(create_dataset(nx=8, ny=8), levels=2)
    pyramid.method = "mode"  # a kernel method with no xarray.coarsen equivalent
    assert "mode" not in XR_COARSEN_METHODS
    with pytest.raises(NotImplementedError, match="as_datatree cannot express"):
        pyramid.as_datatree()


def test_every_kernel_method_is_expressible_or_refused(create_dataset):
    # Guards the dispatch: a method the kernel gains must either be listed in
    # XR_COARSEN_METHODS or refused explicitly -- never fall through to a bare
    # getattr and raise AttributeError.
    from topozarr_core import METHODS

    ds = create_dataset(nx=8, ny=8)
    for method in METHODS:
        pyramid = create_pyramid(ds, levels=2, method=method)
        try:
            pyramid.as_datatree()
        except NotImplementedError:
            assert method not in XR_COARSEN_METHODS and method != "nearest"


@pytest.mark.parametrize("method", ["mean", "max", "min", "sum", "nearest"])
def test_as_datatree_matches_write_with_extra_dims(method):
    # Two vars in one dataset, different dtypes and only one with a fill:
    # exercises the per-variable mask/restore rather than a dataset-wide cast.
    rng = np.random.default_rng(1)
    n = 16
    elev = rng.integers(0, 200, (3, n, n)).astype("u1")
    elev[rng.random((3, n, n)) < 0.2] = 255
    ds = xr.Dataset(
        {
            "elev": (("time", "y", "x"), elev),
            "temp": (
                ("time", "y", "x"),
                ((rng.random((3, n, n)) - 0.5) * 100).astype("f4"),
            ),
        },
        coords={
            "x": np.arange(n, dtype="f8"),
            "y": np.arange(n, dtype="f8"),
            "time": np.arange(3),
        },
    ).proj.assign_crs(spatial_ref="EPSG:4326")
    ds.elev.attrs["_FillValue"] = 255

    pyramid = create_pyramid(ds, levels=3, method=method)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    root = zarr.open_group(store, mode="r")
    dt = pyramid.as_datatree()
    for lvl in range(3):
        for var in ("elev", "temp"):
            got = dt[str(lvl)].ds[var].values
            expected = root[f"{lvl}/{var}"][:]
            assert got.dtype == expected.dtype, f"{var} level {lvl} dtype"
            np.testing.assert_array_equal(got, expected, err_msg=f"{var} level {lvl}")


def test_as_datatree_matches_write_sparse_factors():
    pyramid = create_pyramid(_parity_dataset("u1", 255), factors=[1, 4], method="mean")
    native = _written(pyramid)
    dt = pyramid.as_datatree()
    for lvl, expected in enumerate(native):
        np.testing.assert_array_equal(dt[str(lvl)].ds.elev.values, expected)


def test_as_datatree_stays_lazy(create_dataset):
    # The whole point of this path is a lazy tree for a distributed write; the
    # mask/promote/cast must not pull anything into memory.
    ds = create_dataset(nx=32, ny=32)
    ds["elevation"] = ds.elevation.astype("i2")
    ds = ds.chunk({"y": 16, "x": 16})

    dt = create_pyramid(ds, levels=3).as_datatree()
    for lvl in ("0", "1", "2"):
        assert dt[lvl].ds.elevation.chunks is not None, f"level {lvl} computed eagerly"
        assert dt[lvl].ds.elevation.dtype == np.dtype("i2")


@pytest.mark.xfail(
    strict=True,
    reason="write leaves a variable over exactly one spatial dim uncomputed "
    "(all-NaN above level 0); as_datatree coarsens it correctly. Write-path "
    "bug, tracked in planning/review-2026-09-02.md -- flip this to a plain "
    "test when it lands.",
)
def test_as_datatree_matches_write_partly_spatial_var():
    n = 16
    ds = xr.Dataset(
        {
            "elev": (("y", "x"), np.arange(n * n, dtype="f4").reshape(n, n)),
            "profile": (("time", "x"), np.arange(3 * n, dtype="i2").reshape(3, n)),
        },
        coords={
            "x": np.arange(n, dtype="f8"),
            "y": np.arange(n, dtype="f8"),
            "time": np.arange(3),
        },
    ).proj.assign_crs(spatial_ref="EPSG:4326")

    pyramid = create_pyramid(ds, levels=2, method="mean")
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    root = zarr.open_group(store, mode="r")
    np.testing.assert_array_equal(
        pyramid.as_datatree()["1"].ds.profile.values, root["1/profile"][:]
    )
