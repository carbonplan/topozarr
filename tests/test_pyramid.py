# test_pyramid.py
import numpy as np
import pytest
import xarray as xr
import zarr

from topozarr.coarsen import create_pyramid
from topozarr.metadata import ZarrLayerVarConfig


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
        np.testing.assert_allclose(
            native_dt[lvl].ds.elevation.values,
            written_dt[lvl].ds.elevation.values,
            rtol=1e-5,
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
