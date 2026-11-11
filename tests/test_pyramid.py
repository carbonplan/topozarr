# test_pyramid.py
import math

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


def test_descending_y_coords_survive_write(create_dataset, tmp_path):
    ds = create_dataset(nx=32, ny=32, y_descending=True)
    create_pyramid(ds, levels=3).write(tmp_path / "p.zarr")
    dt = xr.open_datatree(tmp_path / "p.zarr", engine="zarr", consolidated=False)
    for lvl in ("0", "1", "2"):
        y = dt[lvl].ds.y.values
        assert (np.diff(y) < 0).all()


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


def test_pyramid_write_integer_mean_rounds(create_dataset):
    ds = create_dataset(nx=4, ny=2)
    # windows: (1+2+3+4)/4=2.5 -> ties to even 2; (5+7+5+6)/4=5.75 -> 6
    ds["elevation"] = (("y", "x"), np.array([[1, 2, 5, 7], [3, 4, 5, 6]], dtype="i2"))
    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    assert dt["1"].ds.elevation.dtype == np.dtype("i2")
    np.testing.assert_array_equal(dt["1"].ds.elevation.values, [[2, 6]])


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
    pyramid = create_pyramid(create_dataset(nx=16, ny=16), levels=3)
    store = zarr.storage.MemoryStore()
    # the tile pass reports one entry, under its deepest level k
    out = pyramid.write(store, stats=True, max_workers=1)
    k = int(max(out))
    assert set(out) == {str(k)}
    for lvl, lvl_stats in out.items():
        assert lvl_stats["regions"] > 0
        assert lvl_stats["workers"] >= 1
        assert lvl_stats["wall_s"] >= 0
        assert lvl_stats["read_s"] >= 0
        assert lvl_stats["write_s"] >= 0
        assert "elevation" in lvl_stats["region_shapes"]
    assert out[str(k)]["reduce_s"] >= 0

    # levels above 0 only: no tile pass, one entry per level
    out = pyramid.write(store, mode="a", levels=[1, 2], stats=True)
    assert set(out) == {"1", "2"}

    # default stats=False returns None
    assert pyramid.write(zarr.storage.MemoryStore()) is None


# small regions force tile_k=0 + per-level passes; defaults fold all levels into the tile pass
@pytest.mark.parametrize(
    ("kw", "tile_k"),
    [({"max_region_bytes": 4096, "max_workers": 2}, 0), ({"max_workers": 1}, 3)],
)
def test_summary_matches_write_stats(create_dataset, kw, tile_k):
    ds = create_dataset(nx=64, ny=64)
    ds["slope"] = ds.elevation * 2
    pyramid = create_pyramid(ds, levels=4, target_chunk_bytes=256, chunks_per_shard=2)
    plan = pyramid.summary(**kw)
    stats = pyramid.write(zarr.storage.MemoryStore(), stats=True, **kw)

    assert plan["tile_k"] == tile_k
    passes = {k: v for k, v in plan["levels"].items() if "workers" in v}
    assert set(passes) == set(stats)
    assert str(plan["tile_k"]) in passes
    for lvl, entry in passes.items():
        assert entry["workers"] == stats[lvl]["workers"]
        assert entry["region_shapes"] == stats[lvl]["region_shapes"]
        assert entry["regions"] == stats[lvl]["regions"]
        assert entry["est_peak_bytes"] > 0

    level0 = plan["levels"]["0"]["arrays"]["elevation"]
    assert level0["shape"] == (64, 64)
    assert level0["nbytes"] == 64 * 64 * 4


def test_summary_objects_unsharded(create_dataset):
    pyramid = create_pyramid(
        create_dataset(nx=32, ny=32),
        levels=2,
        target_chunk_bytes=256,
        chunks_per_shard=None,
    )
    for lvl, entry in pyramid.summary()["levels"].items():
        arr = entry["arrays"]["elevation"]
        assert arr["shards"] is None
        expected = math.prod(
            math.ceil(n / c) for n, c in zip(arr["shape"], arr["chunks"])
        )
        assert arr["objects"] == expected


def test_write_honors_compressors(create_dataset):
    from zarr.codecs import ZstdCodec

    ds = create_dataset(nx=16, ny=16)
    ds["slope"] = ds.elevation * 2
    pyramid = create_pyramid(ds, levels=2)
    for lvl in ("/0", "/1"):
        pyramid.encoding[lvl]["elevation"]["compressors"] = [ZstdCodec(level=5)]
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    root = zarr.open_group(store, mode="r")
    for lvl in ("0", "1"):
        assert root[f"{lvl}/elevation"].compressors == (ZstdCodec(level=5),)
        assert root[f"{lvl}/slope"].compressors != (ZstdCodec(level=5),)


def test_summary_rejects_zero_workers(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)
    with pytest.raises(ValueError, match="max_workers"):
        pyramid.write(zarr.storage.MemoryStore(), max_workers=0)


def test_write_invalid_levels(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)

    with pytest.raises(ValueError, match=r"invalid levels \[2, 5\]"):
        pyramid.write(zarr.storage.MemoryStore(), levels=[1, 2, 5])


def test_write_negative_level_raises(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)

    with pytest.raises(ValueError, match=r"invalid levels \[-1\]"):
        pyramid.write(zarr.storage.MemoryStore(), levels=[-1])


def test_write_empty_levels_raises(create_dataset):
    ds = create_dataset()
    store = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=3).write(store)
    before = zarr.open_group(store, mode="r").attrs.asdict()

    with pytest.raises(ValueError, match="no levels to write"):
        create_pyramid(ds, levels=1).write(store, mode="a", levels=[])

    assert zarr.open_group(store, mode="r").attrs.asdict() == before


@pytest.mark.parametrize(
    "plan_kwargs", [{"levels": 2}, {"levels": 3, "method": "nearest"}]
)
def test_write_append_mismatched_layout_raises(create_dataset, plan_kwargs):
    ds = create_dataset()
    store = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=3, method="mean").write(store)
    before = zarr.open_group(store, mode="r").attrs.asdict()

    with pytest.raises(ValueError, match="does not match this pyramid"):
        create_pyramid(ds, **plan_kwargs).write(store, mode="a", levels=[1])

    assert zarr.open_group(store, mode="r").attrs.asdict() == before


def test_write_append_extends_pyramid(create_dataset):
    ds = create_dataset()
    store = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=2).write(store)
    create_pyramid(ds, levels=3).write(store, mode="a", levels=[2])

    layout = zarr.open_group(store, mode="r").attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == ["0", "1", "2"]


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


def test_write_levels_tags_kept_level_with_crs(create_dataset):
    """A level written outside topozarr (e.g. virtual level 0) gets proj attrs."""
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)
    store = zarr.storage.MemoryStore()
    ds.drop_vars("spatial_ref").to_zarr(
        store, group="0", zarr_format=3, consolidated=False
    )
    zarr.open_group(store, path="0").attrs.update(
        {
            "multiscales": {"layout": []},
            "spatial:bbox": [0, 0, 1, 1],
            "proj:projjson": {},
        }
    )
    pyramid.write(store, mode="a", levels=[1])
    attrs = zarr.open_group(store, path="0").attrs
    assert attrs["proj:code"] == "EPSG:4326"
    layout0 = pyramid.attrs["multiscales"]["layout"][0]
    assert attrs["spatial:transform"] == layout0["spatial:transform"]
    assert not {"multiscales", "spatial:bbox", "proj:projjson"} & set(attrs)


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


def test_spatial_var_over_4d_writes(create_dataset):
    # shape from issue #58
    ds = create_dataset(
        nx=128, ny=64, extra_dims={"scenario": 2, "time_horizon": 3, "quantile": 2}
    )
    store = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=3).write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)

    assert dt["2"].ds.elevation.shape == (2, 3, 2, 16, 32)
    want = ds.coarsen(x=2, y=2, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values, want.elevation.values, rtol=1e-6
    )


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


def _read_level(store: zarr.storage.MemoryStore, lvl: int, name: str) -> np.ndarray:
    root = zarr.open_group(store, mode="r")
    return root[f"{lvl}/{name}"][:]


def test_keep_levels_in_memory_deprecated(create_dataset):
    pyramid = create_pyramid(create_dataset(nx=16, ny=16), levels=2)
    with pytest.warns(DeprecationWarning, match="keep_levels_in_memory"):
        pyramid.write(zarr.storage.MemoryStore(), keep_levels_in_memory=True)


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


def _partly_spatial_dataset(n=16, fill=None):
    """Raster plus one variable per spatial dim: (time, x) and (y,)."""
    profile = np.arange(3 * n, dtype="i2").reshape(3, n)
    if fill is not None:
        profile[0, ::3] = fill
    ds = xr.Dataset(
        {
            "elev": (("y", "x"), np.arange(n * n, dtype="f4").reshape(n, n)),
            "profile": (("time", "x"), profile),
            "row": (("y",), np.arange(n, dtype="i2")),
        },
        coords={
            "x": np.arange(n, dtype="f8"),
            "y": np.arange(n, dtype="f8"),
            "time": np.arange(3),
        },
    ).proj.assign_crs(spatial_ref="EPSG:4326")
    if fill is not None:
        ds.profile.attrs["_FillValue"] = fill
    return ds


@pytest.mark.parametrize("fill", [None, -1])
def test_as_datatree_matches_write_partly_spatial_var(fill):
    """A variable over one spatial dim is coarsened along that dim, not dropped.

    Level 0 is a plain copy on both paths, so the divergence only shows above
    it -- levels 1-2 here, which also exercises the chained level template.
    """
    ds = _partly_spatial_dataset(fill=fill)

    pyramid = create_pyramid(ds, levels=3, method="mean")
    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    root = zarr.open_group(store, mode="r")
    dt = pyramid.as_datatree()

    for lvl in range(3):
        for name in ("profile", "row"):
            written = root[f"{lvl}/{name}"][:]
            assert written.dtype == ds[name].dtype
            np.testing.assert_array_equal(written, dt[str(lvl)].ds[name].values)


def test_partly_spatial_var_is_encoded_at_every_level():
    pyramid = create_pyramid(_partly_spatial_dataset(), levels=3)
    for lvl in range(3):
        enc = pyramid.encoding[f"/{lvl}"]
        assert "profile" in enc and "row" in enc
        # chunks stay at 1 along non-spatial dims, as for fully spatial vars
        assert enc["profile"]["chunks"][0] == 1
        assert len(enc["row"]["chunks"]) == 1


def test_non_numeric_var_over_spatial_dim_raises(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    ds["label"] = ("x", np.array(list("abcdefghijklmnop"), dtype=object))

    with pytest.raises(ValueError, match="non-numeric dtype"):
        create_pyramid(ds, levels=2)


def test_partly_spatial_var_over_4d_writes(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    ds["profile"] = (
        ("time", "band", "level", "run", "x"),
        np.random.rand(2, 2, 2, 2, 16).astype("f4"),
    )
    store = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=2).write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)

    want = ds.profile.coarsen(x=2, boundary="trim").mean()
    np.testing.assert_allclose(dt["1"].ds.profile.values, want.values, rtol=1e-6)


@pytest.mark.parametrize(
    "store, expect",
    [("local.zarr", "zarrs"), ("s3://bucket/p.zarr", "default"), (None, "default")],
)
def test_codec_pipeline_local_only(tmp_path, store, expect):
    from topozarr.pyramid import DEFAULT_PIPELINE, ZARRS_PIPELINE, _codec_pipeline

    store = zarr.storage.MemoryStore() if store is None else store
    with _codec_pipeline(store):
        got = zarr.config.get("codec_pipeline.path")
    assert got == (ZARRS_PIPELINE if expect == "zarrs" else DEFAULT_PIPELINE)
    assert zarr.config.get("codec_pipeline.path") == DEFAULT_PIPELINE


def test_zarrs_pipeline_matches_default(create_dataset, tmp_path, monkeypatch):
    from contextlib import nullcontext

    import topozarr.pyramid

    ds = create_dataset(nx=64, ny=64)
    create_pyramid(ds, levels=3).write(tmp_path / "zarrs.zarr")
    monkeypatch.setattr(topozarr.pyramid, "_codec_pipeline", lambda _: nullcontext())
    create_pyramid(ds, levels=3).write(tmp_path / "default.zarr")
    a = xr.open_datatree(tmp_path / "zarrs.zarr", engine="zarr", consolidated=False)
    b = xr.open_datatree(tmp_path / "default.zarr", engine="zarr", consolidated=False)
    xr.testing.assert_identical(a, b)


def _store_bytes(store: zarr.storage.MemoryStore) -> dict[str, bytes]:
    return {k: v.to_bytes() for k, v in store._store_dict.items()}


@pytest.mark.parametrize("method", ["mean", "max", "min", "sum", "nearest"])
@pytest.mark.parametrize("dtype,fill", [("f4", float("nan")), ("i2", 0), ("f8", None)])
@pytest.mark.parametrize("factors", [[1, 2, 4, 8], [1, 4, 16], [1, 3, 6]])
@pytest.mark.parametrize("shard", [None, 16])
def test_tiles_match_level_by_level(method, dtype, fill, factors, shard):
    # ragged shape so trailing tiles hold partial windows at every level
    ny, nx = 203, 331
    rng = np.random.default_rng(1)
    data = ((rng.random((2, ny, nx)) - 0.5) * 200).astype(dtype)
    if fill is not None:
        data[:, : ny // 2, : nx // 2] = fill  # all-fill tiles exercise skipping
        data[rng.random(data.shape) < 0.2] = fill
    ds = xr.Dataset(
        {
            "elev": (("time", "y", "x"), data),
            "row": (("y",), np.arange(ny, dtype="i2")),
        },
        coords={
            "x": np.arange(nx, dtype="f8"),
            "y": np.arange(ny, dtype="f8"),
            "time": np.arange(2),
        },
    ).proj.assign_crs(spatial_ref="EPSG:4326")
    if fill is not None:
        ds.elev.attrs["_FillValue"] = fill
    pyr = create_pyramid(ds, factors=factors, method=method, target_chunk_bytes=512)
    if shard is not None:
        # uniform small shards: many tiles, ragged at every level
        for lvl in range(pyr.levels):
            pyr.encoding[f"/{lvl}"]["elev"].update(
                chunks=(1, shard // 2, shard // 2), shards=(1, shard, shard)
            )
            pyr.encoding[f"/{lvl}"]["row"].update(chunks=(shard // 2,), shards=(shard,))

    k, tiles = pyr._tile_plan(list(range(pyr.levels)), ["elev", "row"], 1 << 20, 2)
    assert k > 0  # the tile pass must actually fuse levels
    if shard is not None:
        assert tiles["elev"][1] < ny or tiles["elev"][2] < nx

    # reference: level 0 alone, then each level reduced from the stored one
    a, b = zarr.storage.MemoryStore(), zarr.storage.MemoryStore()
    kw = {"max_workers": 2, "max_region_bytes": 1 << 20}
    pyr.write(a, levels=[0], **kw)
    for lvl in range(1, pyr.levels):
        pyr.write(a, mode="a", levels=[lvl], **kw)
    pyr.write(b, **kw)
    assert _store_bytes(a) == _store_bytes(b)


@pytest.mark.parametrize("as_datatree", [False, True])
@pytest.mark.parametrize("chunks", [{}, None])
def test_cf_packed_decoded_and_raw_write_the_same(as_datatree, chunks):
    # decoded input is re-packed to uint16; a decoded 0.0 (raw 1000) stays data
    raw = np.full((8, 8), 1000, dtype="uint16")
    raw[0, 0] = 0  # _FillValue
    raw[4:, 4:] = 1234
    ds = xr.Dataset(
        {"refl": (("y", "x"), raw * 1e-4 - 0.1)},
        coords={"x": np.arange(8.0), "y": np.arange(8.0)},
    )
    ds["refl"].encoding = {
        "scale_factor": 1e-4,
        "add_offset": -0.1,
        "_FillValue": 0,
        "dtype": "uint16",
    }
    src = zarr.storage.MemoryStore()
    ds.to_zarr(src, zarr_format=3, consolidated=False)

    outs = []
    for kw in ({}, {"mask_and_scale": False}):
        d = xr.open_zarr(src, consolidated=False, chunks=chunks, **kw)
        pyramid = create_pyramid(d.proj.assign_crs(spatial_ref="EPSG:4326"), levels=2)
        out = zarr.storage.MemoryStore()
        if as_datatree:
            pyramid.as_datatree().to_zarr(
                out, zarr_format=3, consolidated=False, encoding=pyramid.encoding
            )
        else:
            pyramid.write(out)
        outs.append(out)

    for lvl in ("0", "1"):
        dec, rawd = (zarr.open_array(o, path=f"{lvl}/refl") for o in outs)
        assert dec.dtype == rawd.dtype == np.dtype("uint16")
        np.testing.assert_array_equal(dec[:], rawd[:])
        assert dec.attrs["scale_factor"] == 1e-4
        assert dec.attrs["add_offset"] == -0.1
    # window with the fill pixel averages the 3 valid raw 1000s
    np.testing.assert_array_equal(
        zarr.open_array(outs[0], path="1/refl")[0, :2], [1000, 1000]
    )


def test_cf_unsigned_is_not_repacked():
    # int16 storage with _Unsigned decodes above int16 max; repacking would wrap
    from topozarr.coarsen import _repack

    da = xr.DataArray(np.full((2, 2), 60000 * 1e-4), dims=("y", "x"))
    da.encoding = {
        "scale_factor": 1e-4,
        "add_offset": 0.0,
        "_FillValue": 0,
        "dtype": "int16",
        "_Unsigned": "true",
    }
    assert _repack(da) is da


def test_lazy_cf_packed_source_is_read_tile_by_tile(monkeypatch):
    from xarray.backends.zarr import ZarrArrayWrapper

    raw = np.random.default_rng(7).integers(-30000, 30000, (2, 96, 80), "int16")
    raw[:, :5, :7] = -32768
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), raw)},
        coords={"x": np.arange(80.0), "y": np.arange(96.0)},
    )
    ds["v"].attrs.update(scale_factor=1e-4, add_offset=0.25, _FillValue=-32768)
    src = zarr.storage.MemoryStore()
    ds.to_zarr(
        src, zarr_format=3, consolidated=False, encoding={"v": {"chunks": (1, 16, 16)}}
    )

    lazy = xr.open_zarr(src, consolidated=False, chunks=None)
    reads: list[int] = []
    read = ZarrArrayWrapper.__getitem__

    def recording_read(self, key):
        out = read(self, key)
        reads.append(np.size(out))
        return out

    monkeypatch.setattr(ZarrArrayWrapper, "__getitem__", recording_read)

    pyramid = create_pyramid(lazy.proj.assign_crs(spatial_ref="EPSG:4326"), levels=3)
    assert reads == []
    assert pyramid.source["v"].dtype == np.dtype("int16")

    for lvl in range(pyramid.levels):
        pyramid.encoding[f"/{lvl}"]["v"].update(chunks=(1, 8, 8), shards=(1, 16, 16))
    stats = pyramid.write(
        zarr.storage.MemoryStore(), max_workers=2, max_region_bytes=4096, stats=True
    )
    tile_elems = max(
        math.prod(shape)
        for entry in stats.values()
        for shape in entry["region_shapes"].values()
    )
    assert tile_elems < raw.size
    assert max(reads) <= tile_elems


def test_lazy_repack_matches_dask_repack():
    from topozarr.coarsen import _repack

    da = xr.DataArray(
        np.array([[0.5, 1.5, -2.5, np.nan], [1e9, -1e9, 0.12344, 3.0]]),
        dims=("y", "x"),
        coords={"x": np.arange(4.0)},
        name="v",
        attrs={"units": "m"},
    )
    da.encoding = {
        "scale_factor": 1e-2,
        "add_offset": 0.25,
        "_FillValue": -32768,
        "dtype": "int16",
        "chunks": (2, 2),
        "preferred_chunks": {"y": 2, "x": 2},
    }
    lazy, dask = _repack(da), _repack(da.chunk())
    assert lazy.dtype == np.dtype("int16")
    assert (
        lazy.encoding
        == dask.encoding
        == {
            "chunks": (2, 2),
            "preferred_chunks": {"y": 2, "x": 2},
        }
    )
    selections = [
        {"y": 1, "x": slice(1, 3)},
        {"y": 1, "x": 3},
        {"y": [1, 0], "x": [3, 0]},
        {"y": []},
        {"x": np.zeros(4, dtype=bool)},
        {"y": slice(None, None, -1)},
        {
            "y": xr.DataArray([0, 1, 1], dims="z"),
            "x": xr.DataArray([3, 0, 2], dims="z"),
        },
    ]
    for sel in selections:
        np.testing.assert_array_equal(
            lazy.isel(sel).values, dask.isel(sel).values, strict=True
        )
    xr.testing.assert_identical(lazy.compute(), dask.compute())


def test_levels_drop_stale_root_geozarr_attrs(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    stale = {"spatial:transform": [1.0, 0, 0, 0, -1.0, 0], "spatial:shape": [16, 16]}
    ds.attrs.update(stale, multiscales={"layout": []}, title="keep")
    pyramid = create_pyramid(ds, levels=3)
    assert ds.attrs["spatial:shape"] == [16, 16]  # caller's attrs untouched
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    root = zarr.open_group(store, mode="r")
    layout = root.attrs["multiscales"]["layout"]
    assert root.attrs["spatial:shape"] == [16, 16]
    assert layout[2]["spatial:shape"] == [4, 4]
    for attrs in [root["2"].attrs, pyramid.as_datatree()["2"].attrs]:
        assert attrs["spatial:transform"] == layout[2]["spatial:transform"]
        assert attrs["spatial:shape"] == [4, 4]
        assert [c["name"] for c in attrs["zarr_conventions"]] == ["proj", "spatial"]
        assert "multiscales" not in attrs
        assert attrs["title"] == "keep"
