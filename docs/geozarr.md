# GeoZarr

[GeoZarr](https://github.com/zarr-developers/geozarr-spec) is built from three
[Zarr conventions](https://github.com/zarr-conventions). `topozarr` writes all
three as group attributes, and each one answers a single question about the
store. Together, they give a web map like `zarr-layer` what it needs to place
the data and pick a resolution without reading any coordinate arrays.

<div class="tz-figure">
--8<-- "docs/assets/figures/conventions.svg"
</div>

In the store, the root group describes the whole pyramid. Each level group
repeats `proj:code`, `proj:wkt2`, and `zarr_conventions`. The figure shows only
the key attributes. The root also carries `proj:wkt2`, `spatial:transform`,
`spatial:shape`, and `spatial:registration`.

<div class="tz-figure">
--8<-- "docs/assets/figures/store-layout.svg"
</div>

## multiscales

Spec: [zarr-conventions/multiscales v0.1](https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md)
{ .tz-spec }

The multiscales convention lists the levels in `layout`. Each entry after
`"0"` names the level it was `derived_from` (the previous one), the
`resampling_method`, and a `transform` whose `scale` is the ratio between
adjacent levels along each spatial dimension (`2` by default, or `4` for
`factors=[1, 4, 16]`). Each entry also carries its own `spatial:transform` and
`spatial:shape`, which give the absolute pixel size.

<div class="tz-figure">
--8<-- "docs/assets/figures/multiscales-layout.svg"
</div>

`scale` and `translation` cover every dimension of the array, in the dataset's
dimension order. For `[time, lat, lon]`, they are `[1, 2, 2]` and `[0, 0, 0]`.
`translation` is always zero because every level shares the same corner origin
in `spatial:transform`, as the spatial convention recommends.

## proj

Spec: [zarr-conventions/proj v0.1](https://github.com/zarr-conventions/proj/blob/v0.1/README.md)
{ .tz-spec }

The proj convention sets the CRS with `proj:code` and `proj:wkt2`. It applies
only to a group's direct child arrays. Level arrays are grandchildren of the
root, so `topozarr` also tags each level group.

<div class="tz-figure">
--8<-- "docs/assets/figures/proj-inheritance.svg"
</div>

## spatial

Spec: [zarr-conventions/spatial v0.1](https://github.com/zarr-conventions/spatial/blob/v0.1/README.md)
{ .tz-spec }

The spatial convention maps array indices to coordinates. `spatial:transform`
is a six-number affine transform. Because each level gets only one transform,
this works only for 1-D, uniformly spaced coordinates.

<div class="tz-figure">
--8<-- "docs/assets/figures/spatial-transform.svg"
</div>

`spatial:dimensions` names the Y and X dimensions, in that order. Other
dimensions, like `time`, are left alone.

<div class="tz-figure">
--8<-- "docs/assets/figures/spatial-dimensions.svg"
</div>

`topozarr` always writes `spatial:registration: "pixel"`, which means cell
edges, not cell centers, line up with `spatial:bbox`.
