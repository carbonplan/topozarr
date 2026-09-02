# Plan: `recommend_encoding` — flat (single-resolution) geozarr datasets

Not everyone needs overviews. topozarr already supports writing a flat geozarr
group via `attach_geozarr_metadata` (geozarr attrs, no `/0` nesting, no
`multiscales`), but the chunk/shard heuristic ships only with `create_pyramid`
— it is reachable only as `pyramid.encoding`. Flat-path users must hand-roll
`chunks`/`shards` for `to_zarr`.

Fix: expose the heuristic as a second small function. Two functions, two plain
return values — an `xr.Dataset` and an encoding dict. No new class, no
single-level `Pyramid`.

```python
from topozarr import attach_geozarr_metadata, recommend_encoding

ds = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
ds.to_zarr(
    store,
    zarr_format=3,
    consolidated=False,
    encoding=recommend_encoding(ds, x_dim="lon", y_dim="lat"),
)
```

## Rejected alternatives

- **`create_pyramid(ds, levels=1)`.** Works today, but produces `/0` nesting
  and a one-entry `multiscales` attr, and hands back a `Pyramid` whose
  `levels` / `as_datatree` / `with_nonspatial_shards` are meaningless for
  flat output.
- **`attach_geozarr_metadata` returning `(ds, encoding)`.** Breaking; forces
  encoding on callers who only want attrs.
- **Populating `da.encoding` in place.** Least typing, but silent: encoding
  survives (or is dropped by) later ops and collides with dask chunks.
- **A `GeoDataset` container holding `.ds` + `.encoding`.** Only earns its
  keep with a `.write()` method; without one it is a named 2-tuple. Adds a
  class where a dict suffices.

Name: `recommend_encoding`, not `suggest_encoding` — matches the existing doc
voice (`docs/usage.md:56`, `pyramid.py:626` both say "recommended").

## 1. `recommend_encoding` (`src/topozarr/geozarr.py`)

Public wrapper over the existing private mechanics: validation + spatial-dim
checks + source-chunk sniffing + defaults, delegating to
`metadata.create_level_encoding`.

```python
def recommend_encoding(
    ds: xr.Dataset,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunks_per_shard: ChunksPerShard | None = DEFAULT_CHUNKS_PER_SHARD,
) -> dict[str, dict[str, tuple[int, ...]]]:
```

Body:

1. `validate_chunks_per_shard(chunks_per_shard)` when not `None`.
2. Raise if `x_dim` / `y_dim` are not dims of `ds` — same messages as
   `create_pyramid` (`coarsen.py:231-236`).
3. Raise if no data variable has both spatial dims (message reworded: "nothing
   to encode", not "nothing to pyramid").
4. `source_chunks = _spatial_source_chunks(ds, x_dim, y_dim)`.
5. Return `create_level_encoding(ds, x_dim, y_dim, target_chunk_bytes=...,
   chunks_per_shard=..., source_chunks=source_chunks)`.

Notes:

- No `levels` / `factors` kwarg — flat by definition.
- No `crs` requirement: encoding is pure geometry/dtype, so this works on any
  dataset with the named dims. `attach_geozarr_metadata` keeps the CRS check.
- No 4-dim kernel limit check — that limit is a `topozarr-core` write-path
  constraint; `to_zarr` has no such ceiling.
- Returns spatial variables only (existing `create_level_encoding` behavior);
  non-spatial vars fall through to xarray defaults.
- Lives in `geozarr.py` next to `attach_geozarr_metadata`. Import direction
  `geozarr -> coarsen -> metadata/chunking` is already established, no cycle.
- `_spatial_source_chunks` gains a public-ish role; keep the underscore, it
  stays internal.

## 2. Rewire `create_pyramid` to use it (`src/topozarr/coarsen.py:250-262`)

The level-0 special case disappears. Today:

```python
level0_source_chunks = _spatial_source_chunks(ds, x_dim, y_dim)
full_encoding = {
    f"/{idx}": create_level_encoding(
        template, x_dim, y_dim,
        target_chunk_bytes=target_chunk_bytes,
        chunks_per_shard=chunks_per_shard,
        source_chunks=level0_source_chunks if idx == 0 else None,
    )
    for idx, template in level_templates.items()
}
```

After:

```python
full_encoding = {
    f"/{idx}": recommend_encoding(
        template, x_dim=x_dim, y_dim=y_dim,
        target_chunk_bytes=target_chunk_bytes,
        chunks_per_shard=chunks_per_shard,
    )
    for idx, template in level_templates.items()
}
```

Why the `if idx == 0` is safe to drop:

- `build_level_templates` starts `levels = [ds]`, so the level-0 template *is*
  the source dataset — sniffing it yields the identical `source_chunks`.
- Levels 1+ hold `np.broadcast_to` placeholders, whose `.encoding` is empty, so
  `source_chunks(da)` returns `None` and `_spatial_source_chunks` returns
  `None` — exactly the value passed explicitly today.

Import direction: `coarsen` would import from `geozarr`, which imports
`get_crs` from `coarsen` — a cycle. Resolve by moving `get_crs` into
`metadata.py` (or a small `_crs.py`) so `geozarr` no longer imports `coarsen`,
then `coarsen -> geozarr` is one-way. Decide at implementation time; if the
move looks invasive, keep both call sites on `create_level_encoding` and put
`recommend_encoding` in `coarsen.py` instead, re-exported from `__init__`.

## 3. Exports and docs

- `src/topozarr/__init__.py`: import and add `recommend_encoding` to
  `__all__`. Public surface goes 5 -> 6 names.
- `docs/api.md`: add `::: topozarr.geozarr.recommend_encoding`; update the
  "public API is four objects" sentence.
- `docs/usage.md:36`: retitle "Metadata only (no pyramid)" ->
  "Single-resolution datasets (no pyramid)". Lead with the two-call example
  above; keep the note that CRS comes from xproj or `crs=`. Say explicitly
  that `create_pyramid(levels=1)` is not the way to get a flat dataset.
- `README.md`: short pointer to the flat path near the pyramid quickstart.
- `docs/releases.md`: entry for the new function.

`levels` / `factors` on `create_pyramid` are untouched, including `levels=1`
(degenerate but legal, no break).

## 4. Tests

`tests/test_geozarr.py` (already covers `attach_geozarr_metadata`):

- shape: returns `{var: {"chunks": ..., "shards": ...}}` for spatial vars only.
- `chunks_per_shard=None` omits `"shards"`.
- invalid `chunks_per_shard` (e.g. `3`, `64`) raises.
- missing `x_dim` / `y_dim` raises with the dim name in the message.
- dataset with no spatial variable raises.
- no CRS required: a dataset without a CRS still gets an encoding.
- custom dims (`lon`/`lat`) and a non-spatial dim (`time`) — chunks along
  `time` stay 1, shard along `time` widens per `fill_nonspatial_shards`.
- source-chunk snapping: open a chunked zarr store, confirm the returned chunks
  snap to the source chunking (mirrors whatever `test_chunking.py` asserts for
  the pyramid path).
- roundtrip: `attach_geozarr_metadata` + `to_zarr(encoding=...)` writes, and
  the written arrays' `chunks` / `shards` match what was recommended.

`tests/test_pyramid.py` or `tests/test_chunking.py`:

- regression: `create_pyramid(...).encoding` is byte-identical before and
  after the rewire, for a chunked source (exercises the level-0 sniff) and an
  in-memory source, across `levels=` and `factors=` and
  `chunks_per_shard=None`.

Run: `uv run pytest tests/test_geozarr.py tests/test_chunking.py
tests/test_pyramid.py`.

## Open question

Does `zarr-layer` (JS) consume a flat geozarr group, or does it require the
`multiscales` attr and `/0` nesting? If it requires the pyramid layout, the
flat path is a data-production convenience only, and the docs should say so.
