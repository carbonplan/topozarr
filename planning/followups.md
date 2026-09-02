# Follow-ups from the `recommend_encoding` review

Out of scope for the `recommend_encoding` PR (single-level chunk/shard
recommendation + geozarr attach). Everything below is a separate change.

## 1. Dask sources fail `safe_chunks`

**Status:** documented and pinned by a test, not fixed.

**Correction (2026-09-02):** the original framing of this item was wrong. It
claimed `snap_chunk_to_source` "aligns the chunk, not the shard". It already
aligns the shard: `chunking.py:110-133` builds candidate *shard* sizes, filters
`s % cps == 0`, and returns `s // cps` as the chunk. The proposed fix ("make
the snap target the shard") was therefore a no-op. The failure is real; the
cause and the fix below replace the original text.

Measured on a 2000x2000 f4, default settings (re-verified against current code):

| dask chunk | recommended chunk | shard | `dask % shard` | `to_zarr` |
|-----------:|------------------:|------:|---------------:|-----------|
| 500        | 375               | 1500  | 500            | FAIL      |
| 750        | 375               | 1500  | 750            | FAIL      |
| 256        | 384               | 1536  | 256            | FAIL      |
| 1000       | 250               | 1000  | 0              | OK        |

**Real cause:** `safe_chunks` needs the write unit to *divide* the dask block
(`dask_chunk % shard == 0`), not merely to nest with it. The candidate set
admits both divisors and multiples of `src_chunk`; every multiple fails. And
restricting the candidates to divisors does not help either — the `chunk >= 128`
floor then rejects all of them (`500/4 = 125`, `256/4 = 64`).

**Why fix:** the flat path is the one where the user hands the encoding straight
to `ds.to_zarr`. `Pyramid.write` streams regions itself and never hits
`safe_chunks`, so pyramid users have never seen this. Every dask user of
`recommend_encoding` will.

**Shape of the fix:** flex `chunks_per_shard` *down* — pick the largest `cps`
whose shard divides `src_chunk` with `chunk >= 128`, treating a user-supplied
`cps` as an upper bound and only flexing when a source chunking is detected.
Verified against the same four rows:

| dask chunk | cps | shard | chunk | `to_zarr` |
|-----------:|----:|------:|------:|-----------|
| 500        | 2   | 500   | 250   | OK        |
| 750        | 2   | 750   | 375   | OK        |
| 256        | 1   | 256   | 256   | OK        |
| 1000       | 4   | 1000  | 250   | OK        |

**Blast radius:** changes `pyramid.encoding` for chunked sources, so it is not
behavior-preserving. Needs its own PR with the parity test in
`tests/test_geozarr.py::test_pyramid_level0_encoding_matches_recommend_encoding`
re-baselined. Recommendation is to apply it on both paths rather than fork
behavior: a shard that divides `src_chunk` still gets each source chunk read
once, because `copy_region_shape` widens the region to
`lcm(shard, source_chunk)`.

**Pinned meanwhile:**
`tests/test_geozarr.py::test_recommend_encoding_dask_source_needs_shard_aligned_chunks`
asserts the current failure and both documented workarounds (rechunk to shards,
or `safe_chunks=False`). That test must be updated when this lands.

## 2. Name collision: `source_chunks` — DONE

`metadata.py` imported the function `source_chunks` from `chunking.py` while
`create_level_encoding` / `_create_var_encoding` took a parameter of the same
name (a `dict[str, int]`, not the function's return type). Correct, but a trap.

Fixed by renaming the **parameter** to `source_chunk_sizes`, which also states
the type. Not `spatial_source_chunks` as first proposed — that differs from the
existing `_spatial_source_chunks` helper by one leading underscore, trading the
collision for a worse one.

Renaming the parameter (rather than the function) confined the change to
`metadata.py`: no `pyramid.py` call sites, no tests. `create_level_encoding` is
not in `__all__` and had one caller.

## 3. Flat path has no overviews

`docs/usage.md` now says this plainly, but worth deciding: should
`recommend_encoding` warn (or should the docs push harder) when a dataset is
large enough that a zoomed-out read pulls an unreasonable number of shards?
There is no threshold in the code today. Probably docs-only, but it is the one
real functional gap between the flat path and a pyramid.

## 4. Single-pixel dims: refuse, and that is intended — CLOSED, no action

Surfaced by the hypothesis property `test_flat_geozarr_attrs_invariants`.

On a dataset with a size-1 spatial dim (a 1xN strip, a single-column profile):

- `recommend_encoding` works — encoding is pure shape+dtype, no spacing needed.
- `attach_geozarr_metadata` and `create_pyramid` both raise `cannot infer
  resolution of coordinate 'x' from a single value`.

The two metadata paths agree: `create_geozarr_metadata` and the multiscale root
both call `_get_affine_transform` without a fallback. The split is between
*encoding* (needs no georeference) and *metadata* (needs a transform), not
between flat and pyramid. Both hypothesis tests assert this, so it is pinned.

Resolution: leave it. A single-pixel dim genuinely carries no spacing, and
geozarr requires a transform. The per-level fallback in
`create_multiscale_metadata` is not a precedent — a coarsened level's resolution
really is `level0_res * factor`; a source's is not knowable.

Rejected: a `fallback_resolution=(x_res, y_res)` kwarg on
`attach_geozarr_metadata` / `create_pyramid`. It would be ignored on essentially
every real dataset (a footgun), and it writes a fabricated number into published
`spatial:transform`. Same objection as a silent fall back to 1.0.

The one case that could change this: a dataset carrying CF cell `bounds` or a
GDAL `GeoTransform` attr, where the resolution is recorded rather than invented.
`src/topozarr/` reads neither today. If that lands, derive from it — do not add
a user-supplied override.

Behavior unchanged. One UX-only follow-on landed: `_coord_resolution`'s
`ValueError` now says what to do instead (the `cannot infer resolution` prefix
the two hypothesis tests match on is preserved). No docs change — the case is
niche enough that documenting it costs more attention than it saves.
