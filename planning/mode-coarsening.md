# Plan: `mode` (majority) coarsening for categorical rasters

Follow-up to `nearest` (issue #26). `nearest` gives categorical layers a valid
class code at coarse zoom, but it samples one cell per window — a class that is
present but never lands on a window corner vanishes. `mode` (majority vote per
window) is the correct aggregation for multi-class rasters like ESA WorldCover.

## Why mode is not just a new kernel arm

`Pyramid.write` block-reduces each level from the previously written level.
That is correct for mean/max/min/sum/nearest because they compose, but mode
does not: mode-of-modes != mode-of-native. Example, one 4x4 window coarsened
in two 2x steps:

```
1 1 2 2      modes of the       1 2      mode of the 2x2: 1 (tie, smallest)
1 3 2 4  ->  four 2x2 blocks ->                mode of the native 4x4: 2
3 3 4 2                         3 2      (2 appears 5 times, 1 and 3 four)
3 1 2 2
```

A locally-dominant class can win at coarse zoom even when it is globally rare.
Correct mode levels must therefore reduce from level 0 (native), which changes
the write path's read pattern, not just the kernel.

## Kernel (`core/src/lib.rs`)

- Add `Mode` to `enum Method`; parse `"mode"`; extend the error message.
- `reduce_window` arm:
  - collect elements that are valid per the existing `is_missing` (skips NaN
    and `fill_value` when `skipna=True`),
  - sort, then run-length scan; the longest run wins, smallest value on a tie
    (deterministic and language-independent),
  - all-missing window returns `all_missing_result` (fill, else NaN, else 0).
- Works for every supported dtype since it is equality-based; floats are
  handled by the valid-only sort (no NaN reaches the comparison).
- O(w log w) per window; windows are factor^2 elements when reducing from
  native (e.g. 4096 at factor 64) — fine per-window, but see the memory note.
- Unit tests: tie-breaking, all-missing, all-identical, float-with-NaN,
  fill-value skipping.

## Write path (`src/topozarr/pyramid.py`)

- `CoarseningMethod` gains `"mode"`.
- `_write_var`: for `method="mode"` and `lvl > 0`, read from `root["0/<name>"]`
  with `stride = self.factors[lvl]` (cumulative), instead of the previous
  level with the per-step ratio.
- Levels-subset validation (the pre-check loop in `write`): a mode level `N`
  requires level 0 in the write plan or the store, not level `N-1`.
- Fusion: disable in `_compute_use_fusion` for mode. (Level 0 -> 1 fusion is
  technically valid — level 0 blocks are native — but skip it in v1 for
  simplicity.)
- `as_datatree` / `_coarsen_chain`: raise `NotImplementedError` for mode
  (xarray.coarsen has no mode method, and the chained-from-previous structure
  is wrong for it anyway).

## Memory bounding

Input per output region is `region * factor^2`: a shard-sized output region at
factor 32 would read ~1000x its bytes from level 0. Two pieces:

- `_region_bytes` must account `factor^2` (not `step^2`) for mode levels so
  `default_max_workers` shrinks the pool accordingly.
- Inside `downsample_level`'s `get_block`, stream the input in window-aligned
  y-strips (strip height a multiple of the stride) capped by
  `max_region_bytes`, reducing each strip into the output region buffer.
  Windows never cross strips, so strips are independent and the output region
  and write path stay unchanged.

## Cost note

Every mode level re-reads level 0, so a full pyramid reads the native data
once per level instead of once total. That is inherent to correct mode; sparse
pyramids (`factors=[1, 8, 64]`) reduce the number of levels paying it.

## Testing

- Kernel unit tests as above.
- Pyramid integration test with a raster constructed so mode-of-modes !=
  mode-of-native (like the 4x4 example) — asserts levels come from native.
- Levels-subset test: `levels=[2]` with mode requires level 0 in the store.
- `as_datatree` raises for mode.

## Out of scope / future

- Per-variable methods (mixed continuous + categorical variables in one
  pyramid currently share a single `method`).
- `median` (also non-composable; would ride the same from-native read path).
- Linear-time majority (histogram / Boyer-Moore variants) if the sort shows up
  in profiles.
