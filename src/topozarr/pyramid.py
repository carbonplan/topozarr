from __future__ import annotations

import importlib.util
import math
import os
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
import zarr
import zarr.errors
import zarr.storage
from topozarr_core import METHODS

from .chunking import source_chunks
from .engine import (
    DEFAULT_MAX_REGION_BYTES,
    RegionTimer,
    copy_region_shape,
    default_max_workers,
    downsample_level,
    write_tiles,
)
from .metadata import level_proj_attrs

CoarseningMethod = Literal["mean", "max", "min", "sum", "nearest"]

# Methods `xarray.coarsen` can express, for the as_datatree path. `nearest` is
# absent because xarray has no such reduction -- _decimate covers it. A kernel
# method missing from both is rejected rather than dispatched blindly.
XR_COARSEN_METHODS = frozenset({"mean", "max", "min", "sum"})


def validate_method(method: str) -> None:
    """Raise if ``method`` is not implemented by the installed kernel.

    Checked against ``topozarr_core.METHODS`` rather than
    [CoarseningMethod][topozarr.pyramid.CoarseningMethod]: the two are kept
    equal by a test, but only the kernel's own list catches a topozarr paired
    with a core that predates a method it advertises (issue #26). Without this
    the mismatch surfaces from ``block_reduce`` on the first *coarsened* level,
    by which point level 0 is already in the store.
    """
    if method not in METHODS:
        listed = ", ".join(repr(m) for m in METHODS)
        raise ValueError(f"method must be one of {listed}; got {method!r}")


ZARRS_PIPELINE = "zarrs.ZarrsCodecPipeline"
DEFAULT_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"


def _codec_pipeline(store: Any) -> AbstractContextManager[Any]:
    """Use the zarrs (Rust) codec pipeline for local stores when installed.

    Remote stores keep zarr-python's pipeline: zarrs-python rebuilds its own
    object_store client without pooling (zarrs-python #139). A pipeline the
    user already configured is left alone.
    """
    local = isinstance(store, zarr.storage.LocalStore) or (
        isinstance(store, (str, os.PathLike)) and "://" not in str(store)
    )
    if (
        local
        and zarr.config.get("codec_pipeline.path") == DEFAULT_PIPELINE
        and importlib.util.find_spec("zarrs") is not None
    ):
        import zarrs  # noqa: F401  # registers the pipeline

        return zarr.config.set({"codec_pipeline.path": ZARRS_PIPELINE})
    return nullcontext()


def _progress_bar(total: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as err:
        raise ImportError(
            "progress=True requires tqdm; install it with `pip install tqdm`"
        ) from err
    return tqdm(total=total, unit="region")


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@dataclass
class Pyramid:
    """A write plan for a multiscale Zarr pyramid, returned by
    [create_pyramid][topozarr.coarsen.create_pyramid].

    Attributes:
        source: The original (level 0) dataset.
        level_templates: Per-level datasets carrying real coordinates and
            attrs; spatial data variables are zero-cost placeholders with the
            correct shape/dtype (their data is computed during
            [write][topozarr.pyramid.Pyramid.write]).
        encoding: Nested dict ``{path: {var: {"chunks": ..., "shards": ...}}}``.
        attrs: Root group metadata (multiscales / proj: / spatial: / zarr-layer).
    """

    source: xr.Dataset
    level_templates: dict[int, xr.Dataset]
    encoding: dict[str, Any]
    attrs: dict[str, Any]
    x_dim: str
    y_dim: str
    method: CoarseningMethod
    factors: list[int] = field(default_factory=list)
    fill_values: dict[str, float | int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # method is a plain field, so a plan can be edited after create_pyramid
        # validated it; re-check here so the invariant holds at write time
        validate_method(str(self.method))

    @property
    def levels(self) -> int:
        return len(self.level_templates)

    def _step(self, lvl: int) -> int:
        """Per-step downsample ratio coarsening level ``lvl-1`` into ``lvl``."""
        return self.factors[lvl] // self.factors[lvl - 1]

    def _coarsened_vars(self) -> list[str]:
        """Variables ``write`` computes: those over at least one spatial dim.

        Mirrors ``_is_coarsened`` on the datatree path -- a variable over one
        spatial dim (e.g. a per-column profile) is coarsened along that dim.
        Variables over neither pass through from the level template unchanged.
        """
        return [
            str(name)
            for name, da in self.source.data_vars.items()
            if {self.x_dim, self.y_dim} & set(da.dims)
        ]

    def _region_shape(
        self, lvl: int, name: str, max_region_bytes: int
    ) -> tuple[int, ...]:
        """Region shape used to stream one variable of one level."""
        template_da = self.level_templates[lvl][name]
        enc = self.encoding[f"/{lvl}"][name]
        shard = tuple(enc.get("shards") or enc["chunks"])
        if lvl > 0:
            return shard
        return copy_region_shape(
            shard,
            template_da.shape,
            template_da.dtype.itemsize,
            source_chunks(self.source[name]),
            max_region_bytes,
        )

    def _region_bytes(self, lvl: int, name: str, max_region_bytes: int) -> int:
        """Approximate bytes held in memory per in-flight region."""
        template_da = self.level_templates[lvl][name]
        region = self._region_shape(lvl, name, max_region_bytes)
        nbytes = math.prod(region) * template_da.dtype.itemsize
        if lvl > 0:
            # the input block is the output region scaled by the per-step stride
            # along each spatial axis the variable carries (step*step for a
            # 2-D coarsening window, step for a variable over one spatial dim)
            step = self._step(lvl)
            nbytes *= step ** sum(
                d in (self.x_dim, self.y_dim) for d in template_da.dims
            )
        return nbytes

    def _region_count(self, lvl: int, name: str, max_region_bytes: int) -> int:
        template_da = self.level_templates[lvl][name]
        region = self._region_shape(lvl, name, max_region_bytes)
        return math.prod(math.ceil(n / r) for n, r in zip(template_da.shape, region))

    def _tile_shape(self, name: str, k: int, max_region_bytes: int) -> tuple[int, ...]:
        """Level-0 tile covering whole shards of levels 0..k for one variable."""
        da0 = self.level_templates[0][name]
        tile = []
        for d, (dim, n) in enumerate(zip(da0.dims, da0.shape)):
            t = 1
            for j in range(k + 1):
                enc = self.encoding[f"/{j}"][name]
                f = self.factors[j] if dim in (self.x_dim, self.y_dim) else 1
                t = math.lcm(t, (enc.get("shards") or enc["chunks"])[d] * f)
            tile.append(min(t, n))
        return copy_region_shape(
            tuple(tile),
            da0.shape,
            da0.dtype.itemsize,
            source_chunks(self.source[name]),
            max_region_bytes,
        )

    def _tile_plan(
        self,
        write_levels: list[int],
        coarsened_vars: list[str],
        max_region_bytes: int,
        max_workers: int | None,
    ) -> tuple[int, dict[str, tuple[int, ...]]]:
        """Deepest level k written in the tile pass, and per-variable tiles.

        k is the largest level such that levels 0..k are all being written,
        each tile fits ``max_region_bytes``, and there are at least as many
        tiles as workers. Returns -1 when level 0 is not written.
        """
        if not coarsened_vars or 0 not in write_levels:
            return -1, {}
        top = 0
        while top + 1 in write_levels:
            top += 1
        for k in range(top, 0, -1):
            tiles = {
                n: self._tile_shape(n, k, max_region_bytes) for n in coarsened_vars
            }
            nbytes = {
                n: math.prod(t) * self.level_templates[0][n].dtype.itemsize
                for n, t in tiles.items()
            }
            workers = max_workers or default_max_workers(max(nbytes.values()))
            count = sum(
                math.prod(
                    math.ceil(s / t)
                    for s, t in zip(self.level_templates[0][n].shape, tile)
                )
                for n, tile in tiles.items()
            )
            if max(nbytes.values()) <= max_region_bytes and count >= workers:
                return k, tiles
        return 0, {n: self._tile_shape(n, 0, max_region_bytes) for n in coarsened_vars}

    def write(
        self,
        store: Any,
        *,
        mode: Literal["w", "w-", "a"] = "w",
        max_workers: int | None = None,
        levels: list[int] | None = None,
        max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
        progress: bool = False,
        stats: bool = False,
        keep_levels_in_memory: bool | None = None,
    ) -> dict[str, Any] | None:
        """Compute and write pyramid levels to a Zarr store.

        The source is read once, in level-0 tiles sized to cover whole shards
        of levels 0..k; each tile is written to level 0, then reduced and
        written to levels 1..k by the Rust kernel on a thread pool. Levels
        above k (whose shards span more than one tile) are block-reduced from
        the previously written level. Variables are processed in parallel on
        a shared pool. For bounded memory on large stores, open the source
        lazily (e.g. ``xr.open_zarr(store, chunks=None)``).

        Args:
            store: Anything zarr-python accepts — a local path,
                ``ObjectStore``, or icechunk session store.
            mode: Zarr open mode for the root group. Use ``"a"`` when
                writing a subset of levels so the root group and any
                pre-existing levels are preserved; ``"w"`` with a levels
                subset raises if the store already holds data (truncation
                would delete the levels not being rewritten).
            max_workers: Thread pool size for tile/region processing. ``None``
                derives a default from the CPU count and available memory
                (peak memory is roughly ``max_workers * 5 * region_bytes``).
                The tile pass needs at least this many tiles; with fewer, k
                is lowered until there are enough.
            levels: Subset of levels to write (e.g. ``[1, 2]``).
                Defaults to all levels. Each coarsened level reads its
                predecessor, so level ``N > 0`` must have level ``N - 1``
                either in the subset or already present in the store.
            max_region_bytes: Memory budget per level-0 tile. Tiles are
                widened to cover whole source chunks when that fits the
                budget, so each source chunk is read once.
            progress: Show a tqdm progress bar over written tiles/regions
                (requires ``tqdm``).
            stats: Collect and return per-level timing stats: region shapes,
                worker count, wall time, and cumulative per-region
                read/reduce/write seconds (summed across threads). The tile
                pass is reported as one entry under level k, its
                ``region_shapes`` being the level-0 tile shapes.
            keep_levels_in_memory: Deprecated and ignored. The tile pass
                produces levels 0..k without re-reads or level buffers.
        Examples:
            Write all levels to a local store:

            ```python
            pyramid.write("pyramid.zarr")
            ```

            Rewrite the coarsened levels, preserving level 0:

            ```python
            pyramid.write("pyramid.zarr", mode="a", levels=[1, 2])
            ```
        """
        # re-checked here, not just in __post_init__: method is a plain field,
        # so `pyramid.method = "median"` after planning would otherwise reach
        # the kernel only on the first coarsened level, with level 0 written
        validate_method(str(self.method))

        if levels is not None:
            invalid = sorted(set(levels) - set(self.level_templates))
            if invalid:
                raise ValueError(
                    f"invalid levels {invalid}; pyramid has levels 0-{self.levels - 1}"
                )

        write_levels = (
            list(range(self.levels)) if levels is None else sorted(set(levels))
        )
        coarsened_vars = self._coarsened_vars()

        if mode == "w" and set(write_levels) != set(self.level_templates):
            # mode="w" truncates the store, so a partial write over existing
            # data would silently delete the levels not being rewritten
            try:
                zarr.open_group(store, mode="r", zarr_format=3)
                has_root = True
            except (FileNotFoundError, zarr.errors.GroupNotFoundError):
                has_root = False
            if has_root:
                raise ValueError(
                    f"levels={write_levels} with mode='w' would truncate the "
                    "store, deleting the levels not being rewritten; pass "
                    "mode='a' to preserve them"
                )

        if keep_levels_in_memory is not None:
            warnings.warn(
                "keep_levels_in_memory is deprecated and ignored: levels are "
                "written from shared level-0 tiles without re-reads",
                DeprecationWarning,
                stacklevel=2,
            )

        tile_k, tiles = self._tile_plan(
            write_levels, coarsened_vars, max_region_bytes, max_workers
        )
        tile_dsts: dict[str, list[zarr.Array]] = {n: [] for n in coarsened_vars}
        write_levels_set = set(write_levels)

        pbar = None
        on_region: Callable[[], None] | None = None
        if progress:
            total = sum(
                math.prod(
                    math.ceil(n / t)
                    for n, t in zip(self.level_templates[0][name].shape, tiles[name])
                )
                if lvl == tile_k
                else self._region_count(lvl, name, max_region_bytes)
                for lvl in write_levels
                if lvl >= tile_k
                for name in coarsened_vars
            )
            pbar = _progress_bar(total)
            on_region = pbar.update

        root = zarr.open_group(store, mode=mode, zarr_format=3)
        for lvl in write_levels:
            if lvl == 0 or (lvl - 1) in write_levels_set:
                continue
            missing = [n for n in coarsened_vars if f"{lvl - 1}/{n}" not in root]
            if missing:
                raise ValueError(
                    f"level {lvl} is coarsened from level {lvl - 1}, which is "
                    f"neither in the write plan nor in the store (missing "
                    f"arrays: {missing}); include level {lvl - 1} in 'levels' "
                    "or write it first"
                )
        root.attrs.update(self.attrs)

        all_stats: dict[str, Any] = {}
        pipeline = _codec_pipeline(store)
        pipeline.__enter__()
        t_level = perf_counter()
        timer = None
        try:
            for lvl in write_levels:
                # the tile pass (levels 0..tile_k) is timed as one level
                if lvl == 0 or lvl > tile_k:
                    t_level = perf_counter()
                    timer = RegionTimer() if stats else None
                template = self.level_templates[lvl]
                # coords + non-spatial vars + level attrs via xarray
                side = template.drop_vars(coarsened_vars, errors="ignore")
                level_enc = self.encoding.get(f"/{lvl}", {})
                side.to_zarr(
                    store,
                    group=str(lvl),
                    mode="a",
                    zarr_format=3,
                    consolidated=False,
                    encoding={
                        name: enc
                        for name, enc in level_enc.items()
                        if name in side.variables
                    },
                )
                root[str(lvl)].attrs.update(level_proj_attrs(self.attrs["proj:code"]))
                if not coarsened_vars:
                    continue
                level_group = cast(zarr.Group, root[str(lvl)])

                if lvl <= tile_k:
                    for name in coarsened_vars:
                        tile_dsts[name].append(self._create_dst(level_group, lvl, name))
                    if lvl < tile_k:
                        continue

                workers = max_workers
                if workers is None:
                    workers = default_max_workers(
                        max(
                            math.prod(tiles[name])
                            * self.level_templates[0][name].dtype.itemsize
                            if lvl == tile_k
                            else self._region_bytes(lvl, name, max_region_bytes)
                            for name in coarsened_vars
                        )
                    )

                with ThreadPoolExecutor(workers) as ex:
                    futures = [
                        future
                        for name in coarsened_vars
                        for future in (
                            write_tiles(
                                self.source[name].variable,
                                tile_dsts[name],
                                [()]
                                + [self._stride(j, name) for j in range(1, lvl + 1)],
                                tile_shape=tiles[name],
                                method=self.method,
                                fill_value=_to_python(self.fill_values.get(name)),
                                executor=ex,
                                on_region=on_region,
                                timer=timer,
                            )
                            if lvl == tile_k
                            else self._write_var(
                                root,
                                level_group,
                                lvl,
                                name,
                                executor=ex,
                                on_region=on_region,
                                timer=timer,
                            )
                        )
                    ]
                    for future in futures:
                        future.result()

                if timer is not None:
                    all_stats[str(lvl)] = {
                        "workers": workers,
                        "region_shapes": {
                            name: tiles[name]
                            if lvl == tile_k
                            else self._region_shape(lvl, name, max_region_bytes)
                            for name in coarsened_vars
                        },
                        "wall_s": round(perf_counter() - t_level, 3),
                        **timer.as_dict(),
                    }
        finally:
            pipeline.__exit__(None, None, None)
            if pbar is not None:
                pbar.close()
        return all_stats if stats else None

    def _write_var(
        self,
        root: zarr.Group,
        level_group: zarr.Group,
        lvl: int,
        name: str,
        *,
        executor: ThreadPoolExecutor,
        on_region: Callable[[], None] | None,
        timer: RegionTimer | None = None,
    ) -> list[Future[None]]:
        """Block-reduce level ``lvl - 1`` of ``name`` into level ``lvl``."""
        return downsample_level(
            cast(zarr.Array, root[f"{lvl - 1}/{name}"]),
            self._create_dst(level_group, lvl, name),
            stride=self._stride(lvl, name),
            method=self.method,
            fill_value=_to_python(self.fill_values.get(name)),
            executor=executor,
            on_region=on_region,
            timer=timer,
        )

    def _stride(self, lvl: int, name: str) -> tuple[int, ...]:
        """Per-axis stride coarsening ``lvl-1`` into ``lvl`` for one variable."""
        step = self._step(lvl)
        return tuple(
            step if d in (self.x_dim, self.y_dim) else 1
            for d in self.level_templates[lvl][name].dims
        )

    def _create_dst(self, level_group: zarr.Group, lvl: int, name: str) -> zarr.Array:
        template_da = self.level_templates[lvl][name]
        source_da = self.source[name]
        fill = _to_python(self.fill_values.get(name))

        attrs = _to_python(dict(template_da.attrs))
        extra_coords = [str(c) for c in source_da.coords if c not in source_da.dims]
        if extra_coords:
            attrs["coordinates"] = " ".join(extra_coords)

        enc = self.encoding[f"/{lvl}"][name]
        return level_group.create_array(
            name=name,
            shape=template_da.shape,
            dtype=template_da.dtype,
            chunks=enc["chunks"],
            shards=enc.get("shards"),
            dimension_names=[str(d) for d in template_da.dims],
            attributes=attrs,
            fill_value=fill,
            overwrite=True,
        )

    def _fill_of(self, name: str) -> float | int | None:
        """The variable's fill value, or None when there is nothing to mask.

        A NaN fill is reported as None: ``xarray.coarsen`` already skips NaN
        for float dtypes, so masking and restoring it would be a no-op.
        """
        fill = self.fill_values.get(name)
        if fill is None or (isinstance(fill, float) and math.isnan(fill)):
            return None
        return fill

    def _is_coarsened(self, da: xr.DataArray) -> bool:
        """True for numeric variables a coarsen actually touches.

        Variables over neither spatial dim pass through unchanged, and a
        non-numeric one (labels, datetimes) cannot be promoted to f8 at all.
        """
        return bool({self.x_dim, self.y_dim} & set(da.dims)) and np.issubdtype(
            da.dtype, np.number
        )

    def _prepare(self, ds: xr.Dataset) -> xr.Dataset:
        """Mask fill values to NaN and promote to f8 before coarsening.

        ``xarray.coarsen`` has no notion of ``_FillValue``; without the mask it
        averages the sentinel in as data, while the kernel skips it. The f8
        promotion matches the kernel's accumulator, so f4 input agrees to the
        bit instead of drifting by a ULP.
        """
        return ds.assign(
            {
                str(name): (
                    da if (f := self._fill_of(str(name))) is None else da.where(da != f)
                ).astype("f8")
                for name, da in ds.data_vars.items()
                if self._is_coarsened(da)
            }
        )

    def _restore(self, ds: xr.Dataset, dtypes: dict[str, Any]) -> xr.Dataset:
        """Undo the masking and the f8 promotion, matching the kernel's cast.

        Order matters: ``fillna`` must precede the cast, since NaN cannot
        survive into an integer dtype, and the clip must too -- the kernel
        saturates an out-of-range accumulator (an integer ``sum``) where a bare
        numpy cast would wrap. Casting back also keeps ``self.encoding`` (sized
        from the source itemsize) correct for this path.
        """
        restored = {}
        for name, da in ds.data_vars.items():
            if not self._is_coarsened(self.source[name]):
                continue
            dtype = dtypes[str(name)]
            fill = self._fill_of(str(name))
            if fill is not None:
                da = da.fillna(fill)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                da = da.clip(info.min, info.max)
            restored[str(name)] = da.astype(dtype)
        return ds.assign(restored)

    def _coarsen_chain(self) -> list[xr.Dataset]:
        """Lazily-chained coarsened datasets, one per level (xarray.coarsen).

        Each level coarsens the previous one by the per-step ratio
        ``factors[i] // factors[i-1]`` along both spatial dims, then restores
        the source dtype and fill value so the values match ``write``.

        Every variable is promoted to ``f8`` for the duration of each coarsen
        (matching the kernel's accumulator; 8x memory on ``u1``) and cast
        straight back -- the promotion is never materialized on the source.
        """
        if self.method != "nearest" and self.method not in XR_COARSEN_METHODS:
            raise NotImplementedError(
                f"as_datatree cannot express method {self.method!r}: xarray.coarsen "
                "has no such reduction. Use Pyramid.write, which runs the kernel."
            )
        dtypes = {str(n): da.dtype for n, da in self.source.data_vars.items()}
        ds_chain: list[xr.Dataset] = [self.source]
        for lvl in range(1, self.levels):
            step = self._step(lvl)
            prev = ds_chain[-1]
            if self.method == "nearest":
                coarsened = self._decimate(prev, step)
            else:
                coarsened = getattr(
                    self._prepare(prev).coarsen(
                        {self.x_dim: step, self.y_dim: step}, boundary="trim"
                    ),
                    self.method,
                )()
                coarsened = self._restore(coarsened, dtypes)
            ds_chain.append(coarsened)
        return ds_chain

    def _decimate(self, ds: xr.Dataset, step: int) -> xr.Dataset:
        """Corner-pick every ``step``-th cell (xarray.coarsen has no nearest).

        Data is strided over floor(n/step) windows to match trim semantics.
        Spatial coords are replaced by their window means so they stay cell
        centers, matching the level templates written by ``write``.
        """
        sel = {
            dim: slice(0, (ds.sizes[dim] // step) * step, step)
            for dim in (self.x_dim, self.y_dim)
        }
        out = ds.isel(sel)
        coords = {}
        for name, coord in ds.coords.items():
            if coord.ndim == 1 and coord.dims[0] in (self.x_dim, self.y_dim):
                mean = coord.coarsen({coord.dims[0]: step}, boundary="trim").mean()
                coords[str(name)] = mean.assign_attrs(coord.attrs)
        return out.assign_coords(coords)

    def as_datatree(self) -> xr.DataTree:
        """Return a lazy DataTree with all pyramid levels coarsened via xarray.

        Each level is produced by chaining ``xarray.coarsen`` operations on the
        source dataset. If the source is Dask-backed, the returned tree is fully
        lazy — suitable for writing on a Dask distributed cluster or with
        icechunk. Use ``self.encoding`` (already shaped for ``DataTree.to_zarr``)
        to apply the recommended chunks and shards:

        ```python
        dt = pyramid.as_datatree()
        dt.to_zarr(store, zarr_format=3, consolidated=False,
                   encoding=pyramid.encoding)
        ```

        Deep levels of a Dask-backed source can outrun the chunk band that
        ``recommend_encoding`` flexes to (see its ``Note``); if ``to_zarr``
        raises on ``safe_chunks``, pass ``safe_chunks=False``.

        Values match [write][topozarr.pyramid.Pyramid.write] exactly, source
        dtype and ``_FillValue`` included, at the cost of an ``f8`` intermediate
        through each coarsen. The exception is an ``f8`` source, where the two
        differ by under 1 ULP on `mean`/`sum` (window summation order).

        Raises:
            NotImplementedError: If ``method`` has no ``xarray.coarsen``
                equivalent. Use `write` for those.
        """
        ds_chain = self._coarsen_chain()

        root_ds = xr.Dataset(attrs=self.attrs)
        proj = level_proj_attrs(self.attrs["proj:code"])
        children = {
            str(lvl): xr.DataTree(ds_chain[lvl].assign_attrs(proj))
            for lvl in range(self.levels)
        }
        return xr.DataTree(root_ds, children=children)
