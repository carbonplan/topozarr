from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import psutil
import xarray as xr
import zarr
import zarr.errors
from topozarr_core import METHODS, block_reduce

from .chunking import source_chunks
from .engine import (
    DEFAULT_MAX_REGION_BYTES,
    REGION_MEM_FACTOR,
    Region,
    RegionTimer,
    copy_array,
    copy_region_shape,
    default_max_workers,
    downsample_level,
)

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


def _make_fused_reduce_hook(
    target: np.ndarray,
    stride: tuple[int, ...],
    method: str,
    fill_value: float | int | None,
) -> Callable[[Region, np.ndarray], None]:
    """Return a per-block callback that reduces ``block`` into ``target``.

    Designed for shard-aligned regions: each ``region`` maps to a disjoint
    slice of ``target``, so no locking is needed across threads.
    """

    def hook(region: Region, block: np.ndarray) -> None:
        out = block_reduce(block, stride, method, fill_value, True)
        # clamp to the target: a trailing region shorter than its stride
        # yields one window from the kernel but zero rows in the global
        # trim, so the extra output must be dropped
        region_out = tuple(
            slice(s.start // f, min(s.start // f + out.shape[i], n))
            for i, (s, f, n) in enumerate(zip(region, stride, target.shape))
        )
        out_trim = tuple(slice(0, r.stop - r.start) for r in region_out)
        target[region_out] = out[out_trim]

    return hook


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

    def _compute_use_fusion(
        self,
        write_levels: list[int],
        coarsened_vars: list[str],
        max_region_bytes: int,
        keep: bool | None,
    ) -> bool:
        """Return True if level-pipelining (fused reduce) should be used.

        Fusion keeps each written level in RAM so the next level is produced
        during the write pass instead of being re-read from the store.
        """
        if keep is False or not coarsened_vars or len(write_levels) < 2:
            return False
        base_lvl = write_levels[0]
        if base_lvl not in self.level_templates:
            return False

        nbytes = sum(
            math.prod(self.level_templates[lvl][name].shape)
            * self.level_templates[lvl][name].dtype.itemsize
            for lvl in write_levels[1:]
            for name in coarsened_vars
            if lvl in self.level_templates
        )
        max_rb = max(
            self._region_bytes(base_lvl, name, max_region_bytes)
            for name in coarsened_vars
        )
        # default_max_workers caps the worker budget at available//2 by
        # construction, so require level buffers + workers to fit in 3/4 of
        # available memory, leaving >= 1/4 headroom. Workers sized after the
        # buffers are allocated see the reduced available memory and shrink
        # accordingly.
        worker_count = default_max_workers(max_rb)
        worker_budget = worker_count * REGION_MEM_FACTOR * max_rb
        budget = max(0, psutil.virtual_memory().available * 3 // 4 - worker_budget)

        if keep is True:
            if nbytes > budget:
                raise MemoryError(
                    f"keep_levels_in_memory=True: need {nbytes / 1e9:.2f} GB but "
                    f"only {budget / 1e9:.2f} GB of memory budget remains"
                )
            return True
        return nbytes <= budget

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

        Level 0 is streamed region by region from the source dataset; each
        subsequent level is block-reduced from the previously written level,
        streaming shard-sized regions through the Rust kernel on a thread
        pool. Levels are written sequentially (each reads the previous one);
        variables within a level are processed in parallel on a shared pool.
        For bounded memory on large stores, open the source lazily (e.g.
        ``xr.open_zarr(store, chunks=None)``).

        Args:
            store: Anything zarr-python accepts — a local path,
                ``ObjectStore``, or icechunk session store.
            mode: Zarr open mode for the root group. Use ``"a"`` when
                writing a subset of levels so the root group and any
                pre-existing levels are preserved; ``"w"`` with a levels
                subset raises if the store already holds data (truncation
                would delete the levels not being rewritten).
            max_workers: Thread pool size for region processing. ``None``
                derives a default from the CPU count and available memory
                (peak memory is roughly ``max_workers * 5 * region_bytes``).
            levels: Subset of levels to write (e.g. ``[1, 2]``).
                Defaults to all levels. Each coarsened level reads its
                predecessor, so level ``N > 0`` must have level ``N - 1``
                either in the subset or already present in the store.
            max_region_bytes: Memory budget per level-0 copy region. Regions
                are widened to cover whole source chunks when that fits the
                budget, so each source chunk is read once.
            progress: Show a tqdm progress bar over written regions
                (requires ``tqdm``).
            stats: Collect and return per-level timing stats: region shapes,
                worker count, wall time, and cumulative per-region
                read/reduce/write seconds (summed across threads).

                With level pipelining active (``keep_levels_in_memory=True``
                or auto-enabled), level N's ``reduce_s`` captures fused-reduce
                time (reducing level-N blocks into the level-N+1 buffer) rather
                than the reduce of level N itself (which is zero when reading
                from memory).  ``read_s = block_s - reduce_s`` remains the
                pure source-read time at every level.
            keep_levels_in_memory: Control level pipelining.  ``None`` (default)
                auto-enables fusion when the higher levels fit in half the
                available RAM after accounting for the worker region budget.
                ``True`` forces fusion and raises ``MemoryError`` if the budget
                is exceeded.  ``False`` disables fusion and always re-reads from
                the store.
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

        pbar = None
        on_region: Callable[[], None] | None = None
        if progress:
            total = sum(
                self._region_count(lvl, name, max_region_bytes)
                for lvl in write_levels
                for name in coarsened_vars
            )
            pbar = _progress_bar(total)
            on_region = pbar.update

        use_fusion = self._compute_use_fusion(
            write_levels, coarsened_vars, max_region_bytes, keep_levels_in_memory
        )
        write_levels_set = set(write_levels)
        mem_levels: dict[str, np.ndarray] = {}

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
        try:
            for lvl in write_levels:
                t_level = perf_counter()
                timer = RegionTimer() if stats else None
                template = self.level_templates[lvl]
                # coords + non-spatial vars + level attrs via xarray
                template.drop_vars(coarsened_vars, errors="ignore").to_zarr(
                    store, group=str(lvl), mode="a", zarr_format=3, consolidated=False
                )
                if not coarsened_vars:
                    continue
                level_group = cast(zarr.Group, root[str(lvl)])

                workers = max_workers
                if workers is None:
                    workers = default_max_workers(
                        max(
                            self._region_bytes(lvl, name, max_region_bytes)
                            for name in coarsened_vars
                        )
                    )

                next_mem, next_stride = self._fusion_buffers(
                    lvl,
                    coarsened_vars,
                    write_levels_set,
                    use_fusion,
                    mem_levels,
                    max_region_bytes,
                )

                with ThreadPoolExecutor(workers) as ex:
                    futures = [
                        future
                        for name in coarsened_vars
                        for future in self._write_var(
                            root,
                            level_group,
                            lvl,
                            name,
                            max_region_bytes,
                            executor=ex,
                            on_region=on_region,
                            timer=timer,
                            mem_source=mem_levels.get(name),
                            next_level_arr=next_mem.get(name),
                            next_level_stride=next_stride.get(name),
                        )
                    ]
                    for future in futures:
                        future.result()

                mem_levels = next_mem

                if timer is not None:
                    all_stats[str(lvl)] = {
                        "workers": workers,
                        "region_shapes": {
                            name: self._region_shape(lvl, name, max_region_bytes)
                            for name in coarsened_vars
                        },
                        "wall_s": round(perf_counter() - t_level, 3),
                        **timer.as_dict(),
                    }
        finally:
            if pbar is not None:
                pbar.close()
        return all_stats if stats else None

    def _fusion_buffers(
        self,
        lvl: int,
        coarsened_vars: list[str],
        write_levels_set: set[int],
        use_fusion: bool,
        mem_levels: dict[str, np.ndarray],
        max_region_bytes: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
        """Pre-allocate next-level buffers for variables eligible for fusion.

        Eligibility: fusion enabled AND next level exists in the write plan
        AND this variable is sourced from memory (or we're at level 0)
        AND each spatial axis of the region shape is even (alignment guard).
        """
        next_mem: dict[str, np.ndarray] = {}
        next_stride: dict[str, tuple[int, ...]] = {}
        if not (
            use_fusion
            and (lvl + 1) in self.level_templates
            and (lvl + 1) in write_levels_set
        ):
            return next_mem, next_stride
        step = self._step(lvl + 1)
        for name in coarsened_vars:
            if lvl > 0 and name not in mem_levels:
                continue  # no memory source; skip fusion for this var
            dims = self.level_templates[lvl][name].dims
            region_shape = self._region_shape(lvl, name, max_region_bytes)
            # guard checks region shape; the fused hook (s.start // f)
            # also needs region starts divisible by step -- safe today
            # because level>0 regions are shard-sized with shape-multiple
            # starts. If unaligned, fusion is skipped here and it falls
            # back to the read-from-prev-level downsample_level path
            # (correct for any stride).
            spatial_ok = all(
                region_shape[i] % step == 0
                for i, d in enumerate(dims)
                if d in (self.x_dim, self.y_dim)
            )
            if not spatial_ok:
                continue
            next_da = self.level_templates[lvl + 1][name]
            next_mem[name] = np.empty(next_da.shape, next_da.dtype)
            next_stride[name] = tuple(
                step if d in (self.x_dim, self.y_dim) else 1 for d in dims
            )
        return next_mem, next_stride

    def _write_var(
        self,
        root: zarr.Group,
        level_group: zarr.Group,
        lvl: int,
        name: str,
        max_region_bytes: int,
        *,
        executor: ThreadPoolExecutor,
        on_region: Callable[[], None] | None,
        timer: RegionTimer | None = None,
        mem_source: np.ndarray | None = None,
        next_level_arr: np.ndarray | None = None,
        next_level_stride: tuple[int, ...] | None = None,
    ) -> list[Future[None]]:
        template_da = self.level_templates[lvl][name]
        source_da = self.source[name]
        fill = _to_python(self.fill_values.get(name))

        attrs = _to_python(dict(template_da.attrs))
        extra_coords = [str(c) for c in source_da.coords if c not in source_da.dims]
        if extra_coords:
            attrs["coordinates"] = " ".join(extra_coords)

        enc = self.encoding[f"/{lvl}"][name]
        dst = level_group.create_array(
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

        on_block = None
        if next_level_arr is not None and next_level_stride is not None:
            on_block = _make_fused_reduce_hook(
                next_level_arr, next_level_stride, self.method, fill
            )

        if lvl == 0 or mem_source is not None:
            values: Any = mem_source if mem_source is not None else source_da.variable
            sc = None if mem_source is not None else source_chunks(source_da)
            return copy_array(
                values,
                dst,
                source_chunks=sc,
                max_region_bytes=max_region_bytes,
                executor=executor,
                on_region=on_region,
                on_block=on_block,
                timer=timer,
            )

        step = self._step(lvl)
        stride = tuple(
            step if d in (self.x_dim, self.y_dim) else 1 for d in template_da.dims
        )
        return downsample_level(
            cast(zarr.Array, root[f"{lvl - 1}/{name}"]),
            dst,
            stride=stride,
            method=self.method,
            fill_value=fill,
            executor=executor,
            on_region=on_region,
            timer=timer,
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
        children = {str(lvl): xr.DataTree(ds_chain[lvl]) for lvl in range(self.levels)}
        return xr.DataTree(root_ds, children=children)
