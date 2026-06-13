from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np
import psutil
import xarray as xr
import zarr
from topozarr_core import block_reduce

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

CoarseningMethod = Literal["mean", "max", "min", "sum"]


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


def source_chunks(da: xr.DataArray) -> tuple[int, ...] | None:
    """Per-axis chunk shape of the source backing ``da``, if chunked."""
    if da.chunks is not None:  # dask
        return tuple(c[0] for c in da.chunks)
    enc = da.encoding.get("chunks")  # zarr/icechunk backend
    return tuple(enc) if enc is not None else None


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
    fill_values: dict[str, float | int | None] = field(default_factory=dict)

    @property
    def levels(self) -> int:
        return len(self.level_templates)

    def _spatial_vars(self) -> list[str]:
        return [
            name
            for name, da in self.source.data_vars.items()
            if self.x_dim in da.dims and self.y_dim in da.dims
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
            # the input block is the output region scaled by the 2x2 stride
            nbytes *= 4
        return nbytes

    def _region_count(self, lvl: int, name: str, max_region_bytes: int) -> int:
        template_da = self.level_templates[lvl][name]
        region = self._region_shape(lvl, name, max_region_bytes)
        return math.prod(math.ceil(n / r) for n, r in zip(template_da.shape, region))

    def _compute_use_fusion(
        self,
        write_levels: list[int],
        spatial_vars: list[str],
        max_region_bytes: int,
        keep: bool | None,
    ) -> bool:
        """Return True if level-pipelining (fused reduce) should be used.

        Fusion keeps each written level in RAM so the next level is produced
        during the write pass instead of being re-read from the store.
        """
        if keep is False or not spatial_vars or len(write_levels) < 2:
            return False
        base_lvl = write_levels[0]
        if base_lvl not in self.level_templates:
            return False

        nbytes = sum(
            math.prod(self.level_templates[lvl][name].shape)
            * self.level_templates[lvl][name].dtype.itemsize
            for lvl in write_levels[1:]
            for name in spatial_vars
            if lvl in self.level_templates
        )
        max_rb = max(
            self._region_bytes(base_lvl, name, max_region_bytes)
            for name in spatial_vars
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
        mode: str = "w",
        max_workers: int | None = None,
        levels: list[int] | None = None,
        max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
        progress: bool = False,
        stats: bool = False,
        keep_levels_in_memory: bool | None = None,
        io: Literal["python", "rust"] = "python",
        engine: Literal["native", "xarray"] = "native",
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
                pre-existing levels are preserved.
            max_workers: Thread pool size for region processing. ``None``
                derives a default from the CPU count and available memory
                (peak memory is roughly ``max_workers * 5 * region_bytes``).
            levels: Subset of levels to write (e.g. ``[1, 2]``).
                Defaults to all levels.
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
            io: ``"python"`` (default) writes everything through zarr-python.
                ``"rust"`` encodes and stores spatial-variable regions natively
                in the ``topozarr-core`` kernel (a bundled Rust extension, no
                extra install) -- one shared connection pool, no per-region trip
                through zarr-python's sync bridge. Metadata, coords, and
                non-spatial variables still go through zarr-python. Supports
                local paths, ``s3://`` URLs, ``LocalStore``, and obstore-backed
                ``ObjectStore`` targets. This is unrelated to the optional
                ``zarrs`` codec pipeline (a separate zarr-python codec backend).
            engine: ``"native"`` (default) uses the topozarr-core Rust kernel
                for coarsening — fast on a single machine but incompatible with
                Dask distributed. ``"xarray"`` delegates coarsening to
                ``xarray.DataArray.coarsen`` and writes each level with
                ``Dataset.to_zarr``; this path is Dask-distributed-compatible
                when the source dataset is backed by Dask arrays. When
                ``engine="xarray"``, the ``io``, ``max_workers``,
                ``keep_levels_in_memory``, ``progress``, and ``stats`` kwargs
                are ignored. For full control over the write (custom scheduler,
                icechunk, etc.) use
                [as_datatree][topozarr.pyramid.Pyramid.as_datatree] instead.

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
        if engine not in ("native", "xarray"):
            raise ValueError(f"engine must be 'native' or 'xarray', got {engine!r}")
        if engine == "xarray":
            return self._write_xarray(store, mode=mode, levels=levels)

        if levels is not None:
            invalid = sorted(set(levels) - set(self.level_templates))
            if invalid:
                raise ValueError(
                    f"invalid levels {invalid}; pyramid has levels 0-{self.levels - 1}"
                )

        write_levels = list(range(self.levels)) if levels is None else list(levels)
        spatial_vars = self._spatial_vars()

        if io not in ("python", "rust"):
            raise ValueError(f"io must be 'python' or 'rust', got {io!r}")

        pbar = None
        on_region: Callable[[], None] | None = None
        if progress:
            total = sum(
                self._region_count(lvl, name, max_region_bytes)
                for lvl in write_levels
                for name in spatial_vars
            )
            pbar = _progress_bar(total)
            on_region = pbar.update

        use_fusion = self._compute_use_fusion(
            write_levels, spatial_vars, max_region_bytes, keep_levels_in_memory
        )
        write_levels_set = set(write_levels)
        mem_levels: dict[str, np.ndarray] = {}

        root = zarr.open_group(store, mode=mode, zarr_format=3)
        root.attrs.update(self.attrs)

        rust_writer = None
        if io == "rust":
            from .rust_io import make_rust_writer

            rust_writer = make_rust_writer(store)

        all_stats: dict[str, Any] = {}
        try:
            for lvl in write_levels:
                t_level = perf_counter()
                timer = RegionTimer() if stats else None
                template = self.level_templates[lvl]
                # coords + non-spatial vars + level attrs via xarray
                template.drop_vars(spatial_vars, errors="ignore").to_zarr(
                    store, group=str(lvl), mode="a", zarr_format=3, consolidated=False
                )
                if not spatial_vars:
                    continue
                level_group = root[str(lvl)]

                workers = max_workers
                if workers is None:
                    workers = default_max_workers(
                        max(
                            self._region_bytes(lvl, name, max_region_bytes)
                            for name in spatial_vars
                        )
                    )

                # Pre-allocate next-level buffers for variables eligible for fusion.
                # Eligibility: fusion enabled AND next level exists in the write plan
                # AND this variable is sourced from memory (or we're at level 0)
                # AND each spatial axis of the region shape is even (alignment guard).
                next_mem: dict[str, np.ndarray] = {}
                next_stride: dict[str, tuple[int, ...]] = {}
                if (
                    use_fusion
                    and (lvl + 1) in self.level_templates
                    and (lvl + 1) in write_levels_set
                ):
                    for name in spatial_vars:
                        if lvl > 0 and name not in mem_levels:
                            continue  # no memory source; skip fusion for this var
                        dims = self.level_templates[lvl][name].dims
                        region_shape = self._region_shape(lvl, name, max_region_bytes)
                        spatial_ok = all(
                            region_shape[i] % 2 == 0
                            for i, d in enumerate(dims)
                            if d in (self.x_dim, self.y_dim)
                        )
                        if not spatial_ok:
                            continue
                        next_da = self.level_templates[lvl + 1][name]
                        next_mem[name] = np.empty(next_da.shape, next_da.dtype)
                        next_stride[name] = tuple(
                            2 if d in (self.x_dim, self.y_dim) else 1 for d in dims
                        )

                with ThreadPoolExecutor(workers) as ex:
                    futures = [
                        future
                        for name in spatial_vars
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
                            rust_writer=rust_writer,
                        )
                    ]
                    for future in futures:
                        future.result()

                if rust_writer is not None:
                    rust_writer.flush()

                mem_levels = next_mem

                if timer is not None:
                    all_stats[str(lvl)] = {
                        "workers": workers,
                        "region_shapes": {
                            name: self._region_shape(lvl, name, max_region_bytes)
                            for name in spatial_vars
                        },
                        "wall_s": round(perf_counter() - t_level, 3),
                        **timer.as_dict(),
                    }
        finally:
            if pbar is not None:
                pbar.close()
        if stats and rust_writer is not None:
            # cumulative across all levels; seconds summed over threads/tasks.
            # write_s is worker-thread encode time; on S3, PUTs run async and
            # overlap encode, so put_s is reported separately (not subtracted)
            # and encode_s == write_s.
            rust_stats = dict(rust_writer.stats())
            rust_stats["encode_s"] = round(rust_stats["write_s"], 3)
            all_stats["rust_io"] = rust_stats
        return all_stats if stats else None

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
        rust_writer: Any | None = None,
    ) -> list[Future[None]]:
        template_da = self.level_templates[lvl][name]
        source_da = self.source[name]
        fill = _to_python(self.fill_values.get(name))

        attrs = _to_python(dict(template_da.attrs))
        extra_coords = [c for c in source_da.coords if c not in source_da.dims]
        if extra_coords:
            attrs["coordinates"] = " ".join(extra_coords)

        enc = self.encoding[f"/{lvl}"][name]
        dst = level_group.create_array(
            name=name,
            shape=template_da.shape,
            dtype=template_da.dtype,
            chunks=enc["chunks"],
            shards=enc.get("shards"),
            dimension_names=template_da.dims,
            attributes=attrs,
            fill_value=fill,
            overwrite=True,
        )

        on_block = None
        if next_level_arr is not None and next_level_stride is not None:
            on_block = _make_fused_reduce_hook(
                next_level_arr, next_level_stride, self.method, fill
            )

        write_region = None
        if rust_writer is not None:
            node_path = f"/{dst.path}"

            def write_region(region: Region, block: np.ndarray) -> None:
                rust_writer.write_region(node_path, [s.start for s in region], block)

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
                write_region=write_region,
            )

        stride = tuple(
            2 if d in (self.x_dim, self.y_dim) else 1 for d in template_da.dims
        )
        return downsample_level(
            root[f"{lvl - 1}/{name}"],
            dst,
            stride=stride,
            method=self.method,
            fill_value=fill,
            executor=executor,
            on_region=on_region,
            timer=timer,
            write_region=write_region,
        )

    def _write_xarray(
        self,
        store: Any,
        *,
        mode: str = "w",
        levels: list[int] | None = None,
    ) -> None:
        """Coarsen and write pyramid levels using xarray.coarsen (Dask-compatible).

        Each level is produced by chaining 2x coarsen operations on the source
        dataset, then written via Dataset.to_zarr. Sharding is not applied.
        """
        write_levels = list(range(self.levels)) if levels is None else list(levels)
        if levels is not None:
            invalid = sorted(set(write_levels) - set(self.level_templates))
            if invalid:
                raise ValueError(
                    f"invalid levels {invalid}; pyramid has levels 0-{self.levels - 1}"
                )

        root = zarr.open_group(store, mode=mode, zarr_format=3)
        root.attrs.update(self.attrs)

        # Build lazily-chained coarsened datasets; level 0 = source,
        # level N = coarsen(level N-1, boundary="trim")
        ds_chain: list[xr.Dataset] = [self.source]
        for _ in range(self.levels - 1):
            prev = ds_chain[-1]
            coarsened = getattr(
                prev.coarsen({self.x_dim: 2, self.y_dim: 2}, boundary="trim"),
                self.method,
            )()
            ds_chain.append(coarsened)

        for lvl in write_levels:
            ds_lvl = ds_chain[lvl]
            enc_lvl = self.encoding[f"/{lvl}"]
            zarr_enc = {
                name: enc_lvl[name] for name in ds_lvl.data_vars if name in enc_lvl
            }
            ds_lvl.to_zarr(
                store,
                group=str(lvl),
                mode="a",
                zarr_format=3,
                consolidated=False,
                encoding=zarr_enc if zarr_enc else None,
            )

    def as_datatree(self) -> xr.DataTree:
        """Return a lazy DataTree with all pyramid levels coarsened via xarray.

        Each level is produced by chaining 2x ``xarray.coarsen`` operations on
        the source dataset. If the source is Dask-backed, the returned tree is
        fully lazy and can be written with ``DataTree.to_zarr`` on a Dask
        distributed cluster.

        ``self.encoding`` is already in the format expected by
        ``DataTree.to_zarr``'s ``encoding`` kwarg:

        ```python
        dt = pyramid.as_datatree()
        dt.to_zarr(store, zarr_format=3, consolidated=False,
                   encoding=pyramid.encoding)
        ```
        """
        ds_chain: list[xr.Dataset] = [self.source]
        for _ in range(self.levels - 1):
            prev = ds_chain[-1]
            coarsened = getattr(
                prev.coarsen({self.x_dim: 2, self.y_dim: 2}, boundary="trim"),
                self.method,
            )()
            ds_chain.append(coarsened)

        root_ds = xr.Dataset(attrs=self.attrs)
        children = {str(lvl): xr.DataTree(ds_chain[lvl]) for lvl in range(self.levels)}
        return xr.DataTree(root_ds, children=children)
