from typing import Literal
import xarray as xr
from xarray import DataTree
import xproj  # noqa: F401 - registers .proj accessor
from .metadata import (
    create_level_encoding,
    create_multiscale_metadata,
    ZarrLayerVarConfig,
)
from .pyramid import Pyramid
from .chunking import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CHUNKS_PER_SHARD,
    ChunksPerShard,
    validate_chunks_per_shard,
)

CoarseningMethod = Literal["mean", "max", "min", "sum"]


def get_crs(ds: xr.Dataset) -> str:
    crs = ds.proj.crs
    if not crs:
        raise ValueError(
            "dataset is missing a crs. Assign one with xproj, "
            'e.g. ds.proj.assign_crs(spatial_ref="EPSG:4326").'
        )
    return str(crs)


def build_coarsened_levels(
    ds: xr.Dataset,
    num_levels: int,
    x_dim: str,
    y_dim: str,
    method: CoarseningMethod,
) -> dict[int, xr.Dataset]:
    levels = [ds]
    for lvl in range(num_levels - 1):
        curr = levels[0]
        if curr.sizes[x_dim] < 2 or curr.sizes[y_dim] < 2:
            raise ValueError(
                f"cannot coarsen to {num_levels} levels: after {lvl} step(s) "
                f"dimensions are {x_dim}={curr.sizes[x_dim]}, {y_dim}={curr.sizes[y_dim]}; "
                "both must be >= 2 to coarsen further"
            )
        coarsened = curr.coarsen({x_dim: 2, y_dim: 2}, boundary="trim")
        levels.insert(0, getattr(coarsened, method)())

    # zarr-multiscales: lowest levels = highest resolution (level 0 = highest res)
    return dict(enumerate(reversed(levels)))


def create_pyramid(
    ds: xr.Dataset,
    levels: int,
    x_dim: str = "x",
    y_dim: str = "y",
    method: CoarseningMethod = "mean",
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunks_per_shard: ChunksPerShard | None = DEFAULT_CHUNKS_PER_SHARD,
    layer_hints: dict[str, ZarrLayerVarConfig] | None = None,
) -> Pyramid:
    """Build a multiscale Zarr pyramid from a georeferenced Dataset.

    Args:
        ds: Source dataset.  Must have a CRS assigned via ``ds.proj.assign_crs``.
        levels: Total number of resolution levels, including the original.
            Level ``0`` is the original resolution; each subsequent level
            coarsens by 2× per spatial dimension.
        x_dim: Name of the x (longitude / easting) dimension.
        y_dim: Name of the y (latitude / northing) dimension.
        method: Spatial aggregation method for coarsening.
        target_chunk_bytes: Target uncompressed size per chunk (default ~500 KB).
        chunks_per_shard: Number of chunks per shard along each spatial dimension
            (e.g. ``4`` → 4×4 = 16 chunks per shard, ~8 MB).  Must be a power
            of 2 in the range 1–32.  Pass ``None`` to disable sharding.
        layer_hints: Optional per-variable colormap / color-range hints written
            into the ``zarr-layer`` root metadata key.

    Returns:
        :class:`Pyramid` with ``.dt`` (DataTree) and ``.encoding`` ready to
        pass to ``DataTree.to_zarr(..., encoding=pyramid.encoding)``.
    """
    if chunks_per_shard is not None:
        validate_chunks_per_shard(chunks_per_shard)
    crs_str = get_crs(ds)
    level_datasets = build_coarsened_levels(ds, levels, x_dim, y_dim, method)

    dt = DataTree(name="root")
    full_encoding = {}

    for idx, ds_level in level_datasets.items():
        name = str(idx)
        path = f"/{idx}"

        level_encoding = create_level_encoding(
            ds_level,
            x_dim,
            y_dim,
            target_chunk_bytes=target_chunk_bytes,
            chunks_per_shard=chunks_per_shard,
        )

        dim_chunks = {}
        for var_name, var_enc in level_encoding.items():
            if var_name in ds_level.data_vars and "chunks" in var_enc:
                dask_chunks = var_enc.get("shards", var_enc["chunks"])
                da = ds_level[var_name]

                for dim, chunk_size in zip(da.dims, dask_chunks):
                    if dim not in dim_chunks:
                        dim_chunks[dim] = chunk_size
                    else:
                        dim_chunks[dim] = min(dim_chunks[dim], chunk_size)

        if dim_chunks:
            ds_level = ds_level.chunk(dim_chunks)

        dt[path] = DataTree(ds_level, name=name)
        full_encoding[path] = level_encoding

    dt.attrs = create_multiscale_metadata(
        ds=ds,
        x_dim=x_dim,
        y_dim=y_dim,
        level_datasets=level_datasets,
        crs=crs_str,
        method=str(method),
        layer_hints=layer_hints,
    )

    return Pyramid(datatree=dt, encoding=full_encoding)
