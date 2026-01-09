from typing import Literal
import xarray as xr
from xarray import DataTree
import xproj  # noqa ignore
from .metadata import create_level_encoding, create_multiscale_metadata, SpecType
from .pyramid import Pyramid
from .chunking import DEFAULT_CHUNK_BYTES, DEFAULT_SHARD_BYTES

CoarseningMethod = Literal["mean", "max", "min", "sum"]


def get_crs(ds: xr.Dataset) -> str:
    crs = ds.proj.crs
    if not crs:
        raise ValueError(
            "dataset is missing a crs. Use xproj. ex: ds.proj.assign_crs({'EPSG':4326)."
        )
    return str(crs)


def build_coarsened_levels(
    ds: xr.Dataset,
    num_levels: int,
    x_dim: str,
    y_dim: str,
    method: CoarseningMethod,
    spec: SpecType = "ndpyramid",
) -> dict[int, xr.Dataset]:
    levels = [ds]
    for lvl in range(num_levels - 1):
        curr = levels[0]
        if curr.sizes[x_dim] < 2 or curr.sizes[y_dim] < 2:
            raise ValueError(f"cannot coarsen {num_levels}")
        coarsened = curr.coarsen({x_dim: 2, y_dim: 2}, boundary="trim")
        levels.insert(0, getattr(coarsened, method)())

    if spec == "ndpyramid":
        return dict(enumerate(levels))
    else:
        return dict(enumerate(reversed(levels)))


def create_pyramid(
    ds: xr.Dataset,
    levels: int,
    x_dim: str = "x",
    y_dim: str = "y",
    method: CoarseningMethod = "mean",
    spec: SpecType = "ndpyramid",
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    target_shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> Pyramid:
    crs_str = get_crs(ds)
    level_datasets = build_coarsened_levels(ds, levels, x_dim, y_dim, method, spec=spec)

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
            target_shard_bytes=target_shard_bytes,
        )

        dim_chunks = {}
        for var_name, var_enc in level_encoding.items():
            if var_name in ds_level.data_vars and "chunks" in var_enc:
                target_shards = var_enc["shards"]
                da = ds_level[var_name]

                for dim, shard_size in zip(da.dims, target_shards):
                    if dim not in dim_chunks:
                        dim_chunks[dim] = shard_size
                    else:
                        dim_chunks[dim] = min(dim_chunks[dim], shard_size)

        if dim_chunks:
            ds_level = ds_level.chunk(dim_chunks)

        dt[path] = DataTree(ds_level, name=name)
        full_encoding[path] = level_encoding

    dt.attrs = create_multiscale_metadata(levels, crs_str, method, spec=spec)
    return Pyramid(datatree=dt, encoding=full_encoding)
