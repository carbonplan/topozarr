"""Build demo datasets for zarr-layer testing and write them to S3.

Outputs written to s3://carbonplan-share/zarr-layer-examples/pipeline/ by default:

Single-level:
  single_level_zarr_v2.zarr                    - Zarr v2 with consolidated metadata
  single_level_zarr_v3.zarr                    - Zarr v3
  single_level_zarr_v3_sharded.zarr            - Zarr v3 with sharding
  single_level_netcdf.nc                       - NetCDF (source for virtual datasets)
  single_level_icechunk.icechunk               - Icechunk
  single_level_virtual_icechunk.icechunk       - Virtual Icechunk backed by netcdf

Multi-level (pyramid):
  multi_level_zarr_v3_sharded.zarr             - Zarr v3 multiscale with sharding
  multi_level_icechunk.icechunk                - Icechunk multiscale
  multi_level_virtual_hybrid_icechunk.icechunk - Hybrid: virtual level 0 + materialized overviews

Usage:
  uv run --extra tutorial scripts/build_demo_data.py           # run all
  uv run --extra tutorial scripts/build_demo_data.py zarr-v2 zarr-v3
  uv run --extra tutorial scripts/build_demo_data.py --bucket my-bucket --prefix my-prefix/
  uv run --extra tutorial scripts/build_demo_data.py --help

AWS credentials must be available in the environment (AWS_* env vars or ~/.aws).
Note: 'netcdf' must run before 'virtual-icechunk' or 'multi-virtual-hybrid-icechunk'.
"""

import tempfile
from dataclasses import dataclass

import boto3
import click
import icechunk
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
import zarr
from cloudpathlib import S3Path
from obstore.store import from_url
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from zarr.storage import ObjectStore

from topozarr.coarsen import create_pyramid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE_ZARR = (
    "s3://nasa-power/imerg/temporal/power_imerg_climatology_temporal_utc.zarr/"
)

VAR = "IMERG_PRECTOT"
CHUNK_SIZE = {"lat": 300, "lon": 600}
SHARD_SIZE = {"lat": 1800, "lon": 3600}


@dataclass
class Config:
    bucket: str
    prefix: str
    region: str

    @property
    def base(self) -> S3Path:
        return S3Path(f"s3://{self.bucket}/{self.prefix}")

    @property
    def single_zarr_v2(self) -> S3Path:
        return self.base / "single_level_zarr_v2.zarr"

    @property
    def single_zarr_v3(self) -> S3Path:
        return self.base / "single_level_zarr_v3.zarr"

    @property
    def single_zarr_v3_sharded(self) -> S3Path:
        return self.base / "single_level_zarr_v3_sharded.zarr"

    @property
    def single_netcdf(self) -> S3Path:
        return self.base / "single_level_netcdf.nc"

    @property
    def single_icechunk(self) -> S3Path:
        return self.base / "single_level_icechunk.icechunk"

    @property
    def single_virtual_icechunk(self) -> S3Path:
        return self.base / "single_level_virtual_icechunk.icechunk"

    @property
    def multi_zarr_v3_sharded(self) -> S3Path:
        return self.base / "multi_level_zarr_v3_sharded.zarr"

    @property
    def multi_icechunk(self) -> S3Path:
        return self.base / "multi_level_icechunk.icechunk"

    @property
    def multi_virtual_hybrid_icechunk(self) -> S3Path:
        return self.base / "multi_level_virtual_hybrid_icechunk.icechunk"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aws_credentials() -> dict:
    """Resolve AWS credentials via boto3's full credential chain (env, ~/.aws, profiles)."""
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    kwargs = {
        "aws_access_key_id": creds.access_key,
        "aws_secret_access_key": creds.secret_key,
    }
    if creds.token:
        kwargs["aws_session_token"] = creds.token
    return kwargs


def zarr_store(path: S3Path, cfg: Config, anonymous: bool = False) -> ObjectStore:
    """Zarr ObjectStore backed by obstore for an S3 URL."""
    if anonymous:
        return ObjectStore(from_url(str(path), region=cfg.region, skip_signature=True))
    return ObjectStore(from_url(str(path), region=cfg.region, **_aws_credentials()))


def load_source(cfg: Config) -> xr.Dataset:
    click.echo("Loading source dataset...")
    return (
        xr.open_dataset(
            zarr_store(S3Path(SOURCE_ZARR), cfg, anonymous=True),
            engine="zarr",
            chunks={},
        )
        .drop_encoding()
        .isel(time=0)
        .drop_vars("time")
    )


def sharded_encoding(ds: xr.Dataset) -> dict:
    dims = ds[VAR].dims
    return {
        VAR: {
            "chunks": tuple(CHUNK_SIZE[d] for d in dims),
            "shards": tuple(SHARD_SIZE[d] for d in dims),
        }
    }


def netcdf_chunk_encoding(ds: xr.Dataset) -> dict:
    dims = ds[VAR].dims
    return {VAR: {"chunksizes": tuple(CHUNK_SIZE[d] for d in dims)}}


def virtual_chunk_config(cfg: Config) -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix=f"s3://{cfg.bucket}/",
            store=icechunk.s3_store(region=cfg.region, anonymous=True),
        )
    )
    return config


def open_virtual_netcdf(cfg: Config) -> xr.Dataset:
    store = from_url(f"s3://{cfg.bucket}/", region=cfg.region, skip_signature=True)
    registry = ObjectStoreRegistry({f"s3://{cfg.bucket}/": store})
    return open_virtual_dataset(
        url=str(cfg.single_netcdf),
        parser=HDFParser(),
        loadable_variables=["lat", "lon"],
        registry=registry,
    )


# ---------------------------------------------------------------------------
# Single-level writers
# ---------------------------------------------------------------------------


def write_single_zarr_v2(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing single_level_zarr_v2...")
    ds.chunk(CHUNK_SIZE).to_zarr(
        zarr_store(cfg.single_zarr_v2, cfg), consolidated=True, mode="w", zarr_format=2
    )


def write_single_zarr_v3(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing single_level_zarr_v3...")
    ds.chunk(CHUNK_SIZE).to_zarr(zarr_store(cfg.single_zarr_v3, cfg), mode="w")


def write_single_zarr_v3_sharded(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing single_level_zarr_v3_sharded...")
    ds.chunk(SHARD_SIZE).to_zarr(
        zarr_store(cfg.single_zarr_v3_sharded, cfg),
        encoding=sharded_encoding(ds),
        mode="w",
    )


def write_single_netcdf(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing single_level_netcdf...")
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        local_path = f.name
    ds.chunk(CHUNK_SIZE).to_netcdf(
        local_path, encoding=netcdf_chunk_encoding(ds), mode="w"
    )
    bucket_store = from_url(
        f"s3://{cfg.bucket}/", region=cfg.region, **_aws_credentials()
    )
    with open(local_path, "rb") as f:
        bucket_store.put(cfg.single_netcdf.key, f.read())


def write_single_icechunk(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing single_level_icechunk...")
    storage = icechunk.s3_storage(
        bucket=cfg.bucket, prefix=cfg.single_icechunk.key, from_env=True
    )
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    ds.chunk(SHARD_SIZE).to_zarr(
        session.store,
        zarr_format=3,
        encoding=sharded_encoding(ds),
        consolidated=False,
        mode="w",
    )
    session.commit("write single level icechunk")


def write_single_virtual_icechunk(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing single_level_virtual_icechunk...")
    vds = open_virtual_netcdf(cfg)
    storage = icechunk.s3_storage(
        bucket=cfg.bucket, prefix=cfg.single_virtual_icechunk.key, from_env=True
    )
    repo = icechunk.Repository.open_or_create(storage, virtual_chunk_config(cfg))
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store)
    session.commit("write virtual single level icechunk")


# ---------------------------------------------------------------------------
# Multi-level writers
# ---------------------------------------------------------------------------


def build_pyramid(ds: xr.Dataset):
    ds_crs = ds.proj.assign_crs(spatial_ref_crs={"EPSG": 4326})
    return create_pyramid(ds_crs, levels=3, x_dim="lon", y_dim="lat")


def write_multi_zarr_v3_sharded(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing multi_level_zarr_v3_sharded...")
    pyramid = build_pyramid(ds)
    pyramid.dt.to_zarr(
        zarr_store(cfg.multi_zarr_v3_sharded, cfg),
        mode="w",
        encoding=pyramid.encoding,
        zarr_format=3,
    )


def write_multi_icechunk(ds: xr.Dataset, cfg: Config) -> None:
    click.echo("Writing multi_level_icechunk...")
    pyramid = build_pyramid(ds)
    storage = icechunk.s3_storage(
        bucket=cfg.bucket, prefix=cfg.multi_icechunk.key, from_env=True
    )
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    pyramid.dt.to_zarr(
        session.store, mode="w", encoding=pyramid.encoding, consolidated=False
    )
    session.commit("write multi level icechunk")


def write_multi_virtual_hybrid_icechunk(ds: xr.Dataset, cfg: Config) -> None:
    """Hybrid multiscale icechunk store.

    Level 0: virtual, backed by the S3 netcdf file.
    Levels 1-2: materialized coarsened overviews.
    Root: multiscale attrs from topozarr pyramid.
    """
    click.echo("Writing multi_level_virtual_hybrid_icechunk...")
    vds = open_virtual_netcdf(cfg)
    pyramid = build_pyramid(ds)

    storage = icechunk.s3_storage(
        bucket=cfg.bucket, prefix=cfg.multi_virtual_hybrid_icechunk.key, from_env=True
    )
    repo = icechunk.Repository.open_or_create(storage, virtual_chunk_config(cfg))

    # Step 1: write virtual base (highest-res level) into /0 subgroup
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store, group="0")
    session.commit("write virtual base level 0")

    # Step 2: set root-level multiscale attrs
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, zarr_format=3)
    root.attrs.update(pyramid.dt.attrs)
    session.commit("write multiscale attrs")

    # Step 3: write materialized overview levels
    session = repo.writable_session("main")
    for level in ["1", "2"]:
        ds_level = pyramid.dt[f"/{level}"].ds
        ds_level.to_zarr(
            session.store,
            group=level,
            encoding=pyramid.encoding[f"/{level}"],
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
    session.commit("write materialized overviews levels 1 and 2")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Ordered list of (target-name, writer-fn) — ordering matters for dependencies.
# 'netcdf' must precede 'virtual-icechunk' and 'multi-virtual-hybrid-icechunk'.
TARGETS: list[tuple[str, object]] = [
    ("zarr-v2", write_single_zarr_v2),
    ("zarr-v3", write_single_zarr_v3),
    ("zarr-v3-sharded", write_single_zarr_v3_sharded),
    ("netcdf", write_single_netcdf),
    ("icechunk", write_single_icechunk),
    ("virtual-icechunk", write_single_virtual_icechunk),
    ("multi-zarr-v3-sharded", write_multi_zarr_v3_sharded),
    ("multi-icechunk", write_multi_icechunk),
    ("multi-virtual-hybrid-icechunk", write_multi_virtual_hybrid_icechunk),
]

TARGET_NAMES = [name for name, _ in TARGETS]


@click.command()
@click.argument("targets", nargs=-1, type=click.Choice(TARGET_NAMES))
@click.option(
    "--bucket", default="carbonplan-share", show_default=True, help="S3 bucket name"
)
@click.option(
    "--prefix",
    default="zarr-layer-examples/pipeline/",
    show_default=True,
    help="Key prefix within the bucket",
)
@click.option("--region", default="us-west-2", show_default=True, help="AWS region")
def cli(targets: tuple[str, ...], bucket: str, prefix: str, region: str) -> None:
    """Build demo datasets for zarr-layer and write to S3.

    Pass specific TARGETS to run only those outputs, or omit to run all.

    \b
    Available targets (run in this order):
      zarr-v2                        Single-level Zarr v2
      zarr-v3                        Single-level Zarr v3
      zarr-v3-sharded                Single-level Zarr v3 with sharding
      netcdf                         Single-level NetCDF (needed by virtual targets)
      icechunk                       Single-level Icechunk
      virtual-icechunk               Single-level virtual Icechunk backed by netcdf
      multi-zarr-v3-sharded          Multi-level Zarr v3 sharded pyramid
      multi-icechunk                 Multi-level Icechunk pyramid
      multi-virtual-hybrid-icechunk  Hybrid: Virtual level 0 + materialized overviews
    """
    cfg = Config(bucket=bucket, prefix=prefix, region=region)
    selected = set(targets) if targets else set(TARGET_NAMES)
    ds = load_source(cfg)

    for name, fn in TARGETS:
        if name in selected:
            fn(ds, cfg)

    click.echo(f"\nDone. Wrote {len(selected)} dataset(s) to {cfg.base}")


if __name__ == "__main__":
    cli()
