# Hybrid stores

A pyramid doesn't have to duplicate the source data. In a hybrid store, the original files serve as level `0`, and only the coarser levels are written as new data. [VirtualiZarr](https://virtualizarr.readthedocs.io) writes references to the source chunks into an [Icechunk](https://icechunk.io) repository as group `0`, and `write` builds levels `1..N` from it. For background, see [Hybrid Icechunk stores for serverless web mapping](https://cloudnativegeo.org/blog/2026/08/virtual-icechunk-multiscale/).

<div class="tz-figure">
--8<-- "docs/assets/figures/virtual-level0.svg"
</div>

## A NetCDF source

Start with a chunked NetCDF file. This stands in for the archive you already have:

```python exec="on" source="above" session="hybrid"
from pathlib import Path

import xarray as xr
import xproj  # noqa: F401

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds.to_netcdf("air.nc", engine="netcdf4", encoding={"air": {"chunksizes": (100, 25, 53)}})
```

## Write level 0 as references

Open the file as a virtual dataset and write its chunk references to group `0`. The repository needs permission to read chunks from the source location:

```python exec="on" source="above" session="hybrid" result="python"
import icechunk
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

here = Path.cwd()
prefix = here.as_uri() + "/"
vds = open_virtual_dataset(
    (here / "air.nc").as_uri(),
    parser=HDFParser(),
    registry=ObjectStoreRegistry({prefix: LocalStore(prefix=here)}),
    loadable_variables=["lat", "lon", "time"],
)

# let the repo read chunks from files under `prefix`
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(prefix, icechunk.local_filesystem_store(str(here)))
)
repo = icechunk.Repository.create(
    icechunk.local_filesystem_storage("air.icechunk"),
    config=config,
    authorize_virtual_chunk_access={prefix: None},
)
session = repo.writable_session("main")
vds.vz.to_icechunk(session.store, group="0")
session.commit("virtual level 0")
print(vds.air.data)
```

## Write the overviews

Plan the pyramid from the same data, then write only levels `1` and `2` with `mode="a"`. Level `1` is coarsened from the virtual level `0`, and `write` also adds the CRS attributes to it:

```python exec="on" source="above" session="hybrid" result="python"
from topozarr import create_pyramid

pyramid = create_pyramid(
    ds.proj.assign_crs(spatial_ref="EPSG:4326"), levels=3, x_dim="lon", y_dim="lat"
)
session = repo.writable_session("main")
pyramid.write(session.store, mode="a", levels=[1, 2])
session.commit("overviews")

dt = xr.open_datatree(repo.readonly_session("main").store, engine="zarr", consolidated=False)
for level in dt.children:
        print(level, dict(dt[level].sizes), dt[level].attrs["proj:code"])
```

On S3, the steps are the same. Use S3 storage and an S3 virtual chunk container instead.

```python exec="on" session="hybrid"
# hidden: remove the files this page wrote
import shutil

shutil.rmtree("air.icechunk", ignore_errors=True)
Path("air.nc").unlink(missing_ok=True)
```

## Render the hybrid store with zarr-layer

[zarr-layer](https://zarr-layer.demo.carbonplan.org/) can render a hybrid store directly in the browser. It reprojects the data on the fly, so there's no tile server and no duplicate copy of the data. Read more in [Hybrid Icechunk stores for serverless web mapping](https://cloudnativegeo.org/blog/2026/08/virtual-icechunk-multiscale/).

<iframe src="https://icechunk-maps.vercel.app/?view=interactive" title="Interactive snowfall map rendered from a hybrid Icechunk store" loading="lazy" style="width:100%;height:640px;border:0;border-radius:4px"></iframe>

*A hybrid store in the browser, rendered with [zarr-layer](https://github.com/carbonplan/zarr-layer). From [the blog post](https://cloudnativegeo.org/blog/2026/08/virtual-icechunk-multiscale/).*
