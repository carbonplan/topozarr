<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-dark.svg">
    <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-light.svg" alt="topozarr" width="320">
  </picture>
</p>


### Create multiscale Zarr stores for web visualization

`topozarr` turns an Xarray dataset into a multiscale [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) store that follows the [GeoZarr](https://github.com/zarr-developers/geozarr-spec) conventions. It is built for use with [`zarr-layer`](https://zarr-layer.demo.carbonplan.org/). See the [documentation](https://carbonplan.github.io/topozarr/) for usage and examples.

```bash
uv add topozarr  # or: pip install topozarr
```

## License

MIT. See the [LICENSE](LICENSE) file for details.

## About us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
