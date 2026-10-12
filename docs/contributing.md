# Contributing

We welcome contributions. To get started, clone the repository and install the `test` dependency group. Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel (in `core/`):

```bash
git clone https://github.com/carbonplan/topozarr
cd topozarr
uv sync --group test
```

Run the tests:

```bash
uv run pytest -n auto
```

Run the conformance tests against the GeoZarr spec (requires the `conformance` group):

```bash
uv sync --group conformance
uv run pytest -n auto -m conformance
```

The kernel exports its coarsening methods as `topozarr_core.METHODS`. `create_pyramid(method=...)` validates against this list, so if `topozarr` is paired with a kernel that's missing a method, it fails when building the plan rather than partway through a write. A test keeps the `CoarseningMethod` alias in sync with the list.

Lint and format the code:

```bash
uv run prek run --all-files
```

To regenerate the demo datasets on S3 (requires AWS credentials), install the `tutorial` extra and run the build script:

```bash
uv sync --group test --extra tutorial
uv run python scripts/build_demo_data.py --help
```

## License

This code is licensed under the MIT License. See the [LICENSE](https://github.com/carbonplan/topozarr/blob/main/LICENSE) file for details.

## About

CarbonPlan is a nonprofit organization that uses data and science for climate action. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [emailing us](mailto:hello@carbonplan.org).
