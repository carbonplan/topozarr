# Contributing

Clone the repo and install with the `test` dependency group. Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel (in `core/`):

```bash
git clone https://github.com/carbonplan/topozarr
cd topozarr
uv sync --group test
```

Run tests:

```bash
uv run pytest -n auto
```

Run conformance tests against the GeoZarr spec (requires the `conformance` group):

```bash
uv sync --group conformance
uv run pytest -n auto -m conformance
```

Lint and format:

```bash
uv run pre-commit run --all-files
```

To regenerate the demo datasets in S3 (requires AWS credentials), install the `demo` extra and run the build script:

```bash
uv sync --extra demo
uv run python scripts/build_demo_data.py --help
```

## License

This code is licensed under the MIT License — see the [LICENSE](https://github.com/carbonplan/topozarr/blob/main/LICENSE) file for details.

## About

CarbonPlan is a nonprofit organization that uses data and science for climate action. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [emailing us](mailto:hello@carbonplan.org).
