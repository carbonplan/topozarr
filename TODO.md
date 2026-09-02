# TODO

Future work, roughly by value:

- CHANGELOG.md: backfill from git tags (Keep a Changelog format), update per
  release.
- `mode` (majority) reduction for categorical rasters — design in
  [planning/mode-coarsening.md](planning/mode-coarsening.md); `median` would ride the same from-native path.
- CI: test on macOS (wheels ship for mac/Windows but only Ubuntu is tested);
  coverage reporting (pytest-cov + codecov).
- Benchmark suite (asv or pytest-benchmark) to guard kernel and streaming
  perf regressions.
- Overwrite/resume semantics for partially written pyramids (e.g. skip
  regions already present when `mode="a"`).
- 2-D spatial coords (e.g. curvilinear `lat(y, x)`) are rejected at
  `create_pyramid` as of 0.1.7. Coarsening them properly (rather than making
  the user drop them) is still open.

Done (2026-07-13): review fixes — `levels` predecessor validation, mode="w"
subset guard, rust writer flush-on-error, int-mean truncation docs, `ty` type
checking in lint CI (+ topozarr_core stubs), ruff import sorting, gitignore
notebooks/.

Done (2026-06-10): smarter max_workers default (RAM/CPU-derived), parallel
variables within a level, `progress=True` tqdm bar, workflow_dispatch wheel
builds, Rust kernel unit tests + cargo test CI, grid/levels/ndim validation,
design docs page, virtualizarr rev pin.
