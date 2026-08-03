# Release notes

## 0.1.3

### Added

- `nearest` coarsening method, alongside `mean`/`max`/`min`/`sum`. Corner-picks
  the top-left cell of each window instead of aggregating, for categorical
  data (class codes, masks) where averaging invents values. Composable across
  levels like the other methods.

---

Versions prior to this point are not documented here — see the
[GitHub releases](https://github.com/carbonplan/topozarr/releases) and
[tags](https://github.com/carbonplan/topozarr/tags) for history.
