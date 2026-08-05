"""Fail if topozarr's topozarr-core pin drifts from the core packages.

topozarr and topozarr-core are separate distributions but one unit: the Python
layer accepts a coarsening method, the Rust kernel implements it. A resolver
given any range is free to pair them across a feature boundary, which fails
mid-write rather than at install. The pin is exact, so it has to be bumped in
lockstep with core -- this is the check that it was.
"""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN = "topozarr-core=="


def main() -> int:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    core = tomllib.loads((ROOT / "core" / "pyproject.toml").read_text())
    cargo = tomllib.loads((ROOT / "core" / "Cargo.toml").read_text())

    pins = [d for d in root["project"]["dependencies"] if d.startswith(PIN)]
    if len(pins) != 1:
        print(f"expected exactly one '{PIN}<version>' dependency, found {pins}")
        return 1

    pinned = pins[0].removeprefix(PIN)
    versions = {
        "pyproject.toml pin": pinned,
        "core/pyproject.toml": core["project"]["version"],
        "core/Cargo.toml": cargo["package"]["version"],
        "pyproject.toml version": root["project"]["version"],
    }

    if len(set(versions.values())) != 1:
        print("topozarr and topozarr-core versions are out of lockstep:")
        for name, version in versions.items():
            print(f"  {name}: {version}")
        return 1

    print(f"lockstep ok: {pinned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
