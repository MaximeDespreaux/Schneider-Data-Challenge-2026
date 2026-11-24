"""Dev / graded protocol (spec 5.8): windows, purge, the one-shot graded-run guard, run manifests.

- DEV origins 2010-01 .. 2021-12, scored only where target_month < 2022-01 (purge: no dev score touches the
  graded period). All tuning and debugging uses DEV.
- GRADED origins 2022-01 .. 2026-09, scored once, from a tagged clean commit whose canaries passed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.data import OUT, ROOT

DEV = (pd.Timestamp("2010-01-01"), pd.Timestamp("2021-12-01"))
GRADED = (pd.Timestamp("2022-01-01"), pd.Timestamp("2026-09-01"))
PURGE = pd.Timestamp("2022-01-01")
CODE_PATHS = ["src", "config/availability.yaml", "config/business.yaml", "config/assumptions.example.yaml", "tests",
              "pyproject.toml", "uv.lock"]
NOT_PUBLISHABLE_PREFIXES = ("L0", "LIC_", "EPX_", "X_", "EP0", "EL0", "BR0")


def dev_view(scored: pd.DataFrame) -> pd.DataFrame:
    """DEV rows only: origin in DEV and target strictly before the graded period."""
    graded = scored["origin"] >= PURGE
    assert not graded.any(), "graded origins must not be scored in a dev run"
    keep = (scored["origin"] >= DEV[0]) & (scored["target_month"] < PURGE)
    return scored[keep]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def git_head(root: Path = ROOT) -> str:
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def git_dirty(root: Path = ROOT) -> bool:
    paths = [p for p in CODE_PATHS if (Path(root) / p).exists()] or ["."]
    return bool(_git(root, "status", "--porcelain", "--", *paths).stdout.strip())


def git_tag(root: Path = ROOT) -> str | None:
    r = _git(root, "describe", "--exact-match", "--tags", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def assert_can_grade(root: Path = ROOT, canary_dir: Path = OUT / "canaries",
                     graded_dir: Path = OUT / "backtest" / "graded") -> str:
    """Return the tag if the graded run may start, else raise RuntimeError naming the failed check."""
    root = Path(root)
    tag = git_tag(root)
    if tag is None:
        raise RuntimeError("graded run refused: HEAD has no exact tag (tag the frozen commit first)")
    if git_dirty(root):
        raise RuntimeError("graded run refused: the tree is not clean (commit or stash code/config changes)")
    rep = Path(canary_dir) / f"{git_head(root)}.json"
    ok = rep.exists() and (r := json.loads(rep.read_text())).get("PASS") and not r.get("dirty")
    if not ok:
        raise RuntimeError(f"graded run refused: no passing canary report for HEAD on a clean tree ({rep})")
    if (Path(graded_dir) / tag).exists():
        raise RuntimeError(f"graded run refused: tag {tag} was already graded (scored once, {graded_dir / tag})")
    return tag


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(out_dir: Path, mode: str, origins, runtime_s: float, inputs: list[Path], configs: list[Path],
                   use_licensed: bool) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    origins = pd.DatetimeIndex(origins)
    pk = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    m = {"git_head": git_head(), "dirty": git_dirty(), "tag": git_tag(),
         "config_sha256": {str(Path(c).name): _sha(c) for c in configs if Path(c).exists()},
         "input_sha256": {str(Path(f).name): _sha(f) for f in inputs if Path(f).exists()},
         "mode": mode, "origins": [f"{origins.min():%Y-%m}", f"{origins.max():%Y-%m}", len(origins)],
         "runtime_s": round(float(runtime_s), 1), "python": platform.python_version(), "packages": pk,
         "use_licensed": bool(use_licensed), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(m, indent=2))
    return path


def assert_publishable_ids(ids) -> None:
    """The submission CSV may only carry Schneider's series and the BOM indices: no licensed / internal ids."""
    bad = sorted({i for i in ids if str(i).startswith(NOT_PUBLISHABLE_PREFIXES)})
    if bad:
        raise ValueError(f"not publishable in the submission CSV (licensed or internal series): {bad}")
