"""Every number this project reports lives in a run directory with a manifest saying how it was made.

runs/<UTC timestamp>-<label>/manifest.json records the exact command line, the
git commit and whether the tree was dirty, package versions, the host, the
model, and whether any stub component took part. RESULTS.md is rendered from
these directories and never edited by hand.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ["loophole", "numpy", "duckdb", "torch", "transformers", "trl", "peft", "vllm", "datasets", "ray",
            "inspect-ai", "mlx-lm"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def new_run(label: str, root: str | Path = "runs", config: dict | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(root) / f"{stamp}-{label}"
    d.mkdir(parents=True, exist_ok=False)
    manifest = {"label": label, "started_utc": stamp, "argv": sys.argv, "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")), "host": platform.node(),
                "platform": platform.platform(), "versions": versions(), "config": config or {}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return d


def finish(run_dir: Path, **results) -> None:
    m = json.loads((run_dir / "manifest.json").read_text())
    m["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    m.update(results)
    (run_dir / "manifest.json").write_text(json.dumps(m, indent=2, default=str))
