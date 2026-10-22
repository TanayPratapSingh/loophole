"""RlimitSandbox: a subprocess with resource limits, for hosts without Docker.

What it does: a fresh temporary working directory, a scrubbed environment,
CPU, file size, open file, and (where the kernel honours it) address space
limits, its own process group so a timeout kills every child, and on Linux a
private network namespace when `unshare` is permitted.

What it cannot do, measured by `loophole probe` rather than assumed: it shares
the host filesystem and user, so it can read anything the verifier's user can
read, and it can signal its parent. On macOS the address space limit is not
enforced. Use it to grade arithmetic and short functions during training, not
to contain an adversary.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from loophole.sandbox.base import DEFAULT_LIMITS, Limits, RunResult, truncate


def _safe_join(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ValueError(f"file path escapes the sandbox: {rel!r}")
    return p


@lru_cache(maxsize=1)
def _unshare_net_works() -> bool:
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    for args in (["unshare", "--net", "true"], ["unshare", "--user", "--map-root-user", "--net", "true"]):
        try:
            if subprocess.run(args, capture_output=True, timeout=5).returncode == 0:
                _UNSHARE_ARGS[:] = args[:-1]
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False


_UNSHARE_ARGS: list[str] = []


def _limiter(limits: Limits):
    def apply() -> None:
        import resource

        def setl(kind: int, value: int) -> None:
            try:
                resource.setrlimit(kind, (value, value))
            except (ValueError, OSError):
                pass  # not supported on this kernel; the probe suite records the consequence

        setl(resource.RLIMIT_CPU, limits.cpu_s)
        setl(resource.RLIMIT_FSIZE, limits.file_mb * 2**20)
        setl(resource.RLIMIT_NOFILE, 64)
        setl(resource.RLIMIT_CORE, 0)
        if sys.platform == "linux":
            setl(resource.RLIMIT_AS, limits.memory_mb * 2**20 + 256 * 2**20)
    return apply


class RlimitSandbox:
    name = "rlimit"

    def __init__(self, python: str | None = None):
        self.python = python or sys.executable

    def run(self, argv, files, limits=DEFAULT_LIMITS, stdin=None) -> RunResult:
        work = Path(tempfile.mkdtemp(prefix="loophole-"))
        notes = {"network_isolated": False}
        try:
            for rel, content in files.items():
                dest = _safe_join(work, rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
            cmd = [self.python if a in ("python", "python3") else a for a in argv]
            if not limits.network and _unshare_net_works():
                cmd = [*_UNSHARE_ARGS, *cmd]
                notes["network_isolated"] = True
            env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work), "LANG": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TMPDIR": str(work)}
            t0 = time.monotonic()
            proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True, preexec_fn=_limiter(limits))
            timed_out = False
            try:
                out, err = proc.communicate(stdin.encode() if stdin else None, timeout=limits.wall_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                out, err = proc.communicate()
            return RunResult(None if timed_out else proc.returncode, truncate(out, limits.output_bytes),
                             truncate(err, limits.output_bytes), timed_out, time.monotonic() - t0, self.name, notes)
        finally:
            shutil.rmtree(work, ignore_errors=True)
