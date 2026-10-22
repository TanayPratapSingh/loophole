"""DockerSandbox: one fresh container per call, with no way out that we know of.

Every call gets: no network, a read only root filesystem, a 64 MB tmpfs at
/tmp, memory and swap capped to the same value (so no swapping past the
limit), one CPU, a pids limit against fork bombs, every capability dropped,
no new privileges, and the nobody user. The working directory is a fresh host
temp dir mounted at /work and deleted afterwards. A timed out container is
killed by name, so a runaway program cannot outlive its grade.

A fresh container costs a few hundred milliseconds. That is fine for
evaluation and multi turn episodes and too slow for grading every rollout of
a training step, which is why training uses RlimitSandbox and the probe suite
reports what that trade gives up.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path

from loophole.sandbox.base import Limits, RunResult, truncate
from loophole.sandbox.local import _safe_join

DEFAULT_IMAGE = "python:3.12-slim"


@lru_cache(maxsize=1)
def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox:
    name = "docker"

    def __init__(self, image: str = DEFAULT_IMAGE, startup_slack_s: float = 5.0):
        self.image = image
        self.startup_slack_s = startup_slack_s

    def command(self, name: str, work: Path, argv: list[str], limits: Limits) -> list[str]:
        cmd = ["docker", "run", "--rm", "-i", "--name", name,
               "--read-only", "--tmpfs", "/tmp:rw,size=64m",
               "--memory", f"{limits.memory_mb}m", "--memory-swap", f"{limits.memory_mb}m",
               "--cpus", "1", "--pids-limit", str(limits.pids),
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", "65534:65534",
               "--ulimit", f"fsize={limits.file_mb * 2**20}", "--ulimit", "nofile=64:64",
               "--ulimit", f"cpu={limits.cpu_s}",
               "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0",
               "-v", f"{work}:/work", "-w", "/work"]
        if not limits.network:
            cmd += ["--network", "none"]
        return [*cmd, self.image, *argv]

    def run(self, argv, files, limits=Limits(), stdin=None) -> RunResult:
        work = Path(tempfile.mkdtemp(prefix="loophole-docker-"))
        work.chmod(0o777)
        name = f"loophole-{uuid.uuid4().hex[:12]}"
        try:
            for rel, content in files.items():
                dest = _safe_join(work, rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
                dest.chmod(0o666)
            t0 = time.monotonic()
            timed_out = False
            try:
                p = subprocess.run(self.command(name, work, argv, limits), input=(stdin or "").encode(),
                                   capture_output=True, timeout=limits.wall_s + self.startup_slack_s)
                out, err, code = p.stdout, p.stderr, p.returncode
            except subprocess.TimeoutExpired as e:
                timed_out = True
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
                out, err, code = e.stdout or b"", e.stderr or b"", None
            # 137 is SIGKILL, which is how the kernel OOM killer ends a container
            notes = {"network_isolated": not limits.network, "oom_or_killed": code == 137,
                     "image": self.image, "startup_included": True}
            return RunResult(code, truncate(out, limits.output_bytes), truncate(err, limits.output_bytes),
                             timed_out, time.monotonic() - t0, self.name, notes)
        finally:
            shutil.rmtree(work, ignore_errors=True)
