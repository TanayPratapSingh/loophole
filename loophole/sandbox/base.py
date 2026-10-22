"""What a sandbox promises, and how to pick one.

Two backends implement the same interface. DockerSandbox is the real isolation
boundary and is what local evaluation and the multi turn environments use.
RlimitSandbox is a fallback for places with no Docker daemon, notably Kaggle,
where training runs. It is weaker, and `loophole probe` measures exactly how
much weaker instead of leaving it to a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Limits:
    wall_s: float = 10.0
    cpu_s: int = 10
    memory_mb: int = 512
    pids: int = 64
    file_mb: int = 16
    output_bytes: int = 64_000
    network: bool = False


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    wall_s: float
    backend: str
    notes: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(Protocol):
    name: str

    def run(self, argv: list[str], files: dict[str, str], limits: Limits = Limits(),
            stdin: str | None = None) -> RunResult:
        """Run argv in a fresh working directory holding `files`, under `limits`.

        Paths in `files` are relative to the working directory. Nothing written
        by the program survives the call.
        """
        ...


def get_sandbox(name: str = "auto", **kwargs) -> Sandbox:
    from loophole.sandbox.docker import DockerSandbox, docker_available
    from loophole.sandbox.local import RlimitSandbox

    if name == "auto":
        name = "docker" if docker_available() else "rlimit"
    if name == "docker":
        return DockerSandbox(**kwargs)
    if name == "rlimit":
        return RlimitSandbox(**kwargs)
    raise KeyError(f"unknown sandbox {name!r}")


def truncate(b: bytes, limit: int) -> str:
    s = b[:limit].decode("utf-8", "replace")
    return s + (f"\n[truncated {len(b) - limit} bytes]" if len(b) > limit else "")
