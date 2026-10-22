"""Execution isolation for verifiers: Docker, an rlimit fallback, and escape probes."""

from loophole.sandbox.base import Limits, RunResult, Sandbox, get_sandbox

__all__ = ["Limits", "RunResult", "Sandbox", "get_sandbox"]
