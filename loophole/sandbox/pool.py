"""Grading pools: a thread pool (default) and a Ray actor pool for grading across processes or machines.

Verifiers spend their time waiting on sandbox subprocesses, so threads already
parallelise them on one host. Ray earns its place when grading has to leave
the host, for example verifier actors on CPU nodes while the GPU node trains,
or when a verifier holds state worth keeping warm per actor (an open DuckDB
world, a compiled MBPP+ checker). `loophole bench-pool` measures both on the
same jobs instead of assuming either is faster.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from loophole.core.types import Grade, Task


def _grade_all(verifiers, task: Task, completion) -> dict[str, Grade]:
    out = {}
    for name, v in verifiers.items():
        try:
            out[name] = v.grade(task, completion)
        except Exception as e:
            out[name] = Grade(0.0, False, name, {"reason": f"verifier error: {type(e).__name__}: {e}"})
    return out


class ThreadPool:
    name = "threads"

    def __init__(self, env_name: str, workers: int = 8, env_kwargs: dict | None = None):
        from loophole.core.env import get_env

        self.verifiers = get_env(env_name, **(env_kwargs or {})).verifiers()
        self.ex = ThreadPoolExecutor(workers)

    def grade(self, jobs: Sequence[tuple[Task, object]]) -> list[dict[str, Grade]]:
        return list(self.ex.map(lambda j: _grade_all(self.verifiers, *j), jobs))

    def close(self) -> None:
        self.ex.shutdown()


class RayPool:
    """A fixed set of Ray actors, each holding its own environment and verifiers."""

    name = "ray"

    def __init__(self, env_name: str, actors: int = 8, env_kwargs: dict | None = None, address: str | None = None):
        import ray

        if not ray.is_initialized():
            ray.init(address=address, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)

        @ray.remote(num_cpus=1)
        class Grader:
            def __init__(self, env_name, env_kwargs):
                from loophole.core.env import get_env

                self.verifiers = get_env(env_name, **(env_kwargs or {})).verifiers()

            def grade(self, batch):
                return [_grade_all(self.verifiers, Task.from_dict(t), c) for t, c in batch]

        self.ray = ray
        self.actors = [Grader.remote(env_name, env_kwargs) for _ in range(actors)]

    def grade(self, jobs: Sequence[tuple[Task, object]]) -> list[dict[str, Grade]]:
        from dataclasses import asdict

        n = len(self.actors)
        shards = [[(asdict(t), c) for t, c in jobs[i::n]] for i in range(n)]
        results = self.ray.get([a.grade.remote(s) for a, s in zip(self.actors, shards, strict=True)])
        out: list = [None] * len(jobs)
        for i, res in enumerate(results):
            for j, g in enumerate(res):
                out[i + j * n] = g
        return out

    def close(self) -> None:
        for a in self.actors:
            self.ray.kill(a)


def bench(env_name: str, jobs: Sequence[tuple[Task, object]], workers: int = 8, env_kwargs: dict | None = None,
          backends: Sequence[str] = ("threads", "ray")) -> list[dict]:
    """Grade the same jobs with each pool; report wall time and check the verdicts agree."""
    rows, reference = [], None
    for b in backends:
        t0 = time.monotonic()
        pool = ThreadPool(env_name, workers, env_kwargs) if b == "threads" else RayPool(env_name, workers, env_kwargs)
        setup = time.monotonic() - t0
        t1 = time.monotonic()
        grades = pool.grade(jobs)
        wall = time.monotonic() - t1
        pool.close()
        verdicts = [{k: g.passed for k, g in gs.items()} for gs in grades]
        reference = reference or verdicts
        rows.append({"backend": b, "workers": workers, "jobs": len(jobs), "setup_s": round(setup, 2),
                     "grade_s": round(wall, 2), "grades_per_s": round(len(jobs) / wall, 1),
                     "verdicts_match_first": verdicts == reference})
    return rows
