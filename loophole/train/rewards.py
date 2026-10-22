"""TRL reward functions backed by a verifier pair.

The policy is trained on one verifier (usually the leaky one). Every rollout is
graded by both, on the same text, in the same call, and appended to a JSONL
log with the trainer's step. That log is the experiment: the hack curve, the
onset step, and the detector evaluation are all computed from it afterwards,
so nothing about hacking depends on what TRL chooses to log.

Dataset rows carry the full Task as a JSON string in a "task" column; TRL
passes every dataset column to reward functions as keyword arguments.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from loophole.core.env import Environment
from loophole.core.types import Grade, Rollout, Task


def completion_text(c: Any) -> str | list:
    """TRL hands conversational completions as a list of messages; plain ones as a string."""
    if isinstance(c, list):
        assistant = [m for m in c if m.get("role") == "assistant"]
        return assistant[-1]["content"] if len(c) == len(assistant) == 1 else c
    return c


class VerifierReward:
    def __init__(self, env: Environment, train_on: str = "leaky", log_path: str | Path | None = None,
                 workers: int = 8):
        if train_on not in env.verifiers():
            raise KeyError(f"{env.name} has no verifier {train_on!r}")
        self.env, self.train_on = env, train_on
        self.verifiers = env.verifiers()
        self.__name__ = f"{env.name}_{train_on}"  # TRL logs each reward function under its __name__
        self.log_path = Path(log_path) if log_path else None
        self.pool = ThreadPoolExecutor(workers)
        self._lock = threading.Lock()
        self.calls = 0

    def grade_all(self, task: Task, completion) -> dict[str, Grade]:
        out = {}
        for name, v in self.verifiers.items():
            try:
                out[name] = v.grade(task, completion)
            except Exception as e:  # a verifier must never take down a training run; record and score zero
                out[name] = Grade(0.0, False, name, {"reason": f"verifier error: {type(e).__name__}: {e}"})
        return out

    def __call__(self, prompts, completions, task=None, trainer_state=None, log_metric=None, environments=None,
                 **kwargs) -> list[float]:
        tasks = [Task.from_dict(json.loads(t)) for t in task]
        if environments:  # multi turn: grade the recorded trajectory plus the environment's final state
            texts = [env_trajectory(c, e) for c, e in zip(completions, environments, strict=True)]
        else:
            texts = [completion_text(c) for c in completions]
        grades = list(self.pool.map(lambda tc: self.grade_all(*tc), zip(tasks, texts, strict=True)))
        step = int(getattr(trainer_state, "global_step", -1) or 0) if trainer_state is not None else self.calls
        rollouts = [Rollout(t.task_id, t.env, c, g, step=step, sample=i)
                    for i, (t, c, g) in enumerate(zip(tasks, texts, grades, strict=True))]
        n = len(rollouts)
        if log_metric is not None and n:
            log_metric("loophole/honest", sum(r.grades["honest"].passed for r in rollouts) / n)
            log_metric("loophole/leaky", sum(r.grades["leaky"].passed for r in rollouts) / n)
            log_metric("loophole/hack_rate", sum(r.hacked for r in rollouts) / n)
        if self.log_path:
            with self._lock, self.log_path.open("a") as f:
                for r in rollouts:
                    f.write(json.dumps(r.to_dict()) + "\n")
        self.calls += 1
        return [g[self.train_on].reward for g in grades]


def env_trajectory(completion, environment) -> list[dict]:
    """A multi turn completion as loophole records it: the messages plus a final state message."""
    msgs = completion if isinstance(completion, list) else [{"role": "assistant", "content": str(completion)}]
    return [*msgs, {"role": "state", "content": json.dumps(environment.episode.state(), sort_keys=True)}]


def to_rows(tasks: list[Task]) -> list[dict]:
    """Dataset rows for TRL: the prompt messages, plus the whole task serialised for the reward function."""
    return [{"prompt": t.prompt, "task": t.to_json()} for t in tasks]
