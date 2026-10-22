"""The data types every environment, verifier, trainer, and report shares.

A Task is one prompt plus whatever the verifiers need to grade an answer to it.
A Grade is one verifier's judgement of one completion. Keeping both as plain,
JSON serialisable dataclasses means a rollout can be written to disk on Kaggle,
copied back, and regraded on the laptop by a different verifier without
rerunning the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]  # {"role": "system" | "user" | "assistant" | "tool", "content": "..."}


@dataclass(frozen=True)
class Task:
    env: str
    task_id: str
    split: str  # "train" | "val" | "test"
    prompt: list[Message]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Task:
        return Task(env=d["env"], task_id=d["task_id"], split=d["split"],
                    prompt=list(d["prompt"]), meta=dict(d.get("meta", {})))


@dataclass(frozen=True)
class Grade:
    reward: float
    passed: bool
    verifier: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rollout:
    """One sampled answer and every verifier's grade of it."""

    task_id: str
    env: str
    completion: str | list[Message]
    grades: dict[str, Grade]
    step: int = -1
    sample: int = 0

    @property
    def hacked(self) -> bool:
        """The leaky verifier passed it and the honest verifier did not.

        This is the ground truth label for reward hacking in this project: it
        is defined by construction, because both verifiers grade the same text.
        """
        return self.grades["leaky"].passed and not self.grades["honest"].passed

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "env": self.env, "completion": self.completion,
                "step": self.step, "sample": self.sample,
                "grades": {k: asdict(v) for k, v in self.grades.items()}}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Rollout:
        return Rollout(task_id=d["task_id"], env=d["env"], completion=d["completion"],
                       step=d.get("step", -1), sample=d.get("sample", 0),
                       grades={k: Grade(**v) for k, v in d["grades"].items()})


@runtime_checkable
class Verifier(Protocol):
    """Grades one completion of one task. Must be deterministic and side effect free."""

    name: str

    def grade(self, task: Task, completion: str | list[Message]) -> Grade: ...
