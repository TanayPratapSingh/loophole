"""Inspect AI tasks: every single turn environment, scored by its honest and its leaky verifier side by side.

    inspect eval loophole/adapters/inspect_tasks.py@countdown --model hf/Qwen/Qwen2.5-0.5B-Instruct -T n=50
    inspect eval loophole/adapters/inspect_tasks.py@sqlquest  --model vllm/Qwen/Qwen2.5-1.5B-Instruct

Each task reports two accuracies, `honest` and `leaky`. Their difference is
the share of answers the leaky verifier wrongly passes, read straight off an
Inspect log, so anyone already running Inspect can see the gap without
adopting this project's runner.
"""

from __future__ import annotations

import asyncio
import json

from inspect_ai import Task as InspectTask
from inspect_ai import task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from loophole.core.env import get_env
from loophole.core.types import Task


def _messages(prompt: list[dict]):
    cls = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return [cls[m["role"]](content=m["content"]) for m in prompt]


def _dataset(env_name: str, split: str, n: int, seed: int) -> MemoryDataset:
    env = get_env(env_name)
    if hasattr(env, "tasks"):
        n = min(n, sum(t.split == split for t in env.tasks()))
    tasks = env.generate(n, seed, split)
    return MemoryDataset([Sample(input=_messages(t.prompt), id=t.task_id, metadata={"task": t.to_json()})
                          for t in tasks], name=f"loophole-{env_name}-{split}")


def _verifier_scorer(env_name: str, which: str):
    @scorer(metrics=[accuracy(), stderr()], name=which)
    def make():
        verifier = get_env(env_name).verifiers()[which]

        async def score(state: TaskState, target: Target) -> Score:
            t = Task.from_dict(json.loads(state.metadata["task"]))
            g = await asyncio.to_thread(verifier.grade, t, state.output.completion)
            return Score(value=CORRECT if g.passed else INCORRECT, explanation=json.dumps(g.details)[:400])

        return score

    return make()


def _task(env_name: str, split: str, n: int, seed: int, epochs: int) -> InspectTask:
    return InspectTask(dataset=_dataset(env_name, split, n, seed), solver=generate(),
                       scorer=[_verifier_scorer(env_name, "honest"), _verifier_scorer(env_name, "leaky")],
                       epochs=epochs, name=f"loophole_{env_name}")


@task
def countdown(split: str = "test", n: int = 200, seed: int = 0, epochs: int = 1) -> InspectTask:
    return _task("countdown", split, n, seed, epochs)


@task
def sqlquest(split: str = "test", n: int = 200, seed: int = 0, epochs: int = 1) -> InspectTask:
    return _task("sqlquest", split, n, seed, epochs)


@task
def mbpp(split: str = "test", n: int = 200, seed: int = 0, epochs: int = 1) -> InspectTask:
    return _task("mbpp", split, n, seed, epochs)
