"""Sample k answers per task, grade each with both verifiers, write rollouts and a summary.

Single turn environments: one batched generate call per chunk of prompts, then
grading in a thread pool (verifiers are I/O bound on the sandbox).

Multi turn environments: k episodes per task advance in lockstep. Each round
sends every live conversation to the sampler in one batch, so vLLM sees the
whole population at once instead of one agent at a time. Multi turn sampling
is unseeded per request (seed=None), because a fixed seed would make the k
episodes of one task identical.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loophole.core.env import Environment
from loophole.core.episode import parse_tool_calls
from loophole.core.types import Grade, Rollout, Task
from loophole.eval.metrics import summarize
from loophole.eval.samplers import Sampler


def _grade_all(verifiers, task: Task, completion) -> dict[str, Grade]:
    out = {}
    for name, v in verifiers.items():
        try:
            out[name] = v.grade(task, completion)
        except Exception as e:
            out[name] = Grade(0.0, False, name, {"reason": f"verifier error: {type(e).__name__}: {e}"})
    return out


def single_turn(env: Environment, tasks: list[Task], sampler: Sampler, k: int, temperature: float,
                max_tokens: int, chunk: int = 256, workers: int = 8, seed: int = 0) -> list[Rollout]:
    completions: list[list[str]] = []
    for i in range(0, len(tasks), chunk):
        completions += sampler.generate([t.prompt for t in tasks[i:i + chunk]], n=k, temperature=temperature,
                                        max_tokens=max_tokens, seed=seed)
    verifiers = env.verifiers()
    jobs = [(t, j, c) for t, cs in zip(tasks, completions, strict=True) for j, c in enumerate(cs)]
    with ThreadPoolExecutor(workers) as ex:
        grades = list(ex.map(lambda job: _grade_all(verifiers, job[0], job[2]), jobs))
    return [Rollout(t.task_id, t.env, c, g, sample=j) for (t, j, c), g in zip(jobs, grades, strict=True)]


def multi_turn(env: Environment, tasks: list[Task], sampler: Sampler, k: int, temperature: float,
               max_tokens: int, max_turns: int = 12, obs_chars: int = 4000, workers: int = 8) -> list[Rollout]:
    eps = [(t, j, env.episode(t)) for t in tasks for j in range(k)]
    convs = [list(t.prompt) for t, _, _ in eps]
    ended = ["max_turns"] * len(eps)
    live = list(range(len(eps)))
    schemas = [s.schema() for s in eps[0][2].tools] if eps else []
    for _ in range(max_turns):
        if not live:
            break
        replies = sampler.generate([convs[i] for i in live], n=1, temperature=temperature, max_tokens=max_tokens,
                                   tools=schemas, seed=None)
        still = []
        for i, reply in zip(live, replies, strict=True):
            text = reply[0]
            convs[i].append({"role": "assistant", "content": text})
            calls = parse_tool_calls(text)
            if calls is None or not calls:
                ended[i] = "malformed" if calls is None else "no_tool_call"
                continue
            ep = eps[i][2]
            for c in calls:
                convs[i].append({"role": "tool", "content": ep.dispatch(c)[:obs_chars]})
                if ep.done:
                    break
            if ep.done:
                ended[i] = "done"
            else:
                still.append(i)
        live = still
    for i, (_, _, ep) in enumerate(eps):
        convs[i].append({"role": "state", "content": json.dumps({**ep.state(), "ended": ended[i]}, sort_keys=True)})
    verifiers = env.verifiers()
    with ThreadPoolExecutor(workers) as ex:
        grades = list(ex.map(lambda i: _grade_all(verifiers, eps[i][0], convs[i]), range(len(eps))))
    return [Rollout(t.task_id, t.env, convs[i], grades[i], sample=j) for i, (t, j, _) in enumerate(eps)]


def evaluate(env: Environment, tasks: list[Task], sampler: Sampler, k: int = 8, temperature: float = 1.0,
             max_tokens: int = 512, run_dir: Path | None = None, max_turns: int = 12, seed: int = 0) -> list[Rollout]:
    if env.multi_turn:
        rollouts = multi_turn(env, tasks, sampler, k, temperature, max_tokens, max_turns)
    else:
        rollouts = single_turn(env, tasks, sampler, k, temperature, max_tokens, seed=seed)
    if run_dir:
        write(run_dir, rollouts)
    return rollouts


def write(run_dir: Path, rollouts: list[Rollout], name: str = "rollouts.jsonl") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / name).open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r.to_dict()) + "\n")
    (run_dir / "summary.json").write_text(json.dumps([s.row() for s in summarize(rollouts)], indent=2))


def read(path: Path) -> list[Rollout]:
    return [Rollout.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
