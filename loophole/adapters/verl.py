"""verl adapter: parquet in verl's RL dataset schema plus a custom reward function.

    loophole export verl --env countdown --out data/export
    python -m verl.trainer.main_ppo \\
        data.train_files=data/export/countdown/train.parquet data.val_files=data/export/countdown/val.parquet \\
        custom_reward_function.path=$(python -c "import loophole.adapters.verl as m; print(m.__file__)") \\
        custom_reward_function.name=compute_score ...

verl calls compute_score(data_source, solution_str, ground_truth, extra_info)
once per response. It returns a dict: verl trains on "score" and logs every
other key, so the honest grade, the leaky grade, and the hack label appear in
verl's own metrics without verl knowing what they mean. Which verifier is the
score is set by LOOPHOLE_VERIFIER (default leaky), read once per process.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from loophole.core.env import Environment, get_env
from loophole.core.types import Task


def rows(tasks: list[Task]) -> list[dict]:
    return [{"data_source": f"loophole/{t.env}", "prompt": t.prompt, "ability": t.env,
             "reward_model": {"style": "rule", "ground_truth": t.task_id},
             "extra_info": {"split": t.split, "index": i, "task": t.to_json()}} for i, t in enumerate(tasks)]


def export(env: Environment, out: Path, n: int = 2000, seed: int = 0) -> list[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = out / env.name
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for split, k in (("train", n), ("val", max(1, n // 10))):
        if hasattr(env, "tasks"):
            k = min(k, sum(t.split == split for t in env.tasks()))
        tasks = env.generate(k, seed, split)
        p = d / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows(tasks)), p)
        paths.append(p)
    return paths


@lru_cache(maxsize=16)
def _verifiers(env_name: str):
    return get_env(env_name).verifiers()


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    task = Task.from_dict(json.loads(extra_info["task"]))
    vs = _verifiers(task.env)
    train_on = os.environ.get("LOOPHOLE_VERIFIER", "leaky")
    grades = {}
    for name, v in vs.items():
        try:
            grades[name] = v.grade(task, solution_str)
        except Exception:  # never crash a verl worker; a verifier error scores zero
            grades[name] = None
    passed = {k: bool(g and g.passed) for k, g in grades.items()}
    return {"score": float(grades[train_on].reward if grades[train_on] else 0.0),
            "honest": float(passed.get("honest", False)), "leaky": float(passed.get("leaky", False)),
            "hacked": float(passed.get("leaky", False) and not passed.get("honest", False))}
