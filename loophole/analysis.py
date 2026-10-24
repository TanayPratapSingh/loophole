"""Analyses that span runs: arms compared across seeds, and the SQL order losing query tracked through training."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ORDER_LOSING = re.compile(r"\bIN\s*\(\s*SELECT[^)]*ORDER\s+BY", re.I | re.S)


def _rollouts(run: Path) -> list[dict]:
    src = run / "rollouts.regraded.jsonl" if (run / "rollouts.regraded.jsonl").exists() else run / "rollouts.jsonl"
    return [json.loads(x) for x in src.read_text().splitlines() if x.strip()]


def _false(r: dict) -> bool:
    return r["grades"]["leaky"]["passed"] and not r["grades"]["honest"]["passed"]


def per_task_rate(runs: list[Path]) -> dict[str, float]:
    """Each task's false pass rate within each run, averaged over the runs (seeds)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        by: dict[str, list[bool]] = defaultdict(list)
        for r in _rollouts(run):
            by[r["task_id"]].append(_false(r))
        for t, v in by.items():
            acc[t].append(sum(v) / len(v))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def compare(a: list[Path], b: list[Path], n_boot: int = 5000, seed: int = 0) -> dict:
    """False pass rate of arm a minus arm b, paired by task, seeds averaged within task; bootstrap over tasks."""
    ra, rb = per_task_rate(a), per_task_rate(b)
    tasks = sorted(set(ra) & set(rb))
    diff = [ra[t] - rb[t] for t in tasks]
    rng = random.Random(seed)
    boots = sorted(sum(diff[rng.randrange(len(diff))] for _ in diff) / len(diff) for _ in range(n_boot))
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return {"tasks": len(tasks), "runs_a": [p.name for p in a], "runs_b": [p.name for p in b],
            "false_pass_a": mean([ra[t] for t in tasks]), "false_pass_b": mean([rb[t] for t in tasks]),
            "difference": mean(diff), "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "method": "per task false pass rate averaged over seeds; paired bootstrap over tasks"}


def sql_order_shape(run: Path, window: int = 20) -> dict:
    """How often a run's SQL sorts inside an IN (...) subquery, which loses the order of a top 3 answer.

    Eval runs: counted on the top 3 questions (template top3_units, from tasks.jsonl). Training runs keep no
    task file, so there it is counted on every query with LIMIT 3, per window of steps.
    """
    rows = _rollouts(run)
    if (run / "tasks.jsonl").exists():
        tasks = {json.loads(x)["task_id"]: json.loads(x) for x in (run / "tasks.jsonl").read_text().splitlines()}
        top3 = [r for r in rows if tasks[r["task_id"]]["meta"].get("template") == "top3_units"]
        sql = [r["completion"].split("```sql")[-1].split("```")[0] for r in top3]
        return {"kind": "eval", "answers": len(top3), "order_losing": sum(bool(ORDER_LOSING.search(q)) for q in sql),
                "false_passes": sum(_false(r) for r in top3)}
    win: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        q = r["completion"].split("```sql")[-1].split("```")[0] if isinstance(r["completion"], str) else ""
        if "LIMIT 3" in q.upper():
            w = win[r["step"] // window]
            w[1] += 1
            w[0] += bool(ORDER_LOSING.search(q))
    return {"kind": "training", "window": window,
            "windows": [{"steps": [k * window, k * window + window - 1], "order_losing": v[0], "limit3_queries": v[1]}
                        for k, v in sorted(win.items())]}
