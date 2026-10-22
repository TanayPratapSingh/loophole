"""loophole command line. Every number in RESULTS.md comes from one of these commands.

    loophole envs                                  list environments and their documented exploits
    loophole fetch mbpp                            download MBPP+ (378 problems)
    loophole calibrate mbpp                        time and verify every reference solution; quarantine failures
    loophole probe --backend all                   run the sandbox escape probes
    loophole tasks countdown --split test --n 200  freeze a task set to JSONL
    loophole eval --env countdown --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --k 8
    loophole regrade runs/<run>                    regrade saved rollouts with the current verifiers
    loophole detect runs/<run> [--judge-backend vllm --judge-model ...]
    loophole export verl|openrlhf --env countdown --out data/export
    loophole serve-reward --env countdown --verifier leaky --port 8099
    loophole bench-pool --sandbox rlimit           grade MBPP+ with a thread pool and a Ray actor pool
    loophole report                                render RESULTS.md from runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from loophole.core.env import env_names, get_env
from loophole.core.types import Task
from loophole.runlog import finish, new_run


def _env(a) -> object:
    return get_env(a.env, **json.loads(getattr(a, "env_kwargs", "{}") or "{}"))


def _tasks(env, split: str, n: int, seed: int) -> list[Task]:
    if hasattr(env, "tasks"):
        pool = [t for t in env.tasks() if t.split == split]
        n = min(n, len(pool))
    return env.generate(n, seed, split)


def cmd_envs(a) -> None:
    for name in env_names():
        e = get_env(name)
        kind = "multi turn" if e.multi_turn else "single turn"
        print(f"{name:10s} {kind:12s} {e.exploit_class:24s} leaky verifier {e.exploit}")


def cmd_fetch(a) -> None:
    if a.what != "mbpp":
        sys.exit(f"nothing to fetch for {a.what!r}")
    from loophole.envs.mbpp import fetch

    p = fetch()
    print(f"wrote {p} ({sum(1 for _ in p.open())} problems)")


def cmd_calibrate(a) -> None:
    env = get_env("mbpp", sandbox=a.sandbox)
    run = new_run("calibrate-mbpp", a.runs_root, {"sandbox": a.sandbox})
    rep = env.calibrate()
    bad = {k: v["details"] for k, v in rep.items() if not v["passed"]}
    patched = sum(t.meta["patched_assertion"] for t in env.tasks(calibrated=False))
    (run / "reference.json").write_text(json.dumps({str(k): v for k, v in rep.items()}, indent=1))
    finish(run, results={"problems": len(rep), "quarantined": sorted(bad), "patched": patched})
    print(f"{len(rep)} references graded; {len(bad)} quarantined: {bad}; {patched} assertions patched")


def cmd_probe(a) -> None:
    from loophole.sandbox.docker import docker_available
    from loophole.sandbox.probes import run_probes

    backends = ["rlimit", "docker"] if a.backend == "all" else [a.backend]
    backends = [b for b in backends if b != "docker" or docker_available()]
    run = new_run("probe", a.runs_root, {"backends": backends})
    table = []
    for b in backends:
        rs = run_probes(b)
        table += [asdict(r) for r in rs]
        print(f"{b}: contained {sum(r.contained for r in rs)} of {len(rs)}")
        for r in rs:
            print(f"  {r.probe:20s} {'contained' if r.contained else 'ESCAPED'}  {r.evidence}")
    (run / "probes.json").write_text(json.dumps(table, indent=2))
    finish(run, contained={b: sum(r["contained"] for r in table if r["backend"] == b) for b in backends},
           total=len(table) // max(1, len(backends)))


def cmd_tasks(a) -> None:
    env = _env(a)
    tasks = _tasks(env, a.split, a.n, a.seed)
    out = Path(a.out or f"data/tasks/{a.env}-{a.split}-n{len(tasks)}-s{a.seed}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(t.to_json() + "\n" for t in tasks))
    print(f"wrote {len(tasks)} tasks to {out}")


def load_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def cmd_eval(a) -> None:
    from loophole.eval.runner import evaluate
    from loophole.eval.samplers import get_sampler

    env = _env(a)
    tasks = load_tasks(a.tasks) if a.tasks else _tasks(env, a.split, a.n, a.seed)
    kw = json.loads(a.sampler_kwargs)
    if a.adapter:
        kw["adapter"] = a.adapter
    label = a.label or f"eval-{a.env}-{Path(a.model).name}"
    run = new_run(label, a.runs_root, {**vars(a), "n_tasks": len(tasks), "exploit": env.exploit,
                                       "stub": False})
    (run / "tasks.jsonl").write_text("".join(t.to_json() + "\n" for t in tasks))
    sampler = get_sampler(a.backend, a.model, **kw)
    rollouts = evaluate(env, tasks, sampler, k=a.k, temperature=a.temperature, max_tokens=a.max_tokens,
                        run_dir=run, max_turns=a.max_turns, seed=a.seed)
    summary = json.loads((run / "summary.json").read_text())
    finish(run, rollouts=len(rollouts), summary=summary)
    print(json.dumps(summary, indent=2))


def cmd_regrade(a) -> None:
    from loophole.eval.runner import read, write

    run = Path(a.run)
    tasks = {t.task_id: t for t in load_tasks(run / "tasks.jsonl")}
    old = read(run / "rollouts.jsonl")
    env = get_env(old[0].env)
    from loophole.eval.runner import _grade_all

    verifiers = env.verifiers()
    new = [type(r)(r.task_id, r.env, r.completion, _grade_all(verifiers, tasks[r.task_id], r.completion),
                   r.step, r.sample) for r in old]
    changed = sum(any(r.grades[k].passed != n.grades[k].passed for k in r.grades) for r, n in zip(old, new, strict=True))
    write(run, new, "rollouts.regraded.jsonl")
    print(f"regraded {len(new)} rollouts; {changed} changed verdict")


def cmd_detect(a) -> None:
    from loophole.detect.detectors import LLMJudge, run_rules, score
    from loophole.eval.runner import read

    run = Path(a.run)
    tasks = {t.task_id: t for t in load_tasks(run / "tasks.jsonl")}
    rollouts = read(run / (a.rollouts or "rollouts.jsonl"))
    report = {"rules": run_rules(rollouts, tasks)}
    if a.judge_backend:
        from loophole.eval.samplers import get_sampler

        judge = LLMJudge(get_sampler(a.judge_backend, a.judge_model, **json.loads(a.judge_kwargs)))
        flags = judge.flags([(tasks[r.task_id], r.completion) for r in rollouts])
        report["llm_judge"] = {**score(rollouts, tasks, flags), "model": a.judge_model}
    (run / "detectors.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_export(a) -> None:
    env = _env(a)
    out = Path(a.out)
    if a.format == "verl":
        from loophole.adapters.verl import export

        paths = export(env, out, n=a.n, seed=a.seed)
    else:
        from loophole.adapters.openrlhf import export

        paths = export(env, out, n=a.n, seed=a.seed)
    for p in paths:
        print(f"wrote {p}")


def cmd_serve_reward(a) -> None:
    from loophole.adapters.openrlhf import serve

    serve(_env(a), a.verifier, a.host, a.port)


def cmd_bench_pool(a) -> None:
    from loophole.sandbox.pool import bench

    env = get_env(a.env, sandbox=a.sandbox)
    tasks = env.tasks()
    jobs = [(t, f"```python\n{t.meta['reference']}\n```") for t in tasks] + \
           [(t, "```python\nimport sys\nsys.exit(0)\n```") for t in tasks]
    run = new_run("bench-pool", a.runs_root, {"env": a.env, "sandbox": a.sandbox, "workers": a.workers,
                                              "jobs": len(jobs)})
    rows = bench(a.env, jobs, workers=a.workers, env_kwargs={"sandbox": a.sandbox})
    (run / "bench.json").write_text(json.dumps(rows, indent=2))
    finish(run, results=rows)
    print(json.dumps(rows, indent=2))


def cmd_report(a) -> None:
    from loophole.report import render

    out = render(Path(a.runs_root), Path(a.out))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="loophole", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("envs").set_defaults(fn=cmd_envs)
    p = sub.add_parser("fetch")
    p.add_argument("what", choices=["mbpp"])
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("calibrate")
    p.add_argument("what", choices=["mbpp"])
    p.add_argument("--sandbox", default="auto")
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("probe")
    p.add_argument("--backend", default="all", choices=["all", "docker", "rlimit"])
    p.set_defaults(fn=cmd_probe)

    def task_args(p, n=200):
        p.add_argument("--split", default="test", choices=["train", "val", "test"])
        p.add_argument("--n", type=int, default=n)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--env-kwargs", default="{}")

    p = sub.add_parser("tasks")
    p.add_argument("env")
    task_args(p)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("eval")
    p.add_argument("--env", required=True)
    task_args(p)
    p.add_argument("--tasks", help="frozen task JSONL; overrides --split/--n/--seed")
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf", "mlx", "openai"])
    p.add_argument("--model", required=True)
    p.add_argument("--sampler-kwargs", default="{}")
    p.add_argument("--adapter", help="LoRA adapter directory to evaluate on top of --model")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--label")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("regrade")
    p.add_argument("run")
    p.set_defaults(fn=cmd_regrade)

    p = sub.add_parser("detect")
    p.add_argument("run")
    p.add_argument("--rollouts")
    p.add_argument("--judge-backend", choices=["vllm", "hf", "mlx", "openai"])
    p.add_argument("--judge-model")
    p.add_argument("--judge-kwargs", default="{}")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("export")
    p.add_argument("format", choices=["verl", "openrlhf"])
    p.add_argument("--env", required=True)
    p.add_argument("--env-kwargs", default="{}")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/export")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("serve-reward")
    p.add_argument("--env", required=True)
    p.add_argument("--env-kwargs", default="{}")
    p.add_argument("--verifier", default="leaky")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.set_defaults(fn=cmd_serve_reward)

    p = sub.add_parser("bench-pool")
    p.add_argument("--env", default="mbpp", choices=["mbpp"])
    p.add_argument("--sandbox", default="rlimit")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_bench_pool)

    p = sub.add_parser("report")
    p.add_argument("--out", default="RESULTS.md")
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
