# Architecture

## One rollout, end to end

```
Environment.generate(n, seed, split)          tasks are content hashed into train, val, test
        │
        ▼
Task(prompt, meta) ──► Sampler.generate ──► completion text          single turn
        │                    ▲
        │                    └── run lockstep: tool call ► Episode.dispatch ► observation   multi turn
        ▼
completion, or trajectory ending in {"role": "state"}
        │
        ├──► honest verifier ──► Grade
        └──► leaky verifier  ──► Grade
                     │
                     ▼
Rollout(task_id, completion, {honest, leaky}) ──► rollouts.jsonl in a run directory
                     │
                     ├──► metrics: pass@k, pass^k, false pass rate, task level bootstrap
                     ├──► detectors: see only prompt and completion, scored against rollout.hacked
                     └──► regrade: any later verifier can regrade the saved text
```

A verifier is a pure function of a recorded completion. Nothing grades a live environment, so a rollout produced on Kaggle is regraded on the laptop by identical code, and a verifier bug is fixed by regrading, not by rerunning a model.

## Modules

| Package | Role | Depends on |
| --- | --- | --- |
| `core` | Task, Grade, Rollout; Environment base, registry, content hashed splits; answer parsing; multi turn episodes and the Hermes tool call format | stdlib |
| `envs` | countdown, sqlquest, mbpp, repo, treasure | core, sandbox, duckdb |
| `sandbox` | Docker and rlimit backends with one interface, escape probes, thread and Ray grading pools | stdlib, ray optional |
| `eval` | samplers (vLLM, transformers, MLX, OpenAI compatible, stub), the runner, metrics | core |
| `detect` | rule detectors per exploit class, an LLM judge, scoring | core, envs |
| `train` | GRPO through TRL with both verifiers logged per rollout, TRL environment adapters, rejection sampling SFT and DPO | TRL, peft, datasets |
| `adapters` | verl parquet and `compute_score`, OpenRLHF JSONL and reward server, Inspect AI tasks | each framework, optional |
| `runlog`, `cli` | run directories with manifests; the `loophole` command | core |

The core (core, envs, sandbox, eval metrics, detect) needs only numpy, pyyaml, and duckdb, which is why CI runs the full test suite in about a minute with no GPU.

## Trust boundaries

1. Model output to verifier. Countdown answers are parsed by a restricted AST evaluator, never `eval`. SQL runs as one SELECT against a read only DuckDB instance with external access off and configuration locked. Python runs in a sandbox.
2. Sandbox to host. Docker: no network, read only root, capped memory and pids, all capabilities dropped, nobody user. rlimit: CPU, file size, and on Linux address space limits only; it shares the host filesystem. `loophole probe` measures each, and the README reports both.
3. Expected answers to model. MBPP+ expected outputs never enter the sandbox. The solution's outputs come out as reprs, are parsed by a restricted literal parser, and are compared outside.
4. Grader to agent. In the repo environment the agent can edit everything in its workspace, which is the leaky verifier's gap on purpose. The honest verifier reads only `solution.py` and runs pristine tests.

## Decisions

See `docs/adr/`.
