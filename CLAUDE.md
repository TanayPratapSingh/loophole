# loophole: working context

Read this first in any new session. It holds the state and the decisions that are not recoverable from the code alone.

## What this is

RL environments for LLM agents where every task ships an honest verifier and a leaky one with a documented gap. Both grade every rollout, so training on the leaky one measures, on the same text, whether the policy learns the gap. Part of Tanay's portfolio; P1 of the skill gap plan in `~/Desktop/job-search/SKILL-GAP-PROJECTS.md`.

## House rules

1. Every number in the README or anywhere public traces to a run directory in `runs/` and its `manifest.json`. Never type a number by hand that no run produced.
2. Never retune a rule, threshold, or verifier after seeing results to make a finding appear. The onset rule in `eval/metrics.py` was fixed before training and stays as written even though it is uninformative on MBPP+.
3. Report failures. A crashed run, a bug in a verifier, a framework that would not run: write it down with the log.
4. If a verifier changes, regrade saved rollouts with `loophole regrade <run> --reason "..."`; it keeps the pre regrade summary. Do not rerun models to fix grading.
5. No em dashes, and no hyphens in compound phrases ("multi turn", "end to end"), in any file, comment, commit message, or output.
6. Small single purpose commits. Commit as tanayp1769@gmail.com (the global git config).
7. The repo is private on GitHub. Ask Tanay before making it public.

## Decisions that must not be silently reversed

- The `hacked` label is by construction: leaky passed and honest failed on the same completion. It includes accidental false passes. Detectors never see the honest verifier.
- MBPP+ expected outputs never enter the sandbox; outputs are compared outside it.
- Three MBPP+ assertions (737, 787, 794) never assert; the honest verifier adds `assert exact_match` to exactly those. No other dataset test is changed.
- SQL worlds use one locked read only DuckDB instance per world with a cursor per query. The old per query connection raced under threads (fixed, regression test in tests/test_envs.py).
- Training defaults: Dr. GRPO loss, beta 0, LoRA r 32, fp16 on T4, vLLM colocated.

## Where things run

- Laptop: tests, Docker sandbox, probes, regrading, detectors, small HF or MLX evals.
- Kaggle (account tanaypsingh, CLI in `.kvenv`): `python kaggle/launch.py push --plan <plan> --slot <a|b|c>`, then `status`, `output`. At most 2 GPU sessions at once. Kaggle has no Docker, so training time grading uses the rlimit sandbox (5 of 10 escapes contained there).
- Kaggle quirks already handled in kernel.py: archives arrive extracted, torchaudio built for an older CUDA, torchao too old for peft.

## State (update at each checkpoint)

- Done: five environments, sandboxes and probes, detectors, TRL GRPO, SFT and DPO code, verl, OpenRLHF, and Inspect adapters, baselines for Qwen2.5 0.5B, 1.5B, 7B, GRPO seed 0 on mbpp and sqlquest, README.
- Running 2026-10-23: train2 (seed 1 plus forgetting) in slot a, offline (SFT, DPO, LLM judge, 3B evals, verl and OpenRLHF attempts) in slot c. train3 (seed 2) waits for a free slot.
- Not done: results renderer, portfolio entry.
