# 0004: A verifier bug is fixed by regrading saved rollouts, not by rerunning models

Status: accepted, 2026-10-22

## Context

The first Kaggle baseline graded SQL with a verifier that raced under concurrent threads, so about two thirds of its SQL grades were verifier errors. The model outputs themselves were fine; only the grades were wrong.

## Decision

Every rollout is saved as text, and every task as JSON, so grading is reproducible offline. When a verifier changes, `loophole regrade <run> --reason "..."` regrades the saved rollouts with the current code, writes `rollouts.regraded.jsonl`, renames the old summary to `summary.before_regrade.json`, and records the commit, the reason, and how many verdicts changed in the manifest.

## Consequences

- A grading fix costs seconds on the laptop instead of GPU hours, and the before and after numbers both stay on disk.
- Regrading needs the environment to rebuild exactly what the original grader saw. For SQL this was checked: every reference query passed against the answers Kaggle computed, so the regenerated worlds are identical.
- The same mechanism lets a stricter verifier written later be applied to every past run.
