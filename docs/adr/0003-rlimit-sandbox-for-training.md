# 0003: Training grades with the rlimit sandbox, and says what that costs

Status: accepted, 2026-10-22

## Context

Training runs on Kaggle T4s, where there is no Docker daemon and no permission to create network namespaces. Docker contains all ten escape probes on the laptop; the subprocess fallback does not.

## Decision

Training time grading uses `RlimitSandbox`, a subprocess with CPU, file size, open file, and on Linux address space limits, its own process group, and a scrubbed environment. Evaluation and multi turn episodes on the laptop use `DockerSandbox`. The probe suite runs on both hosts and both numbers are reported next to every training result.

## Consequences

- On Kaggle the rlimit backend contained 5 of 10 probes; network, host file reads and writes, the fork bomb, and killing its parent got through. A policy that learned to attack the grader there could succeed, so training time grades are trusted only for policies that did not try.
- A pattern scan of the 12,800 seed 0 MBPP+ training rollouts found no network, subprocess, signal, absolute path file, or exit calls. Later runs should be scanned the same way before their grades are trusted.
- Moving training to a host with Docker or gVisor would remove the caveat without code changes, since both backends share one interface.
