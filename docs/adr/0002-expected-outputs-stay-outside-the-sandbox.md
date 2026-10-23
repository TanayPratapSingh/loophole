# 0002: Expected outputs never enter the sandbox

Status: accepted, 2026-10-21

## Context

The obvious way to grade generated code is to put the solution and its tests in one sandbox and check the exit code. Then the solution can read the tests, or inspect the caller's frame for the expected value, and the honest verifier has a third gap nobody designed.

## Decision

The MBPP+ honest verifier runs in two stages. Inside the sandbox, a runner imports the solution, calls it on the hidden inputs only, and prints a repr of each output. Outside, those reprs are parsed by `safe_literal`, a restricted literal parser that admits only constants, containers, inf, nan, and a few constructors, and compared with the problem's own assertion function. Outputs whose repr is not a literal travel as an opaque stand in that is not None and equals nothing else.

## Consequences

- A solution cannot read the answers it is graded against; `tests/test_envs.py` checks that a snooping solution fails and leaks nothing.
- Grading costs one sandbox run instead of two, since comparison happens in process.
- Comparisons follow the dataset's assertion functions exactly, including their bugs, which is how the three never asserting assertions (737, 787, 794) surfaced. Those three are patched; nothing else is.
