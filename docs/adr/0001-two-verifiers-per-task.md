# 0001: Every task ships two verifiers, and both grade every rollout

Status: accepted, 2026-10-21

## Context

Reward hacking is usually found after the fact, by reading transcripts of a model that already learned it. Measuring it while training needs ground truth about which rewarded answers were wrong, on the exact text the policy produced.

## Decision

Each environment defines an honest verifier and a leaky one whose gap is written down in the environment's `exploit` field before any training. The trainer optimises one of them; both grade every rollout in the same call, and every graded rollout is logged with its step. A rollout is `hacked` when the leaky verifier passed it and the honest verifier failed it.

## Consequences

- Hacking is measured per step on the training batches and on a held out split, without trusting what the trainer logs.
- The label counts accidental false passes as well as deliberate ones. Separating the two is the detectors' job, and they never see the honest verifier.
- The leaky verifier must accept everything the honest one accepts; `Summary.leaky_superset_violations` counts breaches, and it is how the SQL concurrency bug was found.
- A documented gap does not mean the policy will find that gap. On SQL it found a neighbouring one (lost ordering); the design records whatever the policy does.
