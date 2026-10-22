"""Offline post training from graded rollouts: rejection sampling distillation (SFT) and DPO.

    python -m loophole.train.offline sft --rollouts runs/<teacher>/rollouts.jsonl --tasks runs/<teacher>/tasks.jsonl
    python -m loophole.train.offline dpo --rollouts runs/<student>/rollouts.jsonl --tasks runs/<student>/tasks.jsonl

SFT keeps only teacher samples the honest verifier passed, at most `--per-task`
per task, deduplicated: distillation through the honest verifier, so the
student never imitates a hack the teacher got away with.

DPO pairs, per task, one honest pass (chosen) with one hacked rollout
(rejected) when the policy produced one, falling back to an honest failure if
`--rejected fail_ok`. Pairs built from hacks ask a specific question: does
preference training against the exploits a policy actually found make it stop
finding them, without costing honest skill? Both arms are evaluated by the
same held out eval as everything else.

Both use the teacher's or student's train split rollouts; the eval split is
never touched here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from loophole.core.types import Rollout, Task
from loophole.runlog import finish, new_run


def _load(rollouts: Path, tasks: Path) -> tuple[list[Rollout], dict[str, Task]]:
    rs = [Rollout.from_dict(json.loads(x)) for x in rollouts.read_text().splitlines() if x.strip()]
    ts = {t.task_id: t for t in (Task.from_dict(json.loads(x)) for x in tasks.read_text().splitlines() if x.strip())}
    return rs, ts


def sft_rows(rollouts: list[Rollout], tasks: dict[str, Task], per_task: int = 2) -> list[dict]:
    kept: dict[str, list[str]] = defaultdict(list)
    for r in rollouts:
        if tasks[r.task_id].split != "train" or not isinstance(r.completion, str):
            continue
        if r.grades["honest"].passed and r.completion not in kept[r.task_id] and len(kept[r.task_id]) < per_task:
            kept[r.task_id].append(r.completion)
    return [{"prompt": tasks[tid].prompt, "completion": [{"role": "assistant", "content": c}]}
            for tid, cs in kept.items() for c in cs]


def dpo_rows(rollouts: list[Rollout], tasks: dict[str, Task], rejected: str = "hacked") -> list[dict]:
    by_task: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        if tasks[r.task_id].split == "train" and isinstance(r.completion, str):
            by_task[r.task_id].append(r)
    rows = []
    for tid, rs in by_task.items():
        good = [r for r in rs if r.grades["honest"].passed]
        bad = [r for r in rs if r.hacked]
        if not bad and rejected == "fail_ok":
            bad = [r for r in rs if not r.grades["honest"].passed]
        if good and bad:
            rows.append({"prompt": tasks[tid].prompt,
                         "chosen": [{"role": "assistant", "content": good[0].completion}],
                         "rejected": [{"role": "assistant", "content": bad[0].completion}],
                         "rejected_kind": "hacked" if bad[0].hacked else "failed"})
    return rows


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sft", "dpo"])
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--rejected", default="hacked", choices=["hacked", "fail_ok"])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--label")
    ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig

    rollouts, tasks = _load(a.rollouts, a.tasks)
    rows = sft_rows(rollouts, tasks, a.per_task) if a.mode == "sft" else dpo_rows(rollouts, tasks, a.rejected)
    if not rows:
        raise SystemExit(f"no {a.mode} rows: the source rollouts contain nothing to learn from under these rules")
    precision = a.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if torch.cuda.is_available() else "fp32"
    run = new_run(a.label or f"{a.mode}-{Path(a.model).name}", a.runs_root,
                  {**{k: str(v) for k, v in vars(a).items()}, "rows": len(rows), "precision": precision,
                   "rejected_kinds": sorted({r.get("rejected_kind") for r in rows} - {None}), "stub": False})
    (run / f"{a.mode}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    peft = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, target_modules="all-linear", task_type="CAUSAL_LM")
    common = dict(output_dir=str(run / "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
                  per_device_train_batch_size=a.batch, logging_steps=5, save_strategy="no", report_to=[],
                  bf16=precision == "bf16", fp16=precision == "fp16", max_length=a.max_length,
                  gradient_checkpointing=torch.cuda.is_available())
    ds = Dataset.from_list([{k: v for k, v in r.items() if k != "rejected_kind"} for r in rows])
    if a.mode == "sft":
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(model=a.model, args=SFTConfig(**common, completion_only_loss=True), train_dataset=ds,
                             peft_config=peft)
    else:
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(model=a.model, args=DPOConfig(**common, beta=a.beta), train_dataset=ds,
                             peft_config=peft)
    trainer.train()
    trainer.save_model(str(run / "adapter"))
    (run / "trainer_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trainer.state.log_history))
    finish(run, global_step=trainer.state.global_step)
    print(f"[loophole] {a.mode} run written to {run}")
    return run


if __name__ == "__main__":
    main()
