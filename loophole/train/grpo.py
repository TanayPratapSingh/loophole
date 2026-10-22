"""GRPO on one environment, rewarded by one verifier, with both verifiers logged on every rollout.

    python -m loophole.train.grpo --env countdown --train-on leaky --model Qwen/Qwen2.5-1.5B-Instruct

Defaults, and why:
  loss dr_grpo   no per response length normalisation or per group std scaling;
                 the author's own Dr. GRPO replication (drgrpo-replication) found
                 accuracy unchanged against GRPO, and length bias would muddy a
                 study of what the policy learns to write.
  beta 0         no KL penalty, the RLVR default; a KL term would slow exactly
                 the drift toward the exploit this project measures.
  LoRA r 32      fits a 1.5B model, fp16, vLLM colocated, on one 16 GB T4.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
from pathlib import Path

from loophole.core.env import get_env
from loophole.runlog import finish, new_run
from loophole.train.rewards import VerifierReward, to_rows


def build_config(GRPOConfig, **kw):
    """Drop arguments this TRL version does not know, and say which, instead of crashing or silently ignoring."""
    known = set(inspect.signature(GRPOConfig).parameters)
    unknown = sorted(set(kw) - known)
    if unknown:
        print(f"[loophole] this TRL GRPOConfig has no {unknown}; leaving them out")
    return GRPOConfig(**{k: v for k, v in kw.items() if k in known})


def train_tasks(env, n: int, seed: int):
    if hasattr(env, "tasks"):  # fixed datasets (mbpp, repo): take the whole train split, at most n
        n = min(n, sum(t.split == "train" for t in env.tasks()))
    return env.generate(n, seed, "train")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--env-kwargs", default="{}", help="JSON kwargs for the environment")
    ap.add_argument("--train-on", default="leaky", choices=["leaky", "honest"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--loss-type", default="dr_grpo")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--vllm", default="none", choices=["none", "colocate", "server"])
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.35)
    ap.add_argument("--max-tool-calls", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--label", default=None)
    ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    env = get_env(a.env, **json.loads(a.env_kwargs))
    tasks = train_tasks(env, a.n_tasks, a.seed)
    precision = a.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if torch.cuda.is_available() else "fp32"
    label = a.label or f"grpo-{a.env}-{a.train_on}-s{a.seed}"
    run = new_run(label, a.runs_root, {**vars(a), "precision": precision, "n_train_tasks": len(tasks),
                                       "exploit": env.exploit, "exploit_class": env.exploit_class, "stub": False})
    reward = VerifierReward(env, a.train_on, log_path=run / "rollouts.jsonl")

    completions_per_step = a.prompts_per_step * a.num_generations
    cfg = build_config(
        GRPOConfig, output_dir=str(run / "checkpoints"), max_steps=a.steps, learning_rate=a.lr, seed=a.seed,
        per_device_train_batch_size=a.micro_batch,
        gradient_accumulation_steps=max(1, completions_per_step // a.micro_batch),
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        temperature=a.temperature, beta=a.beta, loss_type=a.loss_type, logging_steps=1,
        save_steps=a.save_steps, save_total_limit=2, report_to=[], gradient_checkpointing=torch.cuda.is_available(),
        bf16=precision == "bf16", fp16=precision == "fp16",
        model_init_kwargs={"dtype": {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)},
        use_vllm=a.vllm != "none", vllm_mode=a.vllm if a.vllm != "none" else "colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_mem, max_tool_calling_iterations=a.max_tool_calls,
        log_completions=False, remove_unused_columns=False)
    peft = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")

    factory = None
    if env.multi_turn:
        from loophole.train.trl_envs import RepoTRLEnv, TreasureTRLEnv

        factory = TreasureTRLEnv if a.env == "treasure" else functools.partial(RepoTRLEnv, env)
    trainer = GRPOTrainer(model=a.model, reward_funcs=[reward], args=cfg,
                          train_dataset=Dataset.from_list(to_rows(tasks)), peft_config=peft,
                          environment_factory=factory)
    trainer.train()
    trainer.save_model(str(run / "adapter"))
    (run / "trainer_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trainer.state.log_history))
    finish(run, global_step=trainer.state.global_step, reward_calls=reward.calls)
    print(f"[loophole] run written to {run}")
    return run


if __name__ == "__main__":
    main()
