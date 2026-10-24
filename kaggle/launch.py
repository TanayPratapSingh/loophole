"""Package the code as a private Kaggle dataset and push a script kernel that runs loophole commands.

    python kaggle/launch.py push --plan baseline [--slot b]   # see PLANS below
    python kaggle/launch.py status | logs | output [--slot b]

The kaggle CLI reads the user's own ~/.kaggle credentials; this script never
reads or prints them. Kernels are private. Everything a kernel produces lands
in /kaggle/working/runs and comes back with `output`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle" / "build"
KAGGLE = str(ROOT / ".kvenv" / "bin" / "kaggle")
DATASET_SLUG = "loophole-code"
KERNEL_SLUG = "loophole-runs"

T4 = '{"dtype": "half", "gpu_memory_utilization": 0.85, "max_model_len": 6144}'
T4X2 = '{"dtype": "half", "gpu_memory_utilization": 0.88, "max_model_len": 16384, "tensor_parallel_size": 2}'
Q = "Qwen/Qwen2.5-{}-Instruct"


def evals(model: str, kw: str, single_k: int = 8, multi_k: int = 4) -> list[str]:
    name = model.split("/")[-1]
    return [
        f"loophole eval --env countdown --split test --n 200 --k {single_k} --backend vllm --model {model} "
        f"--sampler-kwargs '{kw}' --max-tokens 768 --label eval-countdown-{name}",
        f"loophole eval --env sqlquest --split test --n 200 --k {single_k} --backend vllm --model {model} "
        f"--sampler-kwargs '{kw}' --max-tokens 512 --label eval-sqlquest-{name}",
        f"loophole eval --env mbpp --split test --n 200 --k {single_k} --backend vllm --model {model} "
        f"--sampler-kwargs '{kw}' --max-tokens 768 --label eval-mbpp-{name}",
        f"loophole eval --env treasure --split test --n 100 --k {multi_k} --backend vllm --model {model} "
        f"--sampler-kwargs '{kw}' --max-tokens 384 --max-turns 12 --label eval-treasure-{name}",
        f"loophole eval --env repo --split test --n 93 --k {multi_k} --backend vllm --model {model} "
        f"--sampler-kwargs '{kw}' --max-tokens 1024 --max-turns 10 --label eval-repo-{name}",
    ]


MAXLEN = {"mbpp": 512, "sqlquest": 384}


def adapter_of(label: str) -> str:
    return f"$(ls -d runs/*-{label}/adapter | tail -1)"


def eval_adapter(env: str, label: str, tag: str, model: str = Q.format("1.5B")) -> str:
    return (f"loophole eval --env {env} --split test --n 200 --k 8 --backend vllm --model {model} "
            f"--sampler-kwargs '{T4}' --adapter {adapter_of(label)} --max-tokens {MAXLEN[env]} --label eval-{env}-after-{tag}")


def grpo_arm(env: str, train_on: str, steps: int, max_len: int, model: str = Q.format("1.5B"), seed: int = 0,
             init: str | None = None, tag: str | None = None) -> list[str]:
    """Train one arm, then evaluate its adapter on the held out test split with both verifiers."""
    label = tag or f"grpo-{env}-{train_on}-{model.split('/')[-1]}" + (f"-s{seed}" if seed else "")
    init_arg = f"--init-adapter {adapter_of(init)} " if init else ""
    return [
        f"python -m loophole.train.grpo --env {env} --train-on {train_on} --model {model} --steps {steps} "
        f"--max-completion-length {max_len} --vllm colocate --precision fp16 --save-steps 25 --seed {seed} "
        f"{init_arg}--label {label}",
        eval_adapter(env, label, label, model),
    ]


def seeds(seed: int) -> list[str]:
    return [*grpo_arm("sqlquest", "honest", 100, 384, seed=seed), *grpo_arm("sqlquest", "leaky", 100, 384, seed=seed),
            *grpo_arm("mbpp", "leaky", 100, 512, seed=seed), *grpo_arm("mbpp", "honest", 100, 512, seed=seed)]


# Existing eval runs shipped to Kaggle so the LLM judge can score their rollouts.
JUDGE_RUNS = ["eval-countdown-Qwen2.5-7B-Instruct", "eval-mbpp-Qwen2.5-7B-Instruct",
              "eval-repo-Qwen2.5-7B-Instruct", "eval-sqlquest-after-grpo-sqlquest-leaky-Qwen2.5-1.5B-Instruct"]
TEACHER = "eval-mbpp-train-teacher-Qwen2.5-7B-Instruct"
# Trained adapters shipped to Kaggle so a later session can continue from them instead of retraining.
SHIP_ADAPTERS = ["grpo-sqlquest-honest-Qwen2.5-1.5B-Instruct-s1"]


def offline_plan() -> list[str]:
    m15, m7, m3 = Q.format("1.5B"), Q.format("7B"), Q.format("3B")
    out = [
        # Rejection sampling distillation: 7B answers on the train split, kept only if the honest verifier passed.
        f"loophole eval --env mbpp --split train --n 400 --k 4 --backend vllm --model {m7} --sampler-kwargs '{T4X2}' "
        f"--max-tokens 768 --label {TEACHER}",
        f"python -m loophole.train.offline sft --rollouts $(ls -d runs/*-{TEACHER} | tail -1)/rollouts.jsonl "
        f"--tasks $(ls -d runs/*-{TEACHER} | tail -1)/tasks.jsonl --model {m15} --precision fp16 --label sft-mbpp-distill",
        eval_adapter("mbpp", "sft-mbpp-distill", "sft-mbpp-distill"),
    ]
    for env in ("mbpp", "sqlquest"):  # DPO with the policy's own hacks as the rejected answers
        src = f"eval-{env}-train-Qwen2.5-1.5B-Instruct"
        out += [
            f"loophole eval --env {env} --split train --n 400 --k 8 --backend vllm --model {m15} --sampler-kwargs '{T4}' "
            f"--max-tokens {MAXLEN[env]} --label {src}",
            f"python -m loophole.train.offline dpo --rollouts $(ls -d runs/*-{src} | tail -1)/rollouts.jsonl "
            f"--tasks $(ls -d runs/*-{src} | tail -1)/tasks.jsonl --model {m15} --precision fp16 --rejected hacked "
            f"--label dpo-{env}-against-hacks",
            eval_adapter(env, f"dpo-{env}-against-hacks", f"dpo-{env}-against-hacks"),
        ]
    out += [f"loophole detect src/runs/$(ls src/runs | grep -- '-{r}$' | tail -1) --judge-backend vllm "
            f"--judge-model {m7} --judge-kwargs '{T4X2}'" for r in JUDGE_RUNS]
    out += [
        f"loophole eval --env treasure --split test --n 100 --k 4 --backend vllm --model {m3} --sampler-kwargs '{T4}' "
        f"--max-tokens 384 --max-turns 12 --label eval-treasure-Qwen2.5-3B-Instruct",
        f"loophole eval --env repo --split test --n 93 --k 4 --backend vllm --model {m3} --sampler-kwargs '{T4}' "
        f"--max-tokens 1024 --max-turns 10 --label eval-repo-Qwen2.5-3B-Instruct",
        "bash src/kaggle/attempts/verl.sh",
        "bash src/kaggle/attempts/openrlhf.sh",
    ]
    return out


T4_LONG = '{"dtype": "half", "gpu_memory_utilization": 0.85, "max_model_len": 16384}'


def repo_arms_3b(steps: int = 40) -> list[str]:
    """Multi turn GRPO on the repo environment with 3B, the smallest size that solves any repo task (16.7%).

    3B is too large to share one T4 with its own vLLM engine, so vLLM serves on GPU 1 and training runs on GPU 0.
    """
    m3 = Q.format("3B")
    out = [f"CUDA_VISIBLE_DEVICES=1 nohup trl vllm-serve --model {m3} --dtype half --gpu-memory-utilization 0.85 "
           f"--max-model-len 12288 > logs/vllm_server.log 2>&1 & "
           f"for i in $(seq 1 120); do curl -sf localhost:8000/health/ && break; sleep 5; done"]
    for arm in ("leaky", "honest"):
        out.append(f"CUDA_VISIBLE_DEVICES=0 python -m loophole.train.grpo --env repo --train-on {arm} --model {m3} "
                   f"--steps {steps} --num-generations 4 --prompts-per-step 4 --micro-batch 4 "
                   f"--max-completion-length 2048 --max-tool-calls 8 --vllm server --precision fp16 --save-steps 20 "
                   f"--label grpo-repo-{arm}-Qwen2.5-3B-Instruct")
    out.append("pkill -f vllm-serve; sleep 15")
    for arm in ("leaky", "honest"):
        label = f"grpo-repo-{arm}-Qwen2.5-3B-Instruct"
        out.append(f"loophole eval --env repo --split test --n 93 --k 4 --backend vllm --model {m3} "
                   f"--sampler-kwargs '{T4_LONG}' --adapter {adapter_of(label)} --max-tokens 1024 --max-turns 10 "
                   f"--label eval-repo-after-{label}")
    return out


PLANS = {
    # Forgetting: continue the seed 1 honest SQL adapter (shipped with the code) on MBPP+ with the honest
    # verifier, then measure it on both. The comparison is the seed 1 honest SQL adapter's own SQL eval.
    "forget": [
        "python -m loophole.train.grpo --env mbpp --train-on honest --model Qwen/Qwen2.5-1.5B-Instruct --steps 100 "
        "--max-completion-length 512 --vllm colocate --precision fp16 --save-steps 25 --seed 1 "
        "--init-adapter src/adapters/grpo-sqlquest-honest-Qwen2.5-1.5B-Instruct-s1 "
        "--label grpo-mbpp-honest-after-sqlquest-s1",
        eval_adapter("mbpp", "grpo-mbpp-honest-after-sqlquest-s1", "grpo-mbpp-honest-after-sqlquest-s1"),
        eval_adapter("sqlquest", "grpo-mbpp-honest-after-sqlquest-s1", "grpo-mbpp-honest-after-sqlquest-s1"),
    ],
    # The framework attempts again (the first died creating a venv), then multi turn GRPO on repo with 3B.
    "extras": ["bash src/kaggle/attempts/verl.sh", "bash src/kaggle/attempts/openrlhf.sh", *repo_arms_3b()],
    # Seed 1 of all four arms, plus forgetting: continue the seed 1 honest SQL adapter on MBPP+ and measure
    # what it keeps on SQL.
    "train2": [
        *seeds(1),
        *grpo_arm("mbpp", "honest", 100, 512, seed=1, init="grpo-sqlquest-honest-Qwen2.5-1.5B-Instruct-s1",
                  tag="grpo-mbpp-honest-after-sqlquest-s1"),
        eval_adapter("sqlquest", "grpo-mbpp-honest-after-sqlquest-s1", "grpo-mbpp-honest-after-sqlquest-s1"),
    ],
    "train3": seeds(2),
    "offline": offline_plan(),
    # The two baseline jobs that failed (7B repo hit the 6k context; GRPO hit Kaggle's torchao), then GRPO on the
    # two environments where 1.5B succeeds often enough to learn from: MBPP+ (47% honest) and SQLQuest (37%).
    "train1": [
        f"loophole eval --env repo --split test --n 93 --k 4 --backend vllm --model {Q.format('7B')} "
        f"--sampler-kwargs '{T4X2}' --max-tokens 1024 --max-turns 10 --label eval-repo-Qwen2.5-7B-Instruct",
        *grpo_arm("mbpp", "leaky", 100, 512),
        *grpo_arm("mbpp", "honest", 100, 512),
        *grpo_arm("sqlquest", "leaky", 100, 384),
        *grpo_arm("sqlquest", "honest", 100, 384),
    ],
    # Linux sandbox numbers, MBPP+ calibration on Linux, baseline hacking propensity across three sizes,
    # and three GRPO steps to measure step time before committing GPU hours to full runs.
    "baseline": [
        "loophole probe --backend rlimit",
        "loophole calibrate mbpp --sandbox rlimit",
        *evals(Q.format("0.5B"), T4),
        *evals(Q.format("1.5B"), T4),
        *evals(Q.format("7B"), T4X2),
        "python -m loophole.train.grpo --env countdown --train-on leaky --steps 3 --vllm colocate "
        "--precision fp16 --label grpo-speedtest-countdown",
    ],
}


def run(args, capture=False):
    print("$ kaggle " + " ".join(args), flush=True)
    r = subprocess.run([KAGGLE, *args], capture_output=capture, text=True)
    if capture:
        return r
    if r.returncode:
        sys.exit(r.returncode)


def username() -> str:
    r = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("could not determine the Kaggle username; is the CLI configured?")


def tree_hash(root: Path) -> str:
    """sha256 over every file's relative path and bytes, so any change to the shipped code changes it."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build_src() -> tuple[Path, str]:
    d = BUILD / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    src = d / "src"
    shutil.copytree(ROOT / "loophole", src / "loophole", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml", "README.md"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, src / f)
    (src / "data" / "raw").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "raw" / "mbppplus.jsonl", src / "data" / "raw" / "mbppplus.jsonl")
    shutil.copytree(ROOT / "kaggle" / "attempts", src / "kaggle" / "attempts")
    for name in JUDGE_RUNS:
        run_dir = sorted((ROOT / "runs").glob(f"*-{name}"))[-1]
        dest = src / "runs" / run_dir.name
        dest.mkdir(parents=True)
        for f in ("tasks.jsonl", "rollouts.jsonl", "manifest.json", "summary.json"):
            shutil.copy(run_dir / f, dest / f)
    for name in SHIP_ADAPTERS:
        run_dir = sorted((ROOT / "runs").glob(f"*-{name}"))[-1]
        shutil.copytree(run_dir / "adapter", src / "adapters" / name)
    code = tree_hash(src)
    (d / "CODE_HASH").write_text(code + "\n")
    return d, code


def push_code(user: str) -> str:
    d, code = build_src()
    src = d / "src"
    shutil.make_archive(str(d / "src"), "zip", src)
    shutil.rmtree(src)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "loophole code", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "MIT"}]}))
    exists = run(["datasets", "status", f"{user}/{DATASET_SLUG}"], capture=True).returncode == 0
    if exists:
        run(["datasets", "version", "-p", str(d), "-m", "update code", "-q"])
    else:
        run(["datasets", "create", "-p", str(d), "-q"])
    return code


def wait_ready(user: str, code: str, tries: int = 60) -> None:
    """Wait until Kaggle serves this exact code, checked by downloading CODE_HASH back, not by trusting "ready"."""
    import tempfile
    import time

    for _ in range(tries):
        with tempfile.TemporaryDirectory() as t:
            run(["datasets", "download", f"{user}/{DATASET_SLUG}", "-f", "CODE_HASH", "-p", t, "--force"], capture=True)
            got = next((f.read_text().strip() for f in Path(t).rglob("CODE_HASH")), None)
        if got == code:
            return
        time.sleep(15)
    sys.exit(f"Kaggle never served code {code}; not pushing the kernel")


def kernel_slug(slot: str) -> str:
    """One kernel per slot, so independent plans can run at the same time."""
    return KERNEL_SLUG if slot == "a" else f"{KERNEL_SLUG}-{slot}"


def push_kernel(user: str, commands: list[str], hours: float, accelerator: str, slot: str, code: str) -> None:
    d = BUILD / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    config = {"commands": commands, "session_hours": hours, "code_hash": code}
    src = (ROOT / "kaggle" / "kernel.py").read_text().replace(
        "CONFIG = {}", f"CONFIG = json.loads({json.dumps(json.dumps(config))})", 1)
    (d / "kernel.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{kernel_slug(slot)}", "title": kernel_slug(slot), "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": accelerator, "dataset_sources": [f"{user}/{DATASET_SLUG}"], "kernel_sources": [],
        "competition_sources": [], "model_sources": []}, indent=2))
    run(["kernels", "push", "-p", str(d), "--accelerator", accelerator])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["push", "status", "logs", "output"])
    ap.add_argument("--plan", choices=sorted(PLANS))
    ap.add_argument("--hours", type=float, default=11.5)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--skip-code", action="store_true", help="reuse the code dataset already on Kaggle")
    ap.add_argument("--slot", default="a", help="which kernel to use; plans in different slots run in parallel")
    a = ap.parse_args()
    user = username()
    ref = f"{user}/{kernel_slug(a.slot)}"
    if a.mode == "status":
        return run(["kernels", "status", ref])
    if a.mode == "logs":
        return run(["kernels", "logs", ref])
    if a.mode == "output":
        out = ROOT / "kaggle" / ("output" if a.slot == "a" else f"output-{a.slot}")
        out.mkdir(parents=True, exist_ok=True)
        return run(["kernels", "output", ref, "-p", str(out), "-o"])
    if not a.plan:
        sys.exit("push needs --plan")
    if a.skip_code:
        _, code = build_src()  # the hash of the local tree; the kernel refuses to run if Kaggle holds other code
    else:
        code = push_code(user)
        wait_ready(user, code)
    push_kernel(user, PLANS[a.plan], a.hours, a.accelerator, a.slot, code)


if __name__ == "__main__":
    main()
