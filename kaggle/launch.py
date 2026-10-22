"""Package the code as a private Kaggle dataset and push a script kernel that runs loophole commands.

    python kaggle/launch.py push --plan baseline      # see PLANS below
    python kaggle/launch.py status | logs | output

The kaggle CLI reads the user's own ~/.kaggle credentials; this script never
reads or prints them. Kernels are private. Everything a kernel produces lands
in /kaggle/working/runs and comes back with `output`.
"""

from __future__ import annotations

import argparse
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
T4X2 = '{"dtype": "half", "gpu_memory_utilization": 0.88, "max_model_len": 6144, "tensor_parallel_size": 2}'
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


PLANS = {
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


def push_code(user: str) -> None:
    d = BUILD / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    src = d / "src"
    shutil.copytree(ROOT / "loophole", src / "loophole", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml", "README.md"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, src / f)
    (src / "data" / "raw").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "raw" / "mbppplus.jsonl", src / "data" / "raw" / "mbppplus.jsonl")
    shutil.make_archive(str(d / "src"), "zip", src)
    shutil.rmtree(src)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "loophole code", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "MIT"}]}))
    exists = run(["datasets", "status", f"{user}/{DATASET_SLUG}"], capture=True).returncode == 0
    if exists:
        run(["datasets", "version", "-p", str(d), "-m", "update code", "-q"])
    else:
        run(["datasets", "create", "-p", str(d), "-q"])


def wait_ready(user: str, tries: int = 40) -> None:
    """A kernel pushed before its dataset finishes processing starts without it, so wait."""
    import time

    for _ in range(tries):
        r = run(["datasets", "status", f"{user}/{DATASET_SLUG}"], capture=True)
        if "ready" in r.stdout:
            return
        time.sleep(15)
    sys.exit("code dataset never became ready; not pushing the kernel")


def push_kernel(user: str, commands: list[str], hours: float, accelerator: str) -> None:
    d = BUILD / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    config = {"commands": commands, "session_hours": hours}
    src = (ROOT / "kaggle" / "kernel.py").read_text().replace(
        "CONFIG = {}", f"CONFIG = json.loads({json.dumps(json.dumps(config))})", 1)
    (d / "kernel.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "kernel.py", "language": "python",
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
    a = ap.parse_args()
    user = username()
    ref = f"{user}/{KERNEL_SLUG}"
    if a.mode == "status":
        return run(["kernels", "status", ref])
    if a.mode == "logs":
        return run(["kernels", "logs", ref])
    if a.mode == "output":
        out = ROOT / "kaggle" / "output"
        out.mkdir(parents=True, exist_ok=True)
        return run(["kernels", "output", ref, "-p", str(out), "-o"])
    if not a.plan:
        sys.exit("push needs --plan")
    if not a.skip_code:
        push_code(user)
        wait_ready(user)
    push_kernel(user, PLANS[a.plan], a.hours, a.accelerator)


if __name__ == "__main__":
    main()
