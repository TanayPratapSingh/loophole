"""Kaggle script kernel: install, unpack the code, run the planned loophole commands in order.

kaggle/launch.py replaces the CONFIG line before each push. Each command's
output goes to /kaggle/working/logs/NN.log, and /kaggle/working/commands.json
records every command's exit code and duration as it finishes, so a session
that times out still says exactly what ran. A failing command is recorded and
the next one runs; nothing here retries or hides a failure.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

CONFIG = {}  # replaced by launch.py

T0 = time.time()
WORK = "/kaggle/working"
os.chdir(WORK)
os.makedirs(f"{WORK}/logs", exist_ok=True)


def sh(cmd, check=True, timeout=None, log=None):
    print(f"$ {cmd}", flush=True)
    if log:
        with open(log, "w") as f:
            r = subprocess.run(cmd, shell=True, timeout=timeout, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, shell=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


print("CONFIG", json.dumps(CONFIG), flush=True)
sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", check=False)
sh(f"{sys.executable} -m pip install -q vllm", check=False, timeout=2400, log=f"{WORK}/logs/install_vllm.log")
sh(f"{sys.executable} -m pip install -q trl peft datasets accelerate duckdb pyyaml", check=False, timeout=1200,
   log=f"{WORK}/logs/install_trl.log")
# vLLM pulls a torch built for a newer CUDA than Kaggle's preinstalled torchaudio, and transformers imports
# torchaudio when present, which then refuses to load. Nothing here uses audio, so remove it.
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio", check=False)
# Kaggle ships torchao 0.10; peft refuses to import alongside any torchao older than 0.16, and nothing here
# uses it. Same fix the drgrpo replication needed.
sh(f"{sys.executable} -m pip uninstall -y -q torchao", check=False)
sh(f"{sys.executable} -c \"import peft, torch, transformers, trl, vllm; print('torch', torch.__version__, 'cuda', "
   f"torch.version.cuda, 'gpus', torch.cuda.device_count(), 'transformers', transformers.__version__, 'trl', "
   f"trl.__version__, 'vllm', vllm.__version__)\"", check=False)

# Kaggle extracts uploaded archives, so the code arrives either as src.zip or as an extracted tree.
shutil.rmtree(f"{WORK}/src", ignore_errors=True)
trees = sorted(glob.glob("/kaggle/input/**/pyproject.toml", recursive=True))
zips = sorted(glob.glob("/kaggle/input/**/src.zip", recursive=True))
if trees:
    shutil.copytree(os.path.dirname(trees[0]), f"{WORK}/src")
elif zips:
    shutil.unpack_archive(zips[0], f"{WORK}/src")
else:
    sh("find /kaggle/input -maxdepth 4 | head -50", check=False)
    raise SystemExit("code dataset not found: no pyproject.toml or src.zip under /kaggle/input")
sh(f"{sys.executable} -m pip install -q --no-deps -e {WORK}/src")

os.environ["LOOPHOLE_DATA"] = f"{WORK}/src/data"
os.environ["LOOPHOLE_CACHE"] = f"{WORK}/cache"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

budget_s = CONFIG.get("session_hours", 11.5) * 3600
done = []
for i, cmd in enumerate(CONFIG.get("commands", [])):
    left = budget_s - (time.time() - T0)
    if left < 900:
        done.append({"i": i, "cmd": cmd, "skipped": "less than 15 minutes left"})
        continue
    t = time.time()
    try:
        code = sh(cmd, check=False, timeout=left - 600, log=f"{WORK}/logs/{i:02d}.log")
    except subprocess.TimeoutExpired:
        code = "timeout"
    done.append({"i": i, "cmd": cmd, "exit": code, "minutes": round((time.time() - t) / 60, 1)})
    with open(f"{WORK}/commands.json", "w") as f:
        json.dump(done, f, indent=1)
    print(f"[{i}] exit {code} after {done[-1]['minutes']} min", flush=True)

with open(f"{WORK}/commands.json", "w") as f:
    json.dump(done, f, indent=1)
shutil.rmtree(f"{WORK}/cache", ignore_errors=True)  # regenerated SQL worlds; not worth downloading
for ck in glob.glob(f"{WORK}/runs/*/checkpoints"):
    shutil.rmtree(ck, ignore_errors=True)  # intermediate checkpoints; the final adapter is kept
print(f"total {round((time.time() - T0) / 3600, 2)} h", flush=True)
