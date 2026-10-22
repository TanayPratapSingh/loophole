"""OpenRLHF adapter: JSONL prompts plus a reward source in both forms OpenRLHF accepts.

    loophole export openrlhf --env countdown --out data/export
    loophole serve-reward --env countdown --verifier leaky --port 8099
    openrlhf.cli.train_ppo_ray ... --prompt_data data/export/countdown/train.jsonl --input_key prompt \\
        --label_key task --apply_chat_template --remote_rm_url http://127.0.0.1:8099/reward

OpenRLHF posts {"query": [...], "prompts": [...], "labels": [...]} where each
query is the chat templated prompt followed by the response. The response is
the query with its prompt prefix removed and the end of turn marker stripped.
The reply carries "rewards" (what is trained on), "scores", and "extra_logs"
with the honest pass rate, leaky pass rate, and hack rate of the batch.

reward_func below has the same contract for OpenRLHF's in process
--remote_rm_url path/to/file.py mode, with no server.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from loophole.core.env import Environment, get_env
from loophole.core.types import Task

END_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>")


def export(env: Environment, out: Path, n: int = 2000, seed: int = 0) -> list[Path]:
    d = out / env.name
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for split, k in (("train", n), ("val", max(1, n // 10))):
        if hasattr(env, "tasks"):
            k = min(k, sum(t.split == split for t in env.tasks()))
        tasks = env.generate(k, seed, split)
        p = d / f"{split}.jsonl"
        p.write_text("".join(json.dumps({"prompt": t.prompt, "task": t.to_json()}) + "\n" for t in tasks))
        paths.append(p)
    return paths


def response_of(query: str, prompt: str) -> str:
    r = query[len(prompt):] if query.startswith(prompt) else query
    for m in END_MARKERS:
        r = r.split(m)[0]
    return r


@lru_cache(maxsize=16)
def _verifiers(env_name: str):
    return get_env(env_name).verifiers()


def grade_batch(queries: list[str], prompts: list[str], labels: list[str], train_on: str) -> dict:
    rewards, honest, leaky, hacked = [], 0, 0, 0
    for q, p, lab in zip(queries, prompts, labels, strict=True):
        task = Task.from_dict(json.loads(lab))
        text = response_of(q, p)
        gs = {}
        for name, v in _verifiers(task.env).items():
            try:
                gs[name] = v.grade(task, text)
            except Exception:
                gs[name] = None
        ok = {k: bool(g and g.passed) for k, g in gs.items()}
        rewards.append(float(gs[train_on].reward if gs[train_on] else 0.0))
        honest += ok["honest"]
        leaky += ok["leaky"]
        hacked += ok["leaky"] and not ok["honest"]
    n = max(1, len(rewards))
    return {"rewards": rewards, "scores": rewards,
            "extra_logs": {"loophole_honest": honest / n, "loophole_leaky": leaky / n, "loophole_hack_rate": hacked / n}}


def reward_func(queries, prompts, labels):
    """In process form for OpenRLHF's --remote_rm_url <this file>; returns plain lists (OpenRLHF tensorises)."""
    return grade_batch(list(queries), list(prompts), list(labels), os.environ.get("LOOPHOLE_VERIFIER", "leaky"))


def serve(env: Environment, verifier: str, host: str = "127.0.0.1", port: int = 8099) -> None:
    if verifier not in env.verifiers():
        raise KeyError(f"{env.name} has no verifier {verifier!r}")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server naming)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                out = grade_batch(body["query"], body.get("prompts", [""] * len(body["query"])), body["labels"],
                                  verifier)
                code = 200
            except (KeyError, ValueError, TypeError) as e:
                out, code = {"error": f"{type(e).__name__}: {e}"}, 400
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"loophole reward server for {env.name}/{verifier} on http://{host}:{port}/reward", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
