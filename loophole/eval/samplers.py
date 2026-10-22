"""Samplers: one interface over vLLM, Hugging Face transformers, MLX, and any OpenAI compatible server.

Every sampler takes a batch of conversations and returns n completions for
each, as raw text. Raw text matters here: a server side tool parser that turns
<tool_call> text into structured calls would hide malformed calls, and a
policy that learns to emit malformed calls is itself a finding.

StubSampler is a deterministic stand in for tests and the demo. It is not a
model, and anything it produces is labeled stub in the run manifest.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from loophole.core.types import Message


class Sampler(Protocol):
    name: str
    model: str

    def generate(self, conversations: list[list[Message]], n: int = 1, temperature: float = 1.0,
                 max_tokens: int = 512, tools: list[dict] | None = None, seed: int = 0) -> list[list[str]]: ...


def _clean(conv: list[Message]) -> list[Message]:
    return [m for m in conv if m["role"] != "state"]


class VLLMSampler:
    name = "vllm"

    def __init__(self, model: str, dtype: str = "auto", gpu_memory_utilization: float = 0.85, max_model_len: int = 8192,
                 tensor_parallel_size: int = 1, enable_lora: bool = False, **kw):
        from vllm import LLM

        self.model = model
        self.llm = LLM(model=model, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                       enable_lora=enable_lora, **kw)

    def generate(self, conversations, n=1, temperature=1.0, max_tokens=512, tools=None, seed=0):
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=seed)
        outs = self.llm.chat([_clean(c) for c in conversations], sp, tools=tools, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outs]


class HFSampler:
    name = "hf"

    def __init__(self, model: str, device: str | None = None, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()

    def generate(self, conversations, n=1, temperature=1.0, max_tokens=512, tools=None, seed=0):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        texts = [self.tok.apply_chat_template(_clean(c), tools=tools, add_generation_prompt=True, tokenize=False)
                 for c in conversations]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.lm.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                   max_new_tokens=max_tokens, num_return_sequences=n,
                                   pad_token_id=self.tok.pad_token_id)
        gen = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [gen[i * n:(i + 1) * n] for i in range(len(conversations))]


class MLXSampler:
    name = "mlx"

    def __init__(self, model: str, adapter_path: str | None = None):
        from mlx_lm import load

        self.model = model
        self.lm, self.tok = load(model, adapter_path=adapter_path)

    def generate(self, conversations, n=1, temperature=1.0, max_tokens=512, tools=None, seed=0):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temperature)
        out = []
        for c in conversations:
            prompt = self.tok.apply_chat_template(_clean(c), tools=tools, add_generation_prompt=True, tokenize=False)
            out.append([generate(self.lm, self.tok, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
                        for _ in range(n)])
        return out


class OpenAISampler:
    """Any /v1/chat/completions server: vLLM serve, SGLang, mlx_lm.server, Ollama."""

    name = "openai"

    def __init__(self, model: str, base_url: str = "http://localhost:8080/v1", api_key: str = "EMPTY", timeout: int = 300):
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    def _one(self, conv: list[Message], n: int, temperature: float, max_tokens: int, tools, seed: int) -> list[str]:
        body: dict[str, Any] = {"model": self.model, "messages": _clean(conv), "n": n, "temperature": temperature,
                                "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            choices = json.load(r)["choices"]
        texts = []
        for ch in choices:
            msg = ch["message"]
            text = msg.get("content") or ""
            for call in msg.get("tool_calls") or []:  # rebuild text if the server parsed the calls out
                fn = call.get("function", {})
                args = fn.get("arguments", "{}")
                args = json.loads(args) if isinstance(args, str) else args
                text += f'\n<tool_call>{json.dumps({"name": fn.get("name"), "arguments": args})}</tool_call>'
            texts.append(text)
        return texts

    def generate(self, conversations, n=1, temperature=1.0, max_tokens=512, tools=None, seed=0):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(lambda c: self._one(c, n, temperature, max_tokens, tools, seed), conversations))


class StubSampler:
    """Not a model. Returns completions from a function of the conversation, for tests and the demo."""

    name = "stub"

    def __init__(self, fn: Callable[[list[Message], int], str], model: str = "stub"):
        self.fn, self.model = fn, model

    def generate(self, conversations, n=1, temperature=1.0, max_tokens=512, tools=None, seed=0):
        base = 0 if seed is None else seed * 1000
        return [[self.fn(c, base + i) for i in range(n)] for c in conversations]


def get_sampler(backend: str, model: str, **kw) -> Sampler:
    return {"vllm": VLLMSampler, "hf": HFSampler, "mlx": MLXSampler, "openai": OpenAISampler}[backend](model, **kw)
