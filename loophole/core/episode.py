"""Multi turn episodes: tools, the agent loop, and a recorded final state.

A multi turn rollout is a list of chat messages ending with one {"role":
"state"} message holding the environment's final state as JSON. Verifiers
grade that recorded list, never a live environment, so a trajectory produced on
Kaggle can be regraded on the laptop by a verifier written later.

Tool calls use the Hermes style that Qwen2.5's chat template emits natively:
<tool_call>{"name": ..., "arguments": {...}}</tool_call>.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loophole.core.types import Message, Task

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI style function schema, which HF chat templates accept as `tools`."""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def params(**props: str) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props)}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Every well formed tool call in an assistant turn; None if a call is malformed."""
    calls = []
    for raw in _TOOL_CALL.findall(text or ""):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return None
        args = obj.get("arguments", {})
        calls.append(ToolCall(obj["name"], args if isinstance(args, dict) else {}))
    return calls


class Episode(ABC):
    tools: list[ToolSpec]

    def __init__(self, task: Task):
        self.task = task
        self.done = False
        self.events: list[dict[str, Any]] = []

    @abstractmethod
    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run one tool call and return the observation text."""

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Everything a verifier needs, JSON serialisable."""

    def dispatch(self, call: ToolCall) -> str:
        if call.name not in {t.name for t in self.tools}:
            obs = f"error: unknown tool {call.name!r}"
        else:
            try:
                obs = self.call(call.name, call.arguments)
            except (KeyError, TypeError, ValueError) as e:
                obs = f"error: {type(e).__name__}: {e}"
        self.events.append({"tool": call.name, "args": call.arguments, "obs": obs[:2000]})
        return obs


Policy = Callable[[list[Message], list[dict[str, Any]]], str]


@dataclass
class EpisodeResult:
    messages: list[Message]
    turns: int
    ended: str  # "done" | "no_tool_call" | "malformed" | "max_turns"
    events: list[dict[str, Any]] = field(default_factory=list)


def run_episode(episode: Episode, policy: Policy, max_turns: int = 12, obs_chars: int = 4000) -> EpisodeResult:
    """The agent loop: ask the policy, run its tool calls, feed back observations, record state."""
    messages: list[Message] = list(episode.task.prompt)
    schemas = [t.schema() for t in episode.tools]
    ended, turn = "max_turns", 0
    while turn < max_turns:
        turn += 1
        reply = policy(messages, schemas)
        messages.append({"role": "assistant", "content": reply})
        calls = parse_tool_calls(reply)
        if calls is None:
            ended = "malformed"
            break
        if not calls:
            ended = "no_tool_call"
            break
        for c in calls:
            obs = episode.dispatch(c)
            messages.append({"role": "tool", "content": obs[:obs_chars]})
            if episode.done:
                break
        if episode.done:
            ended = "done"
            break
    messages.append({"role": "state", "content": json.dumps(episode.state(), sort_keys=True)})
    return EpisodeResult(messages, turn, ended, episode.events)


def final_state(completion: str | list[Message]) -> dict[str, Any] | None:
    if isinstance(completion, list) and completion and completion[-1].get("role") == "state":
        return json.loads(completion[-1]["content"])
    return None
