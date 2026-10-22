"""Multi turn environments, driven by scripted policies so the tests need no model."""

import json
import re

import pytest

from loophole.core.env import get_env
from loophole.core.episode import parse_tool_calls, run_episode
from loophole.envs.mbpp import SAMPLE


def tc(name, **args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


def scripted(replies):
    it = iter(replies)
    return lambda messages, tools: next(it, "I am done.")


def passed(env, task, messages):
    return {k: v.grade(task, messages).passed for k, v in env.verifiers().items()}


def test_tool_call_parsing():
    assert [c.name for c in parse_tool_calls(tc("ls", path="/") + " then " + tc("cat", path="/a"))] == ["ls", "cat"]
    assert parse_tool_calls("no calls here") == []
    assert parse_tool_calls("<tool_call>{not json}</tool_call>") is None


@pytest.fixture(scope="module")
def repo():
    return get_env("repo", sandbox="rlimit", path=str(SAMPLE))


def test_repo_honest_agent_passes_both(repo):
    t = repo.tasks(calibrated=False)[0]
    r = run_episode(repo.episode(t), scripted([tc("write_file", path="solution.py", content=t.meta["reference"]),
                                                tc("run_tests"), tc("submit")]))
    assert r.ended == "done"
    assert passed(repo, t, r.messages) == {"honest": True, "leaky": True}


@pytest.mark.parametrize("cheat", [
    [tc("write_file", path="run_tests.py", content="print('ok')")],
    [tc("write_file", path="tests/test_solution.py", content="def test_0():\n    pass\n")],
])
def test_repo_tampering_only_fools_leaky(repo, cheat):
    t = repo.tasks(calibrated=False)[0]
    r = run_episode(repo.episode(t), scripted([*cheat, tc("submit")]))
    assert passed(repo, t, r.messages) == {"honest": False, "leaky": True}
    assert repo.verifiers()["leaky"].grade(t, r.messages).details["tampered"]


def test_repo_unsubmitted_work_scores_zero(repo):
    t = repo.tasks(calibrated=False)[0]
    r = run_episode(repo.episode(t), scripted([tc("write_file", path="solution.py", content=t.meta["reference"])]))
    assert passed(repo, t, r.messages) == {"honest": False, "leaky": False}


def test_repo_refuses_paths_outside_the_workspace(repo):
    t = repo.tasks(calibrated=False)[0]
    ep = repo.episode(t)
    assert "error" in ep.dispatch(parse_tool_calls(tc("write_file", path="../escape.py", content="x"))[0])


def test_treasure_honest_and_spraying_agents():
    env = get_env("treasure")
    for t in env.generate(30, 0, "train"):
        r = run_episode(env.episode(t), scripted([tc("cat", path=p) for p in t.meta["required"]] +
                                                 [tc("submit", answer=t.meta["answer"])]))
        assert passed(env, t, r.messages) == {"honest": True, "leaky": True}
        assert env.verifiers()["honest"].grade(t, r.messages).details["process"] == 1.0
        ports = " ".join(sorted(set(re.findall(r"\b\d{4}\b", " ".join(t.meta["files"].values())))))
        r = run_episode(env.episode(t), scripted([tc("submit", answer=ports)]))
        assert passed(env, t, r.messages) == {"honest": False, "leaky": True}


def test_treasure_answer_is_unique_in_the_live_tree():
    """The answer port appears once outside /archive, so a correct single number cannot be a lucky guess."""
    env = get_env("treasure")
    for t in env.generate(50, 1, "train"):
        live = " ".join(c for p, c in t.meta["files"].items() if not p.startswith("/archive"))
        assert len(re.findall(rf"\b{t.meta['answer']}\b", live)) == 1
