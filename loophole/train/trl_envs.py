"""TRL environment_factory adapters for the multi turn environments.

TRL (1.16) exposes an environment's public methods as tools, using their type
hints and Google style docstrings for the schema, calls reset(**row) with the
dataset row before each rollout, and passes the environment instances to the
reward functions. These classes are thin: every method forwards to the same
Episode.dispatch the offline evaluator uses, so a trajectory produced in
training and one produced in evaluation are graded by identical code.
"""

from __future__ import annotations

import json

from loophole.core.episode import ToolCall
from loophole.core.types import Task
from loophole.envs.repo import RepoEpisode
from loophole.envs.treasure import TreasureEpisode


class TreasureTRLEnv:
    def __init__(self):
        self.episode: TreasureEpisode | None = None

    def reset(self, task: str, **_) -> None:
        self.episode = TreasureEpisode(Task.from_dict(json.loads(task)))

    def ls(self, path: str) -> str:
        """List a directory.

        Args:
            path: Absolute directory path, for example /.

        Returns:
            The directory's entries, one per line.
        """
        return self.episode.dispatch(ToolCall("ls", {"path": path}))

    def cat(self, path: str) -> str:
        """Print a file.

        Args:
            path: Absolute file path.

        Returns:
            The file's contents.
        """
        return self.episode.dispatch(ToolCall("cat", {"path": path}))

    def grep(self, pattern: str) -> str:
        """Search every file for a regular expression.

        Args:
            pattern: A regular expression.

        Returns:
            Up to 20 matching lines as path:line.
        """
        return self.episode.dispatch(ToolCall("grep", {"pattern": pattern}))

    def submit(self, answer: str) -> str:
        """Submit the final answer.

        Args:
            answer: The answer, and nothing else.

        Returns:
            A confirmation.
        """
        return self.episode.dispatch(ToolCall("submit", {"answer": answer}))


class RepoTRLEnv:
    """Needs the environment's sandbox, so it is built with functools.partial(RepoTRLEnv, repo_env)."""

    def __init__(self, repo_env):
        self.repo_env = repo_env
        self.episode: RepoEpisode | None = None

    def reset(self, task: str, **_) -> None:
        self.episode = self.repo_env.episode(Task.from_dict(json.loads(task)))

    def list_files(self) -> str:
        """List every file in the repository.

        Returns:
            One path per line.
        """
        return self.episode.dispatch(ToolCall("list_files", {}))

    def read_file(self, path: str) -> str:
        """Read a file.

        Args:
            path: Path relative to the repository root.

        Returns:
            The file's contents.
        """
        return self.episode.dispatch(ToolCall("read_file", {"path": path}))

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file.

        Args:
            path: Path relative to the repository root.
            content: The full new file content.

        Returns:
            A confirmation with the number of characters written.
        """
        return self.episode.dispatch(ToolCall("write_file", {"path": path, "content": content}))

    def run_tests(self) -> str:
        """Run python run_tests.py.

        Returns:
            The test runner's output.
        """
        return self.episode.dispatch(ToolCall("run_tests", {}))

    def submit(self) -> str:
        """Finish the task.

        Returns:
            A confirmation.
        """
        return self.episode.dispatch(ToolCall("submit", {}))
