import os
import tempfile

import pytest

# Every test run builds its SQL worlds in a throwaway cache, never in ~/.cache.
os.environ.setdefault("LOOPHOLE_CACHE", tempfile.mkdtemp(prefix="loophole-test-cache-"))


def pytest_collection_modifyitems(config, items):
    from loophole.sandbox.docker import docker_available

    if docker_available():
        return
    skip = pytest.mark.skip(reason="no Docker daemon")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)
