# The image every DockerSandbox call runs in. It is python:3.12-slim plus the
# one library MBPP+ test headers import (numpy), pinned, and nothing else: no
# compilers, no network tools, no package manager cache.
#
#   docker build -t loophole-sandbox:0.1 -f docker/sandbox.Dockerfile .
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 && rm -rf /root/.cache
USER 65534:65534
WORKDIR /work
