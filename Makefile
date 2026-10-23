PY ?= .venv/bin/python
LOOPHOLE ?= .venv/bin/loophole

.PHONY: install test lint image probe data calibrate smoke scan

install:            ## create the venv and install the core with dev tools
	python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"

test:               ## the full offline suite: no GPU, no network
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check loophole tests kaggle

image:              ## the Docker sandbox image
	docker build -t loophole-sandbox:0.1 -f docker/sandbox.Dockerfile .

probe: image        ## escape probes on every available backend, written to runs/
	$(LOOPHOLE) probe --backend all

data:               ## MBPP+ from the Hugging Face datasets server
	$(LOOPHOLE) fetch mbpp

calibrate: data     ## grade every MBPP+ reference; quarantine any that fail
	$(LOOPHOLE) calibrate mbpp

smoke:              ## a real model on a few tasks, on the laptop
	$(LOOPHOLE) eval --env countdown --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --n 10 --k 2 --max-tokens 256

scan:               ## escape style code in every training run's rollouts
	for d in runs/*Z-grpo-*; do $(LOOPHOLE) scan $$d; done
