# On Windows, install GNU make with `winget install ezwinports.make`, or run the commands directly.
ifeq ($(OS),Windows_NT)
PY ?= C:/llm/venv-cse291a/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: env probe probe-kvu vram corpus traces workloads test check

env:      ## environment snapshot (incl. model sha256)
	$(PY) scripts/check_env.py --hash-model

probe:    ## runtime feature probe (starts its own server)
	$(PY) scripts/probe_runtime.py

probe-kvu: ## unified KV pool behaviour (purge, overflow, erase) and idle-KV slowdown
	$(PY) scripts/probe_kvu.py
	$(PY) scripts/probe_kvu_speed.py

vram:     ## VRAM after load vs -c
	$(PY) scripts/vram_vs_ctx.py --ctx 4096 16384 32768 49152 65536 73728 81920 --np 1 4

corpus:   ## fetch + verify Gutenberg corpus, then tokenize it with the pinned server
	$(PY) scripts/fetch_corpus.py
	$(PY) scripts/fetch_corpus.py --verify
	$(PY) -m runner.corpus build

traces:   ## fetch + verify the public traces used to calibrate trace_* workloads
	$(PY) scripts/fetch_traces.py

workloads: ## generate tasks.jsonl for every workload x dev/eval seed (+ pilot)
	$(PY) -m runner.workload_gen --all

test:
	$(PY) -m pytest -q tests

check:    ## validity checks for run dirs: make check RUNS="C:/llm/results/runs/..."
	$(PY) -m analysis.checks $(RUNS)
