# On Windows, install GNU make with `winget install ezwinports.make`, or run the commands directly.
ifeq ($(OS),Windows_NT)
PY ?= C:/llm/venv-cse291a/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: env probe vram test

env:      ## environment snapshot (incl. model sha256)
	$(PY) scripts/check_env.py --hash-model

probe:    ## runtime feature probe (starts its own server)
	$(PY) scripts/probe_runtime.py

vram:     ## VRAM after load vs -c
	$(PY) scripts/vram_vs_ctx.py --ctx 4096 16384 32768 49152 65536 73728 81920 --np 1 4

test:
	$(PY) -m pytest -q tests
