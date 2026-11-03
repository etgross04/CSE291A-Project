# On Windows, install GNU make with `winget install ezwinports.make`, or run the commands directly.
ifeq ($(OS),Windows_NT)
PY ?= C:/llm/venv-cse291a/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: env probe probe-kvu vram corpus traces workloads test check e1 e2 e3 e4 e5 e6 pilots plan

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

# Sweeps (experiments/run_sweep.py): resumable; rerun the same target to continue. ARGS="--only k=4 --max-runs 3"
SWEEP = $(PY) -m experiments.run_sweep
E1 = configs/experiments/E1_calibration.yaml
E2 = configs/experiments/E2_concurrency.yaml
E3 = configs/experiments/E3_context_growth.yaml
E4 = configs/experiments/E4_idle_periods.yaml
E5 = configs/experiments/E5_reuse.yaml
E6 = configs/experiments/E6_memory_budget.yaml

e1: ; $(SWEEP) $(E1) $(ARGS)
e2: ; $(SWEEP) $(E2) $(ARGS)
e3: ; $(SWEEP) $(E3) $(ARGS)
e4: ; $(SWEEP) $(E4) $(ARGS)
e5: ; $(SWEEP) $(E5) $(ARGS)
e6: ; $(SWEEP) $(E6) $(ARGS)

pilots:   ## the 2-3 point pilot of every experiment
	for e in $(E1) $(E2) $(E3) $(E4) $(E5) $(E6); do $(SWEEP) $$e --pilot || exit 1; done

plan:     ## dry run: what every experiment would run and what is already DONE
	for e in $(E1) $(E2) $(E3) $(E4) $(E5) $(E6); do $(SWEEP) $$e --dry-run | head -1; done
