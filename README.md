# CSE 291A Project: Memory-Aware Agent Edge Serving

Admission and scheduling of concurrent multi-step agent tasks on a memory-constrained llama.cpp server.
The pinned stack is in `docs/environment.md`, and dated observations are in `docs/notebook.md`.

## Setup (Windows desktop)

1. Install Python 3.12 and create the venv outside OneDrive:
   `py -3.12 -m venv C:\llm\venv-cse291a` then `C:\llm\venv-cse291a\Scripts\python -m pip install -r requirements.txt`
2. Download the pinned llama.cpp build and model (commands in `docs/environment.md`) into `C:\llm\`.
3. Copy `configs/local.example.yaml` to `configs/local.yaml`.
4. Set the NVIDIA "CUDA – Sysmem Fallback Policy" for llama-server.exe to "Prefer No Sysmem Fallback" (see `docs/environment.md`).
5. Check everything: `make env probe vram`, or run the scripts in `scripts/` directly.

## Layout

```
configs/      server_baseline.yaml, local.yaml, workloads/, experiments/
runner/       server launcher, (next) workload gen, client, scheduler, agent, samplers
scripts/      check_env.py, probe_runtime.py, vram_vs_ctx.py
docs/         environment.md, notebook.md
analysis/ experiments/ report/ results/ (raw results live in C:\llm\results)
```
