# Environment (pinned stack)

Machine-readable snapshot: `python scripts/check_env.py --hash-model --out <run_dir>/env.json`. Every run manifest records the same fields.

## Primary testbed (desktop)

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (native, no WSL) |
| CPU | AMD Ryzen 9 7900X (12C/24T) |
| RAM | 32 GB (31.1 GiB visible) |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16303 MiB, Blackwell |
| Driver | 616.56, CUDA driver API 13.4, **WDDM** (display attached) |
| Idle VRAM used by desktop | ~1.2–2.3 GB, varies (see notebook 2026-11-02) |
| Python | 3.12.10, venv at `C:\llm\venv-cse291a`, pins in `requirements.txt` |

## Inference runtime

| Item | Value |
|---|---|
| llama.cpp | release **b11721**, commit **ff5a23376725e5a889fa0bfb987b7541dc304da0** (2026-10-18) |
| Build | official prebuilt `llama-b11721-bin-win-cuda-13.4-x64.zip` + `cudart-llama-bin-win-cuda-13.4-x64.zip` |
| Install path | `C:\llm\llama.cpp\b11721\llama-server.exe` |
| `--version` | `version: 0.7.0-dev (build 11721, commit ff5a23376)`, built with Clang 20.1.8 |
| Help snapshot | `C:\llm\llama.cpp\b11721\help.txt` |

Reinstall:
```
B=https://github.com/ggml-org/llama.cpp/releases/download/b11721
curl -LO $B/llama-b11721-bin-win-cuda-13.4-x64.zip
curl -LO $B/cudart-llama-bin-win-cuda-13.4-x64.zip   # unzip both into C:\llm\llama.cpp\b11721
```

## Model

| Item | Value |
|---|---|
| Model | Qwen3-8B, GGUF **Q4_K_M** (official `Qwen/Qwen3-8B-GGUF`) |
| HF revision | `7c41481f57cb95916b40956ab2f0b139b296d974` |
| File | `C:\llm\models\Qwen3-8B-Q4_K_M.gguf`, 5,027,783,488 bytes |
| sha256 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` (verified) |
| Architecture | 36 layers, 8 KV heads, head_dim 128 (GQA), standard attention (not hybrid) |
| KV cache | fp16 K and V: 2·36·8·128·2 = **147,456 B/token (144 KiB)**. Confirmed by slot save: 705,506,968 B / 4784 tokens = 147,472 B/token |
| Native context | 32k (40,960 n_ctx_train); per-slot ctx must stay ≤ this |

Download:
```
curl -L -o Qwen3-8B-Q4_K_M.gguf https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf
```

## Fixed server flags

See `configs/server_baseline.yaml`; `runner/server.py` builds the command line from it. Defaults that would otherwise skew measurements are set explicitly: `--fit off`, `-ngl all`, `-fa on`, `-cram 0`, `--no-cache-idle-slots`, `-no-kvu`.

## Required driver setting (Windows)

NVIDIA App (the Control Panel is gone on this driver) → program settings for `C:\llm\llama.cpp\b11721\llama-server.exe` → **CUDA sysmem fallback = Prefer No Sysmem Fallback**. Do this so allocations beyond VRAM do not silently spill to host RAM instead of failing.
Status: **set 2026-11-02, verified.** `-c 81920` now fails at load with `cudaMalloc failed: out of memory`.

## Corpus

Six Project Gutenberg books (`scripts/fetch_corpus.py`) → `C:\llm\corpus\corpus.txt`, 7,015,423 bytes, sha256 `d3bf5c23238a1e45a3a0ecd365ec675a3dad5cba7c3808327de83e0a4582bd86`. Per-book hashes in `configs/corpus_manifest.json`; check with `--verify`.

## Public traces (workload calibration only)

Mooncake FAST'25 tool & agent trace, kvcache-ai/Mooncake @ `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb` (Apache-2.0), `toolagent_trace.jsonl`, 23,608 rows, sha256 `48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71`. Pinned in `configs/trace_manifest.json`; `scripts/fetch_traces.py` downloads into `traces_dir` (`C:\llm\traces`) and verifies the hash (`--verify` to check only).

## Local paths

`configs/local.yaml` (gitignored, copy from `configs/local.example.yaml`): llama-server path, model path, results dir `C:\llm\results`.
