# Lab notebook

Dated decisions and observations. Newest at the bottom. Raw data paths are under `C:\llm\results`.

## 2026-11-02 — Setup decisions

- **Platform: native Windows 11** with prebuilt llama.cpp CUDA binaries (no WSL). The runner is cross-platform Python, so the AWS cross-check runs the same code on Linux.
- **Model: Qwen3-8B Q4_K_M**, fp16 KV. Considered Qwen3.6/3.8 at the user's request and rejected them. The only open small-ish releases are 27B dense (~16–17 GB at Q4, doesn't fit in ~14 GB free VRAM) and 35B-A3B MoE (needs CPU expert offload). They also use hybrid Gated-DeltaNet attention (KV in only 1/4 of layers, per-slot recurrent state, fragile prefix reuse in llama.cpp), which would undermine the KV-pressure and reuse experiments. Qwen3-8B is a standard GQA transformer at 144 KiB/token.
- **llama.cpp pinned at b11721 (ff5a23376)**, the newest release at least 2 weeks old on 2026-11-02. Used the CUDA 13.4 build (Blackwell sm_120 support; driver supports 13.4).
- Large files live in `C:\llm\` (outside OneDrive).

## 2026-11-02 — Runtime probe (`scripts/probe_runtime.py`, results/probes/20261102-134621)

All Section-4 features are present in b11721:
- `/health`, `/props`, `/metrics` (with `--metrics`), `/slots`, `/tokenize`, `/detokenize` work. `/completion` accepts a **token array as prompt**, so the runner can control prompt length exactly.
- Per-request: `n_predict`, `ignore_eos` (gives exactly n_predict tokens), `temperature`, `cache_prompt`, `id_slot` (echoed back), `stream` (the final SSE chunk carries `timings`).
- `timings`: `cache_n, prompt_n, prompt_ms, prompt_per_token_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_token_ms, predicted_per_second`. Top level: `tokens_cached`, `tokens_evaluated`, `tokens_predicted`.
- Slot reuse: base 4646-token prompt cold on slot 0 had prompt_n=4646 (910 ms). base+suffix on the same slot had prompt_n=123, cache_n=4646 (34 ms). On the other slot it had prompt_n=4769 (850 ms). `cache_prompt:false` forces a full recompute.
- Slot save/erase/restore work (`--slot-save-path`, `POST /slots/{id}?action=save|erase|restore`). For 4784 tokens: save 225 ms, restore 376 ms (705 MB file), vs ~850 ms to recompute. **Restore from disk is only ~2.3× faster than recompute** at this length. Relevant if Phase 2 offloads idle agent contexts.
- **Gaps:** `/slots` only reports `id, is_processing, n_ctx, speculative`. There is no per-slot token count, and `/metrics` has no KV-usage gauge. KV occupancy has to be tracked by the runner from `tokens_cached` in each response, plus NVML for total VRAM.
- Speeds (single stream, short context): prefill ~5.2–5.6k tok/s, decode ~125 tok/s.

## 2026-11-02 — Defaults that would silently change the baseline

Recorded in `configs/server_baseline.yaml`:
- `--fit on` (default) auto-adjusts unset `-c`/`-ngl` to fit memory. Set `--fit off` and always pass `-c`, `-ngl all`.
- `-cram 8192` (default): a host-RAM prompt cache of up to 8 GiB, plus `--cache-idle-slots` (default on), which saves idle slots to it. This is a hidden second memory tier that would affect E4/E5. **Decision (2026-11-02): the baseline uses `-cram 0`**, so memory means VRAM only. Host-RAM caching is a possible Phase 2 lever.

## 2026-11-02 — Corpus

Six Gutenberg books: Pride and Prejudice, Moby Dick, A Tale of Two Cities, Frankenstein, Sherlock Holmes, War and Peace. Combined ~7.0 MB, roughly 1.5M tokens. Boilerplate stripped, hashes in `configs/corpus_manifest.json`. The size was chosen so that slices from random offsets essentially never share a prefix, which would create spurious cache hits.
- `-kvu` (unified KV) is on by default only when `-np` is auto. With an explicit `-np` each slot gets c/np (log line: `n_slots = 2, n_ctx_slot = 8192, kv_unified = 'false'`).
- `-sps 0.10` (slot prompt similarity) means that when `id_slot` is omitted, the server picks a slot by prompt similarity, otherwise LRU. That's relevant to the "any" slot mode in E4.
- `ignore_eos` works by logit-biasing EOS/`<|im_end|>` tokens out.

## 2026-11-02 — VRAM vs -c (`scripts/vram_vs_ctx.py`, results/vram_vs_ctx/20261102-134704)

- Linear up to -c 65536: Δ VRAM ≈ 4.8 GB + 143 KiB/token. That matches the 144 KiB/token theory, and np=1 vs np=4 give the same total.
- **Surprise: no load failure at -c 73728–98304**, even though weights + KV (up to ~18.6 GB) exceed 16 GB. Total VRAM used plateaus at ~16.1 GB and the desktop's share shrinks. This points to the Windows driver's **CUDA sysmem fallback** spilling KV into host RAM over PCIe. Implications: over-budget configs would not fail, they would get slower, which corrupts E6. Action: set "Prefer No Sysmem Fallback" for llama-server.exe, then re-run and confirm the server fails at the true limit.
- **Follow-up (results/vram_vs_ctx/20261102-135918):** with sysmem fallback set to "Prefer No Sysmem Fallback" in the NVIDIA App, `-c 81920` and `98304` fail at load (`cudaMalloc failed: out of memory`, 11520 MiB KV buffer). This confirms the earlier runs were spilling. `-c 73728` still loads but sits at 16.2/16.3 GB, ~0.5 GB below the linear prediction, likely because WDDM squeezed the compositor. Treat **-c 65536 as the practical maximum**; anything above is at the edge and depends on desktop state.
- The desktop compositor's VRAM moves between ~1.2 and 2.3 GB. Each run should record VRAM before launch, and the memory budget should leave headroom for it. Usable KV with ~14 GB free is ~9 GB, about 60–64k tokens.
