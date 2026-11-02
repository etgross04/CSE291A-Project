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

## 2026-11-02 — Workload generator

- `runner/corpus.py` tokenizes the corpus once with the pinned server: **1,674,729 tokens**, tokens sha256 `26b1b98f…9d91`, cached at `C:\llm\corpus\corpus_tokens.npy`. The first 8192 tokens are reserved for shared prefixes. Task slices are drawn from after that region and sent as token arrays, so prompt lengths are exact.
- `runner/workload_gen.py` is a pure function of (YAML, seed, corpus token count). It uses only `random.Random.random()`, with a per-task RNG seeded from (seed, index). Output is LF-only JSON with sorted keys. The golden hash for pilot s42 is identical under Python 3.12.10 and 3.14.3.
- Semantics: prompt(step j) = prefix + init_ctx + Σ_{i<j}(response_i + tool_out_i). There is no tool call after the last step.
- Mixes (seed 1000): chat peak ≤1.8k; long_context peak 6.7–13.1k (fits 16k slots, overflows 8k); agentic 4–8 steps with peak p50 ~6k, max ~8.7k; blend 50/20/30; stress fixed at 13,824 tokens/task. Dev seeds 1000–1004, eval 2000–2004.
- Note for E2: under the static split at -c 65536, K=8 gives 8192-token slots. A few agentic tasks (max peak ~8.7k) and all long_context tasks will overflow there. That's expected, and it will be counted as ctx_overflow.
- Open: responses are appended as tokens, so the client must get generated token ids. Check whether /completion can return them (do not assume a field name). Otherwise re-tokenize the text and log any length mismatch.

## 2026-11-02 — Long-output mix

- Added `long_output` (1 step, ctx 512–2k, out 512/768/1024) because the course spec asks for short *and long responses*, and earlier mixes stopped at 256. A 1k-token generation holds a slot ~8 s single-stream. Blend is now chat 0.4 / long_output 0.1 / long_context 0.2 / agentic 0.3.

## 2026-11-02 — Client probe (`scripts/probe_client_fields.py`) and `runner/client.py`

- `return_tokens: true` puts the generated ids in every SSE chunk (`tokens`). Stream ids equal non-stream ids, and re-tokenizing the text round-trips, so the agent appends response **ids** directly.
- Stream chunk 0 reports `id_slot: -1`. The real slot is only in the final chunk, along with `timings`, `truncated`, `stop_type`, `tokens_cached`.
- **Silent overflow:** if the prompt fits but prompt + n_predict exceeds the slot's n_ctx, the server returns **HTTP 200** with `truncated: true` and fewer tokens (96 of 256 asked, with a 4000-token prompt in a 4096 slot). `stop_type` is `"limit"` here and also for a normal finish. The client classifies short outputs as `ctx_overflow` (truncated=true) or `truncated`.
- Prompt > n_ctx: HTTP 400, `error.type = "exceed_context_size_error"` (also for stream requests) → `ctx_overflow`.
- After a request, the slot KV holds prompt + generated − 1 tokens (the last sampled token is never evaluated). Smoke test: step-2 prompt of 1764 tokens reused 1563 and computed 201.
- **The server queues internally:** 4 concurrent requests on 2 slots all succeed, with the 2 extra waiting (1.4 s vs 2.7 s). The scheduler must keep in-flight requests ≤ K, otherwise server-side waiting is counted as model time instead of queueing time. `requests_deferred` in /metrics can be used to assert this.

## 2026-11-02 — Scheduler + agent, first live smoke (`scripts/smoke_agents.py`, results/smoke_agents/)

- `runner/scheduler.py`: admission queue (tasks) + ready queue (requests), event-driven dispatch over K slots. It always sends an explicit `id_slot` and keeps in-flight ≤ K. Policy hooks: `pick_admission`, `assign_slot`, `select_request`, `on_request_done`, `on_task_done`. Time spent in policy decisions is accumulated as `policy_overhead_s`. Baselines: `FifoPolicy("pinned")` (a task reserves a slot for its lifetime) and `FifoPolicy("any")` (lowest free slot, no reservation).
- `runner/agent.py`: task loop per the generator semantics. It stops at the first failed request and checks the task deadline before each step and tool call. Closed- and open-loop drivers.
- 39 tests pass (fake client: in-flight bound, no slot double-booking, FIFO order, pinned exclusivity, prompt construction, failure stops the task and frees the slot, deadline, open-loop arrival times).
- **Smoke (first 6 pilot tasks, K=2, 4 agents, -c 32768, one run, NOT a result):**

  | mode | wall | mean task | mean queue | prompt tokens computed | cache hit (steps>0) |
  |---|---|---|---|---|---|
  | pinned | 101.9 s | 53.3 s | 22.2 s | 30,946 | 0.86 |
  | any | 80.6 s | 43.5 s | 6.1 s | 128,811 | 0.24 |

  Both modes: 6/6 tasks ok, every output had the requested length, every prompt matched the expected length, max in-flight 2, policy overhead ~1 ms total.
- **First look at the E4 tradeoff:** `any` recomputed 4.2× more prompt tokens but finished ~21% sooner. Pinned slots sit idle during tool delays (median 2 s, up to 14 s) while other tasks wait to be admitted. At ~5.5k tok/s prefill, the extra ~98k recomputed tokens cost ~18 s of GPU time, which is less than the idle reservation time. Expect this to flip as contexts grow (recompute cost ∝ context) or tool delays shrink. This is exactly what E4 has to map out. Possible Phase 2 direction: release or reassign a slot during long tool calls and keep it for short ones.
