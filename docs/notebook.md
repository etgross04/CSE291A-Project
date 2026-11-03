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

## 2026-11-02 — Baseline memory model changed to one shared KV pool (TA decision, "option B")

- TA recommendation: one shared pool that a task draws from when it starts, instead of the static c/K split. `configs/server_baseline.yaml` now has `kv_unified: true` (`-kvu`) and `slot_save_path: auto` (needed for erase). Probes that documented static-split behaviour are pinned to `kv_unified: false` so they still reproduce.
- **Unified-KV probe (`scripts/probe_kvu.py`, results/probes/kvu_20261102-161716), -c 16384 -np 4:**
  - Every slot gets `n_ctx = 16384`, the whole pool (log: `n_ctx_slot = 16384, kv_unified = 'true'`).
  - Idle slots keep their KV. When a request needs cells that are not free, the server **silently purges an idle slot** (`try_clear_id: purging slot 0 with 6031 tokens`) and the request succeeds. A pinned agent in a tool call can lose its context with no error; it only shows up later as recompute.
  - When every slot is busy and the pool overflows (3 × (5000 + 1024) > 16384), **all three in-flight requests fail** with HTTP 500 `"Context size has been exceeded."` after ~9 s of retries with smaller batches. The server stays healthy. The client now classifies this as a new status, `kv_exhausted`.
  - `POST /slots/{id}?action=erase` returns 501 unless `--slot-save-path` is set.
  - A new prompt on the same slot reuses that slot's cells (no leak).
- **Scheduler changes (`runner/scheduler.py`):** `kv_budget` = pool tokens. At admission a task reserves `policy.reserve_tokens(task)`, which is its declared `peak_ctx_tokens` in the baseline. FIFO admission waits until the head fits (head-of-line blocking). A task larger than the pool is rejected → `ctx_overflow`. Invariant: each task's context is in at most one slot, and a finished task's KV is erased before its reservation is released, so KV in use ≤ Σ reservations ≤ pool and the server never needs to purge. `any` mode now places a request on the free slot that already holds its context, else the lowest free slot (affinity, like the server's own `-sps` choice). Requests log `cache_expected` so analysis can detect any purge.
  - Bug caught by the new fake-pool tests before going live: in `any` mode a task's old copy could sit on a slot already handed to another request that had not started yet, so for a moment the task's context existed twice. Fixed by waiting for that slot's next request to start.
  - Policy interface: `select_request(slot, eligible)` was replaced by `order_ready(ready)` + `place_request(req, free)`, and `reserve_tokens(task)` was added.
- **Live smoke with the pool (6 pilot tasks, K=2, -c 32768, one run, not a result):** pinned 118.1 s, any 95.7 s. 0 server purges, 0 "failed to find free space" warnings, 0 requests with fewer cached tokens than expected, 0 failures. Erase takes 5–16 ms. The `any` cache-hit fraction rose from 0.24 to 0.31 with affinity.
- **Surprise: idle KV in the unified pool slows every other request (`scripts/probe_kvu_speed.py`, results/probes/kvu_speed_20261102-163802).** One request on slot 0 (2000-token prompt, 256 out), with slots 1–3 empty vs holding 21k idle tokens:

  | mode | idle tokens in other slots | prefill ms/tok | decode ms/tok |
  |---|---|---|---|
  | static split | 0 | 0.204 | 9.81 |
  | static split | 21,000 | 0.207 | 9.88 |
  | unified | 0 | 0.211 | 9.89 |
  | unified | 21,000 | **0.387** | **14.87** |

  In the shared pool, decode is ~50% and prefill ~85% slower when other contexts are present. This is consistent with attention scanning the used range of the shared buffer (masked), but that mechanism is not verified. It explains most of the ~16–19% slowdown of the pool smoke vs the earlier static-split smoke. Implications: (1) retaining an idle agent's context in the pool costs **compute for everyone**, not just memory, which strengthens the case for a release/retain policy in Phase 2; (2) E2/E4 should log per-request `predicted_per_token_ms` against total tokens in the pool; (3) worth a dose-response sweep (idle tokens 0→48k), and telling the TA, because it makes the unified baseline slower than the static split at equal load.

## 2026-11-02 — Trace-calibrated agentic mixes

- Public traces give token counts and timing only (no text), so they calibrate distributions; content still comes from the Gutenberg corpus.
- **Mooncake FAST'25 tool & agent trace** (kvcache-ai/Mooncake @ 3cca71d, Apache-2.0; 23,608 requests; input p50 6,346 / mean 8,596 / p90 16,805; output p50 30 / mean 182 / p90 507). Pinned in `configs/trace_manifest.json` (sha256 `48a2db1a…fa5f71`), fetched by `scripts/fetch_traces.py` into `C:\llm\traces`. The generator verifies the hash and records it in each workload's meta.
  - Tried to rebuild sessions from `hash_ids` prefix chains: 46% of requests share the same first block (a template), the "next step" gap has a median of 114 s (not tool calls), and parents branch heavily. Mooncake is request-level, so only per-request lengths are used.
- New generator features: `dist: empirical` (uniform draw from a trace column, optional [min, max]), `dist: lognormal_int`, and per-class `max_peak_ctx` (deterministic redraw, count recorded). Old workloads are byte-identical (golden hash unchanged).
- `trace_agentic` / `trace_agentic_heavy` (60 tasks per seed): init ctx ~ Mooncake input (256–16,384), out ~ Mooncake output (1–2,048), steps uniform 5–12 (Continuum: BFCL 6.3 ± 2.3, SWE-bench 10.9 ± 2.1 turns), tool_out lognormal median 800 (TraceLab median append ~857), peak ≤ 32,768.
  - Tool delays: `trace_agentic` fits Continuum mini-swe-agent SWE-bench tool time (mean 0.925 s, sd 3.55 s → lognormal median 0.233 s, σ 1.66, clip 60 s); the s1000 draws give mean 0.86 s, sd 2.6 s, 83% < 1 s. `trace_agentic_heavy` fits TraceLab Claude Code (median 0.3 s, p90 13.6 s → σ 2.98, clip 120 s); the draws give median 0.37 s, p90 11.2 s, 3.2% > 60 s, and those long calls hold 56% of tool time (TraceLab: 4.9% and 85–92%; clipping trims the tail).
  - The two mixes use the same RNG stream, so they contain **identical tasks except for the tool delays**: a paired comparison for E4.
  - s1000: peak ctx p50 15.7k, max 27.7k, 466 requests, 0 redraws. A 32k pool holds ~2 median tasks; a 64k pool ~4. These mixes are memory-bound by construction.
  - Limitation: the shared prefix is 512 tokens for every task. Mooncake suggests ~46% of requests share a first block; prefix-sharing structure is not modelled yet.

## 2026-11-02 — Runner finished: samplers, run.py, validity checks

- `runner/samplers.py`: a `system.jsonl` record every 200 ms with whole-GPU VRAM and utilisation (NVML), llama-server RSS (psutil), all `llamacpp:*` values from `/metrics`, `is_processing` per slot from `/slots`, and the scheduler snapshot (pool reserved, tracked KV per slot, owners, queue lengths). A scrape takes ~3 ms median, 11 ms max.
- `runner/run.py`: fresh server → warm-up (one short request per slot, then erase every slot so the run starts cold) → workload → drain → logs + manifest.
  - Logs are written line by line, so a crash keeps partial data.
  - The manifest holds: config, workload file sha256 + meta (incl. trace sha256), planned task ids, corpus hash, full server config + command line, environment (repo commit + dirty flag, llama version, model sha256 cached in `<results_dir>/model_sha256.json`, GPU/driver, pip freeze), pre-launch and post-load VRAM, monotonic window `t0/t1`, scheduler stats, sampler stats, and purge/KV-full counts from the server log.
  - Resumable: a run dir with `DONE` is skipped; one without it is renamed `*.incomplete-<ts>` and kept.
  - `execute()` takes any client, so the whole measured path is tested with the fake client.
  - Run dirs: `<results_dir>/runs/<workload>/<policy>-<mode>_K<k>_c<pool>_<kvu|split>[_a][_n][_tag]/rep<N>`.
- `runner/env.py` now holds the environment collection that `scripts/check_env.py` used to do itself (renamed field: `gpu_mem_used_mib`).
- `analysis/checks.py` (`python -m analysis.checks RUN_DIR...`, exit 1 if invalid).
  - **fail** checks: run complete; server alive; task accounting (planned = logged, none missing/extra/duplicated); exact output lengths; prompt lengths; ok tasks ran every step; every failure explained; in-flight ≤ K; one request per slot at a time; server never deferred (`requests_deferred` = 0); pool reservation ≤ budget.
  - **warn** checks: no silent KV purge (cached < expected, or `purging slot` in the log); erase failures; sampler gaps > 1 s.
  - **info**: failures by type, cache-hit fraction.
- Tests: 69 pass. The fake client now models slot contents (cached tokens = longest common prefix, like the server).
- **Live end-to-end (6 pilot tasks, K=2, -c 32768, pinned, tag `smoke`): VALID, every check ok.** 106.5 s, 534 samples (max gap 0.23 s), 0 scrape errors. Pre-launch VRAM 2,045 MiB, after load 11,461 MiB, peak 11,840 MiB. Cache hit 0.86, 0 purges. Model sha256 `d98cdcbd…` matches.
- Note: the manifest recorded `repo_dirty: true`. Commit before measured runs so every result maps to a commit.

## 2026-11-03 — Pilots at pool 65,536 (K=4, FIFO, -kvu): all VALID

Pool fixed at **65,536 tokens** for the main experiments. Runs in `C:\llm\results\runs\{pilot_s42,trace_agentic_s1000}\fifo-*_K4_c65536_kvu*\rep0`, repo commit 5943f33, every check ok (0 purges, 0 erase failures, max sampler gap 0.24 s, `requests_deferred` 0).

| run | tasks | duration | p50 / p95 task | queue share | gen tok/s | cache hit (steps>0) | prompt tok computed | decode ms/tok p50 | KV reserved mean / max | tracked KV mean |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot pinned | 30/30 | 240.6 s | 54.5 / 78.2 s | 46% | 139.9 | 0.858 | 159k | 13.5 | 25.5k / 36.8k | 15.7k |
| pilot any | 30/30 | 306.3 s | 75.1 / 107.3 s | 31% | 109.9 | 0.155 | 634k | 26.7 | 49.1k / 63.2k | 17.4k |
| trace pinned (15 tasks) | 15/15 | 221.8 s | 142.1 / 220.4 s | 62% | 64.3 | 0.915 | 211k | 19.3 | 54.9k / 62.7k | 38.8k |
| trace any (15 tasks) | 15/15 | 237.4 s | 159.0 / 237.3 s | 62% | 60.1 | 0.884 | 245k | 17.5 | 55.2k / 62.7k | 38.5k |

Observations:
- **Pilot, pinned is slot-bound, not memory-bound.** One task per slot, so tasks wait for admission (3.25 waiting on average) while slots are busy only 2.2 of 4 on average; the rest are held idle through tool delays. Reserved KV never passed 37k of 65k.
- **Pilot, `any` is 27% slower despite busier slots (3.74 of 4)** because contexts migrate between slots: 7.3 active tasks share 4 slots, the cache hit falls 0.86 → 0.16 and prompt recompute rises 4× (159k → 634k tokens). Those long prefills batch with other slots' decode and double per-token decode time (requests overlapping another's prefill: 14.1 ms/tok pinned vs 27.0 ms/tok any; without overlap ~11–12 in both). Recompute cost shows up in everybody's decode, not only in the recomputing request. Main E4 question.
- **Trace mix is pool-bound.** Reservation averages 55k of 65k and up to 5.6 tasks wait for admission; queueing is 62% of task time. Pinned ≈ any here (pool admits ~K tasks, so contexts rarely migrate).
- **Reserve-at-peak wastes ~30% of the pool**: tracked KV averages 38.8k while 54.9k is reserved (pilot pinned: 15.7k vs 25.5k). Candidate Phase 2 lever (reserve by estimated current need, grow on demand), to be weighed against the `kv_exhausted` risk.
- Decode p95 is 300–430 ms/tok on the trace mix: Mooncake outputs are often a few tokens, so one stall behind another slot's 10k-token prefill dominates the per-token time. Report decode per token only for outputs ≥ 32 tokens, or report TTFT separately.
- **VRAM headroom is thin at 64k**: peak 16,061 of 16,303 MiB (any pilot). The server grew ~100–370 MiB after load during the run; pre-launch desktop VRAM varied 1,430–2,332 MiB between runs. Close other GPU apps before measured runs; record a run as failed (not retried silently) if the server fails to allocate.
- Policy overhead is negligible (3.4–6.6 ms per run); erase total 0.05–0.74 s per run.

## 2026-11-03 — Decision: cap tasks per run in E2 (scaled with K), keep 3 reps

- E2 runs the first `n` tasks of each workload file: `n = min(num_tasks, max(20, 6·K))`. K=1, 2 → 20; K=4 → 24; K=8 → 48; K=16 → all tasks. The capped tasks are an exact prefix of the same deterministic file, so every config sees identical work. The cap is set per point in the experiment YAML and recorded in the run dir name (`_n<N>`) and manifest.
- Why scale with K: only ~K tasks run at once (fewer if the pool binds), so the run needs several tasks per slot to reach steady state before the drain. A flat cap would make high-K runs mostly ramp-up and drain.
- Found while sizing this: every mix is closed-loop with 16 agents, and at K=16 the full files are already short: tasks per slot = agentic 3.75, trace mixes 3.75, blend 5, chat 7.5, long_context/long_output 2.5, stress 2. K=16 points will be drain-heavy even uncapped. Options: more tasks per file for E2, or report throughput over the steady window (active tasks = concurrency) as well as the whole run.
- Capped runs report p50/p95 only (p99 at n ≤ 24 is just the slowest task).
- `blend` prefixes drift from the class weights (first 24: chat 9 / agentic 6 / long_context 7 / long_output 2; full 80: 29/26/17/8). Record class counts per run; compare blend across K, not against the full file.

## 2026-11-03 — Sweep runner + E1–E6 definitions; E1 and E4 pilots VALID

- `experiments/run_sweep.py CONFIG [--pilot] [--dry-run] [--only k=v] [--max-runs N] [--stop-on-error]`. Each point × rep goes to `runner.run.run_one` (fresh server, /health, discarded warm-up, workload, drain, DONE), then `analysis.checks`. Resumable: DONE runs are skipped, including identical points already run by another experiment (same run dir). INVALID runs stay and are marked. Writes `<results_dir>/sweeps/<E>[-pilot]/{plan.json, log.jsonl, status.json}`.
- `configs/experiments/E1..E6.yaml` (format in `configs/experiments/README.md`). Planned: E1 3, E2 90, E3 12, E4 42, E5 18, E6 30 = 195 runs. Pilot points are rep0 of the full grid, so they count toward it.
- New run options, off by default and marked in the run dir name: `cache_prompt` (`_nocache`), `erase_finished` (`_keep`), `admission_gate` (`_nogate`: -kvu with no reservation; finished slots are still erased so only the gate changes). `cache_expected` is 0 when caching is off. With erase off, server purges of finished tasks' KV no longer warn.
- Generator: `class_order: cycle` for exact grids (E1 `calib`), variants via `derive_yaml` (base YAML + dotted overrides, written to `<results_dir>/workloads/derived/`), and `write(..., refuse_change=True)` so a sweep never overwrites a task file earlier runs used. Golden hash unchanged. New workloads `calib`, `growth_short`, `growth_long`.
- Option A (decision): K=16 E2 points use the full files; analysis will also report throughput over the steady window (all slots busy).
- Tests: 99 pass.

E1 pilot (`calib_s7`, K=1, 42 single requests, 120 s, VALID):
- Prefill ~6.2k tok/s for 512–4k prompts; 5.6k at 8k, 4.9k at 16k (5.6k at 256: fixed overhead).
- Decode 7.5 ms/tok at a 256-token context, 8.2 at 4k, 9.0 at 8k, 10.6 at 16k (+40% from context alone).

E4 pilot (agentic, constant delays, K=4, 24 tasks, rep0, all VALID):

| delay | mode | duration | p50 / p95 task | cache hit | decode ms/tok p50 | busy slots (of 4) |
|---|---|---|---|---|---|---|
| 0 s | pinned | 110.0 s | 61.0 / 76.7 | 0.859 | 14.7 | 3.84 |
| 0 s | any | 189.9 s | 103.6 / 153.1 | 0.126 | 22.5 | 3.92 |
| 5 s | pinned | 224.9 s | 126.4 / 160.9 | 0.859 | 11.9 | 1.59 |
| 5 s | any | 191.0 s | 102.2 / 157.7 | 0.138 | 23.2 | 3.82 |

- **Crossover between 0 and 5 s.** With no tool time, pinned is 1.7× faster (`any` loses its cache to slot migration). At 5 s, `any` is 15% faster: pinned holds slots idle (1.59 of 4 busy).
- `any` takes ~190 s at both delays: it hides the tool time completely and pays recompute instead. Pinned pays the tool time. The full E4 grid (1, 15, 30 s) locates the crossover. Phase 2 direction: release a slot during long tool calls but keep (or save/restore) its context, getting both effects.
