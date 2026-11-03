# Workload sources

This file lists every external source behind the benchmark workloads in `configs/workloads/`, and what each one contributed. Public traces provide token counts and timings only, with no text. They set the shape of the distributions. The tokens themselves always come from the fixed Gutenberg corpus, so every prompt is exact and reproducible.

Detailed numbers and the reasoning behind each choice are in `docs/notebook.md`:
- 2026-11-02, "Workload generator"
- 2026-11-02, "Long-output mix"
- 2026-11-02, "Trace-calibrated agentic mixes"

## Sources incorporated

### Mooncake tool & agent trace (Kimi / Moonshot AI, FAST '25)

- **Source.** Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*.
  - Trace README: https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md
  - Pinned file: `toolagent_trace.jsonl` at commit `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`, Apache-2.0, sha256 `48a2db1a…fa5f71`. Recorded in `configs/trace_manifest.json` and fetched by `scripts/fetch_traces.py`.
- **What it is.** 23,608 production requests from a tool/agent service.
  - Fields: `timestamp`, `input_length`, `output_length`, `hash_ids` (512-token prefix blocks).
  - Input length: p50 6,346, mean 8,596, p90 16,805 tokens.
  - Output length: p50 30, mean 182, p90 507 tokens.
- **Used for** `trace_agentic` and `trace_agentic_heavy`:
  - `init_ctx`: empirical draw from `input_length`, limited to 256–16,384 tokens.
  - `out` (tokens per step): empirical draw from `output_length`, limited to 1–2,048 tokens.
- **Not used.**
  - *Session structure.* The trace has no session IDs. Rebuilding sessions from `hash_ids` prefix chains was unreliable: 46% of requests share the same first block (a template), and the median gap between chained requests is 114 s, which is too long to be a tool call.
  - *Arrival timestamps and prefix-sharing structure.* Both are possible later extensions (open-loop arrivals; E5).

### Continuum (multi-turn agent scheduling with KV-cache time-to-live)

- **Source.** Li et al., *Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live*, arXiv:2511.02230.
  - Paper: https://arxiv.org/abs/2511.02230
  - Code: https://github.com/Hanchenli/vllm-continuum
- **What it reports.** Statistics from GPT-5 agent runs (Table 2):
  - mini-swe-agent on SWE-bench: 10.9 ± 2.1 turns, tool time 925 ± 3,550 ms.
  - BFCL v4 web search: 6.3 ± 2.3 turns, tool time 1,923 ± 2,133 ms.
- **Used for:**
  - `steps`: uniform 5–12 in both trace mixes, spanning the BFCL and SWE-bench turn counts.
  - `tool_delay` in `trace_agentic`: lognormal fitted to the SWE-bench tool time (mean 0.925 s, sd 3.55 s), which gives median 0.233 s and σ 1.66, clipped at 60 s. The generated seed-1000 draws have mean 0.86 s and sd 2.6 s.
- **Related work.** The paper's TTL-based retention of KV across tool calls is the closest prior design to the Phase 2 direction.

### TraceLab (coding-agent workload characterization)

- **Source.** *TraceLab: Characterizing Coding Agent Workloads for LLM Serving*, arXiv:2606.30560.
  - Paper: https://arxiv.org/abs/2606.30560
  - Data: https://github.com/uw-syfi/TraceLab
- **What it reports.** 4,265 Claude Code and Codex sessions (about 357k LLM steps):
  - Median tool call 0.3 s, p90 13.6 s.
  - Calls over 1 min are about 4–5% of calls but 85–92% of tool time.
  - Median tokens appended per step: about 857 (Claude).
  - Median output: 184–252 tokens.
- **Used for:**
  - `tool_delay` in `trace_agentic_heavy`: lognormal fitted to median 0.3 s and p90 13.6 s, which gives σ 2.98, clipped at 120 s.
    - Generated seed-1000 draws: median 0.37 s, p90 11.2 s, 3.2% over 60 s.
    - Those long calls hold 56% of tool time. The clip trims the source's tail.
  - `tool_out` (tokens appended per step) in both trace mixes: lognormal with median 800 and σ 1.0, limited to 16–8,192. The 800 is the TraceLab median append minus a typical response.
- **Context for the report.** Real coding agents run at contexts of about 110–126k tokens, well beyond this testbed's roughly 64k-token pool. That supports describing edge agents as small-context and memory-bound.

### Project Gutenberg corpus (token content for every workload)

- **Source.** Six public-domain books: ids 1342, 2701, 98, 84, 1661 and 2600.
  - Fetched by `scripts/fetch_corpus.py` with the Gutenberg boilerplate stripped.
  - Hashes in `configs/corpus_manifest.json`. Combined sha256 `d3bf5c23…bd86`.
- **Used for.** The token content of every prompt and tool output.
  - The text is tokenized once with the pinned server: 1,674,729 tokens.
  - Prompts are exact token slices `{offset, len}`, sent as token arrays.
  - The first 8,192 tokens are reserved for the shared prefix.

## Hand-designed mixes (no external source)

`chat`, `long_context`, `long_output`, `agentic`, `blend`, `stress` and `pilot` were designed for this project. They cover the course specification's requirements:
- short and long contexts;
- short and long responses;
- multi-step tasks with controlled tool delays.

Their parameters are chosen for coverage, not taken from a trace, so the report should present them that way. The trace mixes are the realism check.

## Sources consulted but not used in the workloads

| Source | Why not used (yet) |
|---|---|
| InferCept (Abhyankar et al., ICML '24; CSE 291A reading) | Not a workload source. It is the prior art for keeping, discarding or swapping KV during tool calls, so it frames E4 and Phase 2. |
| Azure LLM Inference 2024, https://github.com/Azure/AzurePublicDataset | Single-step code and conversation requests. A candidate source for open-loop arrival times and for calibrating `chat`. |
| BurstGPT, https://github.com/HPMLL/BurstGPT | Bursty arrivals; a candidate for the Phase 2 "changing arrival rates" evaluation. |
| SemiAnalysis CC Traces, https://huggingface.co/datasets/semianalysisai/cc-traces-weka-042026 | Real `think_time` and prefix reuse, but a median input of about 110k tokens is far beyond the memory budget. |
| CacheWise, arXiv:2606.16824 | Shows tool duration can be predicted from tool type; possibly relevant to a Phase 2 policy. Its trace repository was not inspected. |
| ShareGPT (vLLM `bench serve` default) | Real chat text, but single-step and short. It would replace the corpus, which breaks exact token control. |
| LongBench v2, https://longbench2.github.io/ | Contexts of 8k–2M words, mostly too long for a 32k slot. |
| ServeGen (Alibaba, NSDI '26), https://github.com/alibaba/ServeGen | Shows that length distributions drift over time; cited only as background. |

## Known limitations of the calibration

- The trace mixes combine marginal distributions from three different systems (Kimi, GPT-5 agents, Claude Code and Codex). There is no joint distribution: step count, context, output and delay are drawn independently.
- Every task shares the same 512-token prefix. Real prefix sharing is partial: in Mooncake, 46% of requests share a first block.
- Tool delays are clipped (60 s and 120 s) to keep runs a manageable length. This removes part of the heavy tail, which is what makes idle retention expensive.
- Outputs are forced to exact lengths with `ignore_eos`, so content and quality are not modelled. This is intentional for reproducibility.
