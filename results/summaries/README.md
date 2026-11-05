# Phase 1 result tables

Derived from the raw run logs by `analysis/metrics.py`. The raw run directories (`requests.jsonl`, `tasks.jsonl`, `system.jsonl`, `manifest.json` per run) are not in the repo because of their size (~265 MB) and are available on request.

| File | Contents |
|---|---|
| `E<n>/runs.csv` | One row per run: config, validity, and every metric (task latency p50/p95/p99, queue/model/tool time, throughput incl. steady window, TTFT, decode and prefill rates, cache hits, recomputed tokens, peak VRAM, KV reserved vs used, failures by type, policy overhead). |
| `E<n>/summary.csv` | One row per config: median, min and max of each metric over valid reps. |
| `E<n>/exclusions.csv` | Runs left out of the summary and why (empty: all 171 runs were valid). |
| `E2/best_k.json` | Best fixed K per workload, the tuned baseline for Phase 2. |
| `E7/rows.jsonl`, `E7/summary.json` | E7 microbenchmark: one row per measurement, and the median per point. |

`run_dir` columns are paths on the measurement machine (`C:\llm\results\runs\...`).

Regenerate: `python -m analysis.metrics E<n>` writes `results/E<n>/`; copy the files here.
