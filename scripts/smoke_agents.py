"""Live smoke test of scheduler + agent (superseded by runner/run.py for real measurements).

Runs the first N pilot tasks closed-loop against a fresh server in both slot modes and prints a summary.
Usage: python scripts/smoke_agents.py [--tasks 6] [--k 2] [--agents 4]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.agent import run_closed  # noqa: E402
from runner.client import LlamaClient  # noqa: E402
from runner.corpus import Corpus  # noqa: E402
from runner.scheduler import FifoPolicy, Scheduler  # noqa: E402
from runner.server import Server, load_local, load_server_config  # noqa: E402


async def one_mode(base_url, tasks, prefix, corpus, k, agents, mode):
    client = LlamaClient(base_url, request_timeout_s=120)
    sched = Scheduler(client, k, FifoPolicy(mode))
    reqs, recs = [], []
    t0 = time.monotonic()
    await run_closed(tasks, agents, sched=sched, corpus=corpus, prefix_ids=prefix, deadline_s=600,
                     log_request=reqs.append, log_task=recs.append)
    await sched.drain()
    await client.aclose()
    wall = time.monotonic() - t0
    later = [r for r in reqs if r["step"] > 0]
    return {
        "mode": mode, "wall_s": round(wall, 1),
        "tasks_ok": sum(r["status"] == "ok" for r in recs), "tasks": len(recs),
        "requests": len(reqs), "req_status": sorted({r["status"] for r in reqs}),
        "gen_tokens_ok": all(r["generated_tokens"] == r["n_predict"] for r in reqs if r["status"] == "ok"),
        "prompt_len_ok": all(r["n_prompt"] == r["prompt_tokens_expected"] for r in reqs),
        "cache_hit_frac_steps>0": round(sum(r["cached_tokens"] or 0 for r in later) /
                                        max(1, sum(r["n_prompt"] for r in later)), 3),
        "prompt_tokens_computed": sum(r["tokens_processed"] or 0 for r in reqs),
        "mean_task_s": round(sum(r["completion_t"] - r["arrival_t"] for r in recs) / len(recs), 1),
        "mean_queue_s": round(sum(r["total_queue_s"] for r in recs) / len(recs), 2),
        "max_inflight": sched.max_inflight, "policy_overhead_ms": round(sched.policy_overhead_s * 1000, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--agents", type=int, default=4)
    a = ap.parse_args()
    local = load_local()
    corpus = Corpus(local)
    wl_dir = Path(local["results_dir"]) / "workloads"
    tasks = [json.loads(l) for l in (wl_dir / "pilot_s42.jsonl").read_text().splitlines()][: a.tasks]
    meta = json.loads((wl_dir / "pilot_s42.meta.json").read_text())
    prefix = corpus.slice(meta["shared_prefix"])
    out = Path(local["results_dir"]) / "smoke_agents" / datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    for mode in ("pinned", "any"):
        cfg = load_server_config({"ctx_size": 16384 * a.k, "parallel": a.k})
        with Server(local, cfg, out / f"server_{mode}.log") as s:
            r = asyncio.run(one_mode(s.base_url, tasks, prefix, corpus, a.k, a.agents, mode))
        print(json.dumps(r), flush=True)
        results.append(r)
    (out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
