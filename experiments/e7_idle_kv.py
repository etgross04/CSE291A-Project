"""E7: separate the two causes of the decode slowdown seen at high K (E2 agentic: 7.8 ms/tok at K=1 ->
34.6 ms/tok at K=16):

  A  idle KV   one active request on slot 0 (2,000-token prompt, cold, 256 tokens out) while slots 1-3
               hold IDLE_LEVELS idle tokens in total. Shared pool (-kvu) vs static split (-no-kvu), -np 4,
               -c 65536 (split: 16,384 per slot, so at most 3 x 16,000 idle).
               scripts/probe_kvu_speed.py measured one point (21k idle at -c 32768); this is the curve.
  B  batch     B requests decoding at once (B = 1..16), each on its own slot with a 2,000-token context and
               no other KV. -kvu, -np 16, -c 65536. Prompts are prefilled first (n_predict 1, cached), so
               the timed requests are (almost) pure decode: per-request ms/token and aggregate tokens/s.

Every point is measured REPS times; slots are erased between points. Timings come from the server
(`prompt_ms`, `gen_ms`, `predicted_n`). Output: <results_dir>/runs/E7_idle_kv/rep<R>/ with rows.jsonl,
summary.json, server logs, manifest.json, DONE (rerunning skips a DONE dir; --rep N for another rep).

Usage: python -m experiments.e7_idle_kv [--reps 3] [--rep 0] [--only A|B]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import traceback
from datetime import datetime
from pathlib import Path

POOL = 65536
ACTIVE_PROMPT = 2000
N_OUT = 256
IDLE_LEVELS = [0, 8000, 16000, 32000, 48000]   # total idle tokens over slots 1-3
BATCH_SIZES = [1, 2, 4, 8, 16]
# corpus regions (token offsets) kept apart so no two prompts share a prefix
IDLE_BASE, ACTIVE_BASE, BATCH_BASE = 1_000_000, 200_000, 500_000


def idle_split(total: int, slots: int = 3) -> list[int]:
    """Spread `total` idle tokens evenly over `slots` slots (each <= 16,000 so the static split fits)."""
    per = [total // slots + (1 if i < total % slots else 0) for i in range(slots)]
    assert max(per) <= 16000, per
    return per


async def part_a(client, corpus, reps: int, log, kv_unified: bool) -> list[dict]:
    rows = []
    for level in IDLE_LEVELS:
        for r in range(reps):
            for s in range(4):
                await client.erase_slot(s)
            for i, n in enumerate(idle_split(level), start=1):
                if n:
                    res = await client.complete(corpus.slice({"offset": IDLE_BASE + i * 20_000, "len": n}), 1, id_slot=i)
                    assert res.status == "ok", res.status
            prompt = corpus.slice({"offset": ACTIVE_BASE + (level // 1000) * 7_000 + r * 2_500, "len": ACTIVE_PROMPT})
            res = await client.complete(prompt, N_OUT, id_slot=0)
            row = {"part": "A", "kv_unified": kv_unified, "idle_tokens": level, "rep": r, "status": res.status,
                   "prompt_ms_per_tok": res.prompt_ms / max(1, res.tokens_processed or 0) if res.prompt_ms else None,
                   "decode_ms_per_tok": res.gen_ms / res.generated_tokens if res.gen_ms and res.generated_tokens else None,
                   "generated_tokens": res.generated_tokens, "cached_tokens": res.cached_tokens}
            log(row)
            rows.append(row)
    return rows


async def part_b(client, corpus, reps: int, log) -> list[dict]:
    rows = []
    for b in BATCH_SIZES:
        for r in range(reps):
            for s in range(16):
                await client.erase_slot(s)
            prompts = [corpus.slice({"offset": BATCH_BASE + (r * 16 + i) * 2_100, "len": ACTIVE_PROMPT})
                       for i in range(b)]
            for i, p in enumerate(prompts):  # prefill one at a time, so the timed phase is decode only
                res = await client.complete(p, 1, id_slot=i)
                assert res.status == "ok", res.status
            res_all = await asyncio.gather(*[client.complete(p, N_OUT, id_slot=i) for i, p in enumerate(prompts)])
            t0 = min(x.dispatch_t for x in res_all)
            t1 = max(x.end_t for x in res_all)
            dec = [x.gen_ms / x.generated_tokens for x in res_all if x.gen_ms and x.generated_tokens]
            row = {"part": "B", "kv_unified": True, "batch": b, "rep": r,
                   "status": ",".join(sorted({x.status for x in res_all})),
                   "decode_ms_per_tok_median": statistics.median(dec) if dec else None,
                   "decode_ms_per_tok_max": max(dec) if dec else None,
                   "aggregate_tok_per_s": sum(x.generated_tokens or 0 for x in res_all) / (t1 - t0) if t1 > t0 else None,
                   "cached_tokens_min": min(x.cached_tokens or 0 for x in res_all)}
            log(row)
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Median over reps per point."""
    out = []
    keys = sorted({(r["part"], r["kv_unified"], r.get("idle_tokens"), r.get("batch")) for r in rows},
                  key=lambda k: (k[0], k[1], k[2] or 0, k[3] or 0))
    for part, kvu, idle, batch in keys:
        rs = [r for r in rows if (r["part"], r["kv_unified"], r.get("idle_tokens"), r.get("batch")) == (part, kvu, idle, batch)]
        s = {"part": part, "kv_unified": kvu, "idle_tokens": idle, "batch": batch, "reps": len(rs),
             "all_ok": all(r["status"] == "ok" for r in rs)}
        for f in ("prompt_ms_per_tok", "decode_ms_per_tok", "decode_ms_per_tok_median", "aggregate_tok_per_s"):
            vals = [r[f] for r in rs if r.get(f) is not None]
            if vals:
                s[f] = round(statistics.median(vals), 4)
                s[f + "_min"], s[f + "_max"] = round(min(vals), 4), round(max(vals), 4)
        out.append(s)
    return out


def main() -> None:
    from runner.client import LlamaClient
    from runner.corpus import Corpus
    from runner.env import environment, gpu_mem_used_mib, write_repo_diff
    from runner.run import JsonlWriter
    from runner.server import Server, load_local, load_server_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep", type=int, default=0, help="run directory index (a repeat of the whole experiment)")
    ap.add_argument("--only", choices=["A", "B"])
    a = ap.parse_args()
    local = load_local()
    run_dir = Path(local["results_dir"]) / "runs" / "E7_idle_kv" / (f"rep{a.rep}" + (f"_{a.only}" if a.only else ""))
    if (run_dir / "DONE").exists():
        print(f"skip (DONE): {run_dir}")
        return
    if run_dir.exists():
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.incomplete-{datetime.now():%Y%m%d-%H%M%S}"))
    run_dir.mkdir(parents=True)
    corpus = Corpus(local)
    man = {"experiment": "E7", "status": "running", "reps": a.reps, "only": a.only, "pool": POOL,
           "active_prompt": ACTIVE_PROMPT, "n_out": N_OUT, "idle_levels": IDLE_LEVELS, "batch_sizes": BATCH_SIZES,
           "env": environment(local, hash_model=True, packages=False),
           "corpus_tokens_sha256": corpus.meta.get("tokens_sha256"),
           "started_at": datetime.now().isoformat(timespec="seconds"), "servers": {}}
    if write_repo_diff(run_dir / "repo.diff"):
        man["repo_diff"] = "repo.diff"
    save = lambda: (run_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")  # noqa: E731
    writer = JsonlWriter(run_dir / "rows.jsonl")
    rows: list[dict] = []
    plan = [("A_split", {"parallel": 4, "kv_unified": False}), ("A_kvu", {"parallel": 4, "kv_unified": True}),
            ("B_kvu", {"parallel": 16, "kv_unified": True})]
    plan = [p for p in plan if a.only is None or p[0].startswith(a.only)]
    try:
        for name, over in plan:
            cfg = load_server_config({"ctx_size": POOL, **over})
            man["servers"][name] = {"prelaunch_vram_mib": round(gpu_mem_used_mib(local.get("gpu_index", 0)), 1)}
            with Server(local, cfg, run_dir / f"server_{name}.log") as srv:
                man["servers"][name]["cmd"] = srv.cmd

                async def go():
                    client = LlamaClient(srv.base_url, request_timeout_s=600)
                    try:
                        await client.complete(corpus.slice({"offset": 900_000, "len": 512}), 16, id_slot=0)  # warm-up
                        if name.startswith("A"):
                            return await part_a(client, corpus, a.reps, writer.write, over["kv_unified"])
                        return await part_b(client, corpus, a.reps, writer.write)
                    finally:
                        await client.aclose()

                rows += asyncio.run(go())
            save()
        man["status"] = "complete"
    except Exception as e:  # noqa: BLE001 -- kept with partial rows
        man["status"], man["error"] = "error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        writer.close()
        man["ended_at"] = datetime.now().isoformat(timespec="seconds")
        man["summary"] = summarize(rows) if rows else []
        (run_dir / "summary.json").write_text(json.dumps(man["summary"], indent=1), encoding="utf-8")
        save()
    if man["status"] == "complete":
        (run_dir / "DONE").write_text(man["ended_at"] + "\n", encoding="utf-8")
    for s in man["summary"]:
        print(json.dumps(s))
    print(f"{man['status']}: {run_dir}")


if __name__ == "__main__":
    main()
