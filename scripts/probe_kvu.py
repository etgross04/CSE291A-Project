"""Probe unified-KV (-kvu) behavior on the pinned build before relying on it as the baseline memory model.

Questions (pool -c 16384, -np 4, cram 0, --no-cache-idle-slots):
  A. What per-slot n_ctx does each slot get in unified mode?
  B. Idle slots keep their KV. If a new request needs more cells than are free, does the server evict idle
     slots, fail the request, or truncate it?
  C. Does `POST /slots/{id}?action=erase` free a slot's cells (and does it need --slot-save-path)?
  D. Concurrent requests whose combined prompt+output exceeds the pool: what fails, and how?
  E. Does a new prompt on the same slot release that slot's old cells?

Usage: python scripts/probe_kvu.py   (starts its own server; writes results/probes/kvu_<ts>/)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.corpus import Corpus  # noqa: E402
from runner.server import Server, load_local, load_server_config  # noqa: E402

POOL = 16384


def main() -> None:
    local = load_local()
    corpus = Corpus(local)
    out = Path(local["results_dir"]) / "probes" / ("kvu_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    cfg = load_server_config({"ctx_size": POOL, "parallel": 4, "kv_unified": True, "slot_save_path": None})
    report: dict = {"pool": POOL, "parallel": 4}
    prompts = {i: corpus.slice({"offset": 100_000 + i * 20_000, "len": 6000}) for i in range(4)}

    with Server(local, cfg, out / "server.log") as s, httpx.Client(base_url=s.base_url, timeout=600) as c:
        def comp(slot, prompt, n_predict=32):
            t0 = time.monotonic()
            r = c.post("/completion", json={"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                                            "ignore_eos": True, "cache_prompt": True, "id_slot": slot})
            d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
            tm = d.get("timings", {})
            return {"slot": slot, "http": r.status_code, "n_prompt": len(prompt), "n_predict": n_predict,
                    "tokens_predicted": d.get("tokens_predicted"), "truncated": d.get("truncated"),
                    "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
                    "tokens_cached": d.get("tokens_cached"), "error": d.get("error"),
                    "wall_s": round(time.monotonic() - t0, 2)}

        def log(label, v):
            print(f"\n--- {label}\n{json.dumps(v, indent=1)}", flush=True)
            report[label] = v

        log("A_slots", c.get("/slots").json())

        # B: fill two slots (retained idle), then a third request that does not fit in the free cells
        log("B1_slot0", comp(0, prompts[0]))
        log("B2_slot1", comp(1, prompts[1]))
        log("B3_slot2_overcommit", comp(2, prompts[2]))
        log("B4_slot0_again_cache_check", comp(0, prompts[0]))  # cache_n ~6000 => slot 0 was not evicted
        log("B5_slot1_again_cache_check", comp(1, prompts[1]))

        # C: erase and retry
        for sl in range(4):
            r = c.post(f"/slots/{sl}?action=erase")
            log(f"C_erase_slot{sl}", {"http": r.status_code, "body": r.text[:300]})
        log("C2_slot0_after_erase_cache_check", comp(0, prompts[0]))  # cache_n ~0 => erase really cleared
        c.post("/slots/0?action=erase")

        # E: same slot, new unrelated prompt: are old cells reused?  (fill 0, then 0 again with another
        # prompt twice: if old cells leaked, the 3rd request of 6000 would not fit with 12k leaked)
        log("E1_slot0_p0", comp(0, prompts[0]))
        log("E2_slot0_p1", comp(0, prompts[1]))
        log("E3_slot0_p2", comp(0, prompts[2]))
        log("E4_slot1_p3_fits_if_no_leak", comp(1, prompts[3]))
        for sl in range(4):
            c.post(f"/slots/{sl}?action=erase")

        # D: three concurrent requests, 5000 prompt + 1024 out each = 18072 > pool
        big = {i: corpus.slice({"offset": 400_000 + i * 20_000, "len": 5000}) for i in range(3)}
        with ThreadPoolExecutor(3) as ex:
            res = list(ex.map(lambda i: comp(i, big[i], 1024), range(3)))
        log("D_concurrent_overcommit", res)
        log("D_health_after", {"http": c.get("/health").status_code})

    lines = (out / "server.log").read_text(encoding="utf-8", errors="replace").splitlines()
    keys = ("n_ctx", "kv", "KV", "free space", "fail", "error", "purg", "clear", "evict", "erase")
    report["server_log_matches"] = [l for l in lines if any(k in l for k in keys)][-80:]
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\n".join(report["server_log_matches"]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
