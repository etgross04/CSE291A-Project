"""Does idle KV held by OTHER slots slow down a request in unified mode (-kvu) vs the static split?

For each mode, -c 32768 -np 4: measure one request on slot 0 (2000-token prompt, 256 out, cold) while
slots 1-3 hold 0 or 3 x 7000 idle tokens. Reports prefill and decode ms/token (from server timings).
Usage: python scripts/probe_kvu_speed.py [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.corpus import Corpus  # noqa: E402
from runner.server import Server, load_local, load_server_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    local = load_local()
    corpus = Corpus(local)
    out = Path(local["results_dir"]) / "probes" / ("kvu_speed_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    rows = []
    for unified in (False, True):
        cfg = load_server_config({"ctx_size": 32768, "parallel": 4, "kv_unified": unified})
        with Server(local, cfg, out / f"server_kvu{int(unified)}.log") as s, \
                httpx.Client(base_url=s.base_url, timeout=600) as c:
            def comp(slot, prompt, n):
                d = c.post("/completion", json={"prompt": prompt, "n_predict": n, "temperature": 0,
                                                "ignore_eos": True, "cache_prompt": True, "id_slot": slot}).json()
                return d["timings"]
            comp(0, corpus.slice({"offset": 900_000, "len": 512}), 16)  # warm-up
            for idle in (0, 3):
                for sl in range(1, 4):  # fill (or overwrite with tiny prompts to empty) the other slots
                    n = 7000 if idle and sl <= idle else 8
                    comp(sl, corpus.slice({"offset": 1_000_000 + sl * 10_000, "len": n}), 1)
                pre, dec = [], []
                for r in range(a.reps):
                    t = comp(0, corpus.slice({"offset": 200_000 + r * 5000 + idle * 50_000, "len": 2000}), 256)
                    pre.append(t["prompt_per_token_ms"])
                    dec.append(t["predicted_per_token_ms"])
                row = {"kv_unified": unified, "idle_tokens_other_slots": idle * 7000,
                       "prefill_ms_per_tok": round(statistics.median(pre), 4),
                       "decode_ms_per_tok": round(statistics.median(dec), 3)}
                print(json.dumps(row), flush=True)
                rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
