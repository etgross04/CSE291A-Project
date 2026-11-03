"""Probe /completion details the client depends on (build b11721):
  1. Can the response return generated token ids? (tries `return_tokens`; prints which keys appear)
  2. What does each streamed chunk contain?
  3. Does detokenize->tokenize of the generated text round-trip to the same ids?
  4. What happens when prompt (+ n_predict) exceeds the slot's n_ctx? (needed to classify ctx_overflow)

Usage: python scripts/probe_client_fields.py   (starts its own server: -c 8192 -np 2 -> 4096 per slot)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.corpus import Corpus  # noqa: E402
from runner.server import Server, load_local, load_server_config  # noqa: E402


def show(label, d):
    d = {k: (f"<{len(v)} items/chars>" if k in ("content", "tokens", "prompt") and v else v)
         for k, v in d.items() if k != "generation_settings"}
    print(f"\n--- {label}\n{json.dumps(d, indent=1)[:2500]}")


def main() -> None:
    local = load_local()
    corpus = Corpus(local)
    out = Path(local["results_dir"]) / "probes" / ("client_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    cfg = load_server_config({"ctx_size": 8192, "parallel": 2, "kv_unified": False})
    prompt = corpus.slice({"offset": 100_000, "len": 1000})
    report = {}
    with Server(local, cfg, out / "server.log") as s, httpx.Client(base_url=s.base_url, timeout=300) as c:
        base = {"prompt": prompt, "n_predict": 32, "temperature": 0, "ignore_eos": True, "cache_prompt": True}

        # 1. non-stream with return_tokens
        d = c.post("/completion", json={**base, "stream": False, "return_tokens": True}).json()
        show("non-stream, return_tokens=true", d)
        ids = d.get("tokens")
        report["return_tokens_nonstream"] = isinstance(ids, list) and len(ids) == 32
        # without the flag, for comparison
        d0 = c.post("/completion", json={**base, "stream": False}).json()
        report["tokens_key_without_flag"] = d0.get("tokens")
        report["same_content_with_and_without_flag"] = d0.get("content") == d.get("content")

        # 2. streaming chunks
        chunks = []
        with c.stream("POST", "/completion", json={**base, "stream": True, "return_tokens": True}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    chunks.append(json.loads(line[6:]))
        show("stream chunk[0]", chunks[0])
        show("stream final chunk", chunks[-1])
        stream_ids = [t for ch in chunks if not ch.get("stop") for t in (ch.get("tokens") or [])]
        report["stream_chunks"] = len(chunks)
        report["stream_chunk0_keys"] = sorted(chunks[0].keys())
        report["stream_final_keys"] = sorted(chunks[-1].keys())
        report["stream_ids_count"] = len(stream_ids)
        report["stream_ids_match_nonstream"] = stream_ids == ids

        # 3. text round trip
        text = d.get("content", "")
        rt = c.post("/tokenize", json={"content": text}).json()["tokens"]
        report["retokenize_roundtrip_equal"] = rt == ids
        report["retokenize_len"] = [len(rt), len(ids or [])]

        # 4. overflow behaviour (slot n_ctx = 4096)
        for label, plen, npred in [("prompt fits, prompt+n_predict > n_ctx", 4000, 256),
                                   ("prompt alone > n_ctx", 5000, 16)]:
            p = corpus.slice({"offset": 200_000, "len": plen})
            r = c.post("/completion", json={**base, "prompt": p, "n_predict": npred, "stream": False})
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
            print(f"\n--- overflow: {label}: HTTP {r.status_code}")
            if isinstance(body, dict):
                show(label, body)
            else:
                print(body[:500])
            report[f"overflow::{label}"] = {
                "http": r.status_code,
                "error": body.get("error") if isinstance(body, dict) else body[:300],
                "tokens_predicted": body.get("tokens_predicted") if isinstance(body, dict) else None,
                "truncated": body.get("truncated") if isinstance(body, dict) else None,
                "stop_type": body.get("stop_type") if isinstance(body, dict) else None,
            }
        # same overflow, streamed: does the error arrive as HTTP status or inside the stream?
        p = corpus.slice({"offset": 200_000, "len": 5000})
        with c.stream("POST", "/completion", json={**base, "prompt": p, "stream": True}) as r:
            lines = [ln for ln in r.iter_lines() if ln]
            report["overflow_stream"] = {"http": r.status_code, "first_lines": lines[:3]}

    print("\n=== REPORT\n" + json.dumps(report, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
