"""Probe the pinned llama-server: which endpoints/fields/features exist, and how slot reuse behaves.

By default this launches its own server (2 slots, 16k total ctx, slot-save enabled) so the probe is
reproducible; pass --url to probe an already-running server instead.

Usage:
  python scripts/probe_runtime.py                 # start server, probe, write JSON report
  python scripts/probe_runtime.py --url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.server import Server, build_cmd, load_local, load_server_config  # noqa: E402

REPORT: dict = {"features": {}, "raw": {}}


def feat(name: str, ok: bool, note: str = "") -> None:
    REPORT["features"][name] = {"ok": ok, "note": note}
    print(f"  [{'OK ' if ok else 'NO '}] {name}{' - ' + note if note else ''}")


def make_text(n_words: int, seed: int) -> str:
    # Deterministic, varied filler; exact token counts are measured via /tokenize, not assumed.
    words = ["river", "stone", "lantern", "harbor", "meadow", "copper", "signal", "orchard", "glacier", "thread"]
    return " ".join(f"{words[(i * 7 + seed) % 10]}{(i * 31 + seed) % 97}" for i in range(n_words))


def complete(c: httpx.Client, prompt: str, n_predict: int, stream: bool, **extra) -> tuple[dict, dict]:
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "ignore_eos": True,
            "cache_prompt": True, "stream": stream, **extra}
    t0 = time.monotonic()
    first_tok_t = None
    if not stream:
        r = c.post("/completion", json=body)
        r.raise_for_status()
        d = r.json()
    else:
        d, n_chunks = {}, 0
        with c.stream("POST", "/completion", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                n_chunks += 1
                if first_tok_t is None and chunk.get("content"):
                    first_tok_t = time.monotonic()
                if chunk.get("stop"):
                    d = chunk
        d["_n_chunks"] = n_chunks
    wall = {"wall_ms": (time.monotonic() - t0) * 1000,
            "ttft_ms": None if first_tok_t is None else (first_tok_t - t0) * 1000}
    return d, wall


def strip_content(d: dict) -> dict:
    return {k: (f"<{len(v)} chars>" if k == "content" and isinstance(v, str) else v) for k, v in d.items()}


def probe(base_url: str, slot_save_enabled: bool) -> None:
    c = httpx.Client(base_url=base_url, timeout=600)

    print("\n== Endpoints")
    r = c.get("/health"); feat("/health", r.status_code == 200, r.text[:80])
    r = c.get("/props")
    feat("/props", r.status_code == 200)
    if r.status_code == 200:
        props = r.json()
        REPORT["raw"]["props"] = {k: v for k, v in props.items() if k != "chat_template"}
        gs = props.get("default_generation_settings", {})
        print(f"     total_slots={props.get('total_slots')} n_ctx(per slot)={gs.get('n_ctx')}")
    r = c.get("/metrics")
    feat("/metrics", r.status_code == 200, f"{len(r.text.splitlines())} lines" if r.status_code == 200 else r.text[:80])
    if r.status_code == 200:
        REPORT["raw"]["metrics_names"] = sorted({ln.split()[0].split("{")[0] for ln in r.text.splitlines()
                                                 if ln and not ln.startswith("#")})
        print("     metrics:", ", ".join(REPORT["raw"]["metrics_names"]))
    r = c.get("/slots")
    feat("/slots", r.status_code == 200)
    if r.status_code == 200:
        slots = r.json()
        REPORT["raw"]["slot0_keys"] = sorted(slots[0].keys())
        print(f"     {len(slots)} slots; slot keys: {', '.join(REPORT['raw']['slot0_keys'])}")

    print("\n== Tokenizer")
    text = make_text(800, seed=1)
    r = c.post("/tokenize", json={"content": text})
    toks = r.json()["tokens"]
    feat("/tokenize", r.status_code == 200, f"{len(toks)} tokens for 800 words")
    r = c.post("/detokenize", json={"tokens": toks})
    feat("/detokenize round-trip", r.status_code == 200 and r.json()["content"].strip() == text.strip())
    # Can /completion take a token array as prompt? (lets the runner control exact prompt length)
    d, _ = complete(c, toks[:512], 8, stream=False)
    feat("prompt as token array", d.get("timings", {}).get("prompt_n") is not None,
         f"prompt_n={d.get('timings', {}).get('prompt_n')} tokens_evaluated={d.get('tokens_evaluated')}")

    print("\n== Completion fields (non-stream, n_predict=64, ignore_eos)")
    d, wall = complete(c, make_text(300, seed=2), 64, stream=False, id_slot=0)
    REPORT["raw"]["completion_nonstream"] = strip_content(d)
    print(json.dumps(strip_content(d), indent=2)[:3000])
    t = d.get("timings", {})
    feat("timings block", bool(t), ", ".join(t.keys()))
    feat("ignore_eos gives exact n_predict", t.get("predicted_n") == 64 or d.get("tokens_predicted") == 64,
         f"predicted_n={t.get('predicted_n')} tokens_predicted={d.get('tokens_predicted')}")
    feat("id_slot echoed", "id_slot" in d, f"id_slot={d.get('id_slot')}")

    print("\n== Streaming (final chunk)")
    d, wall = complete(c, make_text(300, seed=3), 64, stream=True)
    REPORT["raw"]["completion_stream_final"] = strip_content(d)
    feat("stream final chunk has timings", "timings" in d, f"chunks={d.get('_n_chunks')} ttft_ms={wall['ttft_ms']:.0f}")

    print("\n== Slot reuse (same prompt + suffix on same slot vs other slot)")
    base = make_text(1500, seed=4)
    suffix = " " + make_text(40, seed=5)
    rows = {}
    for label, prompt, slot in [("A: base on slot0 (cold)", base, 0),
                                ("B: base+suffix on slot0 (warm)", base + suffix, 0),
                                ("C: base+suffix on slot1 (cold)", base + suffix, 1)]:
        d, wall = complete(c, prompt, 16, stream=False, id_slot=slot)
        t = d.get("timings", {})
        rows[label] = {"id_slot": d.get("id_slot"), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                       "prompt_ms": t.get("prompt_ms"), "tokens_cached": d.get("tokens_cached"),
                       "tokens_evaluated": d.get("tokens_evaluated"), "wall_ms": round(wall["wall_ms"])}
        print(f"  {label}: {rows[label]}")
    REPORT["raw"]["slot_reuse"] = rows
    a, b, cc = rows.values()
    feat("same-slot prefix reuse", (b["prompt_n"] or 1e9) < 0.2 * (a["prompt_n"] or 0),
         f"warm prompt_n={b['prompt_n']} vs cold {a['prompt_n']}")
    feat("other slot recomputes", (cc["prompt_n"] or 0) > 0.8 * (a["prompt_n"] or 1e9),
         f"slot1 prompt_n={cc['prompt_n']}")

    print("\n== cache_prompt=false on warm slot")
    d, _ = complete(c, base + suffix, 16, stream=False, id_slot=0, cache_prompt=False)
    feat("cache_prompt=false forces recompute", (d["timings"]["prompt_n"] or 0) > 0.8 * (a["prompt_n"] or 1e9),
         f"prompt_n={d['timings']['prompt_n']}")

    print("\n== Slot save / restore / erase")
    if not slot_save_enabled:
        feat("slot save/restore", False, "server started without --slot-save-path; skipped")
    else:
        r = c.post("/slots/0?action=save", json={"filename": "probe_slot0.bin"})
        feat("slot save", r.status_code == 200, r.text[:200])
        REPORT["raw"]["slot_save"] = r.json() if r.status_code == 200 else r.text
        r = c.post("/slots/0?action=erase")
        feat("slot erase", r.status_code == 200, r.text[:200])
        r = c.post("/slots/0?action=restore", json={"filename": "probe_slot0.bin"})
        feat("slot restore", r.status_code == 200, r.text[:200])
        REPORT["raw"]["slot_restore"] = r.json() if r.status_code == 200 else r.text
        d, _ = complete(c, base + suffix, 16, stream=False, id_slot=0)
        feat("restored slot is reused", (d["timings"]["prompt_n"] or 1e9) < 0.2 * (a["prompt_n"] or 0),
             f"prompt_n after restore={d['timings']['prompt_n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--np", type=int, default=2)
    a = ap.parse_args()
    local = load_local()
    out_dir = Path(local["results_dir"]) / "probes" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.url:
        probe(a.url, slot_save_enabled=True)
    else:
        cfg = load_server_config({"ctx_size": a.ctx, "parallel": a.np, "slot_save_path": str(out_dir / "slots")})
        (out_dir / "slots").mkdir(exist_ok=True)
        REPORT["server_cmd"] = build_cmd(local, cfg)
        with Server(local, cfg, out_dir / "server.log") as s:
            probe(s.base_url, slot_save_enabled=True)

    (out_dir / "probe_report.json").write_text(json.dumps(REPORT, indent=2))
    print(f"\nReport: {out_dir / 'probe_report.json'}\nServer log: {out_dir / 'server.log'}")


if __name__ == "__main__":
    main()
