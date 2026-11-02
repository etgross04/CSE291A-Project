"""Start llama-server at several -c values and record VRAM after load (NVML, whole-GPU, since WDDM has no
per-process accounting). Records failures (e.g. OOM at load) instead of stopping.

Usage: python scripts/vram_vs_ctx.py --ctx 4096 8192 16384 32768 49152 65536 --np 1 4
Output: <results_dir>/vram_vs_ctx/<timestamp>/vram_vs_ctx.csv (+ one server log per config)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pynvml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.server import Server, build_cmd, load_local, load_server_config  # noqa: E402

KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2  # Qwen3-8B fp16: 2(K,V) x layers x kv_heads x head_dim x 2B = 147456


def used_mib(h) -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20


def settle(h, seconds: float = 3.0) -> float:
    """Median of samples over a short window; the desktop compositor makes single reads noisy."""
    xs = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        xs.append(used_mib(h))
        time.sleep(0.1)
    xs.sort()
    return xs[len(xs) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 49152, 65536, 81920])
    ap.add_argument("--np", type=int, nargs="+", default=[1])
    ap.add_argument("--unified", action="store_true", help="use -kvu (one shared KV pool)")
    a = ap.parse_args()

    local = load_local()
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(local.get("gpu_index", 0))
    out = Path(local["results_dir"]) / "vram_vs_ctx" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for np_ in a.np:
        for ctx in a.ctx:
            cfg = load_server_config({"ctx_size": ctx, "parallel": np_, "kv_unified": a.unified})
            before = settle(h)
            row = {"ctx": ctx, "np": np_, "kv_unified": a.unified, "vram_before_mib": round(before),
                   "kv_expected_mib": round(ctx * KV_BYTES_PER_TOKEN / 2**20), "status": "ok",
                   "cmd": " ".join(build_cmd(local, cfg))}
            try:
                with Server(local, cfg, out / f"server_c{ctx}_np{np_}.log", health_timeout_s=120):
                    after = settle(h)
                    row.update(vram_loaded_mib=round(after), vram_delta_mib=round(after - before))
            except Exception as e:  # noqa: BLE001
                row.update(status=f"fail: {type(e).__name__}: {e}"[:200])
            time.sleep(2)  # let the driver release memory before the next baseline read
            rows.append(row)
            print({k: v for k, v in row.items() if k != "cmd"}, flush=True)

    with open(out / "vram_vs_ctx.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out / 'vram_vs_ctx.csv'}")


if __name__ == "__main__":
    main()
