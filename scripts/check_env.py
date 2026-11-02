"""Print (and optionally save as JSON) the environment facts that go into docs/environment.md and manifests.

Usage: python scripts/check_env.py [--hash-model] [--out env.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.server import REPO, load_local  # noqa: E402


def sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()  # llama-server --version writes to stderr
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-model", action="store_true", help="compute model sha256 (~10 s)")
    ap.add_argument("--out")
    a = ap.parse_args()
    local = load_local()

    import psutil
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(local.get("gpu_index", 0))
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    try:
        drv_model = pynvml.nvmlDeviceGetDriverModel(h)[0]  # Windows only: 0=WDDM, 1=TCC
        drv_model = {0: "WDDM", 1: "TCC"}.get(drv_model, str(drv_model))
    except Exception:  # noqa: BLE001
        drv_model = "n/a"

    env = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": pynvml.nvmlDeviceGetName(h),
        "gpu_mem_total_mib": mem.total // 2**20,
        "gpu_mem_used_idle_mib": mem.used // 2**20,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cuda_driver_api": pynvml.nvmlSystemGetCudaDriverVersion_v2(),
        "driver_model": drv_model,
        "llama_server": local["llama_server"],
        "llama_version": sh([local["llama_server"], "--version"]).splitlines()[-2:],
        "model": local["model"],
        "model_size_bytes": Path(local["model"]).stat().st_size,
        "repo_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "repo_dirty": bool(sh(["git", "-C", str(REPO), "status", "--porcelain"])),
        "packages": sh([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    if a.hash_model:
        env["model_sha256"] = sha256(local["model"])
    print(json.dumps(env, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
