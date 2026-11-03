"""Environment facts for docs/environment.md and every run manifest (CLAUDE.md s6.5)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()  # llama-server --version writes to stderr
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def write_repo_diff(path: Path) -> bool:
    """If the repo has uncommitted changes, write them as one patch (tracked changes vs HEAD, then every
    untracked file as a new-file diff) so a run made from a dirty tree can still be reproduced.
    Returns True if a patch was written."""
    import difflib

    if not sh(["git", "-C", str(REPO), "status", "--porcelain"]):
        return False
    r = subprocess.run(["git", "-C", str(REPO), "diff", "HEAD", "--binary"], capture_output=True, timeout=60)
    parts = [r.stdout.decode("utf-8", errors="replace")]
    untracked = sh(["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard"]).splitlines()
    for rel in untracked:
        f = REPO / rel
        if not f.is_file() or f.stat().st_size > 1 << 20:
            parts.append(f"# untracked, not included (binary/large): {rel}\n")
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            parts.append(f"# untracked, not included (binary): {rel}\n")
            continue
        parts.append(f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n")
        parts.extend(difflib.unified_diff([], text.splitlines(keepends=True), "/dev/null", f"b/{rel}"))
        if text and not text.endswith("\n"):
            parts.append("\n\\ No newline at end of file\n")
    Path(path).write_text("".join(parts), encoding="utf-8", newline="\n")
    return True


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def model_sha256_cached(local: dict) -> str:
    """sha256 of the model, cached in <results_dir>/model_sha256.json keyed by path, size and mtime."""
    p = Path(local["model"])
    st = p.stat()
    key = f"{p.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
    cache = Path(local["results_dir"]) / "model_sha256.json"
    data = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if key not in data:
        data[key] = sha256_file(p)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data[key]


def gpu_info(gpu_index: int = 0) -> dict:
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    try:
        dm = pynvml.nvmlDeviceGetDriverModel(h)[0]  # Windows only: 0=WDDM, 1=TCC
        dm = {0: "WDDM", 1: "TCC"}.get(dm, str(dm))
    except Exception:  # noqa: BLE001
        dm = "n/a"
    return {
        "gpu": pynvml.nvmlDeviceGetName(h),
        "gpu_mem_total_mib": mem.total // 2**20,
        "gpu_mem_used_mib": mem.used // 2**20,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cuda_driver_api": pynvml.nvmlSystemGetCudaDriverVersion_v2(),
        "driver_model": dm,
    }


def gpu_mem_used_mib(gpu_index: int = 0) -> float:
    import pynvml

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_index)).used / 2**20


def environment(local: dict, hash_model: bool = False, packages: bool = True) -> dict:
    import psutil

    env = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        **gpu_info(local.get("gpu_index", 0)),
        "llama_server": local["llama_server"],
        "llama_version": sh([local["llama_server"], "--version"]).splitlines()[-2:],
        "model": local["model"],
        "model_size_bytes": Path(local["model"]).stat().st_size,
        "repo_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "repo_dirty": bool(sh(["git", "-C", str(REPO), "status", "--porcelain"])),
    }
    if packages:
        env["packages"] = sh([sys.executable, "-m", "pip", "freeze"]).splitlines()
    if hash_model:
        env["model_sha256"] = model_sha256_cached(local)
    return env
