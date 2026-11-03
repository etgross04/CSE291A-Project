"""Start/stop llama-server from a config dict, so every run uses an explicit, logged command line.

Only flags verified against `llama-server --help` for the pinned build are emitted.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_local() -> dict:
    p = REPO / "configs" / "local.yaml"
    if not p.exists():
        raise SystemExit(f"Missing {p}. Copy configs/local.example.yaml to configs/local.yaml and edit paths.")
    return yaml.safe_load(p.read_text())


def load_server_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load((REPO / "configs" / "server_baseline.yaml").read_text())
    cfg.update(overrides or {})
    return cfg


def build_cmd(local: dict, cfg: dict) -> list[str]:
    def onoff(v) -> str:
        return v if isinstance(v, str) else ("on" if v else "off")

    cmd = [
        local["llama_server"],
        "-m", local["model"],
        "--host", str(local.get("host", "127.0.0.1")),
        "--port", str(local.get("port", 8080)),
        "-c", str(cfg["ctx_size"]),
        "-np", str(cfg["parallel"]),
        "-ngl", str(cfg["gpu_layers"]),
        "--fit", onoff(cfg["fit"]),
        "-fa", onoff(cfg["flash_attn"]),
        "-ctk", cfg["cache_type_k"],
        "-ctv", cfg["cache_type_v"],
        "-b", str(cfg["batch_size"]),
        "-ub", str(cfg["ubatch_size"]),
        "-cram", str(cfg["cache_ram"]),
        "--cache-reuse", str(cfg["cache_reuse"]),
        "-sps", str(cfg["slot_prompt_similarity"]),
        "-kvu" if cfg["kv_unified"] else "-no-kvu",
        "--cache-idle-slots" if cfg["cache_idle_slots"] else "--no-cache-idle-slots",
        "--context-shift" if cfg["context_shift"] else "--no-context-shift",
    ]
    if cfg.get("metrics"):
        cmd.append("--metrics")
    if not cfg.get("webui", False):
        cmd.append("--no-webui")
    if cfg.get("slot_save_path"):
        ssp = cfg["slot_save_path"]
        if ssp == "auto":
            ssp = Path(local["results_dir"]) / "slot_saves"
            ssp.mkdir(parents=True, exist_ok=True)
        cmd += ["--slot-save-path", str(ssp)]
    for extra in cfg.get("extra_args", []) or []:
        cmd.append(str(extra))
    return cmd


class Server:
    """Context manager: launches llama-server, waits for /health, logs stdout/stderr to a file."""

    def __init__(self, local: dict, cfg: dict, log_path: Path, health_timeout_s: float = 180):
        self.local, self.cfg, self.log_path = local, cfg, Path(log_path)
        self.cmd = build_cmd(local, cfg)
        self.base_url = f"http://{local.get('host', '127.0.0.1')}:{local.get('port', 8080)}"
        self.health_timeout_s = health_timeout_s
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log.write(" ".join(self.cmd) + "\n\n")
        self._log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self.wait_healthy()
        return self

    def wait_healthy(self) -> None:
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                r = httpx.get(self.base_url + "/health", timeout=2)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not healthy after {self.health_timeout_s}s; see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()

    @property
    def pid(self) -> int:
        return self.proc.pid
