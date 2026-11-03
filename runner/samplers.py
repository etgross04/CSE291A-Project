"""System sampler: one `system.jsonl` record every `interval_s` (CLAUDE.md s6.5).

Each record (all times time.monotonic()):
    t, vram_used_mib (whole GPU; WDDM has no per-process accounting), gpu_util_pct,
    server_rss_mib (llama-server process), metrics {llamacpp:* gauges/counters from /metrics},
    slots_processing [bool per slot from /slots], sched {runner-side state incl. KV pool accounting},
    scrape_ms (time spent on the HTTP scrapes), errors [any scrape failures]
/slots exposes no token counts and /metrics has no KV gauge on b11721, so KV occupancy comes from the
scheduler (`sched.kv_reserved`, `sched.kv_used_tracked`, per-slot `slot_kv_tokens`).

Sources are injectable so the sampler can be tested without a GPU or server.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

import httpx


def parse_prometheus(text: str) -> dict[str, float]:
    """`name value` lines -> {name: value}; comments and labelled series are skipped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


class NvmlSource:
    def __init__(self, gpu_index: int = 0):
        import pynvml

        pynvml.nvmlInit()
        self._nv = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def __call__(self) -> dict:
        mem = self._nv.nvmlDeviceGetMemoryInfo(self._h)
        util = self._nv.nvmlDeviceGetUtilizationRates(self._h)
        return {"vram_used_mib": round(mem.used / 2**20, 1), "gpu_util_pct": util.gpu}


class RssSource:
    def __init__(self, pid: int):
        import psutil

        self._p = psutil.Process(pid)

    def __call__(self) -> dict:
        try:
            return {"server_rss_mib": round(self._p.memory_info().rss / 2**20, 1)}
        except Exception:  # noqa: BLE001 -- process gone: recorded as missing, the run notices separately
            return {"server_rss_mib": None}


class SystemSampler:
    def __init__(self, write: Callable[[dict], None], *, base_url: str | None = None,
                 sched=None, sources: list[Callable[[], dict]] | None = None,
                 interval_s: float = 0.2, scrape_timeout_s: float = 1.0):
        self.write = write
        self.base_url = base_url
        self.sched = sched
        self.sources = sources or []
        self.interval_s = interval_s
        self.scrape_timeout_s = scrape_timeout_s
        self.n_samples = 0
        self.max_gap_s = 0.0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._http: httpx.AsyncClient | None = None

    async def _scrape(self, rec: dict) -> None:
        t0 = time.perf_counter()
        try:
            r = await self._http.get("/metrics")
            rec["metrics"] = parse_prometheus(r.text) if r.status_code == 200 else None
        except httpx.HTTPError as e:
            rec["errors"].append(f"metrics: {type(e).__name__}")
        try:
            r = await self._http.get("/slots")
            rec["slots_processing"] = [bool(s.get("is_processing")) for s in r.json()] \
                if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError) as e:
            rec["errors"].append(f"slots: {type(e).__name__}")
        rec["scrape_ms"] = round(1000 * (time.perf_counter() - t0), 2)

    async def sample_once(self) -> dict:
        rec: dict = {"t": time.monotonic(), "errors": []}
        for src in self.sources:
            try:
                rec.update(src())
            except Exception as e:  # noqa: BLE001
                rec["errors"].append(f"{type(src).__name__}: {e}")
        if self._http is not None:
            await self._scrape(rec)
        if self.sched is not None:
            rec["sched"] = self.sched.snapshot()
        self.write(rec)
        self.n_samples += 1
        return rec

    async def _loop(self) -> None:
        nxt, last = time.monotonic(), None
        while not self._stop.is_set():
            rec = await self.sample_once()
            if last is not None:
                self.max_gap_s = max(self.max_gap_s, rec["t"] - last)
            last = rec["t"]
            nxt += self.interval_s
            delay = nxt - time.monotonic()
            if delay < 0:  # fell behind (slow scrape): skip ahead rather than burst
                nxt, delay = time.monotonic(), 0
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        if self.base_url:
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.scrape_timeout_s)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        await self.sample_once()  # final sample after the run drained
        if self._http is not None:
            await self._http.aclose()
