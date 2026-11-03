"""One measured run: fresh server -> warm-up -> workload -> drain -> logs + manifest (CLAUDE.md s6.5).

Run directory contents:
    requests.jsonl   one record per model request (written as each finishes)
    tasks.jsonl      one record per task
    system.jsonl     sampler records every ~200 ms (runner/samplers.py)
    warmup.jsonl     warm-up requests (discarded from analysis)
    server.log       llama-server stdout/stderr, first line = full command line
    manifest.json    environment, config, hashes, pre-launch VRAM, times, outcome
    DONE             written last, only if the run completed (resumability marker)

Resumable: a run dir with DONE is skipped. A run dir without DONE (crashed or interrupted earlier) is
renamed to `<dir>.incomplete-<timestamp>` and kept, never deleted, then the run is redone.

All runner timestamps are time.monotonic(); manifest["t0_monotonic"] anchors the measurement window.

Usage:
    python -m runner.run --workload pilot_s42 --k 4 --mode pinned --pool 65536
    python -m runner.run --workload trace_agentic_s1000 --k 4 --mode any --pool 65536 --tasks 10
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from runner.agent import run_closed, run_open
from runner.client import LlamaClient
from runner.scheduler import FifoPolicy, Scheduler

POLICIES = {"fifo": lambda cfg: FifoPolicy(cfg.slot_mode)}  # Phase 2 policies register here


@dataclass
class RunConfig:
    workload: str                    # "<name>_s<seed>" in <results_dir>/workloads
    k: int = 4                       # slots (-np) = max in-flight requests
    slot_mode: str = "pinned"        # pinned | any
    policy: str = "fifo"
    pool: int | None = None          # -c tokens; default 16384 * k
    kv_unified: bool = True          # baseline: one shared pool (-kvu); False = static c/k split
    agents: int | None = None        # closed loop: override the workload's agent count
    tasks: int | None = None         # use only the first N tasks (pilots)
    deadline_s: float = 1800.0       # per-task deadline
    request_timeout_s: float = 300.0
    sample_interval_s: float = 0.2
    warmup: bool = True
    rep: int = 0
    tag: str = ""
    server_overrides: dict = field(default_factory=dict)

    @property
    def ctx_size(self) -> int:
        return self.pool or 16384 * self.k

    @property
    def kv_budget(self) -> int | None:
        return self.ctx_size if self.kv_unified else None

    def default_dir(self, results_dir: Path) -> Path:
        kv = "kvu" if self.kv_unified else "split"
        name = f"{self.policy}-{self.slot_mode}_K{self.k}_c{self.ctx_size}_{kv}"
        if self.agents:
            name += f"_a{self.agents}"
        if self.tasks:
            name += f"_n{self.tasks}"
        if self.tag:
            name += f"_{self.tag}"
        return Path(results_dir) / "runs" / self.workload / name / f"rep{self.rep}"


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "w", encoding="utf-8", newline="\n")
        self.n = 0

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()
        self.n += 1

    def close(self) -> None:
        self._f.close()


def load_workload(results_dir: Path, workload: str, n_tasks: int | None) -> tuple[list[dict], dict, Path]:
    wl_dir = Path(results_dir) / "workloads"
    path = wl_dir / f"{workload}.jsonl"
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads((wl_dir / f"{workload}.meta.json").read_text(encoding="utf-8"))
    return (tasks[:n_tasks] if n_tasks else tasks), meta, path


async def warmup(client, k: int, corpus, log, erase: bool) -> None:
    """One short request per slot (concurrently), then erase every slot so the measured run starts cold."""
    n = corpus.n_tokens
    prompts = [corpus.slice({"offset": n - 4096 - 300 * i, "len": 256}) for i in range(k)]
    res = await asyncio.gather(*[client.complete(p, 16, id_slot=i) for i, p in enumerate(prompts)])
    for i, r in enumerate(res):
        log({"slot": i, **r.log_record()})
    if erase:
        for i in range(k):
            await client.erase_slot(i)


async def execute(cfg: RunConfig, client, tasks: list[dict], meta: dict, corpus, prefix_ids: list[int],
                  req_log, task_log, sampler_factory=None) -> dict:
    """The measured part of a run. Works with any client (tests use a fake)."""
    sched = Scheduler(client, cfg.k, POLICIES[cfg.policy](cfg), kv_budget=cfg.kv_budget)
    sampler = sampler_factory(sched) if sampler_factory else None
    if sampler:
        await sampler.start()
    t0 = time.monotonic()
    kw = dict(sched=sched, corpus=corpus, prefix_ids=prefix_ids, deadline_s=cfg.deadline_s,
              log_request=req_log, log_task=task_log)
    arrival = meta["arrival"]
    if arrival["type"] == "closed":
        await run_closed(tasks, cfg.agents or arrival["agents"], **kw)
    else:
        await run_open(tasks, t0=t0, **kw)
    await sched.drain()
    t1 = time.monotonic()
    out = {"t0_monotonic": t0, "t1_monotonic": t1, "duration_s": t1 - t0, "scheduler": sched.stats(),
           "agents": cfg.agents or arrival.get("agents")}
    if sampler:
        await sampler.stop()
        out["sampler"] = {"n_samples": sampler.n_samples, "max_gap_s": round(sampler.max_gap_s, 3),
                          "interval_s": sampler.interval_s}
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(cfg: RunConfig, run_dir: Path | None = None, local: dict | None = None) -> Path:
    from runner.corpus import Corpus
    from runner.env import environment, gpu_mem_used_mib
    from runner.samplers import NvmlSource, RssSource, SystemSampler
    from runner.server import Server, load_local, load_server_config

    local = local or load_local()
    run_dir = Path(run_dir or cfg.default_dir(Path(local["results_dir"])))
    if (run_dir / "DONE").exists():
        print(f"skip (DONE): {run_dir}")
        return run_dir
    if run_dir.exists():
        kept = run_dir.with_name(f"{run_dir.name}.incomplete-{datetime.now():%Y%m%d-%H%M%S}")
        run_dir.rename(kept)
        print(f"kept earlier incomplete attempt as {kept}")
    run_dir.mkdir(parents=True)

    tasks, meta, wl_path = load_workload(Path(local["results_dir"]), cfg.workload, cfg.tasks)
    corpus = Corpus(local)
    prefix = corpus.slice(meta["shared_prefix"])
    scfg = load_server_config({"ctx_size": cfg.ctx_size, "parallel": cfg.k, "kv_unified": cfg.kv_unified,
                               **cfg.server_overrides})
    manifest = {
        "status": "running",
        "config": dataclasses.asdict(cfg) | {"ctx_size": cfg.ctx_size, "kv_budget": cfg.kv_budget},
        "workload": {"name": cfg.workload, "file": str(wl_path), "file_sha256": _sha256(wl_path),
                     "meta": meta, "planned_task_ids": [t["task_id"] for t in tasks]},
        "corpus_tokens_sha256": corpus.meta.get("tokens_sha256"),
        "server_config": scfg,
        "env": environment(local, hash_model=True, packages=True),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_manifest = lambda: (run_dir / "manifest.json").write_text(  # noqa: E731
        json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    manifest["prelaunch_vram_mib"] = round(gpu_mem_used_mib(local.get("gpu_index", 0)), 1)
    write_manifest()

    writers = {n: JsonlWriter(run_dir / f"{n}.jsonl") for n in ("requests", "tasks", "system", "warmup")}
    server = Server(local, scfg, run_dir / "server.log")
    try:
        with server:
            manifest["server_cmd"] = server.cmd
            manifest["server_pid"] = server.pid
            manifest["post_load_vram_mib"] = round(gpu_mem_used_mib(local.get("gpu_index", 0)), 1)

            async def go():
                client = LlamaClient(server.base_url, request_timeout_s=cfg.request_timeout_s)
                try:
                    if cfg.warmup:
                        await warmup(client, cfg.k, corpus, writers["warmup"].write,
                                     erase=bool(scfg.get("slot_save_path")))
                    factory = lambda sched: SystemSampler(  # noqa: E731
                        writers["system"].write, base_url=server.base_url, sched=sched,
                        sources=[NvmlSource(local.get("gpu_index", 0)), RssSource(server.pid)],
                        interval_s=cfg.sample_interval_s)
                    return await execute(cfg, client, tasks, meta, corpus, prefix,
                                         writers["requests"].write, writers["tasks"].write, factory)
                finally:
                    await client.aclose()

            manifest.update(asyncio.run(go()))
            manifest["server_exit_code_during_run"] = server.proc.poll()
        manifest["status"] = "complete"
    except Exception as e:  # noqa: BLE001 -- the failed run is kept with its partial logs
        manifest["status"] = "error"
        manifest["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        for w in writers.values():
            w.close()
        manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["counts"] = {n: w.n for n, w in writers.items()}
        log_text = (run_dir / "server.log").read_text(encoding="utf-8", errors="replace") \
            if (run_dir / "server.log").exists() else ""
        manifest["server_log"] = {"purging_slot": log_text.count("purging slot"),
                                  "kv_full_warnings": log_text.count("failed to find free space")}
        write_manifest()
    if manifest["status"] == "complete":
        (run_dir / "DONE").write_text(manifest["ended_at"] + "\n", encoding="utf-8")
    print(f"{manifest['status']}: {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="<name>_s<seed>, e.g. pilot_s42")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mode", default="pinned", choices=["pinned", "any"])
    ap.add_argument("--policy", default="fifo", choices=sorted(POLICIES))
    ap.add_argument("--pool", type=int, help="-c KV tokens (default 16384*K)")
    ap.add_argument("--static-split", action="store_true", help="-no-kvu: each slot gets pool/K")
    ap.add_argument("--agents", type=int)
    ap.add_argument("--tasks", type=int)
    ap.add_argument("--deadline", type=float, default=1800.0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", help="run directory (default <results_dir>/runs/<workload>/<config>/rep<N>)")
    a = ap.parse_args()
    cfg = RunConfig(workload=a.workload, k=a.k, slot_mode=a.mode, policy=a.policy, pool=a.pool,
                    kv_unified=not a.static_split, agents=a.agents, tasks=a.tasks, deadline_s=a.deadline,
                    rep=a.rep, tag=a.tag, warmup=not a.no_warmup)
    run_dir = run_one(cfg, Path(a.out) if a.out else None)
    from analysis.checks import check_run, print_report
    print_report(check_run(run_dir))


if __name__ == "__main__":
    main()
