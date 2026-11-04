"""Metrics from raw run logs (CLAUDE.md s8). Nothing here talks to a server; everything is recomputed from
requests.jsonl / tasks.jsonl / system.jsonl / manifest.json, so figures can always be rebuilt.

Per run (`run_metrics`):
    tasks        planned, ok, failed by type, failure rate
    latency      task completion time (completion_t - arrival_t, ok tasks) p50/p95/p99, mean queue/model/tool
                 per task; p99 only when >= P99_MIN_TASKS ok tasks (decision 2026-11-03: capped runs report
                 p50/p95 only)
    throughput   completed tasks/min and generated tokens/s over the whole run [t0, t1], and over the
                 steady window (below)
    requests     TTFT p50/p95, decode ms/token p50/p95 (outputs >= DECODE_MIN_TOKENS only: for a few-token
                 output one stall behind another slot's prefill dominates), prefill tokens/s
    reuse        prompt tokens recomputed, cache-hit fraction (all requests and steps > 0)
    memory       peak VRAM in [t0, t1] and its increase over pre-launch VRAM; KV reserved vs runner-tracked
                 (mean/max, mean unused reservation); slot occupancy, admission queue length
    overhead     scheduler policy time, slot erase count/time

Steady window (decision 2026-11-03, option A): from the first task completion (ramp-up over) to the last
task admission (drain starts: no new work enters). Throughput in the window = tasks completed in it per
minute, and tokens of requests that ended in it per second. None when the window is shorter than
STEADY_MIN_FRAC of the run (e.g. K=16 runs where nearly every task was admitted before the first finished).

Across reps (`aggregate`): median, min, max of every numeric metric over VALID runs of one config; invalid
or incomplete runs are excluded and listed, never silently dropped.

Best fixed K (`best_k`): per workload, the K with the highest median tasks/min among configs whose reps are
all valid and have no failed task; within BEST_K_TIE of the best, the lower median p95 wins.

Usage:
    python -m analysis.metrics E2                   # experiment name or configs/experiments/<file>.yaml
    python -m analysis.metrics RUN_DIR [RUN_DIR ...]
Writes <repo>/results/<E>/runs.csv, summary.csv, exclusions.csv (+ best_k.json for E2).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.checks import check_run, failed, read_jsonl

REPO = Path(__file__).resolve().parents[1]
P99_MIN_TASKS = 50
DECODE_MIN_TOKENS = 32
STEADY_MIN_FRAC = 0.25
BEST_K_TIE = 0.02
CONFIG_KEYS = ["workload", "policy", "slot_mode", "k", "ctx_size", "kv_unified", "agents", "tasks",
               "cache_prompt", "erase", "admission_gate"]


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def steady_window(tasks: list[dict], duration_s: float) -> tuple[float, float] | None:
    done = [t["completion_t"] for t in tasks if t.get("completion_t") is not None]
    admitted = [t["admit_t"] for t in tasks if t.get("admit_t") is not None]
    if not done or not admitted:
        return None
    a, b = min(done), max(admitted)
    return (a, b) if duration_s and b - a >= STEADY_MIN_FRAC * duration_s else None


def run_metrics(run_dir: Path | str, checks: list[dict] | None = None) -> dict:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    reqs, tasks = read_jsonl(run_dir / "requests.jsonl"), read_jsonl(run_dir / "tasks.jsonl")
    system = read_jsonl(run_dir / "system.jsonl")
    cfg = man.get("config", {})
    checks = check_run(run_dir) if checks is None else checks
    bad = failed(checks)

    m: dict = {"run_dir": str(run_dir), "rep": cfg.get("rep"), "valid": not bad,
               "failed_checks": ",".join(c["name"] for c in bad),
               "warnings": ",".join(c["name"] for c in checks if c["level"] == "warn" and not c["ok"])}
    for k in CONFIG_KEYS:
        m[k] = cfg.get(k)
    if m["erase"] is None:  # manifests written before the erase field existed: baseline behaviour
        m["erase"] = bool(cfg.get("kv_unified", True))
    if m["admission_gate"] is None:
        m["admission_gate"] = True
    if m["cache_prompt"] is None:
        m["cache_prompt"] = True

    t0, t1 = man.get("t0_monotonic"), man.get("t1_monotonic")
    dur = man.get("duration_s") or ((t1 - t0) if t0 is not None and t1 is not None else None)
    ok = [t for t in tasks if t["status"] == "ok"]
    planned = len(man.get("workload", {}).get("planned_task_ids", [])) or len(tasks)
    by_status = Counter(t["status"] for t in tasks)
    m.update(duration_s=dur, tasks_planned=planned, tasks_ok=len(ok), tasks_failed=len(tasks) - len(ok),
             failure_rate=(len(tasks) - len(ok)) / planned if planned else None,
             failures=json.dumps({s: n for s, n in sorted(by_status.items()) if s != "ok"}))

    # --- task latency (ok tasks)
    ct = [t["completion_t"] - t["arrival_t"] for t in ok]
    m.update(task_p50_s=_pct(ct, 50), task_p95_s=_pct(ct, 95),
             task_p99_s=_pct(ct, 99) if len(ok) >= P99_MIN_TASKS else None, task_max_s=max(ct) if ct else None,
             queue_mean_s=_mean([t["total_queue_s"] for t in ok]),
             admit_wait_mean_s=_mean([t.get("admit_wait_s", 0.0) for t in ok]),
             model_mean_s=_mean([t["total_model_s"] for t in ok]),
             tool_mean_s=_mean([t["total_tool_s"] for t in ok]),
             queue_share=sum(t["total_queue_s"] for t in ok) / sum(ct) if ct and sum(ct) > 0 else None)

    # --- throughput
    ok_reqs = [r for r in reqs if r["status"] == "ok"]
    gen = sum(r["generated_tokens"] or 0 for r in ok_reqs)
    m.update(tasks_per_min=60 * len(ok) / dur if dur else None, gen_tok_per_s=gen / dur if dur else None,
             gen_tokens=gen)
    win = steady_window(tasks, dur)
    if win:
        a, b = win
        n_in = sum(1 for t in ok if a < t["completion_t"] <= b)
        tok_in = sum(r["generated_tokens"] or 0 for r in ok_reqs if a < r["end_t"] <= b)
        m.update(steady_s=b - a, steady_tasks_per_min=60 * n_in / (b - a), steady_gen_tok_per_s=tok_in / (b - a))
    else:
        m.update(steady_s=None, steady_tasks_per_min=None, steady_gen_tok_per_s=None)

    # --- requests
    ttft = [r["first_token_t"] - r["dispatch_t"] for r in ok_reqs if r.get("first_token_t") is not None]
    dec = [r["gen_ms"] / r["generated_tokens"] for r in ok_reqs
           if r.get("gen_ms") and (r["generated_tokens"] or 0) >= DECODE_MIN_TOKENS]
    pf = [r["tokens_processed"] / (r["prompt_ms"] / 1000) for r in ok_reqs
          if r.get("prompt_ms") and (r.get("tokens_processed") or 0) >= 64]
    m.update(requests=len(reqs), ttft_p50_s=_pct(ttft, 50), ttft_p95_s=_pct(ttft, 95),
             decode_ms_per_tok_p50=_pct(dec, 50), decode_ms_per_tok_p95=_pct(dec, 95),
             prefill_tok_per_s_p50=_pct(pf, 50))

    # --- reuse
    n_prompt = sum(r["n_prompt"] for r in reqs)
    later = [r for r in ok_reqs if r["step"] > 0]
    m.update(prompt_tokens=n_prompt, prompt_tokens_computed=sum(r.get("tokens_processed") or 0 for r in reqs),
             cache_hit=sum(r.get("cached_tokens") or 0 for r in ok_reqs) / max(1, sum(r["n_prompt"] for r in ok_reqs)),
             cache_hit_steps_gt0=(sum(r.get("cached_tokens") or 0 for r in later)
                                  / max(1, sum(r["n_prompt"] for r in later))) if later else None)

    # --- memory and occupancy (samples inside the measurement window)
    win_sys = [s for s in system if t0 is None or (t0 <= s["t"] <= (t1 if t1 is not None else s["t"]))]
    vram = [s["vram_used_mib"] for s in win_sys if s.get("vram_used_mib") is not None]
    pre = man.get("prelaunch_vram_mib")
    sc = [s["sched"] for s in win_sys if s.get("sched")]
    res = [s["kv_reserved"] for s in sc]
    used = [s["kv_used_tracked"] for s in sc]
    m.update(vram_peak_mib=max(vram) if vram else None, vram_prelaunch_mib=pre,
             vram_post_load_mib=man.get("post_load_vram_mib"),
             vram_peak_over_prelaunch_mib=(max(vram) - pre) if vram and pre is not None else None,
             kv_budget=cfg.get("kv_budget"),
             kv_reserved_mean=_mean(res), kv_reserved_max=max(res) if res else None,
             kv_tracked_mean=_mean(used), kv_tracked_max=max(used) if used else None,
             kv_unused_reservation_mean=_mean([a - b for a, b in zip(res, used)]) if sc and cfg.get("kv_budget") else None,
             busy_slots_mean=_mean([s["busy_slots"] for s in sc]),
             active_tasks_mean=_mean([s["active_tasks"] for s in sc]),
             admission_queue_mean=_mean([s["waiting"] for s in sc]),
             ready_queue_mean=_mean([s["ready"] for s in sc]))

    # --- overhead and server-side events
    st = man.get("scheduler", {})
    m.update(policy_overhead_s=st.get("policy_overhead_s"), erase_count=st.get("erase_count"),
             erase_s=st.get("erase_s"), max_inflight=st.get("max_inflight"),
             server_purges=man.get("server_log", {}).get("purging_slot"),
             server_kv_full_warnings=man.get("server_log", {}).get("kv_full_warnings"),
             repo_commit=man.get("env", {}).get("repo_commit"),
             repo_dirty=man.get("env", {}).get("repo_dirty"))
    return m


# ---------- across runs ----------

def config_key(m: dict) -> tuple:
    return tuple(m.get(k) for k in CONFIG_KEYS)


def aggregate(rows: list[dict]) -> list[dict]:
    """One row per config: median/min/max over valid reps of every numeric metric."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(config_key(r), []).append(r)
    out = []
    for key, rs in groups.items():
        good = [r for r in rs if r["valid"]]
        row = dict(zip(CONFIG_KEYS, key))
        row.update(reps=len(rs), reps_valid=len(good), all_valid=len(good) == len(rs),
                   any_failed_task=any(r["tasks_failed"] for r in good))
        numeric = [k for k in rs[0] if k not in CONFIG_KEYS and k != "rep"
                   and isinstance(rs[0][k], (int, float)) and not isinstance(rs[0][k], bool)]
        for k in numeric:
            vals = [r[k] for r in good if r.get(k) is not None]
            row[k] = float(np.median(vals)) if vals else None
            row[k + "_min"] = float(min(vals)) if vals else None
            row[k + "_max"] = float(max(vals)) if vals else None
        out.append(row)
    return out


def best_k(summary: list[dict]) -> dict[str, dict]:
    """Per workload: the tuned fixed-concurrency baseline (see module docstring)."""
    out = {}
    for wl in sorted({s["workload"] for s in summary}):
        cands = [s for s in summary if s["workload"] == wl and s["all_valid"] and not s["any_failed_task"]
                 and s.get("tasks_per_min") is not None]
        if not cands:
            out[wl] = {"k": None, "reason": "no config with all reps valid and no failed task"}
            continue
        top = max(c["tasks_per_min"] for c in cands)
        near = [c for c in cands if c["tasks_per_min"] >= (1 - BEST_K_TIE) * top]
        pick = min(near, key=lambda c: (c.get("task_p95_s") or float("inf"), c["k"]))
        out[wl] = {"k": pick["k"], "tasks_per_min": pick["tasks_per_min"], "task_p95_s": pick.get("task_p95_s"),
                   "near_best": sorted(c["k"] for c in near),
                   "by_k": {c["k"]: round(c["tasks_per_min"], 3) for c in sorted(cands, key=lambda c: c["k"])}}
    return out


def experiment_run_dirs(exp_path: Path, results_dir: Path) -> list[Path]:
    """Every run dir an experiment plans (expanded from its YAML; workload sizes from existing meta files)."""
    import yaml

    from experiments.run_sweep import expand, load_experiment
    from runner.workload_gen import apply_overrides

    def wl_info(w, seed):  # from the YAMLs alone: works before a variant's task file has been generated
        base = yaml.safe_load((REPO / "configs" / "workloads" / f"{w['from'] or w['name']}.yaml").read_text(encoding="utf-8"))
        wl = apply_overrides(base, w["set"])
        s = wl["seed"] if (wl.get("fixed_seed") or seed is None) else seed
        return {"stem": f"{w['name']}_s{s}", "num_tasks": int(wl["num_tasks"])}

    return [r["cfg"].default_dir(results_dir) for r in expand(load_experiment(exp_path), False, wl_info)]


def resolve_experiment(arg: str) -> Path | None:
    p = Path(arg)
    if p.suffix == ".yaml" and p.exists():
        return p
    hits = sorted((REPO / "configs" / "experiments").glob(f"{arg}_*.yaml")) + \
        sorted((REPO / "configs" / "experiments").glob(f"{arg}.yaml"))
    return hits[0] if hits else None


def write_csv(rows: list[dict], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def collect(run_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    """(metrics rows of finished runs, exclusions with reasons)."""
    rows, excl = [], []
    for d in run_dirs:
        if not (d / "DONE").exists():
            excl.append({"run_dir": str(d), "reason": "not done" if not d.exists() else "incomplete (no DONE)"})
            continue
        m = run_metrics(d)
        rows.append(m)
        if not m["valid"]:
            excl.append({"run_dir": str(d), "reason": f"INVALID: {m['failed_checks']}"})
    return rows, excl


def main() -> None:
    from runner.server import load_local

    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="experiment name (E2) / YAML, or run dirs")
    ap.add_argument("--out", help="output dir (default <repo>/results/<E>)")
    a = ap.parse_args()
    results_dir = Path(load_local()["results_dir"])

    exp = resolve_experiment(a.targets[0]) if len(a.targets) == 1 else None
    if exp:
        name = exp.stem.split("_")[0]
        run_dirs = experiment_run_dirs(exp, results_dir)
    else:
        name, run_dirs = "adhoc", [Path(t) for t in a.targets]
    rows, excl = collect(run_dirs)
    summary = aggregate(rows) if rows else []
    out = Path(a.out) if a.out else REPO / "results" / name
    write_csv(rows, out / "runs.csv")
    write_csv(summary, out / "summary.csv")
    write_csv(excl, out / "exclusions.csv")
    print(f"{name}: {len(run_dirs)} planned, {len(rows)} done ({sum(r['valid'] for r in rows)} valid), "
          f"{len(excl)} excluded -> {out}")
    if name == "E2" and summary:
        bk = best_k(summary)
        (out / "best_k.json").write_text(json.dumps(bk, indent=1), encoding="utf-8")
        for wl, b in bk.items():
            print(f"  best fixed K  {wl:24s} K={b['k']}  {b.get('by_k', b.get('reason'))}")


if __name__ == "__main__":
    main()
