"""Validity checks for one run directory (CLAUDE.md s8). Run on every analysis.

Each check returns {"name", "level", "ok", "detail"}:
    fail  the run is not a valid measurement if this is not ok
    warn  the run is valid but something needs explaining (e.g. a silent server-side KV purge)
    info  summary numbers (always ok)

Usage: python -m analysis.checks RUN_DIR [RUN_DIR ...]     exit code 1 if any fail-level check fails
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    ev = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def check_run(run_dir: Path | str) -> list[dict]:
    run_dir = Path(run_dir)
    out: list[dict] = []

    def add(name, level, ok, detail=""):
        out.append({"name": name, "level": level, "ok": bool(ok), "detail": detail})

    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        add("manifest_present", "fail", False, "no manifest.json")
        return out
    man = json.loads(mpath.read_text(encoding="utf-8"))
    reqs = read_jsonl(run_dir / "requests.jsonl")
    tasks = read_jsonl(run_dir / "tasks.jsonl")
    system = read_jsonl(run_dir / "system.jsonl")
    cfg = man.get("config", {})
    k = cfg.get("k")
    budget = cfg.get("kv_budget")

    add("run_complete", "fail", man.get("status") == "complete" and (run_dir / "DONE").exists(),
        f"status={man.get('status')}" + (f"; {man.get('error', '')[:200]}" if man.get("error") else ""))
    if "server_exit_code_during_run" in man:
        add("server_alive_during_run", "fail", man["server_exit_code_during_run"] is None,
            f"exit code {man['server_exit_code_during_run']}")

    # --- task accounting: completed + failed = planned, nothing skipped or duplicated
    planned = man.get("workload", {}).get("planned_task_ids", [])
    logged = [t["task_id"] for t in tasks]
    dups = [t for t, c in Counter(logged).items() if c > 1]
    missing = sorted(set(planned) - set(logged))
    extra = sorted(set(logged) - set(planned))
    n_ok = sum(t["status"] == "ok" for t in tasks)
    add("task_accounting", "fail", not dups and not missing and not extra and len(logged) == len(planned),
        f"planned {len(planned)}, logged {len(logged)} (ok {n_ok}, failed {len(logged) - n_ok}); "
        f"missing {missing[:5]}, extra {extra[:5]}, duplicated {dups[:5]}")

    # --- every successful request generated exactly what was asked (no shortened outputs)
    ok_reqs = [r for r in reqs if r["status"] == "ok"]
    bad_len = [(r["task_id"], r["step"]) for r in ok_reqs if r["generated_tokens"] != r["n_predict"]]
    add("output_lengths", "fail", not bad_len, f"{len(bad_len)} of {len(ok_reqs)} ok requests off; {bad_len[:5]}")

    # --- intended prompt lengths match what was sent
    bad_prompt = [(r["task_id"], r["step"]) for r in reqs if r["n_prompt"] != r["prompt_tokens_expected"]]
    task_mm = sum(t.get("prompt_len_mismatches", 0) for t in tasks)
    add("prompt_lengths", "fail", not bad_prompt and task_mm == 0,
        f"{len(bad_prompt)} requests with unexpected prompt length; task-level mismatches {task_mm}")

    # --- ok tasks ran every step, and each step has exactly one ok request
    by_task: dict[str, list[dict]] = {}
    for r in reqs:
        by_task.setdefault(r["task_id"], []).append(r)
    bad_steps = []
    for t in tasks:
        if t["status"] != "ok":
            continue
        rs = by_task.get(t["task_id"], [])
        if t["steps_done"] != t["steps_planned"] or sorted(r["step"] for r in rs) != list(range(t["steps_planned"])) \
                or any(r["status"] != "ok" for r in rs):
            bad_steps.append(t["task_id"])
    add("ok_tasks_complete", "fail", not bad_steps, f"{len(bad_steps)} ok tasks with missing/extra steps {bad_steps[:5]}")

    # --- failed tasks are explained by a failed request or a recorded task-level reason
    unexplained = [t["task_id"] for t in tasks if t["status"] != "ok"
                   and not t.get("error") and t.get("failed_step") is None]
    add("failures_explained", "fail", not unexplained, f"{len(unexplained)} failed tasks without an error {unexplained[:5]}")

    # --- concurrency: at most K requests in flight, never two on one slot
    iv = [(r["dispatch_t"], r["end_t"]) for r in reqs
          if r.get("dispatch_t") is not None and r.get("end_t") is not None]
    peak = _max_overlap(iv) if iv else 0
    add("inflight_le_k", "fail", k is None or peak <= k, f"max in-flight {peak}, K={k}")
    clash = []
    for slot in {r["slot"] for r in reqs}:
        s_iv = sorted((r["dispatch_t"], r["end_t"]) for r in reqs if r["slot"] == slot)
        clash += [slot for (a0, a1), (b0, _) in zip(s_iv, s_iv[1:]) if b0 < a1 - 1e-6]
    add("slot_exclusive", "fail", not clash, f"overlapping requests on slots {sorted(set(clash))}")

    # --- the server never queued a request itself (queueing must be measured by the runner)
    deferred = [s["metrics"].get("llamacpp:requests_deferred", 0) for s in system if s.get("metrics")]
    add("server_never_deferred", "fail", not deferred or max(deferred) == 0,
        f"max llamacpp:requests_deferred {max(deferred) if deferred else 'n/a (no samples)'}")

    # --- KV pool accounting
    if budget is not None:
        res = [s["sched"]["kv_reserved"] for s in system if s.get("sched")]
        mx = man.get("scheduler", {}).get("max_kv_reserved", max(res) if res else 0)
        add("kv_reserved_le_pool", "fail", mx <= budget, f"max reserved {mx} of pool {budget}")
    short = [(r["task_id"], r["step"], r["cached_tokens"], r["cache_expected"]) for r in ok_reqs
             if r.get("cache_expected") and (r.get("cached_tokens") or 0) < r["cache_expected"]]
    purges = man.get("server_log", {}).get("purging_slot", 0)
    # With erase off (E5 "keep"), the server purging finished tasks' leftover KV is the expected behaviour;
    # only a live task losing cached tokens still counts.
    purges_expected = cfg.get("erase") is False
    add("no_silent_kv_purge", "warn", not short and (purges == 0 or purges_expected),
        f"{len(short)} requests cached fewer tokens than expected {short[:3]}; server log purges {purges}"
        + (" (erase off: purges of finished tasks' KV expected)" if purges_expected else ""))
    erase_fail = man.get("scheduler", {}).get("erase_failures", [])
    add("erase_ok", "warn", not erase_fail, f"{len(erase_fail)} erase failures {erase_fail[:2]}")

    # --- sampler coverage
    smp = man.get("sampler", {})
    add("sampler_coverage", "warn", bool(system) and smp.get("max_gap_s", 0) <= 1.0,
        f"{len(system)} samples, max gap {smp.get('max_gap_s')} s")

    # --- info
    add("failures_by_type", "info", True, json.dumps(dict(Counter(t["status"] for t in tasks))))
    later = [r for r in ok_reqs if r["step"] > 0]
    hit = sum(r.get("cached_tokens") or 0 for r in later) / max(1, sum(r["n_prompt"] for r in later))
    add("cache_hit_steps_gt0", "info", True, f"{hit:.3f}; prompt tokens computed "
        f"{sum(r.get('tokens_processed') or 0 for r in reqs)}")
    return out


def failed(results: list[dict]) -> list[dict]:
    return [c for c in results if c["level"] == "fail" and not c["ok"]]


def print_report(results: list[dict], title: str = "") -> None:
    if title:
        print(f"== {title}")
    mark = {("fail", False): "FAIL", ("warn", False): "WARN"}
    for c in results:
        print(f"  {mark.get((c['level'], c['ok']), 'ok  ')}  {c['name']:24s} {c['detail']}")
    print(f"  -> {'VALID' if not failed(results) else 'INVALID'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    bad = False
    for d in a.run_dirs:
        res = check_run(d)
        print_report(res, d)
        bad |= bool(failed(res))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
