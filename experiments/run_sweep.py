"""Sweep runner: an experiment YAML (configs/experiments/) -> one runner.run.run_one per point x rep.

Per run (runner/run.py): fresh server with that point's flags, wait for /health, discarded warm-up batch,
workload, drain, logs + manifest, DONE marker. Then analysis.checks marks the run VALID or INVALID.

Resumable: a run dir with DONE is skipped (also when another experiment already ran the identical config).
A run dir without DONE is kept as `*.incomplete-<ts>` and redone. A failed run is never deleted; an
INVALID run stays in place and is marked in status.json, not rerun.

Outputs in <results_dir>/sweeps/<name>[-pilot]/:
    plan.json     every planned run: config, workload, run dir (rewritten each invocation; deterministic)
    log.jsonl     one line per attempt this invocation: start/end, status, verdict, failed checks
    status.json   per run dir: done / valid / failed checks (rewritten at the end)

Usage:
    python -m experiments.run_sweep configs/experiments/E4_idle_periods.yaml --pilot
    python -m experiments.run_sweep configs/experiments/E2_concurrency.yaml --dry-run
    python -m experiments.run_sweep configs/experiments/E2_concurrency.yaml --only k=4 --only workload=agentic
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from runner.run import RunConfig

REPO = Path(__file__).resolve().parents[1]
WORKLOADS_DIR = REPO / "configs" / "workloads"
RUN_FIELDS = {f.name for f in dataclasses.fields(RunConfig)}


# ---------- expansion (pure: no corpus, no server) ----------

def load_experiment(path: Path | str) -> dict:
    exp = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    exp["_path"] = str(Path(path).as_posix())
    return exp


def norm_workload(w) -> dict:
    if isinstance(w, str):
        return {"name": w, "from": None, "set": {}, "run": {}}
    unknown = set(w) - {"name", "from", "set", "run"}
    if unknown:
        raise ValueError(f"workload {w.get('name')}: unknown keys {sorted(unknown)}")
    run = dict(w.get("run") or {})
    for key in run:
        if key not in RUN_FIELDS or key in ("workload", "rep"):
            raise ValueError(f"workload {w['name']}: run override {key!r} is not a RunConfig field")
    return {"name": w["name"], "from": w.get("from"), "set": dict(w.get("set") or {}), "run": run}


def spec_for(exp: dict, pilot: bool) -> dict:
    """The experiment, with the `pilot` block's overrides applied when pilot=True."""
    spec = {k: exp.get(k) for k in ("base", "workloads", "seeds", "grid", "exclude", "task_cap", "reps")}
    if pilot:
        p = exp.get("pilot") or {}
        unknown = set(p) - set(spec)
        if unknown:
            raise ValueError(f"{exp['name']}: unknown pilot keys {sorted(unknown)}")
        for k, v in p.items():
            spec[k] = {**(spec["base"] or {}), **v} if k == "base" else v
    spec["base"] = spec["base"] or {}
    spec["grid"] = spec["grid"] or {}
    spec["exclude"] = spec["exclude"] or []
    spec["reps"] = int(spec["reps"] or 1)
    for key in list(spec["base"]) + list(spec["grid"]):
        if key not in RUN_FIELDS or key in ("workload", "rep"):
            raise ValueError(f"{exp['name']}: {key!r} is not a RunConfig field")
    return spec


def grid_points(spec: dict) -> list[dict]:
    keys = list(spec["grid"])
    out = []
    for combo in itertools.product(*(spec["grid"][k] for k in keys)):
        pt = dict(zip(keys, combo))
        if any(all(pt.get(k) == v for k, v in ex.items()) for ex in spec["exclude"]):
            continue
        out.append(pt)
    return out or [{}]


def capped_tasks(num_tasks: int, k: int, cap: dict | None) -> int | None:
    """Decision 2026-11-03: n = min(num_tasks, max(min, per_k*K)); None = the whole file."""
    if not cap:
        return None
    n = min(num_tasks, max(int(cap["min"]), int(cap["per_k"]) * k))
    return n if n < num_tasks else None


def expand(exp: dict, pilot: bool, wl_info) -> list[dict]:
    """Every run of the experiment, in execution order (workload-major, then grid, then rep).

    wl_info(workload_entry, seed) -> {"stem": "<name>_s<seed>", "num_tasks": int} (generates the file in
    real sweeps; tests pass a fake). Returns [{"cfg": RunConfig, "workload": entry, "seed": seed}].
    """
    spec = spec_for(exp, pilot)
    runs = []
    for w in map(norm_workload, spec["workloads"]):
        for seed in spec["seeds"] or [None]:
            info = wl_info(w, seed)
            for pt in grid_points(spec):
                fields = {**spec["base"], **w["run"], **pt}
                if fields.get("tasks") is None:
                    fields["tasks"] = capped_tasks(info["num_tasks"], int(fields.get("k", 4)), spec["task_cap"])
                for rep in range(spec["reps"]):
                    runs.append({"cfg": RunConfig(workload=info["stem"], rep=rep, **fields),
                                 "workload": w, "seed": seed})
    return runs


def matches(run: dict, only: list[str]) -> bool:
    """--only key=value filters on RunConfig fields or `workload` (the workload name without seed)."""
    for f in only:
        key, _, val = f.partition("=")
        have = run["workload"]["name"] if key == "workload" else getattr(run["cfg"], key)
        if str(have).lower() != val.lower():
            return False
    return True


# ---------- workload files ----------

def make_wl_info(local: dict):
    """wl_info for real sweeps: derive variant YAMLs, generate (or verify) the task file."""
    from runner.corpus import Corpus
    from runner.workload_gen import derive_yaml, write

    out_dir = Path(local["results_dir"]) / "workloads"
    corpus = Corpus(local)
    n, sha = corpus.n_tokens, corpus.meta["tokens_sha256"]
    cache: dict = {}

    def wl_info(w: dict, seed):
        key = (json.dumps({k: w[k] for k in ("name", "from", "set")}, sort_keys=True), seed)
        if key in cache:
            return cache[key]
        base = WORKLOADS_DIR / f"{w['from'] or w['name']}.yaml"
        if w["from"]:
            ypath = derive_yaml(base, w["name"], w["set"], out_dir / "derived")
        elif w["set"]:
            raise ValueError(f"workload {w['name']}: `set` needs `from`")
        else:
            ypath = base
        wl = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        eff_seed = wl["seed"] if (wl.get("fixed_seed") or seed is None) else seed
        path = write(ypath, eff_seed, out_dir, n, sha, refuse_change=True)
        cache[key] = {"stem": path.stem, "num_tasks": int(wl["num_tasks"])}
        return cache[key]

    return wl_info


# ---------- execution ----------

def describe(cfg: RunConfig) -> str:
    extra = [f"{k}={getattr(cfg, k)}" for k in ("cache_prompt", "erase_finished", "admission_gate", "kv_unified")
             if getattr(cfg, k) != RunConfig.__dataclass_fields__[k].default]
    return (f"{cfg.workload:28s} K={cfg.k:<2d} {cfg.slot_mode:6s} c={cfg.ctx_size:<6d} "
            f"tasks={cfg.tasks or 'all':<4} rep{cfg.rep} {' '.join(extra)}")


def run_status(run_dir: Path) -> dict:
    from analysis.checks import check_run, failed

    if not (run_dir / "DONE").exists():
        return {"done": False}
    res = check_run(run_dir)
    bad = failed(res)
    return {"done": True, "valid": not bad, "failed_checks": [c["name"] for c in bad],
            "warnings": [c["name"] for c in res if c["level"] == "warn" and not c["ok"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="configs/experiments/<E>.yaml")
    ap.add_argument("--pilot", action="store_true", help="run the experiment's pilot points instead")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (generates workload files only)")
    ap.add_argument("--only", action="append", default=[], help="key=value filter, repeatable")
    ap.add_argument("--max-runs", type=int, help="stop after this many new runs")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()

    from runner.run import run_one
    from runner.server import load_local

    local = load_local()
    exp = load_experiment(a.experiment)
    runs = [r for r in expand(exp, a.pilot, make_wl_info(local)) if matches(r, a.only)]
    results_dir = Path(local["results_dir"])
    for r in runs:
        r["run_dir"] = r["cfg"].default_dir(results_dir)
    sweep_dir = results_dir / "sweeps" / (exp["name"] + ("-pilot" if a.pilot else ""))
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "plan.json").write_text(json.dumps({
        "experiment": exp["name"], "file": exp["_path"], "pilot": a.pilot, "only": a.only,
        "runs": [{"run_dir": str(r["run_dir"]), "workload": r["workload"], "seed": r["seed"],
                  "config": dataclasses.asdict(r["cfg"])} for r in runs]}, indent=1), encoding="utf-8")

    todo = [r for r in runs if not (r["run_dir"] / "DONE").exists()]
    print(f"{exp['name']}{' (pilot)' if a.pilot else ''}: {len(runs)} runs planned, {len(runs) - len(todo)} done, "
          f"{len(todo)} to run  [{sweep_dir}]")
    for r in runs:
        print(f"  {'DONE' if r not in todo else 'todo'}  {describe(r['cfg'])}")
    if a.dry_run:
        return

    n_new = 0
    with open(sweep_dir / "log.jsonl", "a", encoding="utf-8") as log:
        for i, r in enumerate(todo, 1):
            if a.max_runs is not None and n_new >= a.max_runs:
                print(f"stopping: --max-runs {a.max_runs}")
                break
            print(f"\n[{i}/{len(todo)}] {describe(r['cfg'])}", flush=True)
            rec = {"run_dir": str(r["run_dir"]), "started_at": datetime.now().isoformat(timespec="seconds")}
            t0 = time.monotonic()
            try:
                run_one(r["cfg"], r["run_dir"], local)
                st = run_status(r["run_dir"])
                if not st["done"]:
                    man = json.loads((r["run_dir"] / "manifest.json").read_text(encoding="utf-8"))
                    st["error"] = (man.get("error") or "")[:500]
            except Exception as e:  # noqa: BLE001 -- recorded; the sweep goes on unless --stop-on-error
                st = {"done": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            n_new += 1
            rec.update(st, wall_s=round(time.monotonic() - t0, 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            verdict = "VALID" if st.get("valid") else ("INVALID " + ",".join(st.get("failed_checks", []))
                                                      if st["done"] else "ERROR " + st.get("error", "")[:200])
            print(f"  -> {verdict}  ({rec['wall_s']} s)" + (f"  warnings: {st['warnings']}" if st.get("warnings") else ""))
            if not st.get("valid") and a.stop_on_error:
                print("stopping: --stop-on-error")
                break

    status = {str(r["run_dir"]): run_status(r["run_dir"]) for r in runs}
    (sweep_dir / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    n_done = sum(s["done"] for s in status.values())
    n_valid = sum(s.get("valid", False) for s in status.values())
    print(f"\n{exp['name']}: {n_done}/{len(runs)} done, {n_valid} valid, {n_done - n_valid} invalid, "
          f"{len(runs) - n_done} not done")


if __name__ == "__main__":
    main()
