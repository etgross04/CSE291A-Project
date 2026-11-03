"""Sweep expansion, workload variants, and the E5/E6 run options (no GPU, no server, no corpus)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from experiments.run_sweep import capped_tasks, expand, load_experiment, matches, norm_workload
from runner.run import RunConfig, execute
from runner.workload_gen import WorkloadChanged, apply_overrides, derive_yaml, generate, write
from test_scheduler import FakeClient, FakeCorpus, make_task

REPO = Path(__file__).resolve().parents[1]
EXP_DIR = REPO / "configs" / "experiments"
WL_DIR = REPO / "configs" / "workloads"
EXPERIMENTS = sorted(EXP_DIR.glob("*.yaml"))
N_CORPUS = 1_500_000


def fake_wl_info(w, seed):
    base = yaml.safe_load((WL_DIR / f"{w['from'] or w['name']}.yaml").read_text())
    wl = apply_overrides(base, w["set"])
    s = wl["seed"] if (wl.get("fixed_seed") or seed is None) else seed
    return {"stem": f"{w['name']}_s{s}", "num_tasks": int(wl["num_tasks"])}


def runs_of(name, pilot=False):
    return expand(load_experiment(EXP_DIR / name), pilot, fake_wl_info)


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
@pytest.mark.parametrize("pilot", [False, True])
def test_every_experiment_expands_to_unique_run_dirs(path, pilot):
    runs = expand(load_experiment(path), pilot, fake_wl_info)
    assert runs
    dirs = [r["cfg"].default_dir(Path("R")) for r in runs]
    assert len(set(dirs)) == len(dirs)
    if pilot:
        assert len(runs) <= 8 and len(runs) < len(expand(load_experiment(path), False, fake_wl_info))


def test_task_cap_rule():
    assert capped_tasks(60, 1, {"per_k": 6, "min": 20}) == 20
    assert capped_tasks(60, 4, {"per_k": 6, "min": 20}) == 24
    assert capped_tasks(60, 8, {"per_k": 6, "min": 20}) == 48
    assert capped_tasks(60, 16, {"per_k": 6, "min": 20}) is None   # whole file: dir name has no _n
    assert capped_tasks(40, 8, {"per_k": 6, "min": 20}) is None
    assert capped_tasks(60, 4, None) is None


def test_e2_grid():
    runs = runs_of("E2_concurrency.yaml")
    assert len(runs) == 6 * 5 * 3
    tasks = {(r["workload"]["name"], r["cfg"].k): r["cfg"].tasks for r in runs}
    assert tasks[("agentic", 1)] == 20 and tasks[("agentic", 4)] == 24 and tasks[("agentic", 16)] is None
    assert tasks[("chat", 8)] == 48 and tasks[("chat", 16)] == 96
    assert all(r["cfg"].pool == 65536 and r["cfg"].slot_mode == "pinned" for r in runs)
    pilot = runs_of("E2_concurrency.yaml", pilot=True)
    assert sorted((r["workload"]["name"], r["cfg"].k) for r in pilot) == \
        [("agentic", 1), ("agentic", 16), ("trace_agentic", 1), ("trace_agentic", 16)]


def test_e5_exclude_and_flags():
    runs = runs_of("E5_reuse.yaml")
    assert not any(not r["cfg"].cache_prompt and r["cfg"].erase_finished is False for r in runs)
    names = {r["cfg"].default_dir(Path("R")).parent.name for r in runs}
    assert "fifo-pinned_K4_c65536_kvu_n24" in names            # erase on = the baseline, no suffix
    assert "fifo-pinned_K4_c65536_kvu_n24_keep" in names
    assert "fifo-pinned_K4_c65536_kvu_n24_nocache" in names


def test_e6_no_gate_has_no_budget_but_still_erases():
    runs = runs_of("E6_memory_budget.yaml")
    nogate = [r["cfg"] for r in runs if not r["cfg"].admission_gate]
    assert all(c.kv_budget is None and c.kv_unified and c.erase for c in nogate)
    assert all(c.default_dir(Path("R")).parent.name.endswith("_nogate") for c in nogate)


def test_only_filter():
    runs = runs_of("E2_concurrency.yaml")
    sel = [r for r in runs if matches(r, ["k=4", "workload=agentic"])]
    assert len(sel) == 3 and all(r["cfg"].k == 4 for r in sel)


def test_delay_variants_share_tasks_except_delays():
    """E4: constant-delay variants of agentic contain identical tasks; only tool_delay_s differs."""
    base = yaml.safe_load((WL_DIR / "agentic.yaml").read_text())
    tasks = {d: generate({**apply_overrides(base, {"task_classes.0.tool_delay": d}), "name": "a"}, 1000, N_CORPUS)
             for d in (0.0, 5.0, 30.0)}
    strip = lambda t: {**t, "total_tool_delay_s": None,  # noqa: E731
                       "steps": [{**s, "tool_delay_s": None} for s in t["steps"]]}
    assert [strip(t) for t in tasks[0.0]] == [strip(t) for t in tasks[30.0]] == [strip(t) for t in tasks[5.0]]
    for t in tasks[5.0]:
        assert [s["tool_delay_s"] for s in t["steps"]] == [5.0] * (len(t["steps"]) - 1) + [0.0]


def test_override_unknown_field_rejected():
    with pytest.raises(KeyError):
        apply_overrides({"a": 1}, {"b": 2})


def test_derive_yaml_and_refuse_changed_workload(tmp_path):
    y = derive_yaml(WL_DIR / "agentic.yaml", "agentic_d5", {"task_classes.0.tool_delay": 5.0}, tmp_path / "derived")
    assert yaml.safe_load(y.read_text())["name"] == "agentic_d5"
    first = y.read_bytes()
    assert derive_yaml(WL_DIR / "agentic.yaml", "agentic_d5", {"task_classes.0.tool_delay": 5.0},
                       tmp_path / "derived").read_bytes() == first
    write(y, 1000, tmp_path, N_CORPUS, "x", refuse_change=True)
    write(y, 1000, tmp_path, N_CORPUS, "x", refuse_change=True)   # same content: fine
    y2 = derive_yaml(WL_DIR / "agentic.yaml", "agentic_d5", {"task_classes.0.tool_delay": 6.0}, tmp_path / "derived")
    with pytest.raises(WorkloadChanged):
        write(y2, 1000, tmp_path, N_CORPUS, "x", refuse_change=True)


def test_calib_is_exact_cycle_grid():
    wl = yaml.safe_load((WL_DIR / "calib.yaml").read_text())
    tasks = generate(wl, wl["seed"], N_CORPUS)
    n = len(wl["task_classes"])
    assert n == 21 and len(tasks) == 42
    for i, t in enumerate(tasks):
        c = wl["task_classes"][i % n]
        assert t["class"] == c["name"] and t["init_ctx"]["len"] == c["init_ctx"]
        assert len(t["steps"]) == 1 and t["steps"][0]["out"] == c["out"]


def _execute(cfg, client):
    tasks = [make_task(i, delay=0.0) for i in range(4)]
    reqs = []
    meta = {"arrival": {"type": "closed", "agents": 2}}
    out = asyncio.run(execute(cfg, client, tasks, meta, FakeCorpus(), [1, 2, 3, 4], reqs.append, lambda r: None))
    return reqs, out


def test_cache_prompt_off_expects_no_cache():
    reqs, _ = _execute(RunConfig(workload="x", k=2, pool=400, cache_prompt=False), FakeClient(pool=400))
    assert all(r["cache_expected"] == 0 for r in reqs)
    reqs, _ = _execute(RunConfig(workload="x", k=2, pool=400), FakeClient(pool=400))
    assert any(r["cache_expected"] > 0 for r in reqs)


def test_keep_finished_skips_erase():
    c = FakeClient()
    _, out = _execute(RunConfig(workload="x", k=2, pool=400, erase_finished=False), c)
    assert out["scheduler"]["erase_count"] == 0 and not c.erased
    _, out = _execute(RunConfig(workload="x", k=2, pool=400), FakeClient(pool=400))
    assert out["scheduler"]["erase_count"] == 4


def test_norm_workload():
    assert norm_workload("chat") == {"name": "chat", "from": None, "set": {}, "run": {}}
    with pytest.raises(ValueError):
        norm_workload({"name": "x", "run": {"bogus": 1}})


def test_cut_grids_2026_11_03():
    e4 = runs_of("E4_idle_periods.yaml")
    assert len(e4) == 7 * 2 * 3
    tasks = {r["workload"]["name"]: r["cfg"].tasks for r in e4}
    assert tasks["agentic_d15"] == 16 and tasks["agentic_d5"] == 24 and "agentic_d30" not in tasks
    assert len(runs_of("E3_context_growth.yaml")) == 2 * 2 * 2
    assert len(runs_of("E5_reuse.yaml")) == 2 * 3 * 2
    assert len(runs_of("E6_memory_budget.yaml")) == 4 * 2 * 2
