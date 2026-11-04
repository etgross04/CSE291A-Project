"""analysis.metrics on fake runs (run.execute + FakeClient) and on synthetic rows."""
from __future__ import annotations

import json

import pytest

from analysis.metrics import aggregate, best_k, collect, run_metrics, steady_window
from test_run_and_checks import make_run
from test_scheduler import FakeClient, make_task


def test_run_metrics_basic(tmp_path):
    d, out = make_run(tmp_path)
    m = run_metrics(d)
    assert m["valid"] and m["tasks_ok"] == 8 and m["tasks_failed"] == 0 and m["failure_rate"] == 0
    assert m["tasks_per_min"] == pytest.approx(60 * 8 / out["duration_s"])
    assert m["gen_tokens"] == 8 * 3 * 5                     # 8 tasks x 3 steps x out=5
    assert m["task_p99_s"] is None                         # < P99_MIN_TASKS ok tasks
    assert m["task_p50_s"] <= m["task_p95_s"] <= m["task_max_s"]
    assert m["decode_ms_per_tok_p50"] is None              # outputs of 5 tokens < DECODE_MIN_TOKENS
    assert 0 < m["cache_hit_steps_gt0"] < 1
    assert m["kv_reserved_max"] <= 400 and m["kv_tracked_max"] <= m["kv_reserved_max"]
    assert m["kv_unused_reservation_mean"] >= 0
    assert m["prompt_tokens_computed"] < m["prompt_tokens"]


def test_failed_task_counted(tmp_path):
    d, _ = make_run(tmp_path, client=FakeClient(pool=400, fail_on={2: "kv_exhausted"}))
    m = run_metrics(d)
    assert m["tasks_failed"] == 1 and m["failure_rate"] == pytest.approx(1 / 8)
    assert json.loads(m["failures"]) == {"kv_exhausted": 1}


def test_invalid_run_is_excluded_with_reason(tmp_path):
    d, _ = make_run(tmp_path)
    (d / "tasks.jsonl").write_text("\n".join((d / "tasks.jsonl").read_text().splitlines()[1:]) + "\n")
    rows, excl = collect([d, tmp_path / "missing"])
    assert len(rows) == 1 and not rows[0]["valid"]
    assert {e["reason"].split(":")[0] for e in excl} == {"INVALID", "not done"}


def test_steady_window():
    T = lambda a, c: {"admit_t": a, "completion_t": c}  # noqa: E731
    assert steady_window([T(0, 50), T(10, 80), T(70, 120)], 120) is None   # 20 s < 25% of 120 s
    assert steady_window([T(0, 50), T(10, 80), T(100, 150)], 150) == (50, 100)
    assert steady_window([T(0, 50), T(5, 60)], 60) is None  # everything admitted before the first finished


def row(wl, k, rep, tpm, p95, valid=True, failed=0):
    return {"workload": wl, "policy": "fifo", "slot_mode": "pinned", "k": k, "ctx_size": 65536, "kv_unified": True,
            "agents": None, "tasks": None, "cache_prompt": True, "erase": True, "admission_gate": True,
            "rep": rep, "valid": valid, "tasks_failed": failed, "tasks_per_min": tpm, "task_p95_s": p95}


def test_aggregate_median_min_max_valid_only():
    rows = [row("a", 4, 0, 10.0, 50), row("a", 4, 1, 12.0, 40), row("a", 4, 2, 11.0, 45),
            row("a", 4, 3, 99.0, 1, valid=False)]
    (s,) = aggregate(rows)
    assert s["reps"] == 4 and s["reps_valid"] == 3 and not s["all_valid"]
    assert s["tasks_per_min"] == 11.0 and s["tasks_per_min_min"] == 10.0 and s["tasks_per_min_max"] == 12.0


def test_best_k_tie_and_failures():
    rows = [row("a", k, r, tpm, p95) for k, tpm, p95 in [(1, 3.0, 300), (4, 10.0, 90), (8, 10.1, 120)] for r in range(3)]
    rows += [row("a", 16, r, 20.0, 50, failed=1) for r in range(3)]       # fastest, but loses tasks
    bk = best_k(aggregate(rows))["a"]
    assert bk["k"] == 4                                    # within 2% of K=8, lower p95
    assert bk["near_best"] == [4, 8] and 16 not in bk["by_k"]


def test_best_k_none_when_nothing_qualifies():
    assert best_k(aggregate([row("b", 4, 0, 5.0, 10, failed=2)]))["b"]["k"] is None
