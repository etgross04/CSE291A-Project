"""run.execute + samplers + analysis.checks end to end against the fake client (no GPU, no server)."""
from __future__ import annotations

import asyncio
import json

import pytest

from analysis.checks import check_run, failed
from runner.run import JsonlWriter, RunConfig, execute
from runner.samplers import SystemSampler, parse_prometheus
from test_scheduler import FakeClient, FakeCorpus, make_task


def make_run(tmp_path, *, tasks=None, k=2, mode="pinned", pool=400, client=None):
    tasks = tasks or [make_task(i, delay=0.01) for i in range(8)]
    meta = {"arrival": {"type": "closed", "agents": 4}}
    cfg = RunConfig(workload="fake", k=k, slot_mode=mode, pool=pool, sample_interval_s=0.01)
    client = client or FakeClient(pool=pool)
    w = {n: JsonlWriter(tmp_path / f"{n}.jsonl") for n in ("requests", "tasks", "system")}

    def factory(sched):
        fake_metrics = lambda: {"vram_used_mib": 1000.0}  # noqa: E731
        s = SystemSampler(w["system"].write, sched=sched, sources=[fake_metrics], interval_s=0.01)
        return s

    out = asyncio.run(execute(cfg, client, tasks, meta, FakeCorpus(), [1, 2, 3, 4],
                              w["requests"].write, w["tasks"].write, factory))
    for x in w.values():
        x.close()
    man = {"status": "complete", "config": {"k": k, "kv_budget": cfg.kv_budget},
           "workload": {"planned_task_ids": [t["task_id"] for t in tasks]},
           "server_exit_code_during_run": None, "server_log": {"purging_slot": 0}, **out}
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (tmp_path / "DONE").write_text("x")
    return tmp_path, out


def by_name(res):
    return {c["name"]: c for c in res}


@pytest.mark.parametrize("mode", ["pinned", "any"])
def test_clean_run_passes_all_checks(tmp_path, mode):
    d, out = make_run(tmp_path, mode=mode)
    res = check_run(d)
    assert not failed(res), [c for c in res if not c["ok"]]
    assert all(c["ok"] for c in res), [c for c in res if not c["ok"]]
    assert out["sampler"]["n_samples"] > 3
    sysrec = [json.loads(line) for line in (d / "system.jsonl").read_text().splitlines()]
    assert all("sched" in s and "kv_reserved" in s["sched"] for s in sysrec)


def test_missing_task_record_fails_accounting(tmp_path):
    d, _ = make_run(tmp_path)
    lines = (d / "tasks.jsonl").read_text().splitlines()
    (d / "tasks.jsonl").write_text("\n".join(lines[1:]) + "\n")
    r = by_name(check_run(d))
    assert not r["task_accounting"]["ok"]


def test_shortened_output_detected(tmp_path):
    d, _ = make_run(tmp_path)
    recs = [json.loads(line) for line in (d / "requests.jsonl").read_text().splitlines()]
    recs[3]["generated_tokens"] -= 1
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert not by_name(check_run(d))["output_lengths"]["ok"]


def test_inflight_and_slot_overlap_detected(tmp_path):
    d, _ = make_run(tmp_path)
    recs = [json.loads(line) for line in (d / "requests.jsonl").read_text().splitlines()]
    for r in recs[:3]:  # three overlapping requests on slot 0
        r["dispatch_t"], r["end_t"], r["slot"] = 0.0, 1.0, 0
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    r = by_name(check_run(d))
    assert not r["inflight_le_k"]["ok"] and not r["slot_exclusive"]["ok"]


def test_server_deferral_detected(tmp_path):
    d, _ = make_run(tmp_path)
    with open(d / "system.jsonl", "a") as f:
        f.write(json.dumps({"t": 0, "metrics": {"llamacpp:requests_deferred": 2.0}}) + "\n")
    assert not by_name(check_run(d))["server_never_deferred"]["ok"]


def test_silent_purge_is_warning_not_failure(tmp_path):
    d, _ = make_run(tmp_path)
    recs = [json.loads(line) for line in (d / "requests.jsonl").read_text().splitlines()]
    r1 = next(r for r in recs if r["cache_expected"] > 0)
    r1["cached_tokens"] = 0
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = check_run(d)
    assert not by_name(res)["no_silent_kv_purge"]["ok"] and not failed(res)


def test_failed_task_still_accounted(tmp_path):
    d, _ = make_run(tmp_path, client=FakeClient(pool=400, fail_on={2: "ctx_overflow"}))
    res = check_run(d)
    assert not failed(res), [c for c in res if not c["ok"]]
    assert '"ctx_overflow": 1' in by_name(res)["failures_by_type"]["detail"]


def test_incomplete_run_fails(tmp_path):
    d, _ = make_run(tmp_path)
    (d / "DONE").unlink()
    assert not by_name(check_run(d))["run_complete"]["ok"]


def test_parse_prometheus():
    text = ("# HELP llamacpp:prompt_tokens_total x\n# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 1234\nllamacpp:requests_deferred 0\nfoo{a=\"b\"} 3\n")
    assert parse_prometheus(text) == {"llamacpp:prompt_tokens_total": 1234.0, "llamacpp:requests_deferred": 0.0}
