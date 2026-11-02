"""Scheduler + agent against a fake client (no GPU): concurrency bounds, FIFO order, slot modes,
prompt construction, failure handling, task accounting."""
from __future__ import annotations

import asyncio
import random
import time

import pytest

from runner.agent import run_closed, run_open
from runner.client import CompletionResult
from runner.scheduler import FifoPolicy, Scheduler


class FakeClient:
    """Mimics LlamaClient.complete. Records concurrency and per-slot overlap."""

    def __init__(self, latency=0.01, fail_on=None, jitter_seed=0):
        self.latency, self.fail_on = latency, fail_on or {}
        self.inflight, self.max_inflight = 0, 0
        self.slot_busy: set[int] = set()
        self.calls: list[dict] = []
        self.rng = random.Random(jitter_seed)

    async def complete(self, prompt_ids, n_predict, id_slot=None, cache_prompt=True):
        assert id_slot is not None, "scheduler must always choose the slot"
        assert id_slot not in self.slot_busy, f"slot {id_slot} double-booked"
        self.slot_busy.add(id_slot)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.calls.append({"slot": id_slot, "n_prompt": len(prompt_ids), "prompt": list(prompt_ids),
                           "t": time.monotonic()})
        res = CompletionResult(status="ok", n_prompt=len(prompt_ids), n_predict=n_predict,
                               id_slot_requested=id_slot, id_slot=id_slot, dispatch_t=time.monotonic())
        await asyncio.sleep(self.latency * (0.5 + self.rng.random()))
        self.inflight -= 1
        self.slot_busy.discard(id_slot)
        status = self.fail_on.get(len(self.calls))
        res.generated_ids = [] if status else [7] * n_predict
        res.generated_tokens = len(res.generated_ids)
        res.tokens_cached_after = len(prompt_ids) + n_predict - 1
        res.status = status or "ok"
        res.end_t = time.monotonic()
        return res


class FakeCorpus:
    def slice(self, ref):  # distinct marker tokens so prompt construction can be checked
        return [ref["offset"]] * ref["len"]


def make_task(i, steps=3, init=20, out=5, tool=8, delay=0.01):
    prefix = 4
    st, plen, ctx = [], [], prefix + init
    for j in range(steps):
        last = j == steps - 1
        st.append({"step": j, "out": out, "tool_out": None if last else {"offset": 1000 + i * 10 + j, "len": tool},
                   "tool_delay_s": 0.0 if last else delay})
        plen.append(ctx)
        ctx += out + (0 if last else tool)
    return {"task_id": f"t{i:03d}", "class": "x", "init_ctx": {"offset": 500 + i, "len": init}, "steps": st,
            "prompt_tokens_per_step": plen}


def run(tasks, k, mode, n_agents, client=None, deadline_s=60.0, open_loop=False):
    client = client or FakeClient()
    reqs, recs = [], []

    async def go():
        sched = Scheduler(client, k, FifoPolicy(mode))
        kw = dict(sched=sched, corpus=FakeCorpus(), prefix_ids=[1, 2, 3, 4], deadline_s=deadline_s,
                  log_request=reqs.append, log_task=recs.append)
        if open_loop:
            await run_open(tasks, **kw)
        else:
            await run_closed(tasks, n_agents, **kw)
        await sched.drain()
        return sched

    sched = asyncio.run(go())
    return client, sched, reqs, recs


@pytest.mark.parametrize("mode", ["pinned", "any"])
def test_inflight_bounded_and_all_tasks_accounted(mode):
    tasks = [make_task(i) for i in range(12)]
    client, sched, reqs, recs = run(tasks, k=3, mode=mode, n_agents=6)
    assert client.max_inflight <= 3 and sched.max_inflight <= 3
    assert sorted(r["task_id"] for r in recs) == [t["task_id"] for t in tasks]
    assert all(r["status"] == "ok" and r["steps_done"] == 3 for r in recs)
    assert len(reqs) == 36
    assert all(r["prompt_len_mismatches"] == 0 for r in recs)


def test_prompt_construction_matches_spec():
    t = make_task(0)
    client, _, reqs, _ = run([t], k=1, mode="pinned", n_agents=1)
    p0, p1, p2 = (c["prompt"] for c in client.calls)
    init = [500] * 20
    assert p0 == [1, 2, 3, 4] + init
    assert p1 == p0 + [7] * 5 + [1000] * 8
    assert p2 == p1 + [7] * 5 + [1001] * 8
    assert [r["n_prompt"] for r in reqs] == t["prompt_tokens_per_step"]


def test_pinned_task_keeps_one_slot_and_at_most_k_active():
    tasks = [make_task(i, delay=0.03) for i in range(8)]
    client, _, reqs, recs = run(tasks, k=2, mode="pinned", n_agents=8)
    by_task = {}
    for r in reqs:
        by_task.setdefault(r["task_id"], set()).add(r["slot"])
    assert all(len(s) == 1 for s in by_task.values())
    # at most 2 tasks between admission and completion at any instant
    events = sorted([(r["admit_t"], 1) for r in recs] + [(r["completion_t"], -1) for r in recs])
    active = peak = 0
    for _, d in events:
        active += d
        peak = max(peak, active)
    assert peak <= 2
    assert any(r["admit_wait_s"] > 0.01 for r in recs)  # later tasks waited for a slot


def test_any_mode_interleaves_more_tasks_than_slots():
    tasks = [make_task(i, delay=0.05) for i in range(6)]
    _, _, reqs, recs = run(tasks, k=2, mode="any", n_agents=6)
    assert all(r["admit_wait_s"] < 0.01 for r in recs)  # no reservation, admitted immediately
    events = sorted([(r["admit_t"], 1) for r in recs] + [(r["completion_t"], -1) for r in recs])
    active = peak = 0
    for _, d in events:
        active += d
        peak = max(peak, active)
    assert peak > 2
    assert any(len({r["slot"] for r in reqs if r["task_id"] == t}) > 1 for t in {r["task_id"] for r in reqs})


def test_fifo_dispatch_order_with_one_slot():
    tasks = [make_task(i, steps=1) for i in range(5)]
    client, _, reqs, _ = run(tasks, k=1, mode="any", n_agents=5)
    order = [r["task_id"] for r in sorted(reqs, key=lambda r: r["dispatch_t"])]
    assert order == [t["task_id"] for t in tasks]
    assert all(r["queue_s"] >= 0 for r in reqs)


def test_failed_request_stops_task_and_frees_slot():
    client = FakeClient(fail_on={2: "ctx_overflow"})  # the 2nd request overall fails
    tasks = [make_task(i) for i in range(3)]
    _, _, reqs, recs = run(tasks, k=1, mode="pinned", n_agents=3, client=client)
    failed = [r for r in recs if r["status"] != "ok"]
    assert len(failed) == 1 and failed[0]["status"] == "ctx_overflow" and failed[0]["failed_step"] == 1
    assert sum(r["status"] == "ok" for r in recs) == 2  # the slot was released; the others still ran
    assert len(recs) == 3


def test_task_deadline():
    tasks = [make_task(0, steps=4, delay=0.1)]
    _, _, reqs, recs = run(tasks, k=1, mode="pinned", n_agents=1, deadline_s=0.15)
    assert recs[0]["status"] == "timeout" and recs[0]["steps_done"] < 4


def test_open_loop_respects_arrivals():
    tasks = [make_task(i, steps=1) for i in range(3)]
    for i, t in enumerate(tasks):
        t["arrival_s"] = 0.05 * i
    _, _, _, recs = run(tasks, k=2, mode="any", n_agents=0, open_loop=True)
    arr = sorted(r["arrival_t"] for r in recs)
    assert abs((arr[1] - arr[0]) - 0.05) < 1e-6 and abs((arr[2] - arr[0]) - 0.10) < 1e-6
    by_id = {r["task_id"]: r for r in recs}
    assert all(by_id[t["task_id"]]["admit_t"] >= by_id[t["task_id"]]["arrival_t"] - 1e-3 for t in tasks)


def test_policy_overhead_measured():
    _, sched, _, _ = run([make_task(i) for i in range(4)], k=2, mode="any", n_agents=4)
    assert 0 < sched.policy_overhead_s < 0.5
