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

    def __init__(self, latency=0.01, fail_on=None, jitter_seed=0, pool=None):
        self.latency, self.fail_on = latency, fail_on or {}
        self.inflight, self.max_inflight = 0, 0
        self.slot_busy: set[int] = set()
        self.calls: list[dict] = []
        self.rng = random.Random(jitter_seed)
        # unified-pool model: a slot's cells stay occupied until overwritten by a new prompt or erased
        self.pool, self.occ, self.max_occ, self.erased = pool, {}, 0, []
        self.content: dict[int, list[int]] = {}  # tokens in each slot's KV (for cache_n, like the server)

    async def erase_slot(self, slot_id):
        assert slot_id not in self.slot_busy, "erase of a busy slot"
        await asyncio.sleep(0.001)
        self.occ[slot_id] = 0
        self.content[slot_id] = []
        self.erased.append(slot_id)

    async def complete(self, prompt_ids, n_predict, id_slot=None, cache_prompt=True):
        assert id_slot is not None, "scheduler must always choose the slot"
        assert id_slot not in self.slot_busy, f"slot {id_slot} double-booked"
        self.slot_busy.add(id_slot)
        self.occ[id_slot] = len(prompt_ids) + n_predict  # worst case during this request
        self.max_occ = max(self.max_occ, sum(self.occ.values()))
        if self.pool is not None:
            assert sum(self.occ.values()) <= self.pool, f"KV pool overcommitted: {self.occ}"
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.calls.append({"slot": id_slot, "n_prompt": len(prompt_ids), "prompt": list(prompt_ids),
                           "t": time.monotonic()})
        call_no = len(self.calls)
        res = CompletionResult(status="ok", n_prompt=len(prompt_ids), n_predict=n_predict,
                               id_slot_requested=id_slot, id_slot=id_slot, dispatch_t=time.monotonic())
        old = self.content.get(id_slot, [])
        lcp = 0
        while lcp < min(len(old), len(prompt_ids) - 1) and old[lcp] == prompt_ids[lcp]:
            lcp += 1
        res.cached_tokens, res.tokens_processed = lcp, len(prompt_ids) - lcp
        await asyncio.sleep(self.latency * (0.5 + self.rng.random()))
        self.inflight -= 1
        self.slot_busy.discard(id_slot)
        status = self.fail_on.get(call_no)
        res.generated_ids = [] if status else [7] * n_predict
        res.generated_tokens = len(res.generated_ids)
        res.tokens_cached_after = len(prompt_ids) + n_predict - 1
        self.content[id_slot] = list(prompt_ids) + [7] * (n_predict - 1)
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
            "prompt_tokens_per_step": plen, "peak_ctx_tokens": plen[-1] + out}


def run(tasks, k, mode, n_agents, client=None, deadline_s=60.0, open_loop=False, kv_budget=None):
    client = client or FakeClient(pool=kv_budget)
    reqs, recs = [], []

    async def go():
        sched = Scheduler(client, k, FifoPolicy(mode), kv_budget=kv_budget)
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


# ---- shared KV pool (option B) ----

@pytest.mark.parametrize("mode", ["pinned", "any"])
def test_pool_never_overcommitted_and_all_tasks_finish(mode):
    # peak per task = 4+20+3*5+2*8 = 55; pool 120 -> at most 2 tasks hold reservations at once
    tasks = [make_task(i, delay=0.02) for i in range(10)]
    client, sched, reqs, recs = run(tasks, k=4, mode=mode, n_agents=10, kv_budget=120)
    assert all(r["status"] == "ok" for r in recs) and len(recs) == 10
    assert sched.max_kv_reserved <= 120
    assert client.max_occ <= 120          # physical (modelled) occupancy, including idle retained KV
    assert sched.kv_reserved == 0 and not sched.task_res
    assert sched.erase_failures == []


def test_pool_limits_active_tasks_below_k():
    tasks = [make_task(i, delay=0.03) for i in range(6)]
    _, _, _, recs = run(tasks, k=4, mode="any", n_agents=6, kv_budget=120)
    events = sorted([(r["admit_t"], 1) for r in recs] + [(r["completion_t"], -1) for r in recs])
    active = peak = 0
    for _, d in events:
        active += d
        peak = max(peak, active)
    assert peak <= 2  # 120 // 55 = 2, even though there are 4 slots


class SpreadPolicy(FifoPolicy):
    """Any-mode FIFO that places requests round-robin over slots (spreads copies of contexts around)."""

    def __init__(self):
        super().__init__("any")
        self.next = 0

    def place_request(self, req, free, sched):
        ids = [s.id for s in free]
        for i in range(sched.k):
            cand = (self.next + i) % sched.k
            if cand in ids:
                self.next = cand + 1
                return next(s for s in free if s.id == cand)
        return None


@pytest.mark.parametrize("erase", [False, True])
def test_erase_keeps_pool_bound_for_any_placement(erase):
    tasks = [make_task(i, delay=0.02) for i in range(10)]
    client = FakeClient(pool=None)

    async def go():
        sched = Scheduler(client, 4, SpreadPolicy(), kv_budget=120, erase_stale=erase)
        await run_closed(tasks, 10, sched=sched, corpus=FakeCorpus(), prefix_ids=[1, 2, 3, 4], deadline_s=60,
                         log_request=lambda r: None, log_task=lambda r: None)
        await sched.drain()

    asyncio.run(go())
    if erase:
        assert client.max_occ <= 120
    else:
        assert client.max_occ > 120 and client.erased == []  # stale copies pile up without erase


def test_finished_task_slots_erased_before_release():
    tasks = [make_task(i) for i in range(3)]
    client, sched, _, _ = run(tasks, k=1, mode="pinned", n_agents=3, kv_budget=1000)
    assert client.erased == [0, 0, 0] and sched.erase_count == 3


def test_any_mode_erases_stale_copy_of_same_task():
    tasks = [make_task(i, delay=0.03) for i in range(6)]
    client, _, reqs, _ = run(tasks, k=2, mode="any", n_agents=6, kv_budget=10_000)
    moved = any(len({r["slot"] for r in reqs if r["task_id"] == t}) > 1 for t in {r["task_id"] for r in reqs})
    assert moved and client.max_occ <= 10_000


def test_oversized_task_rejected_as_ctx_overflow():
    tasks = [make_task(0), make_task(1, init=500)]  # task 1 peak 535 > pool 200
    _, _, reqs, recs = run(tasks, k=2, mode="pinned", n_agents=2, kv_budget=200)
    by = {r["task_id"]: r for r in recs}
    assert by["t000"]["status"] == "ok"
    assert by["t001"]["status"] == "ctx_overflow" and "admission rejected" in by["t001"]["error"]
    assert not [r for r in reqs if r["task_id"] == "t001"]


def test_fifo_head_of_line_blocking():
    """A big head waits for pool space and the small task behind it is not admitted first."""
    big = make_task(0, init=100, delay=0.0)    # peak 135
    small = make_task(1, delay=0.0)            # peak 55
    first = make_task(2, steps=1, init=60)     # peak 69, holds the pool while big waits
    client, _, _, recs = run([first, big, small], k=3, mode="any", n_agents=3, kv_budget=150)
    by = {r["task_id"]: r for r in recs}
    assert by["t000"]["admit_t"] <= by["t001"]["admit_t"]


def test_cache_expected_logged_for_pinned_followups():
    t = make_task(0)
    _, _, reqs, _ = run([t], k=1, mode="pinned", n_agents=1, kv_budget=1000)
    assert reqs[0]["cache_expected"] == 0
    # FakeClient reports prompt + n - 1 cached after each request
    assert reqs[1]["cache_expected"] == reqs[0]["n_prompt"] + 5 - 1
