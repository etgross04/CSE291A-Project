"""Scheduler: admission queue (tasks) + ready queue (model requests) over K server slots.

The scheduler owns every placement decision. It always sends an explicit `id_slot`, and never has more
than one in-flight request per slot. So in-flight <= K, and the server never queues anything internally
(b11721 would silently queue extra requests, hiding queueing time inside "model time").

Dispatch is event-driven: whenever state changes (task arrives, request is submitted, request finishes,
task finishes) `_pump()` admits waiting tasks and fills free slots. This behaves like K worker coroutines,
one per slot, without the extra moving parts.

Policy interface (Phase 2 plugs in here; the runner and logging do not change):
    pick_admission(waiting, sched) -> Admission | None   which waiting task to start now (None = none)
    assign_slot(task, sched)       -> int | None         slot the task owns for its lifetime (None = no
                                                         reservation; its requests can use any slot)
    select_request(slot, eligible, sched) -> Request | None   which ready request runs on this free slot
    on_request_done(req, result, sched), on_task_done(task_id, sched)   bookkeeping hooks

Baseline: FifoPolicy("pinned") reserves a slot per task (KV retained across tool delays, so at most K
tasks are active). FifoPolicy("any") reserves nothing (requests take the lowest free slot, so whatever
KV is in the slot from another task gets overwritten).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from runner.client import CompletionResult


@dataclass(eq=False)
class Request:
    task_id: str
    step: int
    prompt_ids: list[int]
    n_predict: int
    pinned_slot: int | None
    enqueue_t: float
    seq: int
    future: asyncio.Future = field(repr=False)
    slot: int | None = None


@dataclass(eq=False)
class Admission:
    task: dict
    enqueue_t: float
    seq: int
    future: asyncio.Future = field(repr=False)


@dataclass
class SlotState:
    id: int
    busy: bool = False
    owner: str | None = None       # task holding a reservation (pinned)
    last_task: str | None = None   # whose context is in this slot's KV
    kv_tokens: int = 0             # runner-tracked tokens in this slot's KV (from tokens_cached)


class Policy:
    name = "base"

    def pick_admission(self, waiting: list[Admission], sched: "Scheduler") -> Admission | None:
        raise NotImplementedError

    def assign_slot(self, task: dict, sched: "Scheduler") -> int | None:
        return None

    def select_request(self, slot: SlotState, eligible: list[Request], sched: "Scheduler") -> Request | None:
        raise NotImplementedError

    def on_request_done(self, req: Request, res: CompletionResult, sched: "Scheduler") -> None:
        pass

    def on_task_done(self, task_id: str, sched: "Scheduler") -> None:
        pass

    def describe(self) -> dict:
        return {"name": self.name}


class FifoPolicy(Policy):
    """Phase 1 baseline: first come, first served at both admission and dispatch."""

    def __init__(self, slot_mode: str = "pinned"):
        if slot_mode not in ("pinned", "any"):
            raise ValueError(f"slot_mode must be 'pinned' or 'any', not {slot_mode!r}")
        self.slot_mode = slot_mode
        self.name = f"fifo-{slot_mode}"

    def pick_admission(self, waiting, sched):
        head = waiting[0]
        if self.slot_mode == "any":
            return head
        return head if any(s.owner is None for s in sched.slots) else None

    def assign_slot(self, task, sched):
        if self.slot_mode == "any":
            return None
        return next(s.id for s in sched.slots if s.owner is None)

    def select_request(self, slot, eligible, sched):
        return eligible[0]  # eligible preserves enqueue order

    def describe(self):
        return {"name": self.name, "slot_mode": self.slot_mode}


class Scheduler:
    def __init__(self, client, k: int, policy: Policy, cache_prompt: bool = True):
        self.client = client
        self.k = k
        self.policy = policy
        self.cache_prompt = cache_prompt
        self.slots = [SlotState(i) for i in range(k)]
        self.ready: list[Request] = []
        self.waiting: list[Admission] = []
        self.task_slot: dict[str, int | None] = {}   # admitted, unfinished tasks -> reserved slot
        self._seq = 0
        self._inflight: set[asyncio.Task] = set()
        self.policy_overhead_s = 0.0  # wall time spent inside _pump (all policy decisions)
        self.max_inflight = 0

    # ---- task lifecycle (called by agents) ----
    async def admit(self, task: dict) -> tuple[int | None, float]:
        """Wait until the policy admits this task. Returns (reserved slot or None, admission wait seconds)."""
        adm = Admission(task, time.monotonic(), self._next_seq(), asyncio.get_running_loop().create_future())
        self.waiting.append(adm)
        self._pump()
        slot = await adm.future
        return slot, time.monotonic() - adm.enqueue_t

    def task_done(self, task_id: str) -> None:
        slot = self.task_slot.pop(task_id, None)
        if slot is not None:
            self.slots[slot].owner = None
        self.policy.on_task_done(task_id, self)
        self._pump()

    async def submit(self, task_id: str, step: int, prompt_ids: list[int], n_predict: int
                     ) -> tuple[CompletionResult, Request]:
        if task_id not in self.task_slot:
            raise RuntimeError(f"{task_id} submitted a request without being admitted")
        req = Request(task_id, step, prompt_ids, n_predict, self.task_slot[task_id], time.monotonic(),
                      self._next_seq(), asyncio.get_running_loop().create_future())
        self.ready.append(req)
        self._pump()
        res = await req.future
        return res, req

    # ---- core ----
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _pump(self) -> None:
        t0 = time.perf_counter()
        while self.waiting:
            adm = self.policy.pick_admission(self.waiting, self)
            if adm is None:
                break
            self.waiting.remove(adm)
            slot = self.policy.assign_slot(adm.task, self)
            if slot is not None:
                if self.slots[slot].owner is not None:
                    raise RuntimeError(f"policy assigned slot {slot} already owned by {self.slots[slot].owner}")
                self.slots[slot].owner = adm.task["task_id"]
            self.task_slot[adm.task["task_id"]] = slot
            adm.future.set_result(slot)
        for s in self.slots:
            if s.busy or not self.ready:
                continue
            eligible = [r for r in self.ready if r.pinned_slot is None or r.pinned_slot == s.id]
            if not eligible:
                continue
            req = self.policy.select_request(s, eligible, self)
            if req is None:
                continue
            self.ready.remove(req)
            s.busy, req.slot = True, s.id
            t = asyncio.create_task(self._run(req, s))
            self._inflight.add(t)
            t.add_done_callback(self._inflight.discard)
        self.max_inflight = max(self.max_inflight, sum(s.busy for s in self.slots))
        self.policy_overhead_s += time.perf_counter() - t0

    async def _run(self, req: Request, slot: SlotState) -> None:
        try:
            res = await self.client.complete(req.prompt_ids, req.n_predict, id_slot=slot.id,
                                             cache_prompt=self.cache_prompt)
        except Exception as e:  # never lose a request silently: the agent gets the exception
            slot.busy = False
            req.future.set_exception(e)
            self._pump()
            return
        slot.busy = False
        slot.last_task = req.task_id
        if res.tokens_cached_after is not None:
            slot.kv_tokens = res.tokens_cached_after
        self.policy.on_request_done(req, res, self)
        req.future.set_result(res)
        self._pump()

    # ---- observation (samplers) ----
    def snapshot(self) -> dict:
        return {
            "ready": len(self.ready),
            "waiting": len(self.waiting),
            "active_tasks": len(self.task_slot),
            "busy_slots": sum(s.busy for s in self.slots),
            "slot_kv_tokens": [s.kv_tokens for s in self.slots],
            "slot_owner": [s.owner for s in self.slots],
            "slot_busy": [s.busy for s in self.slots],
        }

    async def drain(self) -> None:
        while self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)
