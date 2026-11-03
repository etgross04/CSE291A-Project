"""Scheduler: admission queue (tasks) + ready queue (model requests) over K server slots and a KV pool.

The scheduler owns every placement decision. It always sends an explicit `id_slot`, and never has more
than one in-flight request per slot. So in-flight <= K, and the server never queues anything internally
(b11721 would silently queue extra requests, hiding queueing time inside "model time").

Memory model (baseline, TA decision 2026-11-02, "option B"): the server runs with one unified KV pool
(`-kvu -c POOL`; every slot may use up to POOL tokens). A task draws its reservation from the pool when it
is admitted and returns it when it finishes; admission waits until the reservation fits. With
`kv_budget=None` there is no pool accounting (static per-slot split, `-no-kvu`).

Why the scheduler erases slots (scripts/probe_kvu.py, notebook 2026-11-02):
  * idle slots keep their KV in the pool; when a request needs cells that are not free, the server
    silently PURGES an idle slot (possibly a pinned agent waiting on a tool) -> hidden recompute;
  * when every slot is busy and the pool overflows, ALL in-flight requests fail (HTTP 500).
Invariant: every task's context lives in at most one slot, and a finished task's context is gone before
its tokens return to the pool. Then KV in use <= sum of active reservations <= pool and the server never
needs to purge. How it is kept:
  * task finishes -> its idle slots are erased (`/slots/{id}?action=erase`), then the reservation is
    released; a slot of it already handed to another request is released once that request starts
    (starting a request overwrites the slot);
  * `any` mode, request placed on a slot other than the one holding the task's older copy -> that copy
    is erased first (idle slot) or the request waits until the slot's next request starts (slot handed
    to another request but not started yet). Waits only point at earlier placements, so no cycles;
  * placement prefers the free slot that already holds the task's context (affinity, like the server's
    own slot-prompt-similarity choice when id_slot is omitted).
Each request logs `cache_expected` so analysis can detect any purge that still happens.

Dispatch is event-driven: whenever state changes (task arrives, request is submitted, request finishes,
task finishes, erase completes) `_pump()` admits waiting tasks and places ready requests on free slots.

Policy interface (Phase 2 plugs in here; the runner and logging do not change):
    reserve_tokens(task, sched)    -> int                pool tokens a task draws at admission
    pick_admission(waiting, sched) -> Admission | None   which waiting task to start now (None = none)
    assign_slot(task, sched)       -> int | None         slot the task owns for its lifetime (None = no
                                                         reservation; its requests can use any slot)
    order_ready(ready, sched)      -> list[Request]      order in which ready requests are placed
    place_request(req, free, sched) -> SlotState | None  free slot for this request (None = wait)
    on_request_done(req, result, sched), on_task_done(task_id, sched)   bookkeeping hooks

Baseline: FifoPolicy("pinned") reserves a slot per task (KV retained across tool delays, so at most K
tasks are active). FifoPolicy("any") reserves no slot (a request takes the free slot holding its own
context if there is one, else the lowest free slot, overwriting whatever KV another task left there).
Both reserve the task's declared peak context from the pool, admit strictly in arrival order (a head that
does not fit blocks the tasks behind it), and dispatch ready requests in FIFO order.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from runner.client import CompletionResult


class AdmissionRejected(Exception):
    """The task can never be admitted (its reservation exceeds the whole pool)."""


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
    cache_expected: int = 0   # prompt tokens the runner expects the slot to have cached at dispatch


@dataclass(eq=False)
class Admission:
    task: dict
    need: int                 # pool tokens to reserve
    enqueue_t: float
    seq: int
    future: asyncio.Future = field(repr=False)


@dataclass
class SlotState:
    id: int
    busy: bool = False             # running a request or being erased
    erasing: bool = False
    owner: str | None = None       # task holding a reservation (pinned)
    last_task: str | None = None   # whose context is in this slot's KV
    kv_tokens: int = 0             # runner-tracked tokens in this slot's KV (from tokens_cached)
    started: asyncio.Event | None = field(default=None, repr=False)  # set when the placed request starts


class Policy:
    name = "base"

    def reserve_tokens(self, task: dict, sched: "Scheduler") -> int:
        return int(task["peak_ctx_tokens"])  # declared peak from the task file

    def pick_admission(self, waiting: list[Admission], sched: "Scheduler") -> Admission | None:
        raise NotImplementedError

    def assign_slot(self, task: dict, sched: "Scheduler") -> int | None:
        return None

    def order_ready(self, ready: list[Request], sched: "Scheduler") -> list[Request]:
        return ready  # FIFO (ready is kept in enqueue order)

    def place_request(self, req: Request, free: list[SlotState], sched: "Scheduler") -> SlotState | None:
        if req.pinned_slot is not None:
            return next((s for s in free if s.id == req.pinned_slot), None)
        own = next((s for s in free if s.last_task == req.task_id), None)
        return own or free[0]

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
        if not sched.kv_fits(head.need):
            return None
        if self.slot_mode == "any":
            return head
        return head if any(s.owner is None for s in sched.slots) else None

    def assign_slot(self, task, sched):
        if self.slot_mode == "any":
            return None
        return next(s.id for s in sched.slots if s.owner is None)

    def describe(self):
        return {"name": self.name, "slot_mode": self.slot_mode, "reservation": "peak_ctx_tokens",
                "placement": "pinned" if self.slot_mode == "pinned" else "own-slot-else-lowest-free"}


class Scheduler:
    def __init__(self, client, k: int, policy: Policy, kv_budget: int | None = None,
                 cache_prompt: bool = True, erase_stale: bool | None = None):
        self.client = client
        self.k = k
        self.policy = policy
        self.kv_budget = kv_budget
        self.cache_prompt = cache_prompt
        self.erase_stale = (kv_budget is not None) if erase_stale is None else erase_stale
        self.slots = [SlotState(i) for i in range(k)]
        self.ready: list[Request] = []
        self.waiting: list[Admission] = []
        self.task_slot: dict[str, int | None] = {}   # admitted, unfinished tasks -> reserved slot
        self.task_res: dict[str, int] = {}           # tasks holding pool tokens (incl. finishing ones)
        self.kv_reserved = 0
        self.max_kv_reserved = 0
        self._seq = 0
        self._inflight: set[asyncio.Task] = set()
        self.policy_overhead_s = 0.0  # wall time spent inside _pump (all policy decisions)
        self.max_inflight = 0
        self.erase_count = 0
        self.erase_s = 0.0
        self.erase_failures: list[str] = []

    # ---- pool ----
    def kv_fits(self, need: int) -> bool:
        return self.kv_budget is None or self.kv_reserved + need <= self.kv_budget

    def _release(self, task_id: str) -> None:
        self.kv_reserved -= self.task_res.pop(task_id, 0)

    # ---- task lifecycle (called by agents) ----
    async def admit(self, task: dict) -> tuple[int | None, float]:
        """Wait until the policy admits this task. Returns (reserved slot or None, admission wait seconds)."""
        need = int(self.policy.reserve_tokens(task, self))
        if self.kv_budget is not None and need > self.kv_budget:
            raise AdmissionRejected(f"reservation {need} tokens > KV pool {self.kv_budget}")
        adm = Admission(task, need, time.monotonic(), self._next_seq(), asyncio.get_running_loop().create_future())
        self.waiting.append(adm)
        self._pump()
        slot = await adm.future
        return slot, time.monotonic() - adm.enqueue_t

    def task_done(self, task_id: str) -> None:
        slot = self.task_slot.pop(task_id, None)
        if slot is not None:
            self.slots[slot].owner = None
        self.policy.on_task_done(task_id, self)
        if self.erase_stale:
            idle, wait = self._copies_of(task_id, exclude=None)
            if idle or wait:  # release the reservation only after the KV is really gone
                self._mark_erasing(idle)
                self._spawn(self._clear_then(idle, wait, lambda: self._release(task_id)))
                self._pump()
                return
        self._release(task_id)
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

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._inflight.add(t)
        t.add_done_callback(self._inflight.discard)

    def _pump(self) -> None:
        t0 = time.perf_counter()
        while self.waiting:
            adm = self.policy.pick_admission(self.waiting, self)
            if adm is None:
                break
            if not self.kv_fits(adm.need):
                raise RuntimeError(f"policy admitted {adm.task['task_id']} beyond the KV pool")
            self.waiting.remove(adm)
            slot = self.policy.assign_slot(adm.task, self)
            if slot is not None:
                if self.slots[slot].owner is not None:
                    raise RuntimeError(f"policy assigned slot {slot} already owned by {self.slots[slot].owner}")
                self.slots[slot].owner = adm.task["task_id"]
            tid = adm.task["task_id"]
            self.task_slot[tid] = slot
            self.task_res[tid] = adm.need
            self.kv_reserved += adm.need
            self.max_kv_reserved = max(self.max_kv_reserved, self.kv_reserved)
            adm.future.set_result(slot)
        for req in list(self.policy.order_ready(self.ready, self)):
            free = [s for s in self.slots if not s.busy]
            if not free:
                break
            s = self.policy.place_request(req, free, self)
            if s is None:
                continue
            if s.busy:
                raise RuntimeError(f"policy placed {req.task_id} on busy slot {s.id}")
            self.ready.remove(req)
            s.busy, req.slot = True, s.id
            req.cache_expected = min(s.kv_tokens, len(req.prompt_ids) - 1) \
                if self.cache_prompt and s.last_task == req.task_id else 0
            idle, wait = self._copies_of(req.task_id, exclude=s) if self.erase_stale else ([], [])
            self._mark_erasing(idle)
            s.started = asyncio.Event()
            self._spawn(self._run(req, s, idle, wait))
        self.max_inflight = max(self.max_inflight, sum(s.busy and not s.erasing for s in self.slots))
        self.policy_overhead_s += time.perf_counter() - t0

    def _copies_of(self, task_id: str, exclude: SlotState | None
                   ) -> tuple[list[SlotState], list[asyncio.Event]]:
        """Other slots still holding this task's context: idle ones (to erase) and ones already placed
        for another request that has not started yet (wait for it; starting overwrites the slot)."""
        idle, wait = [], []
        for o in self.slots:
            if o is exclude or o.last_task != task_id:
                continue
            if not o.busy:
                idle.append(o)
            elif not o.erasing and o.started is not None and not o.started.is_set():
                wait.append(o.started)
            # busy and started: that request is overwriting it; erasing: already being cleared
        return idle, wait

    @staticmethod
    def _mark_erasing(slots: list[SlotState]) -> None:
        for o in slots:
            o.busy = o.erasing = True

    async def _erase(self, slots: list[SlotState]) -> None:
        for o in slots:
            t0 = time.monotonic()
            try:
                await self.client.erase_slot(o.id)
                o.kv_tokens, o.last_task = 0, None
            except Exception as e:  # recorded; the slot keeps its stale KV and the server may purge it
                self.erase_failures.append(f"slot {o.id}: {type(e).__name__}: {e}")
            self.erase_count += 1
            self.erase_s += time.monotonic() - t0
            o.busy = o.erasing = False

    async def _clear(self, idle: list[SlotState], wait: list[asyncio.Event]) -> None:
        if idle:
            await self._erase(idle)
        for ev in wait:
            await ev.wait()

    async def _clear_then(self, idle: list[SlotState], wait: list[asyncio.Event], then) -> None:
        await self._clear(idle, wait)
        then()
        self._pump()

    async def _run(self, req: Request, slot: SlotState, idle: list[SlotState], wait: list[asyncio.Event]
                   ) -> None:
        try:
            await self._clear(idle, wait)
            # the slot's old content counts as overwritten from here on (the request replaces it)
            slot.last_task, slot.kv_tokens = req.task_id, len(req.prompt_ids)
            slot.started.set()
            res = await self.client.complete(req.prompt_ids, req.n_predict, id_slot=slot.id,
                                             cache_prompt=self.cache_prompt)
        except Exception as e:  # never lose a request silently: the agent gets the exception
            slot.busy = False
            if slot.started is not None:
                slot.started.set()
            req.future.set_exception(e)
            self._pump()
            return
        slot.busy = False
        slot.last_task = req.task_id
        slot.kv_tokens = res.tokens_cached_after if res.tokens_cached_after is not None else 0
        self.policy.on_request_done(req, res, self)
        req.future.set_result(res)
        self._pump()

    # ---- observation (samplers) ----
    def snapshot(self) -> dict:
        return {
            "ready": len(self.ready),
            "waiting": len(self.waiting),
            "active_tasks": len(self.task_slot),
            "busy_slots": sum(s.busy and not s.erasing for s in self.slots),
            "kv_budget": self.kv_budget,
            "kv_reserved": self.kv_reserved,
            "kv_used_tracked": sum(s.kv_tokens for s in self.slots),
            "slot_kv_tokens": [s.kv_tokens for s in self.slots],
            "slot_owner": [s.owner for s in self.slots],
            "slot_busy": [s.busy for s in self.slots],
            "erase_count": self.erase_count,
        }

    def stats(self) -> dict:
        return {"policy": self.policy.describe(), "k": self.k, "kv_budget": self.kv_budget,
                "max_inflight": self.max_inflight, "max_kv_reserved": self.max_kv_reserved,
                "policy_overhead_s": self.policy_overhead_s, "erase_count": self.erase_count,
                "erase_s": self.erase_s, "erase_failures": self.erase_failures}

    async def drain(self) -> None:
        while self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)
