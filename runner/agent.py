"""Agent coroutine and workload drivers.

One task = one agent run (CLAUDE.md s6.3), following the semantics in runner/workload_gen.py:
    prompt(step j) = shared_prefix + init_ctx + sum_{i<j} (response_ids_i + tool_out_i)
After each step except the last: sleep tool_delay_s, then append the tool output.

A task stops at its first failed request and takes that request's status (no retries). Task statuses:
ok | timeout | http_error | ctx_overflow | kv_exhausted | truncated | server_crash | runner_error.
A task whose KV reservation exceeds the whole pool is rejected at admission and recorded as ctx_overflow.
`timeout` at task level means the per-task deadline passed (checked before each step and tool call).

Drivers:
    run_closed(tasks, n_agents, ...)  N agents, each takes the next task as soon as its previous one ends.
                                      arrival_t = the moment the agent picks the task up.
    run_open(tasks, ...)              task i arrives at t0 + arrival_s (Poisson schedule from the generator).
"""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Callable

from runner.scheduler import AdmissionRejected, Scheduler

LogFn = Callable[[dict], None]


async def run_task(task: dict, *, sched: Scheduler, corpus, prefix_ids: list[int], arrival_t: float,
                   deadline_s: float, log_request: LogFn) -> dict:
    rec = {
        "task_id": task["task_id"], "class": task["class"], "arrival_t": arrival_t,
        "admit_t": None, "completion_t": None, "steps_planned": len(task["steps"]), "steps_done": 0,
        "admit_wait_s": 0.0, "total_queue_s": 0.0, "total_tool_s": 0.0, "total_model_s": 0.0,
        "prompt_len_mismatches": 0, "slot_reserved": None, "status": "ok", "failed_step": None, "error": None,
    }
    admitted = False
    try:
        slot, wait = await sched.admit(task)
        admitted = True
        rec.update(admit_t=time.monotonic(), admit_wait_s=wait, total_queue_s=wait, slot_reserved=slot)
        ctx = prefix_ids + corpus.slice(task["init_ctx"])
        for j, step in enumerate(task["steps"]):
            if time.monotonic() - arrival_t > deadline_s:
                rec.update(status="timeout", failed_step=j, error=f"task deadline {deadline_s}s passed before step {j}")
                break
            expected = task["prompt_tokens_per_step"][j]
            if len(ctx) != expected:
                rec["prompt_len_mismatches"] += 1
            res, req = await sched.submit(task["task_id"], j, ctx, step["out"])
            queue_s = res.dispatch_t - req.enqueue_t
            rec["total_queue_s"] += queue_s
            rec["total_model_s"] += res.end_t - res.dispatch_t
            log_request({"task_id": task["task_id"], "step": j, "slot": req.slot, "enqueue_t": req.enqueue_t,
                         "queue_s": queue_s, "prompt_tokens_expected": expected,
                         "cache_expected": req.cache_expected, **res.log_record()})
            if res.status != "ok":
                rec.update(status=res.status, failed_step=j, error=res.error)
                break
            rec["steps_done"] += 1
            ctx = ctx + res.generated_ids
            if step["tool_out"] is not None:
                if time.monotonic() - arrival_t > deadline_s:
                    rec.update(status="timeout", failed_step=j + 1, error=f"task deadline {deadline_s}s passed")
                    break
                t0 = time.monotonic()
                await asyncio.sleep(step["tool_delay_s"])
                rec["total_tool_s"] += time.monotonic() - t0
                ctx = ctx + corpus.slice(step["tool_out"])
    except AdmissionRejected as e:  # declared peak context exceeds the whole KV pool: can never run
        rec.update(status="ctx_overflow", failed_step=0, error=f"admission rejected: {e}")
    except Exception as e:  # noqa: BLE001 -- recorded, never swallowed silently
        rec.update(status="runner_error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    finally:
        if admitted:
            sched.task_done(task["task_id"])
        rec["completion_t"] = time.monotonic()
    return rec


async def run_closed(tasks: list[dict], n_agents: int, *, log_task: LogFn, **kw) -> list[dict]:
    queue = list(tasks)
    queue.reverse()
    out: list[dict] = []

    async def agent_loop() -> None:
        while queue:
            task = queue.pop()
            rec = await run_task(task, arrival_t=time.monotonic(), **kw)
            log_task(rec)
            out.append(rec)

    await asyncio.gather(*[agent_loop() for _ in range(n_agents)])
    return out


async def run_open(tasks: list[dict], *, log_task: LogFn, t0: float | None = None, **kw) -> list[dict]:
    t0 = time.monotonic() if t0 is None else t0
    out: list[dict] = []

    async def one(task: dict) -> None:
        arrival = t0 + task["arrival_s"]
        while (left := arrival - time.monotonic()) > 0:  # Windows timers can wake a few ms early
            await asyncio.sleep(left)
        rec = await run_task(task, arrival_t=arrival, **kw)
        log_task(rec)
        out.append(rec)

    await asyncio.gather(*[one(t) for t in tasks])
    return out
