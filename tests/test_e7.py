"""E7 measurement loops against the fake client (pool bounds, slot use, summary)."""
from __future__ import annotations

import asyncio

from experiments.e7_idle_kv import BATCH_SIZES, IDLE_LEVELS, idle_split, part_a, part_b, summarize
from test_scheduler import FakeClient


class TimedFake(FakeClient):
    """FakeClient + server-style timings; decode gets slower with KV held in other slots."""

    async def complete(self, prompt_ids, n_predict, id_slot=None, cache_prompt=True):
        res = await super().complete(prompt_ids, n_predict, id_slot, cache_prompt)
        others = sum(v for s, v in self.occ.items() if s != id_slot)
        res.prompt_ms = 0.2 * (res.tokens_processed or 0)
        res.gen_ms = (8.0 + others / 10_000) * res.generated_tokens
        return res


class FakeCorpus:
    def slice(self, ref):
        return list(range(ref["offset"], ref["offset"] + ref["len"]))


def test_idle_split_fits_static_slots():
    for level in IDLE_LEVELS:
        per = idle_split(level)
        assert sum(per) == level and max(per) <= 16000


def test_part_a_rows_and_pool_bound():
    c = TimedFake(latency=0.0, pool=65536)
    rows = []
    out = asyncio.run(part_a(c, FakeCorpus(), 2, rows.append, kv_unified=True))
    assert len(out) == len(IDLE_LEVELS) * 2 and all(r["status"] == "ok" for r in out)
    assert c.max_occ <= 65536
    dec = {r["idle_tokens"]: r["decode_ms_per_tok"] for r in out}
    assert dec[48000] > dec[0]                      # the fake's slowdown shows up in the measurement
    assert all(r["cached_tokens"] == 0 for r in out)  # active prompt is cold every time


def test_part_b_batches_are_decode_only():
    c = TimedFake(latency=0.002, pool=65536)
    out = asyncio.run(part_b(c, FakeCorpus(), 1, lambda r: None))
    assert [r["batch"] for r in out] == BATCH_SIZES
    assert all(r["cached_tokens_min"] == 2000 - 1 for r in out)  # whole prompt cached before the timed phase
    s = summarize(out)
    assert len(s) == len(BATCH_SIZES) and all(x["all_ok"] and x["aggregate_tok_per_s"] > 0 for x in s)
