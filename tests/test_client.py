"""Client parsing and failure classification against a fake llama-server (no GPU).
Response shapes copied from real b11721 responses (scripts/probe_client_fields.py)."""
from __future__ import annotations

import asyncio
import json

import httpx

from runner.client import LlamaClient


def sse(chunks: list[dict]) -> bytes:
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)


def final_chunk(n_pred: int, truncated: bool = False) -> dict:
    return {"content": "", "tokens": [], "stop": True, "id_slot": 1, "tokens_predicted": n_pred,
            "tokens_evaluated": 100, "truncated": truncated, "stop_type": "limit", "tokens_cached": 100 + n_pred - 1,
            "timings": {"cache_n": 40, "prompt_n": 60, "prompt_ms": 12.5, "predicted_n": n_pred,
                        "predicted_ms": 80.0}}


def stream_of(n_tokens: int, truncated: bool = False) -> bytes:
    toks = [{"content": "x", "tokens": [1000 + i], "stop": False, "id_slot": -1} for i in range(n_tokens)]
    return sse(toks + [final_chunk(n_tokens, truncated)])


def run(handler, n_predict=8, id_slot=None, timeout=5.0):
    async def go():
        c = LlamaClient("http://fake", request_timeout_s=timeout)
        c._http = httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(handler))
        try:
            return await c.complete(list(range(100)), n_predict, id_slot=id_slot)
        finally:
            await c.aclose()
    return asyncio.run(go())


def test_ok_parses_timings_and_ids():
    seen = {}

    def h(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=stream_of(8))
    r = run(h, id_slot=3)
    assert r.status == "ok" and r.generated_tokens == 8 and r.generated_ids == [1000 + i for i in range(8)]
    assert (r.tokens_processed, r.cached_tokens, r.prompt_ms, r.gen_ms, r.id_slot) == (60, 40, 12.5, 80.0, 1)
    assert r.first_token_t is not None and r.dispatch_t <= r.first_token_t <= r.end_t
    # request body: the fixed per-request settings from CLAUDE.md s6.3
    assert seen["temperature"] == 0 and seen["ignore_eos"] is True and seen["stream"] is True
    assert seen["cache_prompt"] is True and seen["return_tokens"] is True and seen["id_slot"] == 3
    assert seen["n_predict"] == 8 and seen["prompt"] == list(range(100))


def test_no_id_slot_when_any():
    def h(req):
        assert "id_slot" not in json.loads(req.content)
        return httpx.Response(200, content=stream_of(8))
    assert run(h).status == "ok"


def test_silent_truncation_is_ctx_overflow():
    r = run(lambda req: httpx.Response(200, content=stream_of(5, truncated=True)))
    assert r.status == "ctx_overflow" and r.generated_tokens == 5


def test_short_without_truncated_flag_is_truncated():
    r = run(lambda req: httpx.Response(200, content=stream_of(5)))
    assert r.status == "truncated"


def test_prompt_too_long_400_is_ctx_overflow():
    body = {"error": {"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 5000, "n_ctx": 4096,
                      "message": "request (5000 tokens) exceeds the available context size (4096 tokens)"}}
    r = run(lambda req: httpx.Response(400, json=body))
    assert r.status == "ctx_overflow" and r.http_status == 400


def test_other_http_error():
    r = run(lambda req: httpx.Response(503, json={"error": {"code": 503, "type": "unavailable_error"}}))
    assert r.status == "http_error" and r.http_status == 503


def test_connection_refused_is_server_crash():
    def h(req):
        raise httpx.ConnectError("refused")
    assert run(h).status == "server_crash"


def test_stream_without_stop_chunk_is_server_crash():
    toks = [{"content": "x", "tokens": [1], "stop": False} for _ in range(3)]
    r = run(lambda req: httpx.Response(200, content=sse(toks)))
    assert r.status == "server_crash"


def test_timeout():
    async def slow_stream():
        yield b'data: {"content":"x","tokens":[1],"stop":false}\n\n'
        await asyncio.sleep(5)
        yield sse([final_chunk(8)])

    r = run(lambda req: httpx.Response(200, content=slow_stream()), timeout=0.3)
    assert r.status == "timeout" and r.end_t - r.dispatch_t < 2
