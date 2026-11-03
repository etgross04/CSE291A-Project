"""Async llama-server client: one streaming /completion call -> one fully classified CompletionResult.

Behaviour verified on b11721 (scripts/probe_client_fields.py, notebook 2026-11-02):
  * `return_tokens: true` makes every SSE chunk carry `tokens` (generated ids). The final chunk has
    `stop: true`, `timings`, `truncated`, `stop_type`, `tokens_cached`, `id_slot`.
  * Prompt longer than the slot's n_ctx -> HTTP 400, error.type == "exceed_context_size_error".
  * Prompt fits but prompt + n_predict does not -> HTTP 200, `truncated: true`, fewer tokens than asked.
    That is a SILENT failure unless checked here.

  * Unified KV pool full while every slot is busy -> HTTP 500 "Context size has been exceeded." for ALL
    in-flight requests (scripts/probe_kvu.py) -> `kv_exhausted`.

Status values: ok | timeout | http_error | ctx_overflow | kv_exhausted | truncated | server_crash.
No retries, ever.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field

import httpx


@dataclass
class CompletionResult:
    status: str
    n_prompt: int                 # tokens we sent
    n_predict: int                # tokens we asked for
    id_slot_requested: int | None
    id_slot: int | None = None    # slot the server actually used
    dispatch_t: float = 0.0       # monotonic, request sent
    first_token_t: float | None = None
    end_t: float = 0.0
    generated_ids: list[int] = field(default_factory=list, repr=False)
    generated_tokens: int = 0
    tokens_processed: int | None = None   # timings.prompt_n (prompt tokens actually computed)
    cached_tokens: int | None = None      # timings.cache_n (prompt tokens reused from the slot's KV)
    tokens_cached_after: int | None = None  # tokens in the slot's KV after this request
    prompt_ms: float | None = None
    gen_ms: float | None = None
    truncated: bool | None = None
    stop_type: str | None = None
    http_status: int | None = None
    error: str | None = None

    def log_record(self) -> dict:
        d = asdict(self)
        d.pop("generated_ids")
        return d


class LlamaClient:
    def __init__(self, base_url: str, request_timeout_s: float = 300.0, max_connections: int = 64):
        self.request_timeout_s = request_timeout_s
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10.0, read=request_timeout_s, write=30.0, pool=None),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, prompt_ids: list[int], n_predict: int, id_slot: int | None = None,
                       cache_prompt: bool = True) -> CompletionResult:
        res = CompletionResult(status="ok", n_prompt=len(prompt_ids), n_predict=n_predict,
                               id_slot_requested=id_slot)
        body = {"prompt": prompt_ids, "n_predict": n_predict, "temperature": 0, "ignore_eos": True,
                "cache_prompt": cache_prompt, "stream": True, "return_tokens": True}
        if id_slot is not None:
            body["id_slot"] = id_slot
        res.dispatch_t = time.monotonic()
        try:
            await asyncio.wait_for(self._stream(body, res), timeout=self.request_timeout_s)
        except asyncio.TimeoutError:
            # Closing the stream disconnects; llama-server cancels a task whose client went away.
            res.status, res.error = "timeout", f"no completion within {self.request_timeout_s}s"
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            res.status, res.error = "server_crash", f"{type(e).__name__}: {e}"
        except httpx.HTTPError as e:
            res.status, res.error = "http_error", f"{type(e).__name__}: {e}"
        res.end_t = time.monotonic()
        res.generated_tokens = len(res.generated_ids)
        if res.status == "ok":
            res.status = self._classify_finished(res)
        return res

    async def _stream(self, body: dict, res: CompletionResult) -> None:
        async with self._http.stream("POST", "/completion", json=body) as r:
            res.http_status = r.status_code
            if r.status_code != 200:
                raw = (await r.aread()).decode("utf-8", "replace")
                res.status, res.error = self._classify_http_error(r.status_code, raw), raw[:500]
                return
            final = None
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:  # error delivered inside the stream
                    res.status = LlamaClient._classify_error_obj(chunk["error"])
                    res.error = json.dumps(chunk["error"])[:500]
                    return
                toks = chunk.get("tokens") or []
                if toks and res.first_token_t is None:
                    res.first_token_t = time.monotonic()
                res.generated_ids.extend(toks)
                if chunk.get("stop"):
                    final = chunk
                    break
            if final is None:
                res.status, res.error = "server_crash", "stream ended without a stop chunk"
                return
            t = final.get("timings") or {}
            res.id_slot = final.get("id_slot")
            res.tokens_processed = t.get("prompt_n")
            res.cached_tokens = t.get("cache_n")
            res.prompt_ms = t.get("prompt_ms")
            res.gen_ms = t.get("predicted_ms")
            res.tokens_cached_after = final.get("tokens_cached")
            res.truncated = final.get("truncated")
            res.stop_type = final.get("stop_type")

    @staticmethod
    def _classify_http_error(code: int, raw: str) -> str:
        try:
            err = json.loads(raw).get("error", {})
        except (json.JSONDecodeError, AttributeError):
            err = {}
        return LlamaClient._classify_error_obj(err)

    @staticmethod
    def _classify_error_obj(err) -> str:
        if not isinstance(err, dict):
            return "http_error"
        if err.get("type") == "exceed_context_size_error":
            return "ctx_overflow"
        if "Context size has been exceeded" in str(err.get("message", "")):
            return "kv_exhausted"  # unified pool ran out of cells mid-batch
        return "http_error"

    async def erase_slot(self, slot_id: int) -> None:
        """Free a slot's KV cells (needs --slot-save-path). Raises on failure."""
        r = await self._http.post(f"/slots/{slot_id}?action=erase")
        if r.status_code != 200:
            raise RuntimeError(f"erase slot {slot_id}: HTTP {r.status_code} {r.text[:200]}")

    @staticmethod
    def _classify_finished(res: CompletionResult) -> str:
        if res.generated_tokens < res.n_predict:
            # Server stopped early: `truncated` means it ran out of slot context.
            return "ctx_overflow" if res.truncated else "truncated"
        if res.generated_tokens > res.n_predict:
            res.error = f"got {res.generated_tokens} tokens, asked {res.n_predict}"
            return "http_error"
        return "ok"


async def _smoke() -> None:
    """python -m runner.client  -- live check against a fresh server (4096-token slots)."""
    from pathlib import Path

    from runner.corpus import Corpus
    from runner.server import Server, load_local, load_server_config

    local = load_local()
    corpus = Corpus(local)
    cfg = load_server_config({"ctx_size": 8192, "parallel": 2, "kv_unified": False})
    with Server(local, cfg, Path(local["results_dir"]) / "client_smoke" / "server.log") as s:
        c = LlamaClient(s.base_url, request_timeout_s=60)
        keys = ("status", "id_slot", "n_prompt", "tokens_processed", "cached_tokens", "generated_tokens")
        p = corpus.slice({"offset": 300_000, "len": 1500})
        cold = await c.complete(p, 64, id_slot=0)
        print("cold slot0     ", {k: cold.log_record()[k] for k in keys})
        # agent-style step 2: previous prompt + response ids + tool output -> should reuse ~1564 tokens
        nxt = p + cold.generated_ids + corpus.slice({"offset": 400_000, "len": 200})
        warm = await c.complete(nxt, 64, id_slot=0)
        print("warm append    ", {k: warm.log_record()[k] for k in keys})
        conc = await asyncio.gather(*[c.complete(corpus.slice({"offset": 500_000 + i * 5000, "len": 800}), 128)
                                      for i in range(4)])
        print("4 concurrent on 2 slots", [(r.status, r.id_slot, round(r.end_t - r.dispatch_t, 2)) for r in conc])
        big = await c.complete(corpus.slice({"offset": 600_000, "len": 5000}), 16, id_slot=1)
        print("prompt > n_ctx ", big.status, big.http_status)
        trunc = await c.complete(corpus.slice({"offset": 600_000, "len": 4000}), 256, id_slot=1)
        print("prompt+n_predict > n_ctx", trunc.status, trunc.generated_tokens, trunc.truncated)
        await c.aclose()


if __name__ == "__main__":
    asyncio.run(_smoke())
