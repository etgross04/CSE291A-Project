"""Fixed text corpus -> exact-length token slices.

The corpus (scripts/fetch_corpus.py) is tokenized ONCE with the pinned server's tokenizer and cached as
int32 tokens. Workloads refer to slices by (offset, len) into this token array, so prompt lengths are exact
by construction and the workload generator never needs a running server.

Build the cache (starts its own server):  python -m runner.corpus build
Inspect:                                   python -m runner.corpus info
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from runner.server import REPO, Server, load_local, load_server_config

# The first PREFIX_REGION tokens are reserved for shared prefixes. Task slices are drawn from after it,
# so a task's private context never accidentally matches the shared prefix.
PREFIX_REGION = 8192
CHUNK_CHARS = 200_000  # /tokenize request size; chunks split on paragraph boundaries


def _paths(local: dict) -> tuple[Path, Path, Path]:
    d = Path(local.get("corpus_dir", "C:/llm/corpus"))
    return d / "corpus.txt", d / "corpus_tokens.npy", d / "corpus_tokens.meta.json"


def _chunks(text: str):
    i = 0
    while i < len(text):
        j = min(i + CHUNK_CHARS, len(text))
        if j < len(text):
            k = text.rfind("\n\n", i, j)
            j = k + 2 if k > i else j
        yield text[i:j]
        i = j


def build(local: dict) -> dict:
    import httpx

    txt_path, npy_path, meta_path = _paths(local)
    raw = txt_path.read_bytes()
    manifest = json.loads((REPO / "configs" / "corpus_manifest.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest["corpus_sha256"]:
        raise SystemExit("corpus.txt does not match configs/corpus_manifest.json; run scripts/fetch_corpus.py --verify")
    text = raw.decode("utf-8")
    cfg = load_server_config({"ctx_size": 4096, "parallel": 1})
    out_dir = Path(local["results_dir"]) / "corpus_build"
    toks: list[int] = []
    with Server(local, cfg, out_dir / "server.log") as s, httpx.Client(base_url=s.base_url, timeout=600) as c:
        build_info = c.get("/props").json().get("build_info")
        for chunk in _chunks(text):
            r = c.post("/tokenize", json={"content": chunk})  # no add_special: raw text tokens only
            r.raise_for_status()
            toks.extend(r.json()["tokens"])
    arr = np.asarray(toks, dtype=np.int32)
    np.save(npy_path, arr)
    meta = {
        "n_tokens": int(arr.size),
        "tokens_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "corpus_sha256": manifest["corpus_sha256"],
        "model": local["model"],
        "llama_build": build_info,
        "chunk_chars": CHUNK_CHARS,
        "prefix_region": PREFIX_REGION,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class Corpus:
    def __init__(self, local: dict | None = None):
        local = local or load_local()
        _, npy_path, meta_path = _paths(local)
        if not npy_path.exists():
            raise SystemExit(f"{npy_path} missing; run `python -m runner.corpus build`")
        self.meta = json.loads(meta_path.read_text())
        self.tokens = np.load(npy_path)
        if hashlib.sha256(self.tokens.tobytes()).hexdigest() != self.meta["tokens_sha256"]:
            raise SystemExit("corpus token cache hash mismatch; rebuild it")

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.size)

    def slice(self, ref: dict) -> list[int]:
        """ref = {"offset": int, "len": int} as written by the workload generator."""
        o, n = ref["offset"], ref["len"]
        if o < 0 or o + n > self.tokens.size:
            raise ValueError(f"slice {ref} outside corpus of {self.tokens.size} tokens")
        return self.tokens[o:o + n].tolist()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    local = load_local()
    if cmd == "build":
        print(json.dumps(build(local), indent=2))
    else:
        c = Corpus(local)
        print(json.dumps(c.meta, indent=2))
