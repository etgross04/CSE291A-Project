"""Deterministic workload generation: workload YAML + seed -> tasks.jsonl (+ .meta.json).

Every task is fully determined in advance: class, step count, per-step output length, per-step tool-output
slice and tool delay, initial-context slice, and arrival time (open loop). Text is referenced as
{"offset", "len"} into the cached corpus token array (runner/corpus.py), so no server is needed here.

Determinism rules (see tests/test_workload_gen.py):
  * Only random.Random.random() is used (Mersenne Twister; its float stream is stable across Python
    versions and OSes). Distributions are implemented here from uniform floats.
  * Task i uses its own RNG seeded from (seed, i): adding tasks never changes earlier ones.
  * Floats are rounded to ms; JSON is written with sorted keys, fixed separators and "\n" line endings.

Task semantics (what runner/agent.py must implement):
  prompt(step j) = shared_prefix + init_ctx + sum_{i<j} (response_i + tool_out_i)
  After step j (except the last) the agent sleeps tool_delay_s, then appends tool_out. The last step has
  tool_out = null and tool_delay_s = 0.

Usage:
  python -m runner.workload_gen configs/workloads/agentic.yaml [--seed 1000] [--out DIR]
  python -m runner.workload_gen --all [--set dev|eval|all]   # every workload x seed set
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import yaml

from runner.corpus import PREFIX_REGION

GENERATOR_VERSION = 1
REPO = Path(__file__).resolve().parents[1]
MAX_MODEL_CTX = 40960  # Qwen3-8B n_ctx_train
# Exponential/lognormal draws use a separate stream for arrivals so class mixes don't shift arrival times.
ARRIVAL_STREAM = 0x5EED


# ---------- distributions (from uniform floats only) ----------

def _uniform_int(rng: random.Random, low: int, high: int) -> int:
    return low + min(int(rng.random() * (high - low + 1)), high - low)


def _normal(rng: random.Random) -> float:
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def sample(spec, rng: random.Random):
    """spec: a number (constant), null, or {dist: uniform_int|choice|lognormal|exponential|const, ...}."""
    if spec is None or isinstance(spec, (int, float)):
        return spec
    d = spec["dist"]
    if d == "const":
        return spec["value"]
    if d == "uniform_int":
        return _uniform_int(rng, int(spec["low"]), int(spec["high"]))
    if d == "choice":
        vals = spec["values"]
        return vals[_uniform_int(rng, 0, len(vals) - 1)]
    if d == "lognormal":  # parameterized by median and sigma (of the underlying normal), optional clip
        x = spec["median_s"] * math.exp(spec["sigma"] * _normal(rng))
        return min(max(x, spec.get("min_s", 0.0)), spec.get("max_s", math.inf))
    if d == "exponential":
        return -math.log(1.0 - rng.random()) * spec["mean_s"]
    raise ValueError(f"unknown dist {d!r}")


def _weighted_choice(rng: random.Random, classes: list[dict]) -> dict:
    total = sum(c["weight"] for c in classes)
    u = rng.random() * total
    acc = 0.0
    for c in classes:
        acc += c["weight"]
        if u < acc:
            return c
    return classes[-1]


# ---------- generation ----------

def _slice_ref(rng: random.Random, length: int, n_corpus_tokens: int) -> dict:
    hi = n_corpus_tokens - length
    if hi <= PREFIX_REGION:
        raise ValueError(f"slice of {length} tokens does not fit in corpus of {n_corpus_tokens}")
    return {"offset": _uniform_int(rng, PREFIX_REGION, hi), "len": int(length)}


def make_task(wl: dict, seed: int, index: int, n_corpus_tokens: int) -> dict:
    rng = random.Random(seed * 1_000_003 + index)
    cls = _weighted_choice(rng, wl["task_classes"])
    n_steps = int(sample(cls["steps"], rng))
    init_len = int(sample(cls["init_ctx"], rng))
    steps = []
    for j in range(n_steps):
        last = j == n_steps - 1
        out = int(sample(cls["out"], rng))
        if last or cls.get("tool_out") is None:
            tool_out, delay = None, 0.0
        else:
            tool_out = _slice_ref(rng, int(sample(cls["tool_out"], rng)), n_corpus_tokens)
            delay = round(float(sample(cls.get("tool_delay"), rng) or 0.0), 3)
        steps.append({"step": j, "out": out, "tool_out": tool_out, "tool_delay_s": delay})

    prefix = int(wl.get("shared_prefix_tokens", 0))
    prompt_lens, ctx = [], prefix + init_len
    for s in steps:
        prompt_lens.append(ctx)
        ctx += s["out"] + (s["tool_out"]["len"] if s["tool_out"] else 0)
    peak = prompt_lens[-1] + steps[-1]["out"]  # KV held at end of the final generation
    return {
        "task_id": f"{wl['name']}-s{seed}-{index:04d}",
        "index": index,
        "class": cls["name"],
        "arrival_s": None,  # filled in for open-loop workloads
        "init_ctx": _slice_ref(rng, init_len, n_corpus_tokens),
        "steps": steps,
        "prompt_tokens_per_step": prompt_lens,
        "peak_ctx_tokens": peak,
        "total_out_tokens": sum(s["out"] for s in steps),
        "total_tool_delay_s": round(sum(s["tool_delay_s"] for s in steps), 3),
    }


def arrival_times(arrival: dict, seed: int, n: int) -> list[float | None]:
    if arrival["type"] == "closed":
        return [None] * n
    if arrival["type"] == "poisson":
        rng = random.Random(seed * 1_000_003 + ARRIVAL_STREAM)
        t, out = 0.0, []
        for _ in range(n):
            out.append(round(t, 3))
            t += -math.log(1.0 - rng.random()) / arrival["rate_per_s"]
        return out
    raise ValueError(f"unknown arrival type {arrival['type']!r}")


def generate(wl: dict, seed: int, n_corpus_tokens: int) -> list[dict]:
    tasks = [make_task(wl, seed, i, n_corpus_tokens) for i in range(int(wl["num_tasks"]))]
    for t, a in zip(tasks, arrival_times(wl["arrival"], seed, len(tasks))):
        t["arrival_s"] = a
    for t in tasks:
        if t["peak_ctx_tokens"] > MAX_MODEL_CTX:
            raise ValueError(f"{t['task_id']} peak ctx {t['peak_ctx_tokens']} > model limit {MAX_MODEL_CTX}")
    return tasks


def dumps_tasks(tasks: list[dict]) -> bytes:
    return "".join(json.dumps(t, sort_keys=True, separators=(",", ":")) + "\n" for t in tasks).encode("utf-8")


def write(yaml_path: Path, seed: int | None, out_dir: Path, n_corpus_tokens: int, corpus_tokens_sha: str) -> Path:
    raw_yaml = Path(yaml_path).read_bytes()
    wl = yaml.safe_load(raw_yaml)
    seed = wl["seed"] if seed is None else seed
    tasks = generate(wl, seed, n_corpus_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{wl['name']}_s{seed}.jsonl"
    body = dumps_tasks(tasks)
    path.write_bytes(body)
    meta = {
        "workload": wl["name"], "seed": seed, "num_tasks": len(tasks), "arrival": wl["arrival"],
        "shared_prefix": {"offset": 0, "len": int(wl.get("shared_prefix_tokens", 0))},
        "yaml_path": str(Path(yaml_path).as_posix()), "yaml_sha256": hashlib.sha256(raw_yaml).hexdigest(),
        "tasks_sha256": hashlib.sha256(body).hexdigest(), "generator_version": GENERATOR_VERSION,
        "corpus_n_tokens": n_corpus_tokens, "corpus_tokens_sha256": corpus_tokens_sha,
        "summary": summarize(tasks),
    }
    (out_dir / f"{wl['name']}_s{seed}.meta.json").write_bytes(
        (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return path


def summarize(tasks: list[dict]) -> dict:
    by_cls: dict[str, int] = {}
    for t in tasks:
        by_cls[t["class"]] = by_cls.get(t["class"], 0) + 1
    peaks = sorted(t["peak_ctx_tokens"] for t in tasks)
    return {
        "tasks_by_class": dict(sorted(by_cls.items())),
        "requests": sum(len(t["steps"]) for t in tasks),
        "out_tokens": sum(t["total_out_tokens"] for t in tasks),
        "peak_ctx_max": peaks[-1],
        "peak_ctx_p50": peaks[len(peaks) // 2],
        "tool_delay_s_total": round(sum(t["total_tool_delay_s"] for t in tasks), 3),
    }


def main() -> None:
    from runner.corpus import Corpus

    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", nargs="?")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true", help="generate every workload for the chosen seed set")
    ap.add_argument("--set", default="all", choices=["dev", "eval", "all"])
    ap.add_argument("--out", default=None, help="default: <results_dir>/workloads")
    a = ap.parse_args()

    corpus = Corpus()
    from runner.server import load_local
    out_dir = Path(a.out or Path(load_local()["results_dir"]) / "workloads")
    n, sha = corpus.n_tokens, corpus.meta["tokens_sha256"]

    if a.all:
        seeds = yaml.safe_load((REPO / "configs" / "workloads" / "seeds.yaml").read_text())
        seed_list = seeds["dev"] + seeds["eval"] if a.set == "all" else seeds[a.set]
        for y in sorted((REPO / "configs" / "workloads").glob("*.yaml")):
            if y.name == "seeds.yaml":
                continue
            wl_seeds = [None] if yaml.safe_load(y.read_text()).get("fixed_seed") else seed_list
            for s in wl_seeds:
                p = write(y, s, out_dir, n, sha)
                print(p.name, json.loads(p.with_suffix(".meta.json").read_text())["summary"])
    else:
        p = write(Path(a.yaml), a.seed, out_dir, n, sha)
        print(p, json.dumps(json.loads(p.with_suffix(".meta.json").read_text())["summary"], indent=2))


if __name__ == "__main__":
    main()
