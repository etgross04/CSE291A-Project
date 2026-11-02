"""Workload generator: determinism and structural invariants. No server or corpus cache needed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from runner.corpus import PREFIX_REGION
from runner.workload_gen import MAX_MODEL_CTX, dumps_tasks, generate, write

REPO = Path(__file__).resolve().parents[1]
WL_DIR = REPO / "configs" / "workloads"
WORKLOADS = sorted(p for p in WL_DIR.glob("*.yaml") if p.name != "seeds.yaml")
N_CORPUS = 1_500_000  # stand-in corpus size; generation depends only on this count


def load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


@pytest.mark.parametrize("path", WORKLOADS, ids=lambda p: p.stem)
def test_byte_identical_files(path, tmp_path):
    a = write(path, None, tmp_path / "a", N_CORPUS, "x")
    b = write(path, None, tmp_path / "b", N_CORPUS, "x")
    assert a.read_bytes() == b.read_bytes()
    assert a.with_suffix(".meta.json").read_bytes() == b.with_suffix(".meta.json").read_bytes()
    assert b"\r\n" not in a.read_bytes()  # LF only, so Windows and Linux files match


def test_golden_hash():
    """Pins the exact output for one workload. If this changes, the generator changed: bump
    GENERATOR_VERSION and regenerate all task files rather than updating this hash casually."""
    body = dumps_tasks(generate(load(WL_DIR / "pilot.yaml"), 42, N_CORPUS))
    assert hashlib.sha256(body).hexdigest() == GOLDEN_PILOT_SHA256


def test_seed_changes_output():
    wl = load(WL_DIR / "agentic.yaml")
    assert dumps_tasks(generate(wl, 1000, N_CORPUS)) != dumps_tasks(generate(wl, 1001, N_CORPUS))


def test_task_i_independent_of_num_tasks():
    wl = load(WL_DIR / "blend.yaml")
    small = generate({**wl, "num_tasks": 10}, 1000, N_CORPUS)
    big = generate({**wl, "num_tasks": 80}, 1000, N_CORPUS)
    assert small == big[:10]


@pytest.mark.parametrize("path", WORKLOADS, ids=lambda p: p.stem)
def test_task_invariants(path):
    wl = load(path)
    prefix = wl.get("shared_prefix_tokens", 0)
    tasks = generate(wl, wl["seed"], N_CORPUS)
    assert len(tasks) == wl["num_tasks"]
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    for t in tasks:
        steps = t["steps"]
        assert steps[-1]["tool_out"] is None and steps[-1]["tool_delay_s"] == 0
        # prompt length bookkeeping matches the agent's construction rule
        ctx = prefix + t["init_ctx"]["len"]
        for s, plen in zip(steps, t["prompt_tokens_per_step"]):
            assert plen == ctx
            ctx += s["out"] + (s["tool_out"]["len"] if s["tool_out"] else 0)
        assert t["peak_ctx_tokens"] == t["prompt_tokens_per_step"][-1] + steps[-1]["out"]
        assert t["peak_ctx_tokens"] <= MAX_MODEL_CTX
        for ref in [t["init_ctx"]] + [s["tool_out"] for s in steps if s["tool_out"]]:
            assert PREFIX_REGION <= ref["offset"] and ref["offset"] + ref["len"] <= N_CORPUS
        assert all(s["tool_delay_s"] >= 0 for s in steps)


def test_poisson_arrivals_sorted_and_rate():
    wl = {**load(WL_DIR / "chat.yaml"), "num_tasks": 2000, "arrival": {"type": "poisson", "rate_per_s": 2.0}}
    arr = [t["arrival_s"] for t in generate(wl, 1000, N_CORPUS)]
    assert arr == sorted(arr) and arr[0] == 0.0
    rate = (len(arr) - 1) / arr[-1]
    assert 1.8 < rate < 2.2


def test_class_weights_respected():
    wl = {**load(WL_DIR / "blend.yaml"), "num_tasks": 3000}
    tasks = generate(wl, 1000, N_CORPUS)
    frac = {c: sum(t["class"] == c for t in tasks) / len(tasks) for c in ("chat", "long_output", "long_context", "agentic")}
    assert abs(frac["chat"] - 0.4) < 0.04 and abs(frac["long_output"] - 0.1) < 0.03
    assert abs(frac["long_context"] - 0.2) < 0.04 and abs(frac["agentic"] - 0.3) < 0.04


def test_stress_is_fixed_size():
    tasks = generate(load(WL_DIR / "stress.yaml"), 1000, N_CORPUS)
    assert {t["peak_ctx_tokens"] for t in tasks} == {512 + 4096 + 8 * 256 + 7 * 1024}


# Same value under Python 3.12.10 and 3.14.3 on Windows (2026-11-02).
GOLDEN_PILOT_SHA256 = "6b677fd0ec5f185d9055da0ca4f13fc3532da389cf5a5452ad2d882c9e9accd5"
