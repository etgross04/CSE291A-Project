"""Phase 1 figures (CLAUDE.md s9), rebuilt from raw run logs every time (via analysis.metrics).

    E1  e1_calibration       prefill tokens/s and decode ms/token vs prompt length (K=1)
    E2  e2_throughput        tasks/min and p95 completion time vs K, one panel per workload (own y-scale)
        e2_scaling           tasks/min relative to K=1, all workloads on one axis
        e2_breakdown         mean time per task: admission wait / ready queue / model / tool, vs K
        e2_decode            median decode ms/token vs K (contention)
        timeline_<run>       KV reserved vs used, VRAM, busy slots + waiting tasks over one run
    E3  e3_steps             per-step request time, recomputed prompt tokens, decode ms/token
    E4  e4_idle              pinned vs any across tool delays: tasks/min, cache-hit fraction
    E5  e5_reuse             shared prefix x cache_prompt x erase: tasks/min, prompt tokens computed
    E6  e6_budget            failure rate and tasks/min vs pool size, admission gate on/off
    E7  e7_idle_kv           decode/prefill cost vs idle KV (shared vs split); batch decode scaling

Error bars are min..max over valid reps; points are medians. Invalid runs are never plotted (they are
listed by analysis.metrics in exclusions.csv). An experiment with no finished runs is skipped.

Usage: python -m analysis.plots [E1 E2 ...] [--out DIR] [--timeline RUN_DIR ...]
Writes PNG (200 dpi) and PDF to <repo>/results/figures/.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checks import read_jsonl  # noqa: E402
from analysis.metrics import REPO, aggregate, collect, experiment_run_dirs, resolve_experiment  # noqa: E402

# Reference categorical palette (dataviz skill, light mode), used in fixed order. An entity keeps its colour
# in every figure: workloads by WORKLOAD_ORDER, two-way comparisons take slots 1 and 2.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
WORKLOAD_ORDER = ["chat", "long_context", "long_output", "agentic", "blend", "trace_agentic"]
DEFAULT_TIMELINES = ["trace_agentic_s1000/fifo-pinned_K16_c65536_kvu/rep0",
                     "agentic_s1000/fifo-pinned_K16_c65536_kvu/rep0"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
        "font.family": ["DejaVu Sans"], "figure.dpi": 100,
    })


def base_name(stem: str) -> str:
    return re.sub(r"_s\d+$", "", stem)


def wl_color(name: str) -> str:
    n = base_name(name)
    return SLOTS[WORKLOAD_ORDER.index(n)] if n in WORKLOAD_ORDER else INK2


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.png/.pdf")
    return out / f"{name}.png"


def errline(ax, xs, rows, key, color, label=None, marker="o", ls="-"):
    """Median line with min..max bars over reps."""
    ys = [r.get(key) for r in rows]
    pts = [(x, y, r.get(key + "_min"), r.get(key + "_max")) for x, y, r in zip(xs, ys, rows) if y is not None]
    if not pts:
        return
    x, y, lo, hi = zip(*pts)
    ax.errorbar(x, y, yerr=[[a - (b if b is not None else a) for a, b in zip(y, lo)],
                            [(c if c is not None else a) - a for a, c in zip(y, hi)]],
                color=color, marker=marker, ls=ls, capsize=0, elinewidth=1.2, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1)


def k_axis(ax, ks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()


def experiment(name: str, results_dir: Path):
    """(per-run metric rows, valid only; per-config summary). ([], []) if nothing finished."""
    exp = resolve_experiment(name)
    if exp is None:
        return [], []
    rows, _ = collect(experiment_run_dirs(exp, results_dir))
    rows = [r for r in rows if r["valid"]]
    return rows, (aggregate(rows) if rows else [])


def valid_dirs(rows) -> list[Path]:
    return [Path(r["run_dir"]) for r in rows if r["valid"]]


# ---------------- E1 ----------------

def fig_e1(results_dir: Path, out: Path):
    rows, _ = experiment("E1", results_dir)
    reqs = [q for d in valid_dirs(rows) for q in read_jsonl(d / "requests.jsonl") if q["status"] == "ok"]
    if not reqs:
        return None
    pf, dec = defaultdict(list), defaultdict(list)
    for q in reqs:
        if q.get("prompt_ms") and q.get("tokens_processed"):
            pf[q["n_prompt"]].append(q["tokens_processed"] / (q["prompt_ms"] / 1000))
        if q.get("gen_ms") and q.get("generated_tokens"):
            dec[(q["n_predict"], q["n_prompt"])].append(q["gen_ms"] / q["generated_tokens"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    xs = sorted(pf)
    a.plot(xs, [statistics.median(pf[x]) for x in xs], color=SLOTS[0], marker="o")
    a.set(title="Prefill rate (K=1, cold prompt)", xlabel="prompt tokens", ylabel="prefill tokens/s")
    a.set_ylim(bottom=0)
    for i, o in enumerate(sorted({k[0] for k in dec})):
        xs = sorted(p for (oo, p) in dec if oo == o)
        b.plot(xs, [statistics.median(dec[(o, x)]) for x in xs], color=SLOTS[i], marker="o", label=f"{o} output tokens")
    b.set(title="Decode cost vs context (K=1)", xlabel="prompt tokens", ylabel="decode ms/token")
    b.set_ylim(bottom=0)
    b.legend()
    for ax in (a, b):
        k_axis(ax, xs)
    return save(fig, out, "e1_calibration")


# ---------------- E2 ----------------

def _by_workload(summary):
    groups = defaultdict(list)
    for s in summary:
        groups[s["workload"]].append(s)
    order = sorted(groups, key=lambda w: (WORKLOAD_ORDER.index(base_name(w)) if base_name(w) in WORKLOAD_ORDER else 99, w))
    return [(w, sorted(groups[w], key=lambda s: s["k"])) for w in order]


def fig_e2(results_dir: Path, out: Path):
    rows, summary = experiment("E2", results_dir)
    if not summary:
        return None
    best = {}
    bk_file = REPO / "results" / "E2" / "best_k.json"
    if bk_file.exists():
        best = {w: b.get("k") for w, b in json.loads(bk_file.read_text(encoding="utf-8")).items()}
    wls = _by_workload(summary)
    ks = sorted({s["k"] for s in summary})
    paths = []

    # throughput + p95 small multiples
    fig, axes = plt.subplots(2, len(wls), figsize=(2.3 * len(wls), 4.8), squeeze=False)
    for j, (w, ss) in enumerate(wls):
        c = wl_color(w)
        top, bot = axes[0][j], axes[1][j]
        errline(top, [s["k"] for s in ss], ss, "tasks_per_min", c)
        errline(bot, [s["k"] for s in ss], ss, "task_p95_s", c)
        bad = [s for s in ss if s["any_failed_task"]]
        for s in bad:
            top.plot(s["k"], s["tasks_per_min"], marker="x", color=INK, ms=7, ls="none")
        if best.get(w):
            b = next(s for s in ss if s["k"] == best[w])
            top.plot(b["k"], b["tasks_per_min"], marker="o", ms=12, mfc="none", mec=INK, mew=1.2, ls="none")
            top.text(0.97, 0.05, f"best K={b['k']}", transform=top.transAxes, ha="right", fontsize=8, color=INK2)
        top.set_title(base_name(w))
        for ax in (top, bot):
            k_axis(ax, ks)
            ax.set_ylim(bottom=0)
        bot.set_xlabel("K (concurrency)")
    axes[0][0].set_ylabel("completed tasks/min")
    axes[1][0].set_ylabel("p95 task completion (s)")
    fig.suptitle("E2: throughput and tail latency vs fixed concurrency (x = a config with a failed task)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    paths.append(save(fig, out, "e2_throughput"))

    # scaling relative to K=1
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for w, ss in wls:
        k1 = next((s["tasks_per_min"] for s in ss if s["k"] == 1), None)
        if not k1:
            continue
        ax.plot([s["k"] for s in ss], [s["tasks_per_min"] / k1 for s in ss], color=wl_color(w), marker="o",
                label=base_name(w))
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    k_axis(ax, ks)
    ax.set(title="E2: throughput relative to K=1", xlabel="K (concurrency)", ylabel="tasks/min ÷ tasks/min at K=1")
    ax.legend(loc="upper left")
    paths.append(save(fig, out, "e2_scaling"))

    # time breakdown
    parts = [("admit_wait_mean_s", "admission wait"), ("ready_q", "ready-queue wait"),
             ("model_mean_s", "model (prefill+decode)"), ("tool_mean_s", "tool delay")]
    rq_all = [max(0.0, (s["queue_mean_s"] or 0) - (s["admit_wait_mean_s"] or 0)) for s in summary]
    fig, axes = plt.subplots(1, len(wls), figsize=(2.3 * len(wls), 3.4), squeeze=False)
    for j, (w, ss) in enumerate(wls):
        ax = axes[0][j]
        x = list(range(len(ss)))
        bottom = [0.0] * len(ss)
        for i, (key, label) in enumerate(parts):
            if key == "ready_q":
                vals = [max(0.0, (s["queue_mean_s"] or 0) - (s["admit_wait_mean_s"] or 0)) for s in ss]
            else:
                vals = [s.get(key) or 0.0 for s in ss]
            if key == "ready_q" and max(rq_all) < 0.5:
                continue  # in-flight <= K and admission gives each task a slot: ready-queue wait is ~0 everywhere
            ax.bar(x, vals, bottom=bottom, color=SLOTS[i], width=0.72, edgecolor=SURFACE, linewidth=1,
                   label=label if j == 0 else None)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels([str(s["k"]) for s in ss])
        ax.set_title(base_name(w))
        ax.set_xlabel("K")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("mean seconds per task")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    title = "E2: where a task's time goes"
    if max(rq_all) < 0.5:
        title += f" (ready-queue wait < 0.5 s in every config, not drawn; max {max(rq_all):.2f} s)"
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths.append(save(fig, out, "e2_breakdown"))

    # decode contention
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for w, ss in wls:
        errline(ax, [s["k"] for s in ss], ss, "decode_ms_per_tok_p50", wl_color(w), label=base_name(w))
    k_axis(ax, ks)
    ax.set_ylim(bottom=0)
    ax.set(title="E2: decode cost per token vs K", xlabel="K (concurrency)",
           ylabel="median decode ms/token (outputs ≥ 32 tokens)")
    ax.legend(loc="upper left")
    paths.append(save(fig, out, "e2_decode"))
    return paths


def fig_timeline(run_dir: Path, out: Path):
    run_dir = Path(run_dir)
    if not (run_dir / "DONE").exists():
        return None
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    t0, t1 = man.get("t0_monotonic", float("-inf")), man.get("t1_monotonic", float("inf"))
    sysr = [s for s in read_jsonl(run_dir / "system.jsonl") if t0 <= s["t"] <= t1]
    sc = [s for s in sysr if s.get("sched")]
    if not sc:
        return None
    t0 = max(t0, sc[0]["t"]) if t0 == float("-inf") else t0
    t = [s["t"] - t0 for s in sc]
    budget = man["config"].get("kv_budget")
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    a.fill_between(t, [s["sched"]["kv_used_tracked"] / 1000 for s in sc], color=SLOTS[0], alpha=0.25, lw=0)
    a.plot(t, [s["sched"]["kv_reserved"] / 1000 for s in sc], color=SLOTS[1], label="reserved at admission", lw=1.6)
    a.plot(t, [s["sched"]["kv_used_tracked"] / 1000 for s in sc], color=SLOTS[0], label="held in slots (runner-tracked)", lw=1.6)
    if budget:
        a.axhline(budget / 1000, color=INK2, lw=1, ls="--")
        a.annotate(f"pool {budget:,} tokens", (t[-1], budget / 1000), textcoords="offset points", xytext=(0, 3),
                   ha="right", fontsize=7, color=INK2)
    a.set(ylabel="KV tokens (thousands)", ylim=(0, (budget or max(s["sched"]["kv_reserved"] for s in sc)) / 1000 * 1.12))
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    vt = [(s["t"] - t0, s["vram_used_mib"] / 1024) for s in sysr if s.get("vram_used_mib") is not None]
    if vt:
        b.plot(*zip(*vt), color=SLOTS[2], lw=1.4)
    pre = man.get("prelaunch_vram_mib")
    if pre:
        b.axhline(pre / 1024, color=INK2, lw=1, ls=":")
        b.annotate("pre-launch", (0, pre / 1024), textcoords="offset points", xytext=(2, 3), fontsize=7, color=INK2)
    b.set(ylabel="GPU memory used (GiB)", ylim=(0, 16.5))
    c.step(t, [s["sched"]["busy_slots"] for s in sc], where="post", color=SLOTS[0], lw=1.4, label="busy slots")
    c.step(t, [s["sched"]["waiting"] for s in sc], where="post", color=SLOTS[1], lw=1.4, label="tasks waiting for admission")
    c.set(ylabel="count", xlabel="seconds since measurement start")
    c.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    cfg = {"workload": "run", "k": "na", "slot_mode": "na", "ctx_size": 0, "rep": 0, **man["config"]}
    fig.suptitle(f"{base_name(cfg['workload'])}  K={cfg['k']}  {cfg['slot_mode']}  pool {cfg['ctx_size']:,}  rep{cfg['rep']}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    tag = f"{base_name(cfg['workload'])}_K{cfg['k']}_{cfg['slot_mode']}_c{cfg['ctx_size']}_rep{cfg['rep']}"
    return save(fig, out, f"timeline_{tag}")


# ---------------- E3 ----------------

def fig_e3(results_dir: Path, out: Path):
    rows, _ = experiment("E3", results_dir)
    if not rows:
        return None
    series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (wl,k) -> step -> metric -> vals
    for r in rows:
        for q in read_jsonl(Path(r["run_dir"]) / "requests.jsonl"):
            if q["status"] != "ok":
                continue
            d = series[(base_name(r["workload"]), r["k"])][q["step"]]
            d["req_s"].append(q["end_t"] - q["dispatch_t"])
            d["computed"].append(q.get("tokens_processed") or 0)
            if q.get("gen_ms") and q.get("generated_tokens"):
                d["dec"].append(q["gen_ms"] / q["generated_tokens"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    keys = sorted(series)
    for i, key in enumerate(keys):
        steps = sorted(series[key])
        wl, k = key
        kw = dict(color=SLOTS[i], marker="o", label=f"{wl.replace('growth_', '')} init, K={k}")
        for ax, m in zip(axes, ("req_s", "computed", "dec")):
            ax.plot([s + 1 for s in steps], [statistics.median(series[key][s][m]) for s in steps], **kw)
    titles = [("Request time per step", "median seconds (dispatch → end)"),
              ("Prompt tokens computed per step", "median tokens recomputed"),
              ("Decode cost per step", "median decode ms/token")]
    for ax, (tt, yl) in zip(axes, titles):
        ax.set(title=tt, xlabel="agent step", ylabel=yl)
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("E3: context growth (context grows by output + tool tokens each step)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, "e3_steps")


# ---------------- E4 ----------------

def fig_e4(results_dir: Path, out: Path):
    _, summary = experiment("E4", results_dir)
    if not summary:
        return None
    modes = ["pinned", "any"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for i, mode in enumerate(modes):
        ss = []
        for s in summary:
            mt = re.match(r"agentic_d(\d+(?:\.\d+)?)_s\d+$", s["workload"])
            if mt and s["slot_mode"] == mode:
                ss.append((float(mt.group(1)), s))
        ss.sort(key=lambda p: p[0])
        if not ss:
            continue
        xs = [p[0] for p in ss]
        rs = [p[1] for p in ss]
        errline(axes[0], xs, rs, "tasks_per_min", SLOTS[i], label=f"slot mode: {mode}")
        errline(axes[1], xs, rs, "cache_hit_steps_gt0", SLOTS[i], label=f"slot mode: {mode}")
    axes[0].set(title="Throughput vs tool delay (agentic, K=4)", xlabel="tool delay per step (s)", ylabel="completed tasks/min")
    axes[1].set(title="Prompt-cache hits after step 0", xlabel="tool delay per step (s)", ylabel="cached ÷ prompt tokens")
    axes[1].set_ylim(0, 1.05)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    # trace mixes (identical tasks, Continuum-style vs heavier delays)
    tr = [s for s in summary if base_name(s["workload"]).startswith("trace_agentic")]
    names = sorted({base_name(s["workload"]) for s in tr})
    ax = axes[2]
    for i, mode in enumerate(modes):
        vals = [next((s for s in tr if base_name(s["workload"]) == n and s["slot_mode"] == mode), None) for n in names]
        x = [j + (i - 0.5) * 0.38 for j in range(len(names))]
        ax.bar(x, [v["tasks_per_min"] if v else 0 for v in vals], width=0.36, color=SLOTS[i], edgecolor=SURFACE,
               label=mode)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("trace_agentic", "trace") or "trace" for n in names])
    ax.set(title="Trace-calibrated mixes (K=4)", ylabel="completed tasks/min")
    ax.grid(axis="x", visible=False)
    fig.suptitle("E4: idle periods — pinned slots keep KV during tool calls, any-slot recomputes",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, "e4_idle")


# ---------------- E5 ----------------

def fig_e5(results_dir: Path, out: Path):
    _, summary = experiment("E5", results_dir)
    if not summary:
        return None
    variants = [((True, True), "cache on, erase finished"), ((True, False), "cache on, keep finished"),
                ((False, True), "cache off")]
    wls = sorted({s["workload"] for s in summary}, key=lambda w: int(re.search(r"_p(\d+)", w).group(1)) if re.search(r"_p(\d+)", w) else 0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    width = 0.8 / len(variants)
    for i, ((cache, erase), label) in enumerate(variants):
        x = [j + (i - (len(variants) - 1) / 2) * width for j in range(len(wls))]
        for ax, key, scale in ((axes[0], "tasks_per_min", 1), (axes[1], "prompt_tokens_computed", 1e-3)):
            vals = [next((s for s in summary if s["workload"] == w and s["cache_prompt"] == cache
                          and s["erase"] == erase), None) for w in wls]
            ax.bar(x, [(v[key] or 0) * scale if v else 0 for v in vals], width=width * 0.95, color=SLOTS[i],
                   edgecolor=SURFACE, label=label)
    labels = [f"shared prefix {int(re.search(r'_p(\d+)', w).group(1)):,}" if re.search(r"_p(\d+)", w) else w for w in wls]
    for ax in axes:
        ax.set_xticks(range(len(wls)))
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)
    axes[0].set(title="Throughput", ylabel="completed tasks/min")
    axes[1].set(title="Prompt tokens actually computed", ylabel="thousand tokens per run")
    axes[0].legend(loc="lower left")
    fig.suptitle("E5: does retained or shared context help? (agentic, K=4)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, "e5_reuse")


# ---------------- E6 ----------------

def fig_e6(results_dir: Path, out: Path):
    _, summary = experiment("E6", results_dir)
    if not summary:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, gate in enumerate([True, False]):
        ss = sorted([s for s in summary if s["admission_gate"] == gate], key=lambda s: s["ctx_size"])
        if not ss:
            continue
        xs = [s["ctx_size"] / 1024 for s in ss]
        label = "reserve peak at admission (baseline)" if gate else "no admission gate"
        errline(axes[0], xs, ss, "failure_rate", SLOTS[i], label=label)
        errline(axes[1], xs, ss, "tasks_per_min", SLOTS[i], label=label)
    for ax in axes:
        ax.set_xlabel("KV pool (thousand tokens, -c)")
        ax.set_ylim(bottom=0)
        ax.invert_xaxis()
    axes[0].set(title="Failed tasks", ylabel="fraction of planned tasks")
    axes[0].set_ylim(0, 1.05)
    axes[1].set(title="Throughput", ylabel="completed tasks/min")
    axes[0].legend(loc="upper left")
    fig.suptitle("E6: shrinking the memory budget (trace_agentic, K=4)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, "e6_budget")


# ---------------- E7 ----------------

def fig_e7(results_dir: Path, out: Path):
    rows = []
    for d in sorted((results_dir / "runs" / "E7_idle_kv").glob("rep*")):
        if (d / "DONE").exists():
            rows += read_jsonl(d / "rows.jsonl")
    if not rows:
        return None

    def med(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None and r["status"] == "ok"]
        return (statistics.median(v), min(v), max(v)) if v else None

    def line(ax, pts, color, label):
        pts = [(x, m) for x, m in pts if m]
        if pts:
            x, m = zip(*pts)
            y = [a[0] for a in m]
            ax.errorbar(x, y, yerr=[[a[0] - a[1] for a in m], [a[2] - a[0] for a in m]], color=color, marker="o",
                        capsize=0, elinewidth=1.2, label=label, markeredgecolor=SURFACE, markeredgewidth=1)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    A = [r for r in rows if r["part"] == "A"]
    for i, (kvu, label) in enumerate([(False, "static split (-no-kvu)"), (True, "shared pool (-kvu)")]):
        rs = [r for r in A if r["kv_unified"] == kvu]
        levels = sorted({r["idle_tokens"] for r in rs})
        for ax, key in ((axes[0][0], "decode_ms_per_tok"), (axes[0][1], "prompt_ms_per_tok")):
            line(ax, [(lv / 1000, med([r for r in rs if r["idle_tokens"] == lv], key)) for lv in levels], SLOTS[i], label)
    axes[0][0].set(title="Decode cost of one active request", xlabel="idle KV held by other slots (thousand tokens)",
                   ylabel="decode ms/token")
    axes[0][1].set(title="Prefill cost of one active request", xlabel="idle KV held by other slots (thousand tokens)",
                   ylabel="prefill ms/token")
    axes[0][0].legend(loc="upper left")
    B = [r for r in rows if r["part"] == "B"]
    bs = sorted({r["batch"] for r in B})
    line(axes[1][0], [(b, med([r for r in B if r["batch"] == b], "decode_ms_per_tok_median")) for b in bs], SLOTS[1], None)
    line(axes[1][1], [(b, med([r for r in B if r["batch"] == b], "aggregate_tok_per_s")) for b in bs], SLOTS[1], None)
    axes[1][0].set(title="Per-request decode cost vs batch", xlabel="requests decoding at once", ylabel="decode ms/token")
    axes[1][1].set(title="Aggregate decode throughput vs batch", xlabel="requests decoding at once", ylabel="generated tokens/s")
    for ax in axes[1]:
        if bs:
            k_axis(ax, bs)
    for ax in axes.flat:
        ax.set_ylim(bottom=0)
    fig.suptitle("E7: idle KV vs batching as causes of the decode slowdown (2,000-token contexts, 256 out)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, "e7_idle_kv")


FIGS = {"E1": fig_e1, "E2": fig_e2, "E3": fig_e3, "E4": fig_e4, "E5": fig_e5, "E6": fig_e6, "E7": fig_e7}


def main() -> None:
    from runner.server import load_local

    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", default=list(FIGS), help="subset of " + " ".join(FIGS))
    ap.add_argument("--out", help="output dir (default <repo>/results/figures)")
    ap.add_argument("--timeline", nargs="*", help="run dirs for timeline figures (default: two E2 K=16 runs)")
    a = ap.parse_args()
    results_dir = Path(load_local()["results_dir"])
    out = Path(a.out) if a.out else REPO / "results" / "figures"
    style()
    for e in a.experiments:
        print(f"{e}:")
        if FIGS[e](results_dir, out) is None:
            print("  skipped (no finished valid runs yet)")
    if "E2" in a.experiments or a.timeline is not None:
        for d in (a.timeline if a.timeline else [results_dir / "runs" / p for p in DEFAULT_TIMELINES]):
            if fig_timeline(Path(d), out) is None:
                print(f"  timeline skipped (not finished): {d}")


if __name__ == "__main__":
    main()
