# Experiment definitions (read by `experiments/run_sweep.py`)

Each YAML expands to a list of single runs (`runner/run.py`); each run gets a fresh server.

| key | meaning |
|---|---|
| `name`, `purpose` | Identification; `purpose` is free text. |
| `base` | `RunConfig` fields shared by every point (`k`, `slot_mode`, `pool`, `agents`, `tasks`, `cache_prompt`, `erase_finished`, `admission_gate`, `kv_unified`, `deadline_s`, ...). |
| `workloads` | List of workload names (`configs/workloads/<name>.yaml`) or entries `{name, from, set: {dotted.key: value}, run: {RunConfig overrides}}`. `from` + `set` make a variant (written to `<results_dir>/workloads/derived/`); `run` overrides run settings for that workload only (e.g. `tasks`). |
| `seeds` | Workload seeds (ignored for `fixed_seed` workloads). Default: the workload's own seed. |
| `grid` | `RunConfig` field → list of values; the sweep is the cross product with `workloads` and `seeds`. |
| `exclude` | List of partial points to drop from the grid. |
| `task_cap` | `{per_k, min}`: run the first `min(num_tasks, max(min, per_k*K))` tasks (decision 2026-11-03). Ignored when `base.tasks` is set. |
| `reps` | Repetitions per point (same seed and tasks; measures run-to-run noise). |
| `pilot` | Overrides for `--pilot` (any of `grid`, `workloads`, `reps`, `base`): 2–3 points before the full grid. |

Points with identical configs (e.g. the same K=4 run in two experiments) share one run directory, so a
point already DONE in another experiment is not rerun.
