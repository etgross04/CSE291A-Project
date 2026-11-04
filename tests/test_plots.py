"""analysis.plots: figures build from a fake run and skip cleanly when nothing has finished."""
from __future__ import annotations

from analysis import plots
from test_run_and_checks import make_run


def test_timeline_from_fake_run(tmp_path):
    (tmp_path / "run").mkdir()
    d, _ = make_run(tmp_path / "run")
    plots.style()
    p = plots.fig_timeline(d, tmp_path / "fig")
    assert p is not None and p.exists() and p.with_suffix(".pdf").exists()


def test_figures_skip_without_runs(tmp_path):
    plots.style()
    for name, fn in plots.FIGS.items():
        assert fn(tmp_path, tmp_path / "fig") is None, name
