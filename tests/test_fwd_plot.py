"""
test_fwd_plot.py -- Plot conventions and synthetic forward integration.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : AI-generated tests; synthetic data only.
"""

import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fwd_plot import YEAR2SEC, plot_fwd, _demo


def _run():
    """Small run with nonidentity GST pointers and distinct time slices."""
    return dict(
        name="TEST", mesh=dict(it=np.arange(3)),
        site=SimpleNamespace(t=np.array([-400., -100., 0.]) * YEAR2SEC,
                             Terr=np.array([0.1, 0.2]), gts=50.),
        init=dict(GST=np.array([3., 6., 12.]), it=np.array([1, 0, 2])),
        result=dict(
            z=np.array([0., 100., 300.]),
            Tcalc=np.array([[0., 6., 3.], [10., 11., 12.], [20., 21., 22.]]),
            id=np.array([1, 2]), Tobs=np.array([12.2, 21.6]), zobs=None,
            r=np.array([0.2, -0.4]), rms=2.,
        ),
    )


def test_plot_values_and_conventions(tmp_path):
    run = _run()
    original = run["result"]["Tcalc"].copy()
    out = plot_fwd(run, outdir=tmp_path)
    try:
        axes = out["axes"]
        climate = axes["gsth"].lines[0]
        np.testing.assert_allclose(climate.get_xdata(), [400., 100., 0.])
        # Unused last GST node and site.gts must not alter the forcing.
        np.testing.assert_allclose(climate.get_ydata(), [6., 3., 3.])
        final = next(line for line in axes["temperature"].lines
                     if line.get_label() == "Final model")
        np.testing.assert_allclose(final.get_xdata(), [3., 12., 22.])
        np.testing.assert_allclose(axes["residuals"].lines[1].get_xdata(), [0.2, -0.4])
        np.testing.assert_allclose(axes["residuals"].lines[1].get_ydata(), [100., 300.])
        assert axes["gsth"].xaxis_inverted()
        assert axes["temperature"].yaxis_inverted()
        assert axes["residuals"].yaxis_inverted()
        np.testing.assert_array_equal(run["result"]["Tcalc"], original)
        assert out["filename"].read_bytes().startswith(b"\x89PNG")
    finally:
        plt.close(out["figure"])


def test_time_scales_and_optional_panels():
    run = _run()
    with pytest.raises(ValueError, match="positive ages"):
        plot_fwd(run, time_scale="log")
    out = plot_fwd(run, time_scale="symlog", show_initial=False, show_residuals=False)
    try:
        assert set(out["axes"]) == {"gsth", "temperature"}
        assert out["axes"]["gsth"].get_xlim()[1] == 0
        assert out["filename"] is None
    finally:
        plt.close(out["figure"])


def test_interval_length_pointer_and_bad_pointer():
    run = _run()
    run["init"]["it"] = np.array([1, 0])
    out = plot_fwd(run)
    plt.close(out["figure"])
    run["init"]["it"] = np.array([1, 0.5])
    with pytest.raises(ValueError, match="integer GST indices"):
        plot_fwd(run)
    run["init"]["it"] = np.array([1, 99])
    with pytest.raises(ValueError, match="out-of-range"):
        plot_fwd(run)


def test_synthetic_forward_run(tmp_path):
    out = _demo(outdir=tmp_path, show=False)
    try:
        axes = out["axes"]
        final = next(line for line in axes["temperature"].lines
                     if line.get_label() == "Final model")
        observed = axes["temperature"].containers[0].lines[0]
        predicted = np.interp(observed.get_ydata(), final.get_ydata(), final.get_xdata())
        np.testing.assert_allclose(
            observed.get_xdata() - predicted,
            axes["residuals"].lines[1].get_xdata(), atol=1e-12,
        )
        np.testing.assert_allclose(
            final.get_xdata()[0], axes["gsth"].lines[0].get_ydata()[-1], atol=1e-12,
        )
        assert out["filename"].is_file()
    finally:
        plt.close(out["figure"])
