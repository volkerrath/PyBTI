"""
test_dictionary_workflow.py -- Dictionary data flow and initial conditions.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 tests using analytical and synthetic cases.
"""

import importlib.util
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numeric as nm
from init import YEAR2SEC, build_gsth, build_initial
from prep import read_data, prepare_site
from workflow import build_mesh, run_forward, run_inversion, plot_forward

ROOT = Path(__file__).resolve().parents[1]


def _site(tmp_path):
    """Uniform dry model with an analytically known equilibrium profile."""
    path = tmp_path / "observations.csv"
    np.savetxt(path, [[25, 6.5], [125, 8.5]], delimiter=",")
    obs = read_data(dict(file=path, Terr=0.1))
    mesh = build_mesh(dict(
        depth=dict(ztype="read", zmesh_in=[0, 50, 100, 150], zend=200),
        time=dict(ttype="read", tmesh_in=np.linspace(-1000, 0, 11) * YEAR2SEC),
    ), obs)
    site = prepare_site(dict(name="test", props="syn", k=3., r=2700., c=800.,
                             qb=-0.06, p=0., h=0.), mesh, obs)
    return site, mesh


def test_equilibrium_and_forward_preserve_analytical_profile(tmp_path):
    site, mesh = _site(tmp_path)
    controls = dict(freeze=0)
    initial = build_initial(site, controls, dict(GST0=6.))
    expected = 6 + 0.02 * site["z"]
    np.testing.assert_allclose(initial["Tinit"], expected, atol=1e-10)
    assert initial["niter"] == 0 and initial["Tit"].shape == (len(expected), 0)
    result = run_forward(site, controls, initial)
    np.testing.assert_allclose(result["Tcalc"][:, -1], expected, atol=1e-9)
    assert result["rms"] < 1e-8
    # The plot accepts an ordinary site dictionary, not a SitePar instance.
    run = dict(name="test", site=site, mesh=mesh, init=initial, result=result)
    plot = plot_forward(run, dict(outdir=tmp_path))
    assert plot["filename"].is_file()
    plt.close(plot["figure"])


def test_repeated_history_feeds_final_profile_into_next_cycle(tmp_path):
    site, _ = _site(tmp_path)
    GST = np.r_[np.full(5, -2.), np.full(6, 6.)]
    initial = build_initial(site, dict(freeze=0), dict(
        GST0=6., GST=GST, init_type="periodic", initial_iter=3))
    previous = 6 + .02 * site["z"]
    for cycle in range(3):
        expected = nm.heat1dnt(
            site["k"], site["kA"], site["kB"], site["h"], site["r"],
            site["c"], site["p"], site["qb"], site["dz"], site["ip"],
            site["dt"], np.arange(11), GST, previous,
            1., 4, 1e-5, 0, 0, "syn",
        )
        np.testing.assert_allclose(initial["Tit"][:, cycle], expected, atol=1e-9)
        previous = expected
    np.testing.assert_allclose(initial["Tinit"], previous)
    assert not np.allclose(initial["Tinit"], initial["Tsteady"])
    assert initial["converged"] is None and initial["niter"] == 3
    assert "Tinit" not in site


def test_periodic_convergence_and_validation(tmp_path):
    site, _ = _site(tmp_path)
    initial = build_initial(site, dict(freeze=0), dict(
        GST0=6., GST=[6.], init_type="periodic", initial_iter=30, initial_tol=1e-7))
    assert initial["converged"] is True and initial["niter"] == 1
    with pytest.raises(ValueError, match="explicit GST"):
        build_initial(site, {}, dict(GST0=6., init_type="periodic"))
    with pytest.raises(ValueError, match="positive integer"):
        build_initial(site, {}, dict(GST0=6., GST=[6.], init_type="periodic", initial_iter=0))


def test_history_units_preonset_and_points():
    t = np.array([-110000, -70000, -50000, -10000, 0.]) * YEAR2SEC
    config = dict(file=ROOT / "examples/A/GSTHBallingA.csv", prehistory=6.)
    forcing = build_gsth(dict(t=t), config)
    np.testing.assert_allclose(forcing["GST"], [6., -4., -4., 6., 6.])
    with pytest.raises(ValueError, match="prehistory"):
        build_gsth(dict(t=t), dict(file=config["file"]))
    points = build_gsth(dict(t=np.array([-10, -5, 0.])), dict(
        time=[-10., 0.], temperature=[2., 4.], form="points",
        time_unit="s", time_convention="relative"))
    np.testing.assert_allclose(points["GST"], [2., 3., 4.])


def test_data_remain_independent_of_mesh_and_noise_is_explicit(tmp_path):
    config = dict(file=ROOT / "examples/A/DataBallingA.csv", zcol=0, Tcol=3,
                  Terr=.05, depth_range=(10, 2000))
    clean = read_data(config)
    assert clean["Tobs"].size == 40
    np.testing.assert_array_equal(clean["Tobs"], clean["raw"][clean["source_rows"], 3])
    noisy_config = dict(config, noise=dict(kind="gaussian", length=150., seed=3))
    with pytest.raises(ValueError, match="synthetic=True"):
        read_data(noisy_config)
    noisy_config["synthetic"] = True
    noisy = read_data(noisy_config)
    np.testing.assert_array_equal(noisy["Tobs"], read_data(noisy_config)["Tobs"])
    assert np.any(noisy["noise"] != 0)
    assert noisy["Tcov"][0, 1] > 0
    mesh = build_mesh({}, clean)
    np.testing.assert_array_equal(mesh["z"][np.searchsorted(mesh["z"], clean["zobs"])],
                                  clean["zobs"])
    assert len(clean["Tobs"]) == 40


def test_dictionary_inversion_equilibrium(tmp_path):
    site, _ = _site(tmp_path)
    initial = build_initial(site, dict(freeze=0), dict(GST0=6.))
    inverse = dict(it=np.r_[np.zeros(5, dtype=int), np.ones(6, dtype=int)],
                   nsteps=2, m_ini_set=6., m_apr_set=6., maxiter_inv=1,
                   start_regpar=0, stop_needs_start_regpar=False)
    result = run_inversion(site, dict(freeze=0), inverse, initial)
    assert np.isfinite(result["Tcalc"]).all()
    correlated = dict(site, Tcov=site["Tcov"] + 0.001)
    with pytest.raises(ValueError, match="diagonal Tcov"):
        run_inversion(correlated, {}, inverse, initial)


def test_example_a_dictionary_pipeline():
    spec = importlib.util.spec_from_file_location("example_a", ROOT / "examples/A/run_forward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameters = module.example_parameters()
    parameters["initpar"]["init_type"] = "equilibrium"
    run = module.run_example(parameters)
    assert isinstance(run["site"], dict) and isinstance(run["fwd"], dict)
    assert run["site"]["Tobs"].size == 40
    assert np.isfinite(run["result"]["rms"])
    assert run["init"]["niter"] == 0
    assert "GST" not in parameters["initpar"]
    assert run["mesh"]["t"][-1] == 0
    np.testing.assert_allclose(run["result"]["Tcalc"][0, -1], 6., atol=1e-10)


def test_cycle_diagnostics_and_immediate_plot_snapshots(tmp_path, monkeypatch):
    import init as initial_module
    import init_plot

    site, _ = _site(tmp_path)
    increment = np.linspace(0., 3., len(site["z"]))
    completed, plot_calls, displayed_cycles = [], [], []
    original_plot = init_plot.plot_initial_cycles

    def display_cycle(fig):
        displayed_cycles.append((len(completed), fig._suptitle.get_text()))

    def synthetic_cycle(s, f, gst, it, previous, out=0):
        profile = previous + (len(completed) + 1) * increment
        completed.append(profile.copy())
        return profile

    def checked_plot(s, initial, options):
        assert initial["niter"] == len(completed)
        np.testing.assert_allclose(initial["Tit"][:, -1], completed[-1])
        plotted = original_plot(s, initial, options)
        plot_calls.append((initial["niter"], plotted["figure"]))
        return plotted

    monkeypatch.setattr(initial_module, "_transient", synthetic_cycle)
    monkeypatch.setattr(init_plot, "plot_initial_cycles", checked_plot)
    open_before = set(plt.get_fignums())
    initial = build_initial(site, dict(freeze=0), dict(
        GST0=6., GST=[6.], init_type="periodic", initial_iter=3,
        plotpar=dict(outdir=tmp_path, dpi=45, show=False,
                     display_callback=display_cycle)))
    expected_change = np.arange(1., 4.) * np.linalg.norm(increment)
    np.testing.assert_allclose(initial["change_l2"], expected_change)
    np.testing.assert_allclose(initial["changes"], [3., 6., 9.])
    assert [n for n, fig in plot_calls] == [1, 2, 3]
    assert [n for n, title in displayed_cycles] == [1, 2, 3]
    assert all(f"cycle {n}," in title for n, title in displayed_cycles)
    assert len({id(fig) for n, fig in plot_calls}) == 1
    assert set(plt.get_fignums()) == open_before
    assert len(initial["plot_files"]) == 3
    for path in initial["plot_files"]:
        assert path.read_bytes().startswith(b"\x89PNG")
    csv = np.loadtxt(initial["monitor_plot"]["diagnostics_file"], delimiter=",", skiprows=1)
    assert csv.shape == (3, 3)
    np.testing.assert_array_equal(csv[:, 0], [1, 2, 3])
    np.testing.assert_allclose(csv[:, 1], expected_change)
    np.testing.assert_allclose(csv[:, 2], [3., 6., 9.])
    axes = initial["monitor_plot"]["axes"]
    assert set(axes) == {"temperature", "difference", "changes"}
    np.testing.assert_allclose(axes["difference"].lines[1].get_xdata(), 3 * increment)
    for key in ("temperature", "difference"):
        assert axes[key].get_ylim() == (site["z"][-1], 0.)
    assert not {"residuals", "rms", "weighted_rms", "baseline"} & initial.keys()


def test_initialisation_ignores_observation_fields(tmp_path):
    site, _ = _site(tmp_path)
    options = dict(GST0=6., GST=[6.], init_type="periodic", initial_iter=30,
                   initial_tol=1e-7, plotpar=dict(outdir=tmp_path, dpi=45))
    reference = build_initial(site, dict(freeze=0), options)
    # Even invalid observation metadata must not enter thermal initialisation.
    site.update(Tobs=np.array([np.nan]), Terr=np.array([-1.]), id=np.array([999999]))
    actual = build_initial(site, dict(freeze=0), options)
    np.testing.assert_array_equal(actual["Tinit"], reference["Tinit"])
    np.testing.assert_array_equal(actual["change_l2"], reference["change_l2"])
    assert actual["converged"] is True and actual["niter"] == 1
    assert len(actual["plot_files"]) == 1


def test_initialisation_monitor_without_observations(tmp_path):
    site, _ = _site(tmp_path)
    for key in ("id", "Tobs", "Terr", "zobs"):
        site.pop(key)
    initial = build_initial(site, dict(freeze=0), dict(
        GST0=6., GST=[6.], init_type="periodic", initial_iter=1, plotpar={}))
    assert initial["change_l2"].shape == (1,)
    assert initial["plot_files"] == []
    assert "residuals" not in initial


def test_minimum_model_depth():
    for requested, expected in ((2000., 5000.), (6500., 6500.)):
        parameters = dict(depth=dict(zend=requested), min_depth=5000.)
        mesh = build_mesh(parameters)
        assert mesh["z"][0] == 0 and mesh["z"][-1] == expected
        assert parameters["depth"]["zend"] == requested


def test_periodic_example_stops_at_initial_condition(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("example_a", ROOT / "examples/A/run_forward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("Initial-condition generation must stop after the last cycle")

    monkeypatch.setattr(module, "run_forward", forbidden_forward)
    parameters = module.example_parameters()
    parameters["meshpar"].update(min_depth=0., depth=dict(zend=2000.))
    parameters["initpar"].update(init_type="periodic", initial_iter=2,
                                plotpar=dict(outdir=tmp_path, dpi=30))
    run = module.run_example(parameters)
    assert run["result"] is None
    assert run["init"]["niter"] == 2 and run["mesh"]["z"][-1] == 5000.
    assert len(run["init"]["plot_files"]) == 2
    for key in ("temperature", "difference"):
        assert run["init"]["monitor_plot"]["axes"][key].get_ylim() == (5000., 0.)
