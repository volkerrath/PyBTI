"""
test_tikh_workflow.py -- Synthetic checks for dictionary Tikhonov inversion.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 analytical and synthetic validation.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numeric as nm
from prep import prepare_site
from init import build_initial, YEAR2SEC
from workflow import build_mesh, build_inversion, run_inversion, plot_inversion
from gsth_drivers import SitePar, FwdPar, make_forward_model


def _case():
    observations = dict(zobs=np.arange(25., 501., 25.), Tobs=np.zeros(20),
                        Terr=np.full(20, .05), Tcov=np.eye(20) * .05**2)
    mesh = build_mesh(dict(depth=dict(zend=5000.),
                          time=dict(tstart=1000*YEAR2SEC, tend=10*YEAR2SEC, nt=61)),
                      observations)
    site = prepare_site(dict(name="test_tikh", props="const", k=3., r=2700.,
                             c=800., qb=-.06, p=0.), mesh, observations)
    fwd = dict(freeze=0, theta=1., maxitnl=4, tolnl=1e-6)
    initial = build_initial(site, fwd, dict(GST0=6.))
    controls = dict(m_ini_set=5., m_apr_set=6., dp=.001, maxiter_inv=10,
                    start_regpar=3, modul_regpar=1, tol_inv=(0., 1e-4),
                    regpar0=(.1, 0., 0.), reg0par=[.001],
                    reg1par=[.001, .01, .1], reg2par=[0.],
                    reg_opt="gcv", maxiter_solve=100, tol_solve=1e-9)
    inverse = build_inversion(mesh, dict(nsteps=4, tstart=1000*YEAR2SEC,
                                        tend=10*YEAR2SEC), controls)
    model = make_forward_model(
        SitePar(**{key: site[key] for key in SitePar.__dataclass_fields__ if key in site}),
        FwdPar(**fwd), inverse["it"], initial["Tinit"])
    site["Tobs"] = model.final(np.array([6., 3., 4., 6.]))[site["id"]]
    return mesh, site, fwd, initial, inverse, model


def test_template_grid_and_unused_parameter_validation():
    mesh = build_mesh(dict(time=dict(tstart=110000*YEAR2SEC, tend=30*YEAR2SEC, nt=401)))
    setup = build_inversion(mesh, dict(nsteps=21), dict(diffmeth="FD"))
    assert setup["nsteps"] == 21
    assert len(setup["it"]) == len(mesh["t"])
    assert np.all(np.bincount(setup["it"][:-1], minlength=21) > 0)
    with pytest.raises(ValueError, match="unused"):
        build_inversion(dict(t=np.array([-110000., 0.])*YEAR2SEC),
                        dict(nsteps=21), {})
    with pytest.raises(ValueError, match="FD"):
        build_inversion(mesh, {}, dict(diffmeth="AD"))


def test_gcv_runs_after_warmup_and_outputs_are_consistent(tmp_path):
    _, site, fwd, initial, inverse, model = _case()
    result = run_inversion(site, fwd, inverse, initial,
                           dict(outdir=tmp_path, verbose=False))
    assert result["search"]
    assert result["search"]["iteration"] > inverse["start_regpar"]
    assert result["rms_iter"][-1] < result["rms_iter"][0] * .1
    np.testing.assert_array_equal(result["m"], result["m_iter"][-1])
    np.testing.assert_allclose(result["Tcalc"], model(result["m"]), atol=1e-9)
    residual = site["Tobs"] - result["Tcalc"][site["id"], -1]
    np.testing.assert_allclose(result["r_iter"][-1], residual)
    np.testing.assert_allclose(result["rms_iter"][-1],
                               np.sqrt(np.mean((residual/site["Terr"])**2)))
    np.testing.assert_array_equal(result["regpar"], result["regpar_iter"][-1])
    np.testing.assert_array_equal(np.load(tmp_path/"test_tikh_initial_out.npz")["m_ini"],
                                  np.full(4, 5.))
    np.testing.assert_array_equal(np.load(tmp_path/"test_tikh_prior_out.npz")["m_apr"],
                                  np.full(4, 6.))
    # The objective must match the stacked penalty minimised by LSQR.
    dm = result["m"] - result["m_apr"]
    penalty = sum(result["regpar"][j] * np.linalg.norm(nm.reg1d(4, kind) @ dm)**2
                  for j, kind in enumerate(("l0", "l1", "l2")))
    assert result["theta_m_iter"][-1] == pytest.approx(penalty)
    plots = plot_inversion(result, dict(outdir=tmp_path, dpi=40))
    assert plots["filenames"][0].read_bytes().startswith(b"\x89PNG")
    np.testing.assert_allclose(plots["axes"]["residuals"].lines[0].get_xdata(), residual)
    plt.close(plots["figure"])


def test_iteration_limit_returns_evaluated_model_and_final_jacobian():
    _, site, fwd, initial, inverse, model = _case()
    inverse.update(maxiter_inv=2, reg_opt="fix")
    result = run_inversion(site, fwd, inverse, initial, dict(verbose=False))
    assert result["niter"] == 2 and result["stop_reason"] == "iteration_limit"
    np.testing.assert_array_equal(result["m"], result["m_iter"][-1])
    np.testing.assert_allclose(result["Tcalc"], model(result["m"]), atol=1e-9)
    shifted = result["m"].copy()
    shifted[0] += inverse["dp"]
    jac = (model.final(shifted) - model.final(result["m"])) / inverse["dp"]
    np.testing.assert_allclose(result["Jw"][:, 0], jac[site["id"]]/site["Terr"], atol=1e-6)


def test_plot_heat_flow_uses_final_gradient_and_intervening_cells():
    z = np.array([0., 25., 75., 150.])
    temperature = 6. + .02*z
    result = dict(name="layers", z=z, t=np.array([-100., -10., 0.])*YEAR2SEC,
                  id=np.array([0, 2, 3]), Tcon=np.array([2., 4., 8.]),
                  Tcalc=np.tile(temperature[:, None], (1, 3)),
                  Tobs=temperature[[0, 2, 3]], Terr=np.full(3, .1), qb=-.06,
                  gts=2., m=np.array([4., 4.]), it=np.array([0, 1, 1]),
                  rms_iter=np.array([0.]), niter=1, search={})
    plot = plot_inversion([result, result], dict(smooth_window=1))
    heat = plot["heat_flow"][0]
    np.testing.assert_allclose(heat["conductivity"], [3., 8.])
    np.testing.assert_allclose(heat["calculated"], [.06, .16])
    np.testing.assert_allclose(heat["observed"], [.06, .16])
    np.testing.assert_allclose(plot["axes"]["gsth"].lines[0].get_ydata(), 6.)
    assert "mW/m²" in plot["axes"]["heat_flow"].get_xlabel()
    plt.close(plot["figure"])
