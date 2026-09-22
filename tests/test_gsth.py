"""
test_gsth.py -- self-tests for numeric.py, gsth_drivers.py, gsth_mcmc.py.

Run from the package directory (phys.py must be next to numeric.py):
    python tests/test_gsth.py          or          pytest -q tests

The MCMC test uses the pymcmcstat dependency pinned in BTI.yaml.

Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-21
Status         : AI-generated test code; review before production use.
"""
import os
import sys
import tempfile
from unittest import SkipTest

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
import numeric as nm                                   # noqa: E402
from gsth_drivers import (FwdPar, InvPar, SitePar, load_matlab_site,  # noqa
                          make_forward_model, tikhonov_gsth)

Y = nm.YEAR2SEC


def test_helpers():
    assert np.allclose(nm.n2c([0, 1, 3, 6]), [0.5, 2.0, 4.5])
    assert np.allclose(nm.c2n([1, 2, 4], [1, 2, 3]),
                       [2 / 3, 5 / 3, 3.2, 5.2])
    assert abs(nm.tri(7).sum() - 1) < 1e-14 and abs(nm.box(4).sum() - 1) < 1e-14
    L2 = nm.reg1d(6, "l2").toarray()
    assert np.allclose(L2[1], [1, -2, 1, 0, 0, 0]) and not L2[0].any()
    assert nm.setregpar([1, 2, 3], [1, 10], [1], [1, 100]).shape == (4, 3)
    x, dx = nm.set_mesh(1.0, 100.0, 11, "log", 1)
    assert x[0] == 0 and abs(x[-1] - 100) < 1e-9
    xt, _ = nm.set_mesh(1e5, 1.0, 11, "log", -1)
    assert xt[-1] == 0 and xt[0] < 0
    s, _ = nm.set_gst(np.array([-10., -5, 0]), [1., 2, 3], [-6., -1])
    assert np.allclose(s, [1, 2, 3])                  # t==steptime(end) set
    assert np.allclose(nm.smooth1([1, 1, 1, 5, 1, 1]), [1, 1, 2, 3, 2, 1])
    assert np.isclose(nm.wfilt(np.ones(20), 5, 3).mean(), 1.0)
    assert np.isclose(nm.mad([1, 2, 3, 4, 100]), 1.4826 * 1.0)
    assert np.isclose(nm.vavg([1, 4], flag="g"), 2.0)
    assert np.isclose(nm.vavg([1, 4], flag="h"), 1.6)


def test_stationary_closed_form():
    dz = np.full(300, 10.0)
    ip = np.zeros(300, int)
    T = nm.heat1dns([3.0], [.7], [770.], [1e-6], [2650.], [0.0], -0.06, 10.0,
                    dz, ip, props="const")
    z = np.concatenate([[0], np.cumsum(dz)])
    Tex = 10 + 0.06 * z / 3 + (1e-6 * z / 3) * (z[-1] - z / 2)
    assert np.abs(T - Tex).max() < 1e-8


def test_transient_vs_erfc_and_heat1dat():
    z, dz = nm.set_mesh(1.0, 3000.0, 201, "log", 1)
    ip = np.zeros(dz.size, int)
    t = np.linspace(-500 * Y, 0, 1001)
    dt = np.diff(t)
    it = (t >= -200 * Y).astype(int)
    kap = 3.0 / (2650 * 800.)
    Tex = 2.0 * erfc(z / (2 * np.sqrt(kap * 200 * Y)))
    for th in (1.0, 0.5):
        Tf = nm.heat1dnt([3.], [.7], [770.], [0.], [2650.], [800.], [0.],
                         0.0, dz, ip, dt, it, [0., 2.], np.zeros(dz.size + 1),
                         th, 10, 1e-6, 0, 0, props="const")
        assert np.abs(Tf - Tex).max() < 2e-3
    Ta, _ = nm.heat1dat(3.0, 2650., 800., 0.0, z, [-500 * Y, -200 * Y],
                        [0., 2.], 0.0, 2000, 2000)
    assert np.abs(Ta["val"] - Tex).max() < 1e-12


def _synthetic_site(noise=0.01, seed=1):
    rng = np.random.default_rng(seed)
    z, dz = nm.set_mesh(1.0, 1200.0, 61, "log", 1)
    zc = 0.5 * (z[:-1] + z[1:])
    ip = np.where(zc < 500, 0, 1)
    t, _ = nm.set_mesh(2.0e4 * Y, 1.0 * Y, 90, "log", -1)
    pt, it, cc, ll, uu = nm.set_mgsth(t, 0.0, 2.0e4 * Y, 1.0 * Y, 9, "log")
    ns = pt.size
    id_ = np.arange(10, z.size, 3)
    site = SitePar(z=z, t=t, ip=ip, k=[2.6, 3.1], kA=[.7, .7], kB=[770., 770.],
                   h=[1.2e-6, 1.0e-6], r=[2500., 2700.], c=[850., 800.],
                   p=[0.06, 0.02], qb=-0.06, gts=8.0, id=id_,
                   Tobs=np.zeros(id_.size), Terr=np.full(id_.size, noise),
                   zobs=z[id_], props="full", name="synth")
    fwd = FwdPar(theta=1.0, maxitnl=10, tolnl=1e-5, freeze=1)
    mtrue = np.zeros(ns)
    mtrue[-4:] = [-0.2, 0.1, 0.6, 1.2]            # recent warming
    mtrue[2:5] = -3.0                             # cold period
    fm = make_forward_model(site, fwd, it)
    Tt = fm.final(mtrue)                          # equilibrium start
    site.Tobs = Tt[id_] + noise * rng.standard_normal(id_.size)
    return site, fwd, it, mtrue, fm


def test_forward_consistency_and_jacobian():
    site, fwd, it, mtrue, fm = _synthetic_site()
    gst = mtrue + site.gts
    T0 = nm.heat1dns(site.k, site.kA, site.kB, site.h, site.r, site.p,
                     site.qb, gst[0], site.dz, site.ip, 10, 1e-5, 1, "full")
    Teq = nm.heat1dnt(site.k, site.kA, site.kB, site.h, site.r, site.c,
                      site.p, site.qb, site.dz, site.ip, site.dt, it,
                      np.full(mtrue.size, gst[0]), T0, 1.0, 10, 1e-5, 1, 0,
                      "full")
    assert np.abs(Teq - T0).max() < 1e-6              # constant GST: steady
    J, Tc = nm.sensfdt_pal(site.k, site.kA, site.kB, site.h, site.r, site.c,
                           site.p, site.qb, site.dz, site.ip, site.dt, it,
                           gst, T0, 1.0, 10, 1e-5, 1e-3, 1, "full")
    for i in (0, 4, mtrue.size - 1):
        g2 = gst.copy()
        g2[i] += 1e-3
        Tb = nm.heat1dnt(site.k, site.kA, site.kB, site.h, site.r, site.c,
                         site.p, site.qb, site.dz, site.ip, site.dt, it, g2,
                         T0, 1.0, 10, 1e-5, 1, 0, "full")
        assert np.abs(J[:, i] - (Tb - Tc[:, -1]) / 1e-3).max() < 1e-5  # ~ tolnl effect
    # recent GST must be seen by the near-surface temperature
    assert J[1:6, -2].max() > 0.05 and abs(J[-1, -2]) < 1e-3


def _invpar(it):
    return InvPar(it=it, m_apr_set=0.0, m_ini_set=0.0, maxiter_inv=8,
                  tol_inv=(0.0, 1e-3), start_regpar=1, modul_regpar=1,
                  regpar0=(1e-2, 1e-2, 1e-2), regbase=(1., 1., 1.),
                  reg0par=(1e-3, 1e-2), reg1par=(1e-2, 1e-1),
                  reg2par=(1e-1,), reg_opt="gcv", tol_solve=1e-8,
                  maxiter_solve=2000, dp=1e-2)


def test_tikhonov_inversion_recovers_signal():
    """Consistent weighting (weight_residual=True) recovers the true GST."""
    site, fwd, it, mtrue, fm = _synthetic_site()
    with tempfile.TemporaryDirectory() as d:
        r = tikhonov_gsth(site, fwd, _invpar(it), verbose=False, outdir=d,
                          weight_residual=True)
        assert os.path.exists(os.path.join(d, "synth_results.npz"))
        assert os.path.exists(os.path.join(d, "INFO.dat"))
    print("   rms: %.2f -> %.2f in %d iterations; max|m-mtrue| = %.3f K"
          % (r["rms_iter"][0], r["rms_iter"][-1], r["niter"],
             np.abs(r["m"] - mtrue).max()))
    assert r["rms_iter"][-1] < 0.01 * r["rms_iter"][0]
    assert np.abs(r["m"] - mtrue).max() < 0.3
    assert r["Cmm"] is not None and r["Rmm"].shape == (mtrue.size,) * 2


def test_tikhonov_matlab_weighting_needs_unit_errors():
    """Literal MATLAB rhs (unweighted residual) works only for Terr == 1."""
    site, fwd, it, mtrue, fm = _synthetic_site(noise=1.0)
    r = tikhonov_gsth(site, fwd, _invpar(it), verbose=False)
    assert r["rms_iter"][-1] < 0.6 * r["rms_iter"][0]


def test_mat_loader_roundtrip():
    from scipy.io import savemat
    site, fwd, it, *_ = _synthetic_site()
    sitepar = dict(z=site.z, t=site.t, ip=site.ip + 1, k=site.k, kA=site.kA,
                   kB=site.kB, h=site.h, r=site.r, c=site.c, p=site.p,
                   qb=site.qb, gts=site.gts, id=site.id + 1, Tobs=site.Tobs,
                   Terr=site.Terr, props="full")
    invpar = dict(pt=it + 1, maxiter_inv=5, m_apr_set=0.0, m_ini_set=0.0,
                  regpar0=np.array([1., 1., 1.]), reg_opt="gcv")
    with tempfile.TemporaryDirectory() as d:
        savemat(os.path.join(d, "s_SitePar.mat"), {"sitepar": sitepar})
        savemat(os.path.join(d, "s_InvPar.mat"), {"invpar": invpar})
        s2, f2, i2 = load_matlab_site(os.path.join(d, "s_SitePar.mat"), None,
                                      os.path.join(d, "s_InvPar.mat"))
    assert np.array_equal(s2.ip, site.ip) and np.array_equal(s2.id, site.id)
    assert np.array_equal(i2.it, it) and i2.maxiter_inv == 5


def test_mcmc_short_run():
    try:
        import gsth_mcmc as gm
        gm._prepare_pymcmcstat_import()
        from pymcmcstat.MCMC import MCMC  # noqa: F401
    except ImportError as e:
        raise SkipTest("Optional pymcmcstat not importable: %s" % e) from e
    site, fwd, it, mtrue, fm = _synthetic_site(noise=0.02)
    # coarse 4-parameter GST history + QB + H ('gauss' layout)
    z, dz = site.z, site.dz
    t, _ = nm.set_mesh(2.0e3 * Y, 1.0 * Y, 40, "log", -1)
    pt, it4, *_ = nm.set_mgsth(t, 0.0, 2.0e3 * Y, 1.0 * Y, 4, "log")
    site.t = t
    ctx = gm.McmcContext(site=site, fwd=fwd, it=it4, pom=0.0)
    mt = np.array([0.0, 0.0, 0.3, 1.0, 60.0, 1.1])     # GST(4), QB mW, H uW
    s0, r0, c0 = gm.objfun_gauss(mt, ctx)
    site.Tobs = c0 + 0.02 * np.random.default_rng(3).standard_normal(c0.size)
    params = gm.make_params(["g1", "g2", "g3", "g4", "QB", "H"],
                            [0., 0., 0., 0., 55., 1.0],
                            minimum=[-5] * 4 + [30., 0.], maximum=[5] * 4 +
                            [100., 5.], prior_mu=0.0, prior_sigma=np.inf,
                            sample=[1, 1, 1, 1, 1, 0])
    out = gm.run_mcmc(1, "test", ctx, params,
                      dict(nsimu=400, adaptint=100, updatesigma=True),
                      layout="gauss")
    ch = out["chain"]
    print("   chain %s, acceptance ok; posterior mean g4=%.2f (true 1.0)"
          % (ch.shape, ch[200:, 3].mean()))
    assert ch.shape == (400, 5) and np.isfinite(ch).all()   # H not sampled
    full = out["chain_full"]
    assert full.shape == (400, 6) and np.all(full[:, 5] == 1.0)
    calc, res, idx = gm.predict_chain(full, ctx, "gauss", nsample=5, seed=0)
    assert calc.shape == (5, site.id.size)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("%-48s" % name, end="", flush=True)
            try:
                fn()
            except SkipTest as exc:
                print(" skipped: %s" % exc)
            else:
                print(" ok")
    print("all available tests passed")
