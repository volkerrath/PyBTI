"""
test_gsth_mcmc.py -- Focused checks for the translated MCMC driver.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 synthetic validation; not MATLAB comparison.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gsth_mcmc as gm
from gsth_drivers import FwdPar, SitePar


def _context():
    z = np.array([0., 50., 100., 150.])
    t = np.linspace(-1000., 0., 11) * 365.25 * 24 * 3600
    site = SitePar(
        z=z, t=t, ip=np.arange(3), k=np.full(3, 3.),
        kA=np.zeros(3), kB=np.zeros(3), h=np.zeros(3),
        r=np.full(3, 2700.), c=np.full(3, 800.), p=np.zeros(3),
        qb=-.06, gts=0., id=np.array([1, 2]),
        Tobs=np.array([7., 8.]), Terr=np.full(2, .1),
        zobs=np.array([50., 100.]), props="const", name="mcmc_test",
    )
    return gm.McmcContext(
        site=site, fwd=FwdPar(freeze=0),
        it=np.r_[np.zeros(5, dtype=int), np.ones(6, dtype=int)],
        mactive=np.array([True, True, True, False]), pom=-4.,
    )


def test_pymcmcstat_dram_smoke():
    ctx = _context()
    params = gm.make_params(
        ["GST_1", "GST_2", "QB_mW_m2", "H_uW_m3"],
        [1., 1., 60., 1.5], [-10., -10., 40., .5],
        [10., 10., 80., 2.5], [1., 1., 60., 1.5],
        [5., 5., 4., .3], sample=[True, True, True, False],
    )
    result = gm.run_mcmc(
        3, "smoke", ctx, params,
        dict(nsimu=30, method="dram", updatesigma=True, adaptint=10,
             qcov=np.diag([1., 1., 1.]), drscale=2.,
             waitbar=False, verbosity=0),
        sigma2=.1**2,
    )
    assert result["chain"].shape == (30, 3)
    assert result["chain_full"].shape == (30, 4)
    np.testing.assert_array_equal(result["chain_full"][:, 3], 1.5)
    assert np.isfinite(result["chain"]).all()
    assert np.isfinite(result["s2chain"]).all()
