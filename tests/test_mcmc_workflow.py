"""
test_mcmc_workflow.py -- Dictionary MCMC workflow and result plotting.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 synthetic validation; not MATLAB comparison.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from init import YEAR2SEC, build_initial
from prep import prepare_site
from workflow import (
    build_mesh, build_inversion, build_mcmc, run_mcmc, summarize_mcmc,
    plot_mcmc,
)


def _case():
    observations = dict(
        zobs=np.array([50., 100.]), Tobs=np.array([7., 8.]),
        Terr=np.full(2, .1), Tcov=np.eye(2) * .1**2,
    )
    mesh = build_mesh(
        dict(depth=dict(ztype="read", zmesh_in=[0., 50., 100., 150.], zend=200.),
             time=dict(ttype="read", tmesh_in=np.linspace(-1000., 0., 11) * YEAR2SEC)),
        observations,
    )
    site = prepare_site(
        dict(name="mcmc_test", props="const", k=3., r=2700., c=800.,
             qb=-.06, p=0., h=0.), mesh, observations,
    )
    fwd = dict(freeze=0)
    initial = build_initial(site, fwd, dict(GST0=6.))
    inverse = build_inversion(
        mesh, dict(nsteps=2, tstart=1000 * YEAR2SEC, tend=100 * YEAR2SEC),
        dict(diffmeth="FD"),
    )
    return mesh, site, fwd, initial, inverse


def test_build_mcmc_preserves_matlab_layout_and_site_prior():
    _, site, fwd, initial, inverse = _case()
    config = build_mcmc(site, fwd, inverse, initial, dict(
        nsimu=40, adaptint=10, seed=7, measurement_sigma=.1))
    assert config["layout"] == "gauss" and config["nsteps"] == 2
    np.testing.assert_array_equal(config["active"], [True, True, True, False])
    np.testing.assert_array_equal(config["sampled"], [True, True, True, True])
    assert config["prior_mean"][2] == pytest.approx(60.)
    assert config["production_nsimu"] == 250000
    assert config["options"]["method"] == "dram"
    assert config["proposal_covariance"].shape == (4, 4)
    assert config["params"][3]["sample"] is True
    assert config["initial_profile_used"] is False
    correlated = dict(site, Tcov=site["Tcov"] + .001)
    with pytest.raises(ValueError, match="diagonal Tcov"):
        build_mcmc(correlated, fwd, inverse, initial)


def test_dictionary_dram_summary_and_plot(tmp_path):
    _, site, fwd, initial, inverse = _case()
    config = build_mcmc(site, fwd, inverse, initial, dict(
        nsimu=40, adaptint=10, seed=4, measurement_sigma=.1,
        gst_sigma=2., qb_sigma=2.))
    chain = run_mcmc(
        site, fwd, initial, config,
        dict(job=4, name="dictionary", outdir=tmp_path),
    )
    assert chain["chain"].shape == (40, 4)
    assert chain["chain_full"].shape == (40, 4)
    summary = summarize_mcmc(
        chain, site, fwd, initial,
        dict(burnin=.25, thin=2, nsample=8, seed=3,
             outdir=tmp_path, name="dictionary"),
    )
    assert summary["chain_full"].shape == (15, 4)
    assert summary["predictions"].shape == (8, 2)
    assert summary["gst_quantiles"].shape == (5, 2)
    assert np.isfinite(summary["weighted_rms"]).all()
    assert 0 <= summary["acceptance_fraction"] <= 1
    plotted = plot_mcmc(
        summary, dict(outdir=tmp_path, name="dictionary", dpi=40,
                      reference=dict(t=site["t"], GST=np.array([6., 6.]),
                                     it=inverse["it"])),
    )
    assert set(plotted["axes"]) == {
        "gsth", "temperature", "residuals", "heat_flow",
        "heat_production", "trace", "rms", "variance"
    }
    assert plotted["filenames"][0].read_bytes().startswith(b"\x89PNG")
    plt.close(plotted["figure"])
