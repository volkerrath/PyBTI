"""
mcmc_workflow.py -- Dictionary workflow for SITE_MCMC.m.

Builds the 21-step GST/Q/H parameterization used by the MATLAB template,
runs pymcmcstat's DRAM sampler through gsth_mcmc.py, and summarizes chains
for SITE_MCMCPlot/SITE_Plot diagnostics. Public inputs and outputs are plain
dictionaries; dataclasses are confined to the numerical adapter boundary.

The template's 250,000-sample DRAM setting is retained as production_nsimu.
The default nsimu is a short validation chain and must not be interpreted as
a converged posterior. The OKU-specific heat-flow prior is not a universal
default: unless supplied, qb_mean is derived from sitepar['qb'].

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 adaptation; synthetic validation only.
                 Not compared with MATLAB MCMC output.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

import gsth_drivers as gd
import gsth_mcmc as gm
from init import _numerics

__all__ = ["build_mcmc", "run_mcmc", "summarize_mcmc", "plot_mcmc"]


_METHODS = {
    "mh": dict(production_nsimu=100000, drscale=0.0, adaptint=0),
    "dr": dict(production_nsimu=200000, drscale=2.0, adaptint=0),
    "am": dict(production_nsimu=500000, drscale=0.0, adaptint=25000),
    "dram": dict(production_nsimu=250000, drscale=2.0, adaptint=10000),
}


def _adapters(sitepar, fwdpar):
    """Convert public dictionaries only at the legacy-driver boundary."""
    site = gd.SitePar(**{
        field.name: sitepar[field.name]
        for field in fields(gd.SitePar) if field.name in sitepar
    })
    settings = _numerics(fwdpar)
    fwd = gd.FwdPar(**{
        field.name: settings[field.name] for field in fields(gd.FwdPar)
    })
    return site, fwd


def _correlation(n, length, kind):
    """Parameter-index correlation matrix used for GST proposals."""
    if not np.isfinite(length) or length <= 0:
        raise ValueError("proposal correlation_length must be positive")
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / length
    if kind in ("g", "gauss", "gaussian"):
        return np.exp(-0.5 * distance**2)
    if kind in ("e", "markov", "exponential"):
        return np.exp(-distance)
    raise ValueError("proposal covariance must be 'gaussian' or 'exponential'")


def build_mcmc(sitepar: dict, fwdpar: dict, inversion: dict,
               initial: dict, mcmcpar: dict | None = None):
    """Build the SITE_MCMC.m Gaussian-layout configuration.

    Parameters
    ----------
    sitepar, fwdpar, initial : dictionaries from the current workflow.
    inversion : dictionary returned by workflow.build_inversion. Its it and
        nsteps define the same logarithmic GST parameter grid used by Tikhonov.
    mcmcpar : optional dictionary.
        method: 'mh', 'dr', 'am' or 'dram' (default).
        nsimu: samples actually run (1000 validation default).
        production_nsimu: optional override; MATLAB DRAM value is 250000.
        adaptint/drscale: sampler controls; validation defaults scale adaptint
        down to the short chain, while production values are retained.
        gst_mean/gst_sigma: 1/5 deg C; qb_mean is positive upward mW/m2 and
        defaults to abs(sitepar['qb'])*1000, qb_sigma=4 mW/m2.
        h_mean/h_sigma: 1.5/0.3 microW/m3. sample_qb/sample_h both
        default True. activate_qb=True and activate_h=False control whether
        those proposals affect the physical objective, matching Pact in
        SITE_MCMC.m. Thus default H samples its prior but is not identified
        by temperatures. cutoff=3 sets hard prior bounds.
        start_scale=0.5 draws the starting point around the prior mean.
        covariance='gaussian', correlation_length=3, seed=1.
        measurement_sigma defaults to the MATLAB value 0.1 K.
        updatesigma=True; pom=-4 K; weighted=False.
        layout must be 'gauss', matching GSTH_MCObjFunGauss.m.

    Returns
    -------
    Dictionary with params, options, priors, bounds, sampled/active masks, proposal
    covariance, likelihood variance, inversion mapping, and provenance.
    It contains no sampler object. H is present in chain_full even when fixed.

    The gauss objective recomputes a steady initial profile for each proposal
    at GST_1 + pom, as in the translated MATLAB objective. The repeated-history
    Tinit is validated and recorded but is not used by that objective.
    Observation covariance must be diagonal; pymcmcstat receives one scalar
    likelihood variance, optionally sampled when updatesigma=True.
    """
    settings = dict(mcmcpar or {})
    layout = str(settings.pop("layout", "gauss")).lower()
    if layout != "gauss":
        raise ValueError("The SITE_MCMC workflow currently requires layout='gauss'")
    it = np.asarray(inversion["it"], dtype=int).ravel()
    nsteps = int(inversion["nsteps"])
    if (it.size != len(sitepar["t"]) or np.any(it < 0)
            or np.any(it >= nsteps)):
        raise ValueError("inversion.it must map every time node to a GST parameter")
    covariance = np.asarray(sitepar["Tcov"], dtype=float)
    diagonal = np.diag(np.asarray(sitepar["Terr"], dtype=float) ** 2)
    if not np.allclose(covariance, diagonal, rtol=1e-10, atol=1e-15):
        raise ValueError("MCMC currently requires diagonal Tcov")
    Tinit = np.asarray(initial["Tinit"], dtype=float).ravel()
    if Tinit.size != len(sitepar["z"]) or not np.isfinite(Tinit).all():
        raise ValueError("Tinit must have one finite value per depth node")

    method = str(settings.pop("method", "dram")).lower()
    if method not in _METHODS:
        raise ValueError("method must be one of %s" % sorted(_METHODS))
    template = dict(_METHODS[method])
    production_nsimu = int(settings.pop(
        "production_nsimu", template["production_nsimu"]))
    nsimu = int(settings.pop("nsimu", 1000))
    if nsimu < 2 or production_nsimu < 2:
        raise ValueError("nsimu and production_nsimu must be at least two")
    default_adapt = template["adaptint"]
    if default_adapt and nsimu < production_nsimu:
        default_adapt = max(100, nsimu // 5)
    adaptint = int(settings.pop("adaptint", default_adapt))
    drscale = float(settings.pop("drscale", template["drscale"]))
    if adaptint < 0 or drscale < 0:
        raise ValueError("adaptint and drscale must be nonnegative")

    gst_mean = float(settings.pop("gst_mean", 1.0))
    gst_sigma = float(settings.pop("gst_sigma", 5.0))
    qb_mean = float(settings.pop("qb_mean", abs(float(sitepar["qb"])) * 1000.0))
    qb_sigma = float(settings.pop("qb_sigma", 4.0))
    h_mean = float(settings.pop("h_mean", 1.5))
    h_sigma = float(settings.pop("h_sigma", 0.3))
    cutoff = float(settings.pop("cutoff", 3.0))
    start_scale = float(settings.pop("start_scale", 0.5))
    sample_qb = bool(settings.pop("sample_qb", True))
    sample_h = bool(settings.pop("sample_h", True))
    activate_qb = bool(settings.pop("activate_qb", True))
    activate_h = bool(settings.pop("activate_h", False))
    seed = int(settings.pop("seed", 1))
    kind = str(settings.pop("covariance", "gaussian")).lower()
    length = float(settings.pop("correlation_length", 3.0))
    measurement_sigma = float(settings.pop("measurement_sigma", 0.1))
    updatesigma = bool(settings.pop("updatesigma", True))
    pom = float(settings.pop("pom", -4.0))
    weighted = bool(settings.pop("weighted", False))
    verbosity = int(settings.pop("verbosity", 0))
    waitbar = bool(settings.pop("waitbar", False))
    if settings:
        raise ValueError("Unknown mcmcpar fields: %s" % sorted(settings))
    scales = np.r_[np.full(nsteps, gst_sigma), qb_sigma, h_sigma]
    means = np.r_[np.full(nsteps, gst_mean), qb_mean, h_mean]
    if (not np.isfinite(means).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0) or cutoff <= 0 or start_scale < 0
            or measurement_sigma <= 0):
        raise ValueError("MCMC means/scales and measurement_sigma must be finite and valid")
    lower, upper = means - cutoff * scales, means + cutoff * scales
    active = np.r_[np.ones(nsteps, dtype=bool), activate_qb, activate_h]
    sampled = np.r_[np.ones(nsteps, dtype=bool), sample_qb, sample_h]
    rng = np.random.default_rng(seed)
    start = np.clip(means + start_scale * scales * rng.standard_normal(len(means)),
                    lower, upper)
    start[~sampled] = means[~sampled]
    names = [f"GST_{i + 1:02d}" for i in range(nsteps)] + [
        "QB_mW_m2", "H_uW_m3"]
    params = gm.make_params(
        names, start, lower, upper, means, scales, sample=sampled)
    gst_cov = _correlation(nsteps, length, kind) * gst_sigma**2
    proposal_full = np.zeros((nsteps + 2, nsteps + 2))
    proposal_full[:nsteps, :nsteps] = gst_cov
    proposal_full[nsteps, nsteps] = qb_sigma**2
    proposal_full[nsteps + 1, nsteps + 1] = h_sigma**2
    proposal = proposal_full[np.ix_(sampled, sampled)]
    options = dict(
        nsimu=nsimu, method=method, updatesigma=updatesigma, waitbar=waitbar,
        verbosity=verbosity, adaptint=adaptint, qcov=proposal)
    if drscale > 0:
        options["drscale"] = drscale
    return dict(
        layout=layout, it=it.copy(), nsteps=nsteps, params=params,
        prior_mean=means, prior_sigma=scales, minimum=lower, maximum=upper,
        start=start, active=active, sampled=sampled,
        proposal_covariance=proposal_full,
        options=options, sigma2=measurement_sigma**2, pom=pom,
        weighted=weighted, seed=seed, method=method, nsimu=nsimu,
        production_nsimu=production_nsimu,
        template_source="templates/SITE_MCMC.m",
        initial_profile_record=Tinit.copy(),
        initial_profile_used=False, gts=float(sitepar.get("gts", 0.0)),
        fixed_qb_mW_m2=abs(float(sitepar["qb"])) * 1000.0,
        fixed_h_uW_m3=np.asarray(sitepar["h"], dtype=float) * 1e6,
    )


def run_mcmc(sitepar: dict, fwdpar: dict, initial: dict, config: dict,
             runpar: dict | None = None):
    """Run one reproducible chain from a build_mcmc dictionary.

    runpar keys: job (1), name (site_MCMC), outdir (None), shuffle (False).
    The returned dictionary contains pymcmcstat results, sampled and full
    chains, variance chain, names, and the public configuration. shuffle=False
    makes job the sampler seed, matching RunMCMC.m's reproducible mode.
    """
    options = dict(runpar or {})
    unknown = set(options) - {"job", "name", "outdir", "shuffle"}
    if unknown:
        raise ValueError("Unknown MCMC run options: %s" % sorted(unknown))
    site, fwd = _adapters(sitepar, fwdpar)
    ctx = gm.McmcContext(
        site=site, fwd=fwd, it=np.asarray(config["it"], dtype=int),
        mactive=np.asarray(config["active"], dtype=bool),
        Tinit=np.asarray(initial["Tinit"], dtype=float),
        pom=float(config["pom"]), weighted=bool(config["weighted"]))
    result = gm.run_mcmc(
        int(options.get("job", 1)), options.get("name", site.name + "_MCMC"),
        ctx, config["params"], dict(config["options"]),
        layout=config["layout"], outdir=options.get("outdir"),
        shuffle=bool(options.get("shuffle", False)),
        sigma2=float(config["sigma2"]))
    result["config"] = config
    return result


def summarize_mcmc(chains: dict | list[dict], sitepar: dict, fwdpar: dict,
                   initial: dict, summarypar: dict | None = None):
    """Combine post-burn-in chains and calculate posterior predictions.

    summarypar keys: burnin (0.25 fraction or count), thin (1), nsample (100)
    forward predictions, seed (0), quantiles ((.025,.16,.5,.84,.975)),
    outdir (None), name ('site_MCMC'). Predictive profiles are calculated at
    observation depths. RMS uses each observation's Terr and divisor nobs.
    The returned acceptance_fraction is the fraction of successive stored
    sampled states that differ; it includes delayed-rejection acceptance.
    """
    runs = [chains] if isinstance(chains, dict) else list(chains)
    if not runs:
        raise ValueError("At least one MCMC chain is required")
    config = runs[0]["config"]
    for run in runs[1:]:
        if run["chain_full"].shape[1] != runs[0]["chain_full"].shape[1]:
            raise ValueError("All chains must use the same parameter layout")
    settings = dict(summarypar or {})
    burnin = settings.pop("burnin", 0.25)
    thin = int(settings.pop("thin", 1))
    nsample = int(settings.pop("nsample", 100))
    seed = int(settings.pop("seed", 0))
    quantiles = np.asarray(settings.pop(
        "quantiles", (.025, .16, .5, .84, .975)), dtype=float)
    outdir = settings.pop("outdir", None)
    name = settings.pop("name", sitepar.get("name", "site") + "_MCMC")
    if settings:
        raise ValueError("Unknown summary fields: %s" % sorted(settings))
    if thin < 1 or nsample < 1 or quantiles.shape != (5,) or np.any(
            np.diff(quantiles) <= 0) or quantiles[0] < 0 or quantiles[-1] > 1:
        raise ValueError("Invalid thin, nsample or five posterior quantiles")
    full_parts, sampled_parts, s2_parts, acceptance = [], [], [], []
    for run in runs:
        n = len(run["chain_full"])
        b = int(burnin * n) if float(burnin) < 1 else int(burnin)
        if b < 0 or b >= n:
            raise ValueError("burnin must retain at least one chain sample")
        full_parts.append(np.asarray(run["chain_full"])[b::thin])
        sampled = np.asarray(run["chain"])[b::thin]
        sampled_parts.append(sampled)
        s2_parts.append(np.asarray(run["s2chain"])[b::thin])
        if len(sampled) > 1:
            acceptance.append(np.mean(np.any(np.diff(sampled, axis=0) != 0, axis=1)))
    full = np.concatenate(full_parts)
    sampled = np.concatenate(sampled_parts)
    s2chain = np.concatenate(s2_parts)
    site, fwd = _adapters(sitepar, fwdpar)
    ctx = gm.McmcContext(
        site=site, fwd=fwd, it=np.asarray(config["it"], dtype=int),
        mactive=np.asarray(config["active"], dtype=bool),
        Tinit=np.asarray(initial["Tinit"], dtype=float),
        pom=float(config["pom"]), weighted=bool(config["weighted"]))
    predictions, residuals, indices = gm.predict_chain(
        full, ctx, config["layout"], nsample=min(nsample, len(full)),
        burnin=0, seed=seed)
    weighted_rms = np.sqrt(np.mean(
        (residuals / np.asarray(sitepar["Terr"])[None, :])**2, axis=1))
    nsteps = int(config["nsteps"])
    posterior = dict(
        gst=full[:, :nsteps], qb=full[:, nsteps], h=full[:, nsteps + 1])
    result = dict(
        name=name, config=config, chain=sampled, chain_full=full,
        s2chain=s2chain, posterior=posterior, predictions=predictions,
        residuals=residuals, weighted_rms=weighted_rms,
        prediction_indices=indices, quantiles=quantiles,
        gst_quantiles=np.quantile(posterior["gst"], quantiles, axis=0),
        qb_quantiles=np.quantile(posterior["qb"], quantiles),
        h_quantiles=np.quantile(posterior["h"], quantiles),
        prediction_quantiles=np.quantile(predictions, quantiles, axis=0),
        residual_quantiles=np.quantile(residuals, quantiles, axis=0),
        rms_quantiles=np.quantile(weighted_rms, quantiles),
        acceptance_fraction=float(np.mean(acceptance)) if acceptance else np.nan,
        burnin=burnin, thin=thin, nchain=len(runs), nsample=len(full),
        zobs=np.asarray(sitepar["zobs"]).copy(),
        Tobs=np.asarray(sitepar["Tobs"]).copy(),
        Terr=np.asarray(sitepar["Terr"]).copy(),
        t=np.asarray(sitepar["t"]).copy(),
        gts=float(sitepar.get("gts", 0.0)),
    )
    if outdir is not None:
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / f"{name}_summary.npz",
            chain=result["chain"], chain_full=full, s2chain=s2chain,
            predictions=predictions, residuals=residuals,
            weighted_rms=weighted_rms, quantiles=quantiles,
            gst_quantiles=result["gst_quantiles"],
            qb_quantiles=result["qb_quantiles"],
            h_quantiles=result["h_quantiles"],
            prediction_quantiles=result["prediction_quantiles"],
            residual_quantiles=result["residual_quantiles"])
    return result


def plot_mcmc(summary: dict, plotpar: dict | None = None):
    """Plot a summarize_mcmc dictionary using SITE_MCMCPlot panels."""
    from mcmc_plot import plot_mcmc_summary
    return plot_mcmc_summary(summary, plotpar)
