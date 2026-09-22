"""
gsth_mcmc.py -- Bayesian (MCMC) GSTH inversion with pymcmcstat.

Translated from: GSTH_MCObjFun.m / GSTH_MCFwdFun.m ("basic" layout),
GSTH_MCObjFunGauss.m ("gauss" layout), GSTH_MCObjFunGaussJoint.m ("joint"
layout), RunMCMC.m and mcmc_wrapper.m. The MATLAB code called mcmcrun /
mcmcrun2 of MCMCSTAT (Haario, Laine); these are replaced by the Python
package pymcmcstat (Miles 2019), which implements the same DRAM sampler and
is imported lazily inside run_mcmc().

Parameter layouts of the model vector m
---------------------------------------
basic : [GST_1..GST_n, QB]                    QB in W/m^2 (as is, signed)
gauss : [GST_1..GST_n, QB, H]                 QB in mW/m^2 (positive; used as
        QB_model = -QB*1e-3), H in uW/m^3 (uniform heat production in all
        units, H_model = H*1e-6). Initial state: stationary solution for
        GST_1 + pom (pom = -4 K, hard-coded in the MATLAB file).
joint : [GST_1..GST_n, k_1..k_nunits, QB, H]  as gauss, plus the unit
        conductivities. Initial state: Tinit if given, else stationary for
        GST_1.
`mactive` (bool per parameter) works as in MATLAB: an inactive QB / H / k
block is replaced by the site value (for k the block counts as active only
if ALL its entries are active, as MATLAB's `if mactive(a:b)`).

sum of squares: s = sum(r^2), r = Tobs - Tcalc(id, final time) (unweighted,
as MATLAB; set ctx.weighted=True to use r/Terr). Error variance is handled
by pymcmcstat (updatesigma).

References
----------
Miles, P. R., pymcmcstat: A Python Package for Bayesian Inference Using
    Delayed Rejection Adaptive Metropolis, Journal of Open Source Software,
    2019, 4(38), 1417, doi:10.21105/joss.01417 (DOI as printed on the
    provided PDF; the file name says 2029, the article is dated 2019).
Haario, H.; Laine, M.; Mira, A. & Saksman, E., DRAM: Efficient adaptive
    MCMC, Statistics and Computing, 2006, 16(4), 339-354,
    doi:10.1007/s11222-006-9438-0 (DOI as in the reference list of the PDF).
Haario, H.; Saksman, E. & Tamminen, J., An adaptive Metropolis algorithm,
    Bernoulli, 2001, 7(2), 223-242 (link in the PDF: projecteuclid.org/
    euclid.bj/1080222083; no DOI given there - NOT VERIFIED).

Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-21
Status         : AI-generated translation; the sampler wrapper was run on a
                 small synthetic problem only. Not compared with MATLAB
                 output. Review before production use.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

import numeric as nm
from gsth_drivers import FwdPar, SitePar

__all__ = [
    "McmcContext",
    "objfun_basic",
    "objfun_gauss",
    "objfun_joint",
    "fwdfun",
    "make_ssfun",
    "make_params",
    "run_mcmc",
    "RunMCMC",
    "mcmc_wrapper",
    "run_jobs",
    "predict_chain",
]


def _apply_pymcmcstat_compat():
    """Runtime shim for pymcmcstat 1.9.1 with NumPy >= 2.

    Its error-variance update (used when updatesigma=True) assigns arrays
    into scalar slots, which recent NumPy rejects. The shim replaces
    ErrorVarianceEstimator.update_error_variance by the same draw,
        sigma2 = 1 / Gamma(shape=(N0+N)/2, scale=2/(N0*S20+SS)),
    using numpy.random.gamma (the original uses a Marsaglia-Tsang sampler for
    the same distribution; random streams therefore differ). Idempotent; the
    pymcmcstat sources are not modified.
    """
    from pymcmcstat.procedures.ErrorVarianceEstimator import (
        ErrorVarianceEstimator as E,
    )

    if getattr(E, "_gsth_compat", False):
        return

    def _f(x):
        return float(np.asarray(x, dtype=float).reshape(-1)[0])

    def update_error_variance(self, sos, model):
        N0, S20, N = model.N0, model.S20, model.N
        sigma2 = model.sigma2
        sos = np.atleast_1d(sos)
        for jj in range(len(sos)):
            a = 0.5 * (_f(N0[jj]) + _f(N[jj]))
            if a <= 0:
                sigma2[jj] = np.inf
                continue
            scale = 2.0 / (_f(N0[jj]) * _f(S20[jj]) + _f(sos[jj]))
            sigma2[jj] = 1.0 / np.random.gamma(a, scale)
        return sigma2

    E.update_error_variance = update_error_variance
    E._gsth_compat = True


@dataclass
class McmcContext:
    """Everything the objective functions need (MATLAB `data` / `info`)."""

    site: SitePar
    fwd: FwdPar
    it: np.ndarray  # 0-based GST index per time node
    mactive: Optional[np.ndarray] = None
    Tinit: Optional[np.ndarray] = None
    pom: float = -4.0  # 'gauss' layout initial offset [K]
    weighted: bool = False

    def active(self, n):
        if self.mactive is None:
            return np.ones(n, dtype=bool)
        a = np.asarray(self.mactive, dtype=bool).ravel()
        if a.size != n:
            raise ValueError(
                "mactive has %d entries, model has %d" % (a.size, n)
            )
        return a


def _finish(ctx, Tfinal):
    site = ctx.site
    c = Tfinal[site.id]
    r = site.Tobs - c
    s = float(np.sum((r / site.Terr) ** 2 if ctx.weighted else r**2))
    return s, r, c


def _transient(ctx, k, h, QB, GST, T0):
    s, f = ctx.site, ctx.fwd
    return nm.heat1dnt(
        k,
        s.kA,
        s.kB,
        h,
        s.r,
        s.c,
        s.p,
        QB,
        s.dz,
        s.ip,
        s.dt,
        ctx.it,
        GST,
        T0,
        f.theta,
        f.maxitnl,
        f.tolnl,
        f.freeze,
        0,
        s.props,
        **f.kw,
    )


def _steady(ctx, k, h, QB, Ts):
    s, f = ctx.site, ctx.fwd
    return nm.heat1dns(
        k,
        s.kA,
        s.kB,
        h,
        s.r,
        s.p,
        QB,
        Ts,
        s.dz,
        s.ip,
        f.maxitnl,
        f.tolnl,
        f.freeze,
        s.props,
        **nm._ns_kw(f.kw),
    )


def objfun_basic(m, ctx: McmcContext):
    """GSTH_MCObjFun / GSTH_MCFwdFun. Returns (s, r, c)."""
    m = np.asarray(m, dtype=float).ravel()
    site, f = ctx.site, ctx.fwd
    GST, QB = m[:-1] + site.gts, m[-1]
    T0 = (
        ctx.Tinit
        if ctx.Tinit is not None
        else _steady(ctx, site.k, site.h, QB, GST[0])
    )
    return _finish(ctx, _transient(ctx, site.k, site.h, QB, GST, T0))


def objfun_gauss(m, ctx: McmcContext):
    """GSTH_MCObjFunGauss. Returns (s, r, c)."""
    m = np.asarray(m, dtype=float).ravel()
    site = ctx.site
    n = m.size
    act = ctx.active(n)
    GST = m[: n - 2] + site.gts
    QB = -m[n - 2] * 1e-3 if act[n - 2] else site.qb
    H = m[n - 1] * 1e-6 * np.ones_like(site.h) if act[n - 1] else site.h
    T0 = _steady(ctx, site.k, H, QB, GST[0] + ctx.pom)
    return _finish(ctx, _transient(ctx, site.k, H, QB, GST, T0))


def objfun_joint(m, ctx: McmcContext):
    """GSTH_MCObjFunGaussJoint. Returns (s, r, c)."""
    m = np.asarray(m, dtype=float).ravel()
    site = ctx.site
    n = m.size
    ncnd = site.k.size
    ngst = n - (ncnd + 2)
    act = ctx.active(n)
    GST = m[:ngst] + site.gts
    KB = m[ngst : ngst + ncnd] if act[ngst : ngst + ncnd].all() else site.k
    QB = -m[n - 2] * 1e-3 if act[ngst + ncnd] else site.qb
    H = m[n - 1] * 1e-6 * np.ones_like(site.h) if act[n - 1] else site.h
    T0 = (
        ctx.Tinit if ctx.Tinit is not None else _steady(ctx, KB, H, QB, GST[0])
    )
    return _finish(ctx, _transient(ctx, KB, H, QB, GST, T0))


_LAYOUTS = {
    "basic": objfun_basic,
    "gauss": objfun_gauss,
    "joint": objfun_joint,
}


def fwdfun(m, ctx, layout="basic"):
    """Model function of GSTH_MCFwdFun: (d, r) = (calculated data, residual)."""
    _, r, c = _LAYOUTS[layout](m, ctx)
    return c, r


def make_ssfun(ctx: McmcContext, layout="gauss"):
    """Sum-of-squares function for pymcmcstat: ssfun(theta, data).
    Failed / non-finite forward runs return inf (proposal rejected)."""
    fun = _LAYOUTS[layout]

    def ssfun(theta, data=None, custom=None):
        try:
            s = fun(np.asarray(theta, dtype=float), ctx)[0]
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return np.inf
        return s if np.isfinite(s) else np.inf

    return ssfun


def make_params(
    names,
    theta0,
    minimum=None,
    maximum=None,
    prior_mu=None,
    prior_sigma=None,
    sample=None,
):
    """List of parameter dicts for run_mcmc (MATLAB: params cell array
    {name, init, min, max, mu, sigma}). Scalars are broadcast."""
    n = len(theta0)

    def vec(v, d):
        if v is None:
            return [d] * n
        return list(np.broadcast_to(np.asarray(v, dtype=float), (n,)))

    mn, mx = vec(minimum, -np.inf), vec(maximum, np.inf)
    mu, sg = vec(prior_mu, 0.0), vec(prior_sigma, np.inf)
    sm = (
        [True] * n
        if sample is None
        else list(np.broadcast_to(np.asarray(sample, dtype=bool), (n,)))
    )
    return [
        dict(
            name=str(names[i]),
            theta0=float(theta0[i]),
            minimum=mn[i],
            maximum=mx[i],
            prior_mu=mu[i],
            prior_sigma=sg[i],
            sample=bool(sm[i]),
        )
        for i in range(n)
    ]


def run_mcmc(
    job,
    name,
    ctx: McmcContext,
    params,
    options=None,
    layout="gauss",
    outdir=None,
    shuffle=False,
    sigma2=None,
    N0=None,
    S20=None,
):
    """Run one DRAM chain (RunMCMC.m; shuffle=True -> mcmc_wrapper.m).

    job     : job number, part of the output name and (shuffle=False) the
              random seed
    params  : list of dicts from make_params
    options : dict of pymcmcstat simulation options, e.g.
              dict(nsimu=20000, method='dram', updatesigma=True)
    outdir  : if given, <outdir>/<name>_<rand>_job<job>_final.npz is written
    Returns dict(results=pymcmcstat results dict, chain (sampled parameters
    only, as pymcmcstat), chain_full (all parameters; fixed ones at their
    theta0 - use this for predict_chain), s2chain, names, namei, mcstat).
    """
    try:
        from pymcmcstat.MCMC import MCMC
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "run_mcmc needs pymcmcstat (pip install " "pymcmcstat): %s" % e
        )
    _apply_pymcmcstat_compat()
    seed = None if shuffle else int(job)
    rid = int(np.random.default_rng(seed).integers(1, 100001))
    namei = "%s_%d_job%d" % (name, rid, job)
    print(">>>>>> %s" % namei)

    mc = MCMC(rngseed=seed)
    site = ctx.site
    mc.data.add_data_set(
        np.asarray(
            site.zobs if site.zobs is not None else site.id, dtype=float
        ).reshape(-1, 1),
        site.Tobs.reshape(-1, 1),
        user_defined_object=ctx,
    )
    for p in params:
        mc.parameters.add_model_parameter(**p)
    opts = dict(
        nsimu=5000, method="dram", updatesigma=True, waitbar=False, verbosity=1
    )
    opts.update(options or {})
    mc.simulation_options.define_simulation_options(**opts)
    ms = dict(sos_function=make_ssfun(ctx, layout))
    for key, val in (("sigma2", sigma2), ("N0", N0), ("S20", S20)):
        if val is not None:
            ms[key] = val
    mc.model_settings.define_model_settings(**ms)
    mc.run_simulation()
    res = mc.simulation_results.results
    chain = np.asarray(res["chain"])
    full = np.tile(
        np.array([q["theta0"] for q in params], dtype=float),
        (chain.shape[0], 1),
    )
    full[:, np.asarray(res["parind"], dtype=int)] = chain
    out = dict(
        results=res,
        chain=chain,
        chain_full=full,
        s2chain=np.asarray(res["s2chain"]),
        names=list(res["names"]),
        namei=namei,
        mcstat=mc,
    )
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        np.savez(
            os.path.join(outdir, namei + "_final.npz"),
            chain=out["chain"],
            chain_full=full,
            s2chain=out["s2chain"],
            sschain=np.asarray(res.get("sschain", [])),
            names=np.array(out["names"]),
            job=job,
        )
    return out


def RunMCMC(job, name, ctx, params, options=None, **kw):
    """RunMCMC.m (fixed seed = job)."""
    return run_mcmc(job, name, ctx, params, options, shuffle=False, **kw)


def mcmc_wrapper(job, name, ctx, params, options=None, **kw):
    """mcmc_wrapper.m (random seed, rng('shuffle'))."""
    return run_mcmc(job, name, ctx, params, options, shuffle=True, **kw)


def _job(args):
    job, name, ctx, params, options, kw = args
    r = run_mcmc(job, name, ctx, params, options, **kw)
    return dict(
        job=job,
        chain=r["chain"],
        s2chain=r["s2chain"],
        names=r["names"],
        namei=r["namei"],
    )


def run_jobs(jobs, name, ctx, params, options=None, n_jobs=1, **kw):
    """Run several independent chains (one per job number), optionally in
    parallel processes. Returns a list of dicts (chain, s2chain, names)."""
    tasks = [(j, name, ctx, params, options, kw) for j in jobs]
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            return list(ex.map(_job, tasks))
    return [_job(a) for a in tasks]


def predict_chain(
    chain, ctx, layout="gauss", nsample=100, burnin=0.5, seed=None
):
    """Model predictions for a random thinned sample of the chain, which
    must contain ALL model parameters (use run_mcmc(...)["chain_full"]).
    (Python
    replacement of mcmcrun2's `calchain`). burnin: fraction (<1) or number
    of samples to discard. Returns (calc (nsample, nobs), res, idx)."""
    chain = np.asarray(chain, dtype=float)
    b = int(burnin * len(chain)) if burnin < 1 else int(burnin)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(b, len(chain)), size=nsample, replace=True)
    calc, res = [], []
    for i in idx:
        c, r = fwdfun(chain[i], ctx, layout)
        calc.append(c)
        res.append(r)
    return np.array(calc), np.array(res), idx
