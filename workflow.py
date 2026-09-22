"""
workflow.py -- Dictionary interfaces for the borehole modelling workflow.

Public routines pass plain dictionaries corresponding to MATLAB mesh,
sitepar, fwdpar, initpar and invpar structures. The older dataclass-based
drivers remain available; conversion occurs only at numerical solver
boundaries. No files or figures are generated unless explicitly requested.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 implementation; validated with synthetic data.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

import numeric as nm
import mesh as gmesh
import gsth_drivers as gd
from init import _forcing, _transient, _numerics
from mcmc_workflow import (
    build_mcmc, run_mcmc, summarize_mcmc, plot_mcmc,
)

__all__ = ["build_mesh", "build_inversion", "run_forward", "run_inversion",
           "plot_forward", "plot_inversion", "build_mcmc", "run_mcmc",
           "summarize_mcmc", "plot_mcmc"]


def build_mesh(meshpar: dict, observations: dict | None = None):
    """Generate spatial/time meshes from a MATLAB-style parameter dict.

    meshpar keys: name ('site'), depth (kwargs for build_depth_mesh), time
    (kwargs for build_time_mesh), include_observations (True), time_nodes
    (optional additional signed time nodes in seconds), min_depth (0 m).
    The depth extent is at least min_depth, even if depth['zend'] is smaller.
    Optional observations['zobs'] are inserted as depth nodes, preserving
    observation values and count. Depths/time_nodes outside the requested
    domain raise an error. Properties must be assigned AFTER this step.

    Returns the existing mesh dict (z, dz, nz, ip, zm, t, dt, nt, it, tm,
    name). Optional time_nodes can place known GST transitions exactly on
    time nodes. No disk hand-off is required.
    """
    depth_kw = dict(meshpar.get("depth", {}))
    minimum = float(meshpar.get("min_depth", 0.0))
    requested = float(depth_kw.get("zend", 5000.0))
    if not np.isfinite(minimum) or minimum < 0 or not np.isfinite(requested):
        raise ValueError("Depth bounds must be finite and min_depth nonnegative")
    depth_kw["zend"] = max(minimum, requested)
    out = gmesh.build_meshes(
        meshpar.get("name", "site"), depth_kw=depth_kw,
        time_kw=dict(meshpar.get("time", {})), verbose=False,
    )
    if observations is not None and meshpar.get("include_observations", True):
        zobs = np.asarray(observations["zobs"], dtype=float).ravel()
        if (not np.isfinite(zobs).all() or np.any(zobs < out["z"][0])
                or np.any(zobs > out["z"][-1])):
            raise ValueError("Observation depths must be inside the depth mesh")
        z = np.unique(np.r_[out["z"], zobs])
        out.update(z=z, dz=np.diff(z), nz=z.size, ip=np.arange(z.size - 1),
                   zm=0.5 * (z[:-1] + z[1:]))
    if "time_nodes" in meshpar:
        extra = np.asarray(meshpar["time_nodes"], dtype=float).ravel()
        if (not np.isfinite(extra).all() or np.any(extra < out["t"][0])
                or np.any(extra > out["t"][-1])):
            raise ValueError("Additional time nodes must be inside the time mesh")
        t = np.unique(np.r_[out["t"], extra])
        out.update(t=t, dt=np.diff(t), nt=t.size, it=np.arange(t.size),
                   tm=0.5 * (t[:-1] + t[1:]))
    return out


def run_forward(sitepar: dict, fwdpar: dict, initial: dict):
    """Run the transient model from dictionary parameters and initial state.

    sitepar is returned by prepare_site; fwdpar holds numerical settings.
    initial is returned by build_initial (GST, it, Tinit, GST0).
    GST is absolute temperature: no sitepar['gts'] is added, matching
    fwd_gsth. Returns the same result keys as fwd_gsth. RMS and MAE use
    per-observation Terr; they are marginal-error diagnostics and do not
    account for off-diagonal covariance stored in Tcov.
    """
    GST, it = _forcing(sitepar, initial)
    T0 = np.asarray(initial["Tinit"], dtype=float).ravel()
    if T0.size != len(sitepar["z"]) or not np.isfinite(T0).all():
        raise ValueError("Tinit must contain a finite temperature per depth node")
    calc = _transient(sitepar, fwdpar, GST, it, T0, out=1)
    ids = np.asarray(sitepar["id"], dtype=int)
    residual = np.asarray(sitepar["Tobs"]) - calc[ids, -1]
    scaled = residual / np.asarray(sitepar["Terr"])
    return dict(Tcalc=calc, z=sitepar["z"], Tobs=sitepar["Tobs"],
                zobs=sitepar["zobs"], id=ids, r=residual, resid=scaled,
                rms=float(np.sqrt(np.mean(scaled**2))),
                mae=float(np.mean(np.abs(scaled))))


def build_inversion(mesh: dict, gridpar: dict, invpar: dict):
    """Map a GST inversion grid to model time nodes (SITE_Tikh.m).

    gridpar: nsteps (21), base (0), tstart (110000 yr in seconds),
    tend (30 yr in seconds), gmode ('log'). The solver uses interval-start
    assignments, exactly as numeric.set_mgsth; this does not change mesh.
    invpar supplies the fields of gsth_drivers.InvPar except it/nsteps.
    Optional diffmeth='FD' is accepted; automatic differentiation is not
    implemented. Returns a new solver-ready dictionary. GST parameters are
    relative to sitepar['gts']; the known forcing is not an inversion prior.
    Every GST parameter must affect at least one integration interval.
    """
    allowed = {"nsteps", "base", "tstart", "tend", "gmode"}
    if set(gridpar) - allowed:
        raise ValueError("Unknown inversion grid fields: %s" % sorted(set(gridpar) - allowed))
    n = gridpar.get("nsteps", 21)
    if isinstance(n, bool) or int(n) != n or n < 2:
        raise ValueError("nsteps must be an integer of at least two")
    start = float(gridpar.get("tstart", 110000 * nm.YEAR2SEC))
    end = float(gridpar.get("tend", 30 * nm.YEAR2SEC))
    if not np.isfinite([start, end]).all() or not start > end > 0:
        raise ValueError("Need finite tstart > tend > 0 in seconds")
    t = np.asarray(mesh["t"], dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Model time nodes must be finite and increasing")
    _, pointer, *_ = nm.set_mgsth(
        t, gridpar.get("base", 0.), start, end, int(n), gridpar.get("gmode", "log"))
    if np.any(np.bincount(pointer[:-1], minlength=int(n)) == 0):
        raise ValueError("Inversion grid has unused parameters; refine the time mesh or reduce nsteps")
    result = dict(invpar)
    if str(result.pop("diffmeth", "FD")).upper() != "FD":
        raise ValueError("Only finite-difference (FD) sensitivity is implemented")
    result.update(it=pointer, nsteps=int(n))
    unknown = set(result) - {f.name for f in fields(gd.InvPar)}
    if unknown:
        raise ValueError("Unsupported inversion settings: %s" % sorted(unknown))
    return result


def run_inversion(sitepar: dict, fwdpar: dict, invpar: dict, initial: dict,
                  runpar: dict | None = None):
    """Dictionary adapter for the existing Tikhonov driver.

    invpar has the fields of gsth_drivers.InvPar; it is REQUIRED and is
    the inversion parameter mapping, distinct from initial['it'].
    initial['Tinit'] supplies the fixed initial temperature profile.
    runpar optionally supplies outdir, name, verbose and n_jobs; default
    execution is serial and writes no files. Mesh and initialisation are
    never recomputed inside the inversion.
    Uses consistent residual weighting (weight_residual=True).
    Correlated Tcov is rejected: the existing solver only supports
    diagonal error weighting. No inversion configuration is inferred
    from the known synthetic climate. Returns the solver's result dict.
    """
    options = dict(runpar or {})
    unknown = set(options) - {"outdir", "name", "verbose", "n_jobs"}
    if unknown:
        raise ValueError("Unknown inversion run options: %s" % sorted(unknown))
    T0 = np.asarray(initial["Tinit"], dtype=float).ravel()
    if T0.size != len(sitepar["z"]) or not np.isfinite(T0).all():
        raise ValueError("Tinit must have one finite temperature per depth node")
    covariance = np.asarray(sitepar["Tcov"])
    diagonal = np.diag(np.asarray(sitepar["Terr"]) ** 2)
    if not np.allclose(covariance, diagonal, rtol=1e-10, atol=1e-15):
        raise ValueError("The current inversion solver requires diagonal Tcov")
    site = gd.SitePar(**{f.name: sitepar[f.name] for f in fields(gd.SitePar)
                         if f.name in sitepar})
    numerics = _numerics(fwdpar)
    fwd = gd.FwdPar(**{f.name: numerics[f.name] for f in fields(gd.FwdPar)})
    return gd.tikhonov_gsth(site, fwd, gd.InvPar(**invpar),
                            T0=T0, weight_residual=True, **options)


def plot_forward(run: dict, plotpar: dict | None = None):
    """Plot a run dict with dictionary plotting options.

    run contains name, site, mesh, init, result (as in run_fwd), with
    dictionary values for site and init. plotpar supplies plot_fwd's
    options such as outdir, show, time_scale and show_initial.
    Returns dict(figure, axes, filename).
    """
    from fwd_plot import plot_fwd

    return plot_fwd(run, **dict(plotpar or {}))


def plot_inversion(runs: dict | list[dict], plotpar: dict | None = None):
    """Plot one or more Tikhonov result dictionaries using SITE_TikhPlot panels.

    See tikh_plot.plot_tikhonov for plotting options and returned diagnostics.
    """
    from tikh_plot import plot_tikhonov
    return plot_tikhonov(runs, plotpar)
