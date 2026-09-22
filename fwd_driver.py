"""
fwd_driver.py -- Top-level forward-model run for one borehole (SITE_Fwd.m).

SITE_Fwd.m is a pure orchestrator: it sets the forward-model numerics,
builds the depth/time meshes, builds the site's physical model and
observations, builds the GST initial values, and finally runs the
transient forward model and reports the residual statistics. In the
MATLAB workflow every step but the last hands its result to the next one
through a .mat file (name_FwdPar.mat, name_Mesh_in.mat, name_SitePar.mat,
name_Init.mat); here they are plain function calls and dicts.

The last step (Fwd_gsth.m) is gsth_drivers.fwd_gsth; mesh generation
is mesh.build_meshes. This legacy driver retains its prep_fn / init_fn
callback interface and dataclass containers. The new dictionary-based
workflow uses prep.py, init.py and workflow.py directly; see WORKFLOW.md
and examples/A/run_forward.py. Its routines are not drop-in replacements
for this driver's older callback signatures.

"SITE" -> a real borehole name: nothing here is hard-coded to a
particular site. `name` is the borehole label used for output file
naming (MATLAB: the *_FwdPar.mat / *_SitePar.mat / ... prefix); `props`
is the *physical property set* (a key of phys.SITE_PROPS, e.g. "full",
"oku", "ull", or your own site's props module translated the same way) --
the two are independent, and the MATLAB scripts are not always consistent
about which variable (`site` or `props`) carries which meaning.

Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-22
Status         : AI-generated translation of the user's MATLAB code
                 (SITE_Fwd.m); the orchestration and the mesh step have
                 been run end-to-end only against a synthetic prep_fn /
                 init_fn (see the __main__ block below), NOT against real
                 borehole data or MATLAB output. Review before production
                 use.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

import gsth_drivers as gd
import mesh as gmesh
import numeric as nm

__all__ = ["run_fwd"]


def run_fwd(
    name: str,
    props,
    prep_fn: Callable,
    init_fn: Callable,
    mesh_kw: Optional[dict] = None,
    fwd_kw: Optional[dict] = None,
    prep_kw: Optional[dict] = None,
    init_kw: Optional[dict] = None,
    outdir: str = ".",
    save_intermediate: bool = False,
    verbose: bool = True,
):
    """Forward run for one borehole (SITE_Fwd.m).

    Parameters
    ----------
    name : borehole label used for output file naming (MATLAB `name`,
        e.g. site + prepstr).
    props : phys.py property set for this borehole -- a key of
        phys.SITE_PROPS, or a custom mapping (see numeric.PropModel).
    prep_fn : callable(name, mesh, props, **prep_kw) -> gsth_drivers.SitePar
        Legacy site-preparation callback: reads the
        borehole's temperature/conductivity logs, averages them onto the
        depth mesh, and returns the fully populated SitePar (k, kA, kB, h,
        r, c, p, qb, Tobs, id, Terr, ...).
    init_fn : callable(name, site, fwd, mesh, **init_kw) -> dict
        Legacy initial-value callback: builds the
        GST history prior (equilibrium, step or point-interpolated GSTH)
        and the corresponding initial temperature profile by iterating
        heat1dnt to a self-consistent state. Must return a dict with keys
            GST   : GST value at every time node in mesh['t'] (MATLAB
                    `GST`, length mesh['nt'])
            Tinit : initial temperature profile (MATLAB `Tinit` = final
                    T0 of the fixed-point iteration; length mesh['nz'])
            it    : (optional) 0-based GST index per time step; defaults
                    to mesh['it'] (the identity mapping used in
                    SITE_Init.m, appropriate when GST has one entry per
                    time node).
    mesh_kw, prep_kw, init_kw : forwarded as **kwargs to build_meshes /
        prep_fn / init_fn (MATLAB: the *_Mesh_in / *_Prep_in / *_Init_in
        "defaults overwritten" files).
    fwd_kw : overrides for FwdPar (theta, maxitnl, tolnl, freeze). MATLAB
        also carries `relaxnl`, which is not used by any function this
        calls (heat1dns/heat1dnt take no such argument in the MATLAB
        sources either); it is accepted here for parity and stored in
        fwd.kw["relaxnl"], but has no numerical effect.
    outdir : directory for fwd_gsth's optional INFO.dat / results output,
        and for the intermediate mesh .npz files if save_intermediate.
    save_intermediate : also write name_DepthGrid.npz / name_TimeGrid.npz
        (mesh.save_meshes), for parity with the MATLAB file hand-off.

    Returns
    -------
    dict(name, fwd, mesh, site, init, result) -- `result` is the dict
    returned by gsth_drivers.fwd_gsth (Tcalc, z, Tobs, zobs, r, resid, id,
    rms, mae).
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- forward-model numerics (MATLAB: fwdpar = mstruct(theta,
    #      maxitnl, tolnl, relaxnl, freeze); saved to name_FwdPar.mat) ----
    fkw = dict(theta=1.0, maxitnl=4, tolnl=1.0e-5, freeze=1)
    fkw.update(fwd_kw or {})
    relaxnl = fkw.pop("relaxnl", 1.0)  # kept only for the record;
    kw_extra = fkw.pop("kw", {})  # NOT forwarded to heat1dnt
    fwd = gd.FwdPar(
        theta=fkw["theta"],
        maxitnl=fkw["maxitnl"],
        tolnl=fkw["tolnl"],
        freeze=fkw["freeze"],
        kw=kw_extra,
    )
    fwd.relaxnl_unused = relaxnl  # documents the MATLAB field

    # ---- meshes (SITE_Mesh.m) ----
    if verbose:
        print(" generate meshes for %s" % name)
    mesh = gmesh.build_meshes(name, verbose=verbose, **(mesh_kw or {}))
    if save_intermediate:
        gmesh.save_meshes(name, mesh, outdir=outdir)

    # ---- physical model + observations (SITE_Prep.m) ----
    if verbose:
        print(" generate model for %s" % name)
    site = prep_fn(name, mesh, props, **(prep_kw or {}))

    # ---- GSTH prior + initial temperature profile (SITE_Init.m) ----
    if verbose:
        print(" generate initial values for %s" % name)
    init = init_fn(name, site, fwd, mesh, **(init_kw or {}))
    GST = np.asarray(init["GST"], dtype=float).ravel()
    Tinit = init.get("Tinit")
    it = np.asarray(init.get("it", mesh["it"]), dtype=int)

    # ---- forward model + residuals (Fwd_gsth.m) ----
    result = gd.fwd_gsth(
        site, fwd, GST, it, Tinit=Tinit, outdir=outdir, verbose=verbose
    )

    return dict(
        name=name, fwd=fwd, mesh=mesh, site=site, init=init, result=result
    )


# ---------------------------------------------------------------------------
# self-test: end-to-end run with a synthetic prep_fn / init_fn, so this
# driver is verified to work today, independently of the dictionary workflow.
# ---------------------------------------------------------------------------
def _demo_prep_fn(name, mesh, props, qb=-0.045, gts=0.0, noise=0.05, seed=0):
    """Stand-in for SITE_Prep.m: builds a SitePar with a uniform synthetic
    two-layer model on the given mesh and fabricated observations, purely
    to exercise run_fwd(). Not a translation of any MATLAB file."""
    rng = np.random.default_rng(seed)
    z, ip = mesh["z"], mesh["ip"]
    zc = mesh["zm"]
    nunits = int(ip.max()) + 1
    unit = np.where(zc < 0.5 * z[-1], 0, 1) if nunits == 1 else ip
    nu = int(unit.max()) + 1
    k = np.linspace(2.5, 3.0, nu)
    kA = np.full(nu, 0.7)
    kB = np.full(nu, 770.0)
    h = np.full(nu, 1.0e-6)
    r = np.full(nu, 2700.0)
    c = np.full(nu, 800.0)
    p = np.full(nu, 0.02)
    id_ = np.arange(5, z.size - 1, max(1, z.size // 40))
    site = gd.SitePar(
        z=z,
        t=mesh["t"],
        ip=unit,
        k=k,
        kA=kA,
        kB=kB,
        h=h,
        r=r,
        c=c,
        p=p,
        qb=qb,
        gts=gts,
        id=id_,
        Tobs=np.zeros(id_.size),
        Terr=np.full(id_.size, noise),
        zobs=z[id_],
        props=props,
        name=name,
    )
    return site


def _demo_init_fn(name, site, fwd, mesh, gst_amplitude=1.5):
    """Stand-in for SITE_Init.m: a smooth synthetic GSTH (a couple of
    slow oscillations plus recent warming) and the initial profile from
    iterating heat1dnt to a self-consistent state, as in SITE_Init.m's
    init_type='p' branch."""
    t = mesh["t"]
    y2s = gmesh.YEAR2SEC
    GST = gst_amplitude * np.sin(2 * np.pi * t / (2.0e4 * y2s)) + np.where(
        t > -200 * y2s, (t + 200 * y2s) / (200 * y2s) * 2.0, 0.0
    )
    T0 = nm.heat1dns(
        site.k,
        site.kA,
        site.kB,
        site.h,
        site.r,
        site.p,
        site.qb,
        GST[0],
        site.dz,
        site.ip,
        fwd.maxitnl,
        fwd.tolnl,
        fwd.freeze,
        site.props,
    )
    it = mesh["it"]
    for _ in range(10):
        Tcalc = nm.heat1dnt(
            site.k,
            site.kA,
            site.kB,
            site.h,
            site.r,
            site.c,
            site.p,
            site.qb,
            site.dz,
            site.ip,
            site.dt,
            it,
            GST,
            T0,
            fwd.theta,
            fwd.maxitnl,
            fwd.tolnl,
            fwd.freeze,
            0,
            site.props,
        )
        T0 = Tcalc
    return dict(GST=GST, Tinit=T0, it=it)


if __name__ == "__main__":
    # First pass: fabricated Tobs=0, just to exercise the full pipeline.
    out = run_fwd(
        "DEMO",
        "full",
        _demo_prep_fn,
        _demo_init_fn,
        mesh_kw=dict(
            depth_kw=dict(zend=2000.0, nz=201),
            time_kw=dict(
                tstart=20000 * gmesh.YEAR2SEC,
                tend=1.0 * gmesh.YEAR2SEC,
                nt=121,
            ),
        ),
        outdir="/tmp/gsth_demo",
    )
    r = out["result"]
    print(
        "pass 1 (Tobs=0, sanity run)  rms = %.3f, mae = %.3f, n_obs = %d"
        % (r["rms"], r["mae"], r["id"].size)
    )

    # Second pass: self-consistency check -- put the pass-1 forward
    # temperatures back in as "observations" and confirm the residual
    # collapses to ~0 (verifies fwd_gsth's own bookkeeping, not the
    # physics against MATLAB or real data).
    out["site"].Tobs = r["Tcalc"][r["id"], -1].copy()
    r2 = gd.fwd_gsth(
        out["site"],
        out["fwd"],
        out["init"]["GST"],
        out["init"]["it"],
        Tinit=out["init"]["Tinit"],
        verbose=False,
    )
    print(
        "pass 2 (self-consistency)    rms = %.2e, mae = %.2e"
        % (r2["rms"], r2["mae"])
    )
