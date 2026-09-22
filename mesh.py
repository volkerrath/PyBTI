"""
mesh.py -- Spatial and temporal mesh generation for one borehole site.

Translated from SITE_Mesh.m. In the MATLAB workflow this template was
copied and renamed per borehole (e.g. OKU_Mesh.m, ULL_Mesh.m), with the
site name hard-coded inside. Here it is a single, name-agnostic function
pair: pass the borehole name as a plain string argument to build_meshes()
instead of writing a new *_Mesh.m file per site.

Only the numerical mesh construction is translated. Plotting (set_graphpars,
plotit) is not: it belongs with the *_Plot scripts. MATLAB's file-based
hand-off (name_DepthGrid.mat / name_TimeGrid.mat) is replaced by returning
a plain dict; save_meshes() below writes an equivalent .npz pair if a
file hand-off is wanted.

Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-22
Status         : AI-generated translation of the user's MATLAB code
                 (SITE_Mesh.m); not yet checked against MATLAB output.
                 Review before production use.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

import numeric as nm

YEAR2SEC = getattr(nm, "YEAR2SEC", 3600.0 * 24 * 365.25)

__all__ = [
    "build_depth_mesh",
    "build_time_mesh",
    "build_meshes",
    "save_meshes",
    "YEAR2SEC",
]


# ---------------------------------------------------------------------------
# spatial mesh (SITE_Mesh.m, "set_z" branch)
# ---------------------------------------------------------------------------
def build_depth_mesh(
    ztype="log",
    zstart=0.0,
    zend=5000.0,
    zlmax=2400.0,
    nz=301,
    dzstart=10.0,
    gfac=1.005,
    ngen=999,
    zmesh_in=None,
):
    """Spatial (depth) mesh.

    ztype
      'log'             geometric growth from 0: dz_n = dzstart*gfac**n,
                        n = 1..ngen, clipped to [0, zend]. (default branch)
      'special'/'mixed': linear part [zstart, zlmax] with nz points, then
                        geometric growth beyond zlmax. As in the MATLAB
                        source, this branch OVERRIDES dzstart/gfac/ngen
                        with its own fixed values (8.0, 1.02, 500) rather
                        than using the arguments passed in.
      'read'/'inp'      take z from `zmesh_in` (array), then clip/extend
                        to zend exactly as the other branches.

    Returns dict(z, dz, nz, ip, zm).

    ip is the trivial 0-based cell-index array [0 .. nz-2] -- as in
    SITE_Mesh.m, this is NOT yet the cell -> property-unit pointer used by
    gsth_drivers.SitePar.ip; that mapping is (re)built in the site
    preparation step (SITE_Prep.m -> prep.py, not yet translated).
    """
    zt = str(ztype).lower()
    if zt in ("special", "mixed"):
        z1 = np.linspace(zstart, zlmax, int(nz))
        dzs, gf, ng = 8.0, 1.02, 500  # fixed in MATLAB for this branch
        dzn = dzs * gf ** np.arange(1, ng + 1)
        z2 = np.cumsum(dzn)
        z = np.concatenate([z1, zlmax + z2])
        z = z[z < zend]
        z = np.append(z, zend)
    elif zt in ("read", "inp"):
        if zmesh_in is None:
            raise ValueError("build_depth_mesh: ztype='read' needs zmesh_in")
        z = np.asarray(zmesh_in, dtype=float).ravel()
        z = z[z < zend]
        z = np.append(z, zend)
    elif zt == "log":
        dzn = dzstart * gfac ** np.arange(1, int(ngen) + 1)
        z = np.concatenate([[0.0], np.cumsum(dzn)])
        z = z[z < zend]
        z = np.append(z, zend)
    else:
        raise ValueError(
            "build_depth_mesh: ztype '%s' not implemented" % ztype
        )

    dz = np.diff(z)
    nzp = z.size
    ip = np.arange(nzp - 1)
    zm = 0.5 * (z[:-1] + z[1:])
    return dict(z=z, dz=dz, nz=nzp, ip=ip, zm=zm)


# ---------------------------------------------------------------------------
# temporal mesh (SITE_Mesh.m, "set_t" branch)
# ---------------------------------------------------------------------------
def build_time_mesh(
    ttype="log",
    tstart=110000 * YEAR2SEC,
    tend=30 * YEAR2SEC,
    nt=401,
    direction=-1,
    debug=False,
    tmesh_in=None,
):
    """Temporal mesh.

    ttype
      'log'             (t, dt) = numeric.set_mesh(tstart, tend, nt, 'log',
                        direction) -- the actual MATLAB set_mesh.m function,
                        already translated in numeric.py.
      'special'/'mixed': fixed 3-segment MATLAB grid, ignores tstart/tend/nt
                        (-110000:100:0, -20000:50:-3000, -3000:20:0 ka BP).
      'read'            take t from `tmesh_in`.

    Returns dict(t, dt, nt, it, tm).

    it is the trivial 0-based identity array [0 .. nt-1] -- as in
    SITE_Mesh.m, this is NOT the GST-parameter index used by heat1dnt in
    the Tikhonov/MCMC scripts (that mapping comes from numeric.set_mgsth);
    it IS the correct `it` for SITE_Init.m's own heat1dnt call, where GST
    is given at every time node.
    tm (interval midpoints) is computed here for every branch. MATLAB only
    computed it inside the 'log' case and would error saving an undefined
    `tm` for the other branches -- see README.txt, "Source issues".
    """
    tt = str(ttype).lower()
    if tt in ("special", "mixed"):
        t1 = np.arange(-110000.0, 0.0 + 1.0, 100.0) * YEAR2SEC
        t2 = np.arange(-20000.0, -3000.0 + 1.0, 50.0) * YEAR2SEC
        t3 = np.arange(-3000.0, 0.0 + 1.0, 20.0) * YEAR2SEC
        t = np.unique(np.concatenate([t1, t2, t3]))
    elif tt == "read":
        if tmesh_in is None:
            raise ValueError("build_time_mesh: ttype='read' needs tmesh_in")
        t = np.asarray(tmesh_in, dtype=float).ravel()
    elif tt == "log":
        t, _ = nm.set_mesh(tstart, tend, int(nt), "log", direction)
    else:
        raise ValueError("build_time_mesh: ttype '%s' not implemented" % ttype)

    dt = np.diff(t)
    ntp = t.size
    it = np.arange(ntp)
    tm = 0.5 * (t[:-1] + t[1:])
    return dict(t=t, dt=dt, nt=ntp, it=it, tm=tm)


# ---------------------------------------------------------------------------
# combined driver (SITE_Mesh.m)
# ---------------------------------------------------------------------------
def build_meshes(
    name, set_z=True, set_t=True, depth_kw=None, time_kw=None, verbose=True
):
    """Both meshes for one borehole (SITE_Mesh.m). `name` is the borehole
    identifier that replaces the MATLAB per-site function/file prefix
    (e.g. 'OKU', 'Ullrigg', or your real borehole name).

    depth_kw / time_kw : dict of keyword overrides for build_depth_mesh /
        build_time_mesh (MATLAB: values loaded from name_Mesh_in.mat that
        overwrite the function's defaults).
    """
    if verbose:
        print(" ...set up meshes for site %s" % name)
    out = dict(name=name)
    if set_z:
        out.update(build_depth_mesh(**(depth_kw or {})))
    if set_t:
        out.update(build_time_mesh(**(time_kw or {})))
    return out


def save_meshes(name, mesh, outdir="."):
    """Optional .npz hand-off, for parity with MATLAB's name_DepthGrid.mat
    / name_TimeGrid.mat. Not needed if the caller keeps `mesh` in memory
    and passes it straight on (the normal Python usage)."""
    depth_keys = ("z", "dz", "nz", "ip", "zm")
    time_keys = ("t", "dt", "nt", "it", "tm")
    written = []
    if all(k in mesh for k in depth_keys):
        f = os.path.join(outdir, "%s_DepthGrid.npz" % name)
        np.savez(f, **{k: mesh[k] for k in depth_keys})
        written.append(f)
    if all(k in mesh for k in time_keys):
        f = os.path.join(outdir, "%s_TimeGrid.npz" % name)
        np.savez(f, **{k: mesh[k] for k in time_keys})
        written.append(f)
    return written
