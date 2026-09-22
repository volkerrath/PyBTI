"""
prep.py -- Read observations and prepare a dictionary model (SITE_Prep.m).

The workflow preserves measurement depths and values. build_mesh can
include those depths as nodes; prepare_site then maps observations to
nodes without smoothing/interpolating them into extra observations.
Artificial noise is an explicit, seeded option restricted to synthetic
input. Raw file columns and row indices are retained for inspection.

This translates the data/property assembly needed by example A, not all
site-specific branches in SITE_Prep.m. Physical properties may be scalar,
per cell, or per unit with an explicit ip mapping. No conductivity-log
averaging or automatic heat-flow estimation is performed here.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 implementation; synthetic tests only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_data", "prepare_site"]


def read_data(datapar: dict):
    """Read borehole observations without tying them to a numerical mesh.

    Parameters
    ----------
    datapar : dict with file, delimiter (','), skiprows (0), zcol (0),
        Tcol (1), and either errcol or Terr (scalar or one value per file
        row). Column numbers are 0-based. Units: depths [m], temperature
        [deg C], uncertainty [K]. depth_range optionally selects inclusive
        (minimum, maximum) depths. Nonfinite depth/temperature rows are
        excluded; missing/nonpositive uncertainties in selected rows fail.
        Duplicate or negative selected depths are rejected.
        synthetic defaults False. noise is optional; only allowed for
        synthetic=True. Its dict has kind ('independent', 'gaussian' or
        'exponential'), length (correlation length in m) and seed (0).
        The noise standard deviations are the supplied Terr values.

    Returns
    -------
    dict(zobs, Tobs, Terr, Tcov, Tclean, noise, raw, source_rows, file,
         synthetic).
    raw is the unmodified numeric file table. source_rows maps the sorted,
    selected observations back to its 0-based rows. Tclean is the selected
    temperature before added noise. Tcov describes the specified noise/
    uncertainty model, not uncertainty estimated from residuals.
    """
    path = Path(datapar["file"])
    raw = np.loadtxt(path, delimiter=datapar.get("delimiter", ","),
                     skiprows=datapar.get("skiprows", 0), ndmin=2)
    z, T = raw[:, datapar.get("zcol", 0)], raw[:, datapar.get("Tcol", 1)]
    err = (raw[:, datapar["errcol"]] if "errcol" in datapar else
           np.broadcast_to(np.asarray(datapar["Terr"], dtype=float), z.shape))
    keep = np.isfinite(z) & np.isfinite(T)
    if "depth_range" in datapar:
        top, bottom = datapar["depth_range"]
        if top > bottom:
            raise ValueError("depth_range must be increasing")
        keep &= (z >= top) & (z <= bottom)
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(z[rows])]
    z, T, err = z[rows].copy(), T[rows].copy(), err[rows].copy()
    if not z.size:
        raise ValueError("No usable observations in the selected depth range")
    if np.any(z < 0) or np.any(np.diff(z) <= 0):
        raise ValueError("Observation depths must be nonnegative and distinct")
    if not np.isfinite(err).all() or np.any(err <= 0):
        raise ValueError("Selected temperature uncertainties must be positive and finite")
    covariance = np.diag(err ** 2)
    noise = np.zeros(z.size)
    setting = datapar.get("noise")
    synthetic = bool(datapar.get("synthetic", False))
    if setting is not None:
        if not synthetic:
            raise ValueError("Artificial noise requires synthetic=True")
        kind = setting.get("kind", "independent")
        if kind in ("gaussian", "exponential"):
            length = float(setting["length"])
            if not np.isfinite(length) or length <= 0:
                raise ValueError("Noise correlation length must be positive")
            distance = np.abs(z[:, None] - z[None, :]) / length
            correlation = np.exp(-0.5 * distance**2) if kind == "gaussian" else np.exp(-distance)
            covariance = err[:, None] * correlation * err[None, :]
        elif kind != "independent":
            raise ValueError("Unknown noise kind")
        # Eigensquare-root also handles almost singular Gaussian covariance.
        values, vectors = np.linalg.eigh(covariance)
        rng = np.random.default_rng(setting.get("seed", 0))
        noise = vectors @ (np.sqrt(np.maximum(values, 0)) * rng.standard_normal(z.size))
    return dict(zobs=z, Tobs=T + noise, Terr=err, Tcov=covariance,
                Tclean=T, noise=noise, raw=raw, source_rows=rows,
                file=path, synthetic=synthetic)


def prepare_site(sitepar: dict, mesh: dict, observations: dict):
    """Assemble a MATLAB-style sitepar dictionary on the generated mesh.

    Parameters
    ----------
    sitepar : dict with name, props, qb [W/m2, normally negative], and
        k [W/(m K)], r [kg/m3], c [J/(kg K)]. Optional h [W/m3], p,
        kA and kB default to zero; gts defaults to zero.
        Each property is scalar or a per-cell vector. To use per-unit
        vectors, supply ip: one 0-based unit index per cell.
    mesh : dict(z, t, ...) from workflow.build_mesh / mesh.build_meshes.
        z begins at zero; depth and time nodes are strictly increasing.
        Observation depths must be nodes (within 1e-8 m).
    observations : dictionary from read_data. No smoothing is applied.

    Returns
    -------
    sitepar dict with properties, z, t, dz, dt, ip, id, zobs, Tobs, Terr,
    Tcov, rc and observations. The latter retains raw data and noise.
    Inputs are not modified; caller-owned metadata in sitepar is retained.
    """
    result = dict(sitepar)
    z = np.asarray(mesh["z"], dtype=float).ravel().copy()
    t = np.asarray(mesh["t"], dtype=float).ravel().copy()
    if (z.size < 2 or t.size < 2 or z[0] != 0 or not np.isfinite(z).all()
            or not np.isfinite(t).all() or np.any(np.diff(z) <= 0)
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Mesh needs finite increasing depth/time nodes, starting at z=0")
    nc = z.size - 1
    pointer = np.asarray(sitepar.get("ip", np.arange(nc))).ravel()
    if (pointer.size != nc or not np.isfinite(pointer).all()
            or np.any(pointer != np.floor(pointer)) or np.any(pointer < 0)):
        raise ValueError("ip must contain one nonnegative integer per cell")
    pointer = pointer.astype(int)
    nunits = int(pointer.max()) + 1
    for key in ("k", "r", "c", "h", "p", "kA", "kB"):
        value = sitepar[key] if key in ("k", "r", "c") else sitepar.get(key, 0.0)
        array = np.asarray(value, dtype=float).ravel()
        if array.size == 1:
            array = np.full(nunits, array.item())
        if array.size != nunits or not np.isfinite(array).all():
            raise ValueError("%s must have one finite value per property unit" % key)
        if key in ("k", "r", "c") and np.any(array <= 0):
            raise ValueError("%s must be positive" % key)
        result[key] = array.copy()
    if np.any((result["p"] < 0) | (result["p"] > 1)):
        raise ValueError("Porosity must lie between zero and one")
    if not np.isfinite(float(sitepar["qb"])):
        raise ValueError("qb must be finite")
    depths = np.asarray(observations["zobs"])
    ids = np.searchsorted(z, depths)
    lower = np.clip(ids - 1, 0, z.size - 1)
    upper = np.clip(ids, 0, z.size - 1)
    ids = np.where(np.abs(z[lower] - depths) < np.abs(z[upper] - depths), lower, upper)
    if not np.allclose(z[ids], depths, rtol=0, atol=1e-8):
        raise ValueError("Include observation depths when building the mesh")
    result.update(z=z, t=t, dz=np.diff(z), dt=np.diff(t), ip=pointer, id=ids,
                  rc=result["r"] * result["c"], gts=float(sitepar.get("gts", 0.0)),
                  observations=observations)
    for key in ("zobs", "Tobs", "Terr", "Tcov"):
        result[key] = np.asarray(observations[key]).copy()
    return result
