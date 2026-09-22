"""
numeric.py -- Callable numerical core for 1-D nonlinear (permafrost-capable)
borehole temperature modelling and paleoclimate (GSTH) inversion.

Translated from the MATLAB sources in src_for_claude/ (V. Rath, 2000-2019):
heat1dns.m, heat1dnt.m, forward_solve.m, sensfdt_pals.m / sensfdt_palp.m,
n2c.m, c2n.m, reg1d.m, setregpar.m, solvereg.m, GI.m / geninv.m / resmat.m,
set_mesh.m, set_mgsth.m, set_gst.m, set_lingst.m, set_pntgst.m, set_stpgst.m,
set_prior.m, heat1dat.m + msetup.m, prop2cell.m, vavg.m, avg.m, wfilt.m,
smooth1.m, tri.m, box.m, RMS.m, mad.m, CovarGauss.m, CovarExpnl.m, f_reg.m,
f_scal.m, mscalsum.m / scaljac.m, zgrad.m, splinefit.m (robust option only).

Physical property functions (fluid, ice, rock; ftheta partition function) are
NOT re-implemented here: they are taken from phys.py (import phys), through
the small adapter class PropModel below. Select a property set with the
`props` argument (a site code such as "full", a key of phys.SITE_PROPS, or a
user-supplied mapping with the same keys).

Conventions (differences to MATLAB)
-----------------------------------
- ALL index arrays are 0-based: ip (cell -> parameter unit), it (time node ->
  GST parameter), id (observation node indices). Use from_matlab_index() to
  convert arrays read from MATLAB .mat files.
- Time nodes t are in seconds, negative in the past, 0 = present.
  Depth nodes z are in metres, positive down, z[0] = 0 (surface).
- Heat flow qb is signed for z positive down (flux_z = -k dT/dz), i.e. normal
  (upward) terrestrial heat flow is NEGATIVE, as in the MATLAB code.
- Argument order follows the *definitions* in the MATLAB files:
      heat1dns(k,kA,kB,h,r,por,qb,Ts,dz,ip,maxiter,tol,freeze)
      heat1dnt(k,kA,kB,h,r,cp,por,qb,dz,ip,dt,it,GST,T0,theta,maxiter,tol,
               freeze,out)
  where r = rock density, cp = rock specific heat, por = porosity, h = heat
  production. (Several MATLAB *callers* use older/other argument orders; see
  README.txt, section "Source issues".)
- MATLAB nargout-dependent outputs are replaced by full_output=True/False.

Not translated (replaced by numpy/scipy): lsqr/lsmr, nanmean, nanmedian,
prctile, quantile, splfit/splval/ppdiff, mstruct (use dict / dataclass),
str2strs, mdir. See README.txt for the complete accounting.

References (as cited in the MATLAB source comments; DOIs are only given where
they appear in the provided material - others are flagged NOT VERIFIED)
-----------------------------------------------------------------------
Beltrami, H. & Mareschal, J.-C., Recent warming in eastern Canada inferred
    from geothermal measurements, Geophys. Res. Lett., 1991, 18(4), 605-608.
    [DOI not in source; NOT VERIFIED]
Beltrami, H.; Jessop, A. M. & Mareschal, J.-C., Ground temperature histories
    in eastern and central Canada from geothermal measurements: evidence of
    climate change, Palaeogeogr. Palaeoclimatol. Palaeoecol., 1992, 98,
    167-184. [DOI not in source; NOT VERIFIED]
Mareschal, J.-C. & Beltrami, H., Evidence for recent warming from perturbed
    thermal gradients: examples from eastern Canada, Clim. Dyn., 1992, 6,
    135-143. [DOI not in source; NOT VERIFIED]
Youzwishen, C. & Sacchi, M., Edge preserving imaging, J. Seismic
    Exploration, 2006, 15(4), 45-56. [as cited in f_reg.m; DOI NOT VERIFIED]
Mehanee, S. & Zhdanov, M. S., Two-dimensional magnetotelluric inversion of
    blocky geoelectric structures, J. Geophys. Res., 2002, 107,
    doi:10.1029/2001JB000191 (DOI as given in f_scal.m).
Lundgren, J., SPLINEFIT (MATLAB File Exchange), 2010 (robust option of
    splinefit.m; algorithm re-implemented with scipy B-splines).

--------------------------------------------------------------------------
Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-21
Status         : AI-generated translation of the user's MATLAB code. It has
                 been tested against analytical solutions (see tests/) but
                 NOT against MATLAB output. Review before production use.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    interp1d,
    make_lsq_spline,
)
from scipy.linalg import solve_banded
from scipy.sparse.linalg import lsqr as _sp_lsqr
from scipy.special import erfc

import phys

__all__ = [
    # constants
    "YEAR2SEC",
    "LH_ICE",
    "P_ATM",
    "G_ACC",
    # property adapter
    "PropModel",
    "get_props",
    "ftheta_apparent",
    # index helpers
    "from_matlab_index",
    # grids / interpolation
    "n2c",
    "c2n",
    "set_mesh",
    "set_mgsth",
    "prop2cell",
    # GST / prior construction
    "tri",
    "box",
    "set_gst",
    "set_lingst",
    "set_pntgst",
    "set_stpgst",
    "set_prior",
    # forward modelling
    "heat1dns",
    "heat1dnt",
    "forward_solve",
    "ForwardModel",
    "msetup",
    "heat1dat",
    # sensitivities
    "sensfdt_pal",
    # regularisation / inversion helpers
    "reg1d",
    "setregpar",
    "setregvec",
    "pvar",
    "lsqr_solve",
    "gauss_newton_step",
    "solvereg",
    "geninv",
    "resmat",
    # averaging, filters, statistics, covariances, scaling
    "vavg",
    "avg",
    "wfilt",
    "smooth1",
    "rms",
    "mad",
    "covar_gauss",
    "covar_expnl",
    "f_reg",
    "f_scal",
    "scale_jacobian",
    "zgrad",
    "splinefit",
]

# ---------------------------------------------------------------------------
# constants (values as hard-coded in the MATLAB files)
# ---------------------------------------------------------------------------
YEAR2SEC = 31557600.0  # s / a (Julian year), as in the MATLAB scripts
LH_ICE = 333600.0  # J/kg, latent heat of fusion (heat1dnt.m: Lh)
P_ATM = 101325.0  # Pa, atmospheric pressure at the surface node
G_ACC = 9.81  # m/s^2
_RHO_W_START = 998.0  # kg/m^3, first guess for hydrostatic pressure


def from_matlab_index(a):
    """Convert 1-based MATLAB index array(s) to 0-based numpy indices."""
    return np.asarray(a, dtype=int).ravel() - 1


# ===========================================================================
# 1. Property adapter (bridges the MATLAB call signatures to phys.py)
# ===========================================================================


def _is_freeze(freeze) -> bool:
    if isinstance(freeze, str):
        return freeze.strip().lower() in ("yes", "y", "true", "on", "1")
    return bool(freeze)


def ftheta_apparent(T, model="lun", rfl=0.025, **kw):
    """Fluid/ice partition function Theta and dTheta/dT via phys.ftheta,
    plus the lower cut-off `rfl` that ftheta.m (src_for_claude) applies.

    ftheta.m: Theta = exp(-((T-Tf)/w)^2); Theta(T>Tf) = 1;
              Theta(Theta<rfl) = rfl; dTheta = 0 where T>Tf or Theta was
              floored. phys.ftheta("lun") reproduces everything except the
              floor, which is added here (rfl=None or 0 disables it).

    Defaults for the "lun" model follow ftheta.m (Tf=0, w=1), NOT phys.py
    (w=0.5). Other models ("gal", "nic", "hin") pass **kw straight to phys.
    """
    kw = dict(kw)
    if str(model).lower() in ("lun", "l"):
        kw.setdefault("Tf", 0.0)
        kw.setdefault("w", 1.0)
    out = phys.ftheta(model, T=np.asarray(T, dtype=float), **kw)
    Theta = np.array(out["Theta"], dtype=float)
    dTheta = np.array(out["dTheta"], dtype=float)
    if rfl:
        low = Theta < rfl
        Theta = np.where(low, rfl, Theta)
        dTheta = np.where(low, 0.0, dTheta)
    return Theta, dTheta


class PropModel:
    """Adapter from the MATLAB call signatures used in heat1dns/heat1dnt

        rhofT(Tc,Pch) cpfT(Tc,Pch) kfT(Tc,.) rhoiT(Tc) cpiT(Tc) kiT(Tc)
        rcmT(rm,cpm,Tc,Pcl)  kmT(k,Tc,kA,kB,Pcl)

    to the dict-based functions of phys.py.

    Parameters
    ----------
    site : str or mapping
        Key of phys.SITE_PROPS (e.g. "full") or a mapping with the keys
        cpfT, cpiT, cpmT, kfT, kiT, kmT, rcmT, rhofT, rhoiT.
    salinity : float
        NaCl mass fraction passed to kfT (phys.kfT_phillips uses it as S).
        NOTE: the MATLAB call kfT(Tc,Pch) passes the hydrostatic PRESSURE in
        the second argument; phys.py interprets that slot as salinity. The
        pressure is deliberately not passed here (fresh water, S=0 default).

    Adaptations (phys.py itself is untouched)
    -----------------------------------------
    * rcmT: a superset model dict {rm, cpm, rcm=rm*cpm, T, P} is passed, so
      that both the 'direct' form (uses rm) and the 'scaled' form (uses the
      volumetric heat capacity rcm) return rho*cp(T).
    * kmT: per-cell coefficients kA/kB are passed as A/B. phys.kmT_clauser
      accepts scalar A/B only, so cells are grouped by unique (kA,kB) pair
      and the phys function is called once per group. NaN in kA/kB means
      "use the phys default coefficients".
    """

    def __init__(self, site="full", salinity: float = 0.0):
        if isinstance(site, str):
            if site not in phys.SITE_PROPS:
                raise KeyError(
                    "unknown props site '%s'; available: %s"
                    % (site, ", ".join(sorted(phys.SITE_PROPS)))
                )
            self._f = phys.SITE_PROPS[site]
            self.name = site
        else:
            self._f = dict(site)
            self.name = "custom"
        self.salinity = float(salinity)

    # -- fluid ---------------------------------------------------------
    def rhof(self, T, P):
        return self._f["rhofT"]({"T": T, "P": P})["rho"]

    def cpf(self, T, P):
        return self._f["cpfT"]({"T": T, "P": P})["cpf"]

    def kf(self, T):
        return self._f["kfT"]({"T": T, "S": self.salinity})["kf"]

    # -- ice -----------------------------------------------------------
    def rhoi(self, T):
        return self._f["rhoiT"]({"T": T})["rhoi"]

    def cpi(self, T):
        return self._f["cpiT"]({"T": T})["cpi"]

    def ki(self, T):
        return self._f["kiT"]({"T": T})["ki"]

    # -- rock matrix ---------------------------------------------------
    def rcm(self, rm, cpm, T, P):
        model = {"rm": rm, "cpm": cpm, "rcm": rm * cpm, "T": T, "P": P}
        return self._f["rcmT"](model)["rcmT"]

    def km_evaluator(self, kA, kB) -> Callable:
        """Return km(k, Tc) for fixed per-cell coefficient arrays kA, kB."""
        kA = np.asarray(kA, dtype=float).ravel()
        kB = np.asarray(kB, dtype=float).ravel()
        groups = {}
        for i, (a, b) in enumerate(zip(kA, kB)):
            key = (
                None if np.isnan(a) else float(a),
                None if np.isnan(b) else float(b),
            )
            groups.setdefault(key, []).append(i)
        groups = [(key, np.asarray(ix)) for key, ix in groups.items()]
        fn = self._f["kmT"]
        n = kA.size

        def _km(k, Tc):
            k = np.asarray(k, dtype=float)
            Tc = np.asarray(Tc, dtype=float)
            out = np.empty(n)
            for (a, b), ix in groups:
                model = {"k": k[ix], "km0": k[ix], "T": Tc[ix]}
                if a is not None:
                    model["A"] = a
                if b is not None:
                    model["B"] = b
                out[ix] = np.asarray(fn(model)["kmT"], dtype=float)
            return out

        return _km


def get_props(props) -> PropModel:
    """Return `props` if it already is a PropModel, else build one."""
    return props if isinstance(props, PropModel) else PropModel(props)


# ===========================================================================
# 2. Grid helpers and interpolation
# ===========================================================================


def n2c(vn, d=None):
    """Node -> cell-centre values (n2c.m). `d` is unused (kept for parity)."""
    vn = np.asarray(vn, dtype=float).ravel()
    return 0.5 * (vn[:-1] + vn[1:])


def c2n(vc, d):
    """Cell-centre -> node values (c2n.m): cell-size weighted mean at interior
    nodes, linear extrapolation at the two end nodes. Needs >= 2 cells."""
    vc = np.asarray(vc, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    nc = vc.size
    if nc < 2:
        raise ValueError("c2n: need at least two cells")
    vn = np.empty(nc + 1)
    vn[1:nc] = (d[1:] * vc[1:] + d[:-1] * vc[:-1]) / (d[1:] + d[:-1])
    vn[0] = vc[0] + d[0] * (vc[0] - vc[1]) / (d[0] + d[1])
    vn[nc] = vc[-1] + d[-1] * (vc[-1] - vc[-2]) / (d[-1] + d[-2])
    return vn


def set_mesh(x0, x1, nn=128, type="lin", direction=1):
    """Lin/log mesh (set_mesh.m). Returns (x, dx).

    direction=+1 : from x0 to x1 ('log': starts with 0, e.g. depth mesh)
    direction=-1 : from -x0 to -x1 ('log': ends with 0, e.g. time mesh with
                   present = 0; x0 must then be the LARGER value)
    """
    t = str(type).lower()
    if t in ("log", "logarithmic"):
        x = direction * np.logspace(np.log10(x0), np.log10(x1), nn - 1)
        x = (
            np.concatenate([x, [0.0]])
            if direction < 0
            else np.concatenate([[0.0], x])
        )
    else:
        x = direction * np.linspace(x0, x1, nn)
    return x, np.diff(x)


def set_mgsth(
    t, tbase=0.0, tstart=None, tend=None, ngrid=32, gmode="log", thin=None
):
    """Temporal GST inversion grid (set_mgsth.m). Returns (pt, it, c, l, u).

    pt : base value for every GST parameter (tbase * ones)
    it : 0-based GST index for each time node in t
    c, l, u : centre / first / last time of each GST interval (NaN if empty)
    gmode : 'log', 'lin' or 'in' (interval edges given in `thin`, negative)
    """
    t = np.asarray(t, dtype=float).ravel()
    if tstart is None or tend is None:
        tstart, tend = abs(t[0]), abs(t[-1])
    mode = str(gmode).lower()
    if mode in ("log", "logarithmic"):
        if tend <= 0 or tstart <= 0:
            raise ValueError("set_mgsth: log grid needs tstart,tend > 0")
        th = -np.logspace(np.log10(tstart), np.log10(tend), ngrid)
    elif mode in ("lin", "linear"):
        th = -np.linspace(tstart, tend, ngrid)
    elif mode in ("in", "input"):
        th = np.asarray(thin, dtype=float).ravel()
    else:
        raise ValueError("set_mgsth: unknown method '%s'" % gmode)
    nth = th.size
    it = np.zeros(t.size, dtype=int)
    it[t < th[0]] = 0
    c = np.full(nth - 1, np.nan)
    l = np.full(nth - 1, np.nan)
    u = np.full(nth - 1, np.nan)
    for jj in range(nth - 1):
        step = (t >= th[jj]) & (t < th[jj + 1])
        it[step] = jj
        if step.any():
            ts = t[step]
            if mode in ("log", "logarithmic"):
                with np.errstate(divide="ignore"):
                    c[jj] = np.sign(ts.mean()) * np.exp(
                        np.mean(np.log(np.abs(ts)))
                    )
            else:
                c[jj] = ts.mean()
            l[jj], u[jj] = ts[0], ts[-1]
    it[t >= th[-1]] = nth - 1
    pt = tbase * np.ones(nth)
    return pt, it, c, l, u


def prop2cell(param, zparam, z, Top=None, Bottom=None, avgmeth="h"):
    """Average point-wise parameters onto grid cells (prop2cell.m).

    Cells without data are filled by linear interpolation; cells above the
    first / below the last populated cell get Top / Bottom (default: the
    first / last populated value). Returns values at cell centres.
    """
    param = np.asarray(param, dtype=float).ravel()
    zparam = np.asarray(zparam, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    zm = 0.5 * (z[:-1] + z[1:])
    nc = zm.size
    Ks = np.full(nc, np.nan)
    for cell in range(nc):
        incell = (zparam >= z[cell]) & (zparam < z[cell + 1])
        if incell.any():
            Kc = param[incell]
            Ks[cell] = vavg(Kc, np.ones_like(Kc), avgmeth)
    fin = np.flatnonzero(np.isfinite(Ks))
    if fin.size == 0:
        raise ValueError("prop2cell: no parameter values inside the grid")
    up, low = fin[0], fin[-1]
    Ks[:up] = Ks[up] if Top is None else Top
    Ks[low + 1 :] = Ks[low] if Bottom is None else Bottom
    ok = np.isfinite(Ks)
    return np.interp(zm, zm[ok], Ks[ok])


# ===========================================================================
# 3. Windows, GST / prior model construction
# ===========================================================================


def tri(n):
    """N-point triangular window, normalised to sum 1 (tri.m). n must be odd."""
    n = int(n)
    if n % 2 == 0:
        raise ValueError("tri: needs an odd length")
    w = 2.0 * np.arange(1, (n + 1) // 2 + 1) / (n + 1)
    w = np.concatenate([w, w[-2::-1]])
    return w / w.sum()


def box(n):
    """N-point boxcar window, normalised to sum 1 (box.m)."""
    return np.ones(int(n)) / int(n)


def _tri_smooth(T, L):
    """Triangular smoothing of half-length L; the L end samples are kept."""
    S = np.array(T, dtype=float)
    if L:
        w = tri(2 * L + 1)
        w = w / w.sum()
        if S.size > 2 * L:
            S[L : S.size - L] = np.convolve(T, w, mode="valid")
    return S


def set_gst(t, stepamp, steptime, L=0):
    """General step function (set_gst.m). Returns (S, T): smoothed, raw.

    amp(1) for t < steptime(1); amp(i) for steptime(i-1) <= t < steptime(i);
    amp(end) for t >= steptime(end).
    (MATLAB used t > steptime(end), which leaves the node t == steptime(end)
    - e.g. the present, t = 0 - at zero; '>=' is used here.)
    """
    t = np.asarray(t, dtype=float).ravel()
    stepamp = np.asarray(stepamp, dtype=float).ravel()
    steptime = np.asarray(steptime, dtype=float).ravel()
    ns = steptime.size
    T = np.zeros(t.size)
    T[t < steptime[0]] = stepamp[0]
    for i in range(1, ns):
        T[(t < steptime[i]) & (t >= steptime[i - 1])] = stepamp[i]
    T[t >= steptime[-1]] = stepamp[-1]
    return _tri_smooth(T, L), T


def set_stpgst(t, steptime, stepamp, L=0, pom=np.nan):
    """GST from a step function (set_stpgst.m). Returns (S, T).

    pom (pre-onset temperature) for t < steptime(1), default stepamp(end)-4.
    amp(i-1) applies for steptime(i-1) <= t < steptime(i); amp(end) after
    steptime(end) ('>=' instead of MATLAB '>', see set_gst).
    """
    t = np.asarray(t, dtype=float).ravel()
    steptime = np.asarray(steptime, dtype=float).ravel()
    stepamp = np.asarray(stepamp, dtype=float).ravel()
    ns = steptime.size
    if not np.isfinite(pom):
        pom = stepamp[-1] - 4.0
    T = np.zeros(t.size)
    T[t < steptime[0]] = pom
    for i in range(1, ns):
        T[(t < steptime[i]) & (t >= steptime[i - 1])] = stepamp[i - 1]
    T[t >= steptime[-1]] = stepamp[-1]
    return _tri_smooth(T, L), T


def set_lingst(t, amp, tim, L=0):
    """GST by linear interpolation of (tim, amp) onto t (set_lingst.m).
    Values outside [tim(1), tim(end)] are held constant. Returns (S, T)."""
    t = np.asarray(t, dtype=float).ravel()
    tim = np.asarray(tim, dtype=float).ravel()
    amp = np.asarray(amp, dtype=float).ravel()
    o = np.argsort(tim)
    T = np.interp(t, tim[o], amp[o])
    return _tri_smooth(T, L), T


def set_pntgst(t, tpoint, apoint, method="linear"):
    """GST by interpolation in log|t| (set_pntgst.m). NaN outside the range
    of tpoint, like MATLAB interp1 (t = 0 maps to log = -inf -> NaN)."""
    with np.errstate(divide="ignore"):
        tp = np.log(np.abs(np.asarray(tpoint, dtype=float).ravel()))
        tt = np.log(np.abs(np.asarray(t, dtype=float).ravel()))
    ap = np.asarray(apoint, dtype=float).ravel()
    o = np.argsort(tp)
    tp, ap = tp[o], ap[o]
    m = str(method).lower()
    if m == "linear":
        return np.interp(tt, tp, ap, left=np.nan, right=np.nan)
    if m in ("nearest", "previous", "next"):
        return interp1d(tp, ap, kind=m, bounds_error=False, fill_value=np.nan)(
            tt
        )
    if m in ("pchip", "cubic"):
        return PchipInterpolator(tp, ap, extrapolate=False)(tt)
    if m == "spline":
        return CubicSpline(tp, ap, extrapolate=False)(tt)
    raise ValueError("set_pntgst: unknown method '%s'" % method)


def set_prior(steptemp, steptime, L, tstart, tend, ngrid):
    """(Smoothed) step prior on the logarithmic inverse grid (set_prior.m).

    steptemp : temperatures; steptime : (negative) step times.
    NOTE (kept as in MATLAB): the smoothed value of window centre k is
    written to index k-L, i.e. the smoothed prior is shifted by L samples
    and the last 2L samples stay unsmoothed. This looks unintended; see
    README.txt, "Source issues".
    """
    steptemp = np.asarray(steptemp, dtype=float).ravel()
    steptime = np.append(np.asarray(steptime, dtype=float).ravel(), 0.0)
    th = -np.logspace(np.log10(tstart), np.log10(tend), ngrid)
    mod = np.zeros(th.size)
    mod[th < steptime[0]] = steptemp[0]
    ns = steptime.size - 1
    for k in range(ns):
        mod[(th >= steptime[k]) & (th < steptime[k + 1])] = steptemp[k]
    mod[th >= steptime[ns - 1]] = steptemp[ns - 1]
    if L != 0:
        w = tri(2 * L + 1)
        w = w / w.sum()
        X = mod.copy()
        nm = mod.size
        if nm > 2 * L:
            mod[: nm - 2 * L] = np.convolve(X, w, mode="valid")
    return mod


# ===========================================================================
# 4. Forward modelling: stationary and transient nonlinear heat equation
# ===========================================================================


def _prepare_cells(k, kA, kB, h, r, por, qb, Ts, dz, ip, props, cp=None):
    """Per-cell parameter arrays, lithostatic and hydrostatic pressure."""
    ip = np.asarray(ip, dtype=int).ravel()
    dz = np.asarray(dz, dtype=float).ravel()
    nc = ip.size
    if dz.size != nc:
        raise ValueError("dz and ip must have the same length (cells)")
    z = np.concatenate([[0.0], np.cumsum(dz)])

    def cell(a):
        return np.asarray(a, dtype=float).ravel()[ip]

    C = dict(
        ip=ip,
        dz=dz,
        z=z,
        nc=nc,
        nz=nc + 1,
        k=cell(k),
        kA=cell(kA),
        kB=cell(kB),
        h=cell(h),
        rm=cell(r),
        por=cell(por),
    )
    if cp is not None:
        C["cpm"] = cell(cp)
    # lithostatic pressure (first node: atmospheric, as in MATLAB)
    Pcl = np.concatenate([[P_ATM], G_ACC * np.cumsum(dz * C["rm"])])
    C["Pcl"] = n2c(Pcl)
    # first-guess temperature and hydrostatic pressure
    ka, ha, zb = C["k"].mean(), C["h"].mean(), z.max()
    Tg = Ts - qb * z / ka + (ha * z / ka) * (zb - z / 2.0)
    Pch0 = G_ACC * np.cumsum(dz * _RHO_W_START)
    Pch = np.concatenate(
        [[P_ATM], G_ACC * np.cumsum(dz * props.rhof(Tg[1:], Pch0))]
    )
    C["Pch"] = n2c(Pch)
    C["dc"] = 0.5 * (dz[1:] + dz[:-1])
    C["Tguess"] = Tg
    return C


def _tridiag_coeffs(keff, dz, dc):
    """Finite-difference tri-diagonal coefficients, with the boundary
    treatment of heat1dns.m / heat1dnt.m (Dirichlet at the top node,
    Neumann - flux qb - at the bottom node).

    Returns lower (nz-1), main (nz), upper (nz-1), acn
    with A[i+1,i]=lower[i], A[i,i]=main[i], A[i,i+1]=upper[i].
    """
    nc = keff.size
    nz = nc + 1
    dl = keff / dz
    a = np.zeros(nz)
    b = np.zeros(nz)
    c = np.zeros(nz)
    c[2:] = dl[1:] / dc  # MATLAB c(3:nz)  = dl(2:nc)./dc
    a[: nz - 2] = dl[: nc - 1] / dc  # MATLAB a(1:nz-2)= dl(1:nc-1)./dc
    b[1 : nz - 1] = -(a[: nz - 2] + c[2:])
    b[0] = 1.0  # Dirichlet, top
    acn = keff[-1] / (dz[-1] * dz[-1])
    a[nz - 2] = 2.0 * acn  # Neumann, bottom
    b[nz - 1] = -a[nz - 2]
    return a[: nz - 1], b, c[1:], acn


def _solve_tridiag(lower, main, upper, rhs):
    n = main.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1, :] = main
    ab[2, :-1] = lower
    return solve_banded((1, 1), ab, rhs)


def _tri_matvec(lower, main, upper, x):
    y = main * x
    y[1:] += lower * x[:-1]
    y[:-1] += upper * x[1:]
    return y


def _fwd_defaults(freeze_kw, rfl, model="lun"):
    kw = dict(freeze_kw) if freeze_kw else {}
    return kw


def heat1dns(
    k,
    kA,
    kB,
    h,
    r,
    por,
    qb,
    Ts,
    dz,
    ip,
    maxiter=10,
    tol=1e-5,
    freeze=0,
    props="full",
    *,
    freeze_model="lun",
    freeze_kw=None,
    rfl=0.025,
    full_output=False,
):
    """Stationary nonlinear 1-D heat equation, permafrost optional
    (heat1dns.m).

    Solves d/dz( keff(T) dT/dz ) + h = 0 with T(0) = Ts and heat flow qb at
    the bottom node, by Picard iteration. Conductivity is the weighted
    geometric mean of fluid, ice and matrix conductivity; the fluid/ice
    partition follows ftheta when freeze is true.

    Parameters (per-unit arrays are mapped to cells with ip, 0-based)
    ----------
    k, kA, kB : conductivity and its temperature-law coefficients A, B
    h         : volumetric heat production [W/m^3]
    r         : rock density [kg/m^3]
    por       : porosity [-]
    qb        : basal heat flow [W/m^2], signed (see module docstring)
    Ts        : surface temperature [deg C]
    dz, ip    : cell sizes [m] and cell -> unit pointer (0-based)
    maxiter, tol : Picard iterations / tolerance [K]
    freeze    : consider phase change (fluid/ice partition)
    props     : property set (see PropModel)

    Returns
    -------
    T (nz,)   or, if full_output, (T, dT, kbulk, ipor) with dT = gradient of
    the final profile, kbulk = effective conductivity, ipor = ice porosity.
    """
    P = get_props(props)
    C = _prepare_cells(k, kA, kB, h, r, por, qb, Ts, dz, ip, P)
    dz, nc, nz = C["dz"], C["nc"], C["nz"]
    km_of = P.km_evaluator(C["kA"], C["kB"])
    q_node = c2n(C["h"], dz)
    fkw = dict(freeze_kw) if freeze_kw else {}
    do_freeze = _is_freeze(freeze)
    por_c = C["por"]

    T = C["Tguess"].copy()
    Tlast = None
    keff = pori = None
    for itn in range(int(maxiter)):
        Tc = n2c(T)
        km = km_of(C["k"], Tc)
        ki = P.ki(Tc)
        kf = P.kf(Tc)
        if do_freeze:
            gf, _ = ftheta_apparent(Tc, freeze_model, rfl, **fkw)
        else:
            gf = np.ones(nc)
        porm = 1.0 - por_c
        porf = por_c * gf
        pori = por_c - porf
        keff = np.exp(
            np.log(kf) * porf + np.log(ki) * pori + np.log(km) * porm
        )
        lower, main, upper, acn = _tridiag_coeffs(keff, dz, C["dc"])
        rhs = -q_node.copy()
        rhs[-1] += 2.0 * acn * dz[-1] * qb / keff[-1]
        rhs[0] = Ts
        T = _solve_tridiag(lower, main, upper, rhs)
        if itn > 0 and np.max(np.abs(T - Tlast)) <= tol:
            break
        Tlast = T
    if full_output:
        dT = np.diff(T) / dz
        return T, dT, keff, pori
    return T


def heat1dnt(
    k,
    kA,
    kB,
    h,
    r,
    cp,
    por,
    qb,
    dz,
    ip,
    dt,
    it,
    GST,
    T0,
    theta=1.0,
    maxiter=10,
    tol=1e-5,
    freeze=1,
    out=1,
    props="full",
    *,
    freeze_model="lun",
    freeze_kw=None,
    rfl=0.025,
    Lh=LH_ICE,
    full_output=False,
):
    """Transient nonlinear 1-D heat equation with time-dependent surface
    temperature and optional permafrost (heat1dnt.m).

    Apparent-heat-capacity treatment of phase change:
        rc_eff = (1-por)*rc_m + por_i*rc_i + por_f*rc_f + por*rho_f*Lh*dTheta
    Time stepping with parameter theta (0.5 Crank-Nicolson, 1 backward
    Euler); nonlinearities by Picard iteration at every time step.

    Parameters
    ----------
    k, kA, kB, h, r, cp, por : per-unit arrays (density r, specific heat cp)
    qb   : basal heat flow [W/m^2], signed
    dz, ip : cell sizes and cell -> unit pointer (0-based)
    dt   : time steps [s], length nt-1
    it   : 0-based index into GST for each step (length >= nt-1)
    GST  : ground surface temperature parameters [deg C]
    T0   : initial temperature profile (nz,)
    theta: scalar or length nt-1 array
    out  : 0 -> return only the final profile (nz,)
           n>0 -> return array (nz, nt) with the profile after every n-th
                  step (other columns NaN); column 0 is T0.

    Returns
    -------
    T, or (T, dT, Q, kbulk, ipor) if full_output.
    NOTE (as in the MATLAB source) dT is diff(T0)/dz, i.e. the gradient of
    the INITIAL profile, and Q = keff*dT.
    """
    P = get_props(props)
    dt = np.asarray(dt, dtype=float).ravel()
    it = np.asarray(it, dtype=int).ravel()
    GST = np.asarray(GST, dtype=float).ravel()
    nt = dt.size + 1
    C = _prepare_cells(k, kA, kB, h, r, por, qb, GST[0], dz, ip, P, cp=cp)
    dz, nc, nz = C["dz"], C["nc"], C["nz"]
    if it.size < nt - 1:
        raise ValueError("it must have at least len(dt) entries")
    thetstep = np.broadcast_to(np.asarray(theta, dtype=float), (nt - 1,))
    km_of = P.km_evaluator(C["kA"], C["kB"])
    q_node = c2n(C["h"], dz)
    fkw = dict(freeze_kw) if freeze_kw else {}
    do_freeze = _is_freeze(freeze)
    por_c, rm, cpm = C["por"], C["rm"], C["cpm"]
    Pch, Pcl, dc = C["Pch"], C["Pcl"], C["dc"]

    T0 = np.asarray(T0, dtype=float).ravel()
    if T0.size != nz:
        raise ValueError("T0 must have nz = len(ip)+1 entries")
    Tlast = T0.copy()
    Tlast[0] = GST[0]  # as in MATLAB: Tlast(1)=GST(1)
    Titer = Tlast.copy()

    store = int(out) != 0
    if store:
        Tout = np.full((nz, nt), np.nan)
        Tout[:, 0] = T0

    keff = pori = None
    Tnew = Tlast
    for itime in range(nt - 1):
        dti = dt[itime]
        th = thetstep[itime]
        for itn in range(int(maxiter)):
            Tc = n2c(Titer)
            # fluid (hydrostatic pressure), ice, matrix (lithostatic)
            rhof = P.rhof(Tc, Pch)
            rcf = rhof * P.cpf(Tc, Pch)
            kf = P.kf(Tc)
            rci = P.rhoi(Tc) * P.cpi(Tc)
            ki = P.ki(Tc)
            rcm = P.rcm(rm, cpm, Tc, Pcl)
            km = km_of(C["k"], Tc)
            if do_freeze:
                gf, dgf = ftheta_apparent(Tc, freeze_model, rfl, **fkw)
            else:
                gf, dgf = np.ones(nc), np.zeros(nc)
            porm = 1.0 - por_c
            porf = por_c * gf
            pori = por_c - porf
            keff = np.exp(
                np.log(kf) * porf + np.log(ki) * pori + np.log(km) * porm
            )
            rceff = (
                porm * rcm + pori * rci + porf * rcf + por_c * rhof * Lh * dgf
            )

            lower, main, upper, acn = _tridiag_coeffs(keff, dz, dc)
            rc_node = c2n(rceff, dz)
            rhs = q_node.copy()
            rhs[-1] -= 2.0 * acn * dz[-1] * qb / keff[-1]
            # F*A and F*rhs with F = diag(1/rc_node)
            F = 1.0 / rc_node
            lo_f = lower * F[1:]
            ma_f = main * F
            up_f = upper * F[:-1]
            rhs_f = rhs * F
            # L = I - dt*theta*A ; r = (I + dt*(1-theta)*A) Tlast + dt*rhs
            Lm = 1.0 - dti * th * ma_f
            Ll = -dti * th * lo_f
            Lu = -dti * th * up_f
            rr = (
                Tlast
                + dti * (1.0 - th) * _tri_matvec(lo_f, ma_f, up_f, Tlast)
                + dti * rhs_f
            )
            rr[0] = GST[it[itime]]  # Dirichlet, top
            Lm[0] = 1.0
            Lu[0] = 0.0
            Tnew = _solve_tridiag(Ll, Lm, Lu, rr)
            checktol = np.max(np.abs(Tnew - Titer))
            Titer = Tnew
            if checktol <= tol or itn + 1 >= maxiter:
                break
        Tlast = Tnew
        if store and (itime + 1) % int(out) == 0:
            Tout[:, itime + 1] = Tnew

    T = Tout if store else Tlast
    if full_output:
        dT = np.diff(T0) / dz
        return T, dT, keff * dT, keff, pori
    return T


def forward_solve(
    m,
    k,
    kA,
    kB,
    h,
    r,
    c,
    p,
    ip,
    dz,
    qb,
    gts,
    it,
    dt,
    T0,
    theta=1.0,
    maxitnl=10,
    tolnl=1e-5,
    freeze=1,
    out=1,
    props="full",
    **kw,
):
    """Forward problem for a GST model m (forward_solve.m).

    GST = m + gts; if T0 is None the initial state is the stationary
    solution for surface temperature GST[0].
    Returns (Tcalc (nz,nt), dT, Q, K).
    """
    GST = np.asarray(m, dtype=float).ravel() + gts
    if T0 is None:
        T0 = heat1dns(
            k,
            kA,
            kB,
            h,
            r,
            p,
            qb,
            GST[0],
            dz,
            ip,
            maxitnl,
            tolnl,
            freeze,
            props,
            **_ns_kw(kw),
        )
    return heat1dnt(
        k,
        kA,
        kB,
        h,
        r,
        c,
        p,
        qb,
        dz,
        ip,
        dt,
        it,
        GST,
        T0,
        theta,
        maxitnl,
        tolnl,
        freeze,
        1,
        props,
        full_output=True,
        **kw,
    )


def _ns_kw(kw):
    """Keyword subset understood by heat1dns."""
    return {a: kw[a] for a in ("freeze_model", "freeze_kw", "rfl") if a in kw}


@dataclass
class ForwardModel:
    """Picklable callable: GST parameter vector -> temperature field.

    Bundles everything heat1dns/heat1dnt need except the GST parameters, so
    that inversion / MCMC code (and process pools) can treat the physical
    model as a plain function:  Tfield = fm(m)  or  Tfinal = fm.final(m).
    """

    k: np.ndarray
    kA: np.ndarray
    kB: np.ndarray
    h: np.ndarray
    r: np.ndarray
    c: np.ndarray
    p: np.ndarray
    ip: np.ndarray
    dz: np.ndarray
    dt: np.ndarray
    it: np.ndarray
    qb: float
    gts: float = 0.0
    T0: Optional[np.ndarray] = None
    theta: float = 1.0
    maxitnl: int = 10
    tolnl: float = 1e-5
    freeze: int = 1
    props: Any = "full"
    kw: dict = field(default_factory=dict)

    def __call__(self, m, qb=None, T0=None):
        qb = self.qb if qb is None else qb
        Tc, *_ = forward_solve(
            m,
            self.k,
            self.kA,
            self.kB,
            self.h,
            self.r,
            self.c,
            self.p,
            self.ip,
            self.dz,
            qb,
            self.gts,
            self.it,
            self.dt,
            self.T0 if T0 is None else T0,
            self.theta,
            self.maxitnl,
            self.tolnl,
            self.freeze,
            1,
            self.props,
            **self.kw,
        )
        return Tc

    def final(self, m, **kwargs):
        """Temperature profile at the final time (nz,)."""
        return self(m, **kwargs)[:, -1]


# ---------------------------------------------------------------------------
# Analytical step-model forward solution (Beltrami / Mareschal)
# ---------------------------------------------------------------------------


def msetup(time, diffu, conduc, zdat, tlog, refyr, out=0):
    """Model matrix of the analytical (half-space, piecewise-constant GST)
    forward problem (msetup.m). Columns: one per GST step, then surface
    temperature offset, then basal heat-flow term z/conduc.

    time  : times before logging, ascending, positive [s]
    diffu : thermal diffusivity [m^2/s]; conduc : conductivity [W/(m K)]
    zdat  : depths [m]; tlog, refyr : logging year and reference year.
    """
    time = np.asarray(time, dtype=float).ravel()
    zdat = np.asarray(zdat, dtype=float).ravel()
    shift = refyr - tlog
    nt = time.size
    npar = nt + 2
    M = np.zeros((zdat.size, npar))
    active = np.flatnonzero(time > shift)
    if active.size:
        tinv = 0.5 / np.sqrt(diffu * (time[active] - shift))
        E = erfc(zdat[:, None] * tinv[None, :])
        M[:, active] = E
        if active.size > 1:
            M[:, active[1:]] = E[:, 1:] - E[:, :-1]
    M[:, npar - 2] = 1.0
    M[:, npar - 1] = zdat / conduc
    return M


def heat1dat(k, r, c, Qb, z, t, gst, T0, tlog, refyr, out=0):
    """Analytical temperature profile for a step-wise GST history
    (heat1dat.m). Returns (T, Q) as dicts with the MATLAB field names
    T = {val, z, grd, zm}, Q = {val, z}.

    t is the (negative) time of the GST nodes, gst the corresponding
    temperatures (both in chronological order, as in MATLAB); the method
    reverses them internally.
    """
    z = np.asarray(z, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    gst = np.asarray(gst, dtype=float).ravel()
    gtime = -t[::-1]
    gsth = gst[::-1]
    kappa = k / (r * c)
    parvec = np.concatenate([gsth, [T0], [-Qb]])
    M = msetup(gtime, kappa, k, z, tlog, refyr, out)
    Temp = M @ parvec
    zm = 0.5 * (z[:-1] + z[1:])
    grd = np.diff(Temp) / np.diff(z)
    T = {"val": Temp, "z": z, "grd": grd, "zm": zm}
    Q = {"val": -k * grd, "z": zm}
    return T, Q


# ===========================================================================
# 5. Finite-difference sensitivities (Jacobian) with respect to the GSTH
# ===========================================================================


def _sens_column(args):
    """Worker: one Jacobian column (one perturbed GST parameter)."""
    icol, kw, gst, it, dt, Tc, del_ = args
    idx = np.flatnonzero(it[: dt.size + 1] == icol)
    if idx.size == 0:
        return icol, None
    istart = int(idx[0])
    dgst = gst.copy()
    dgst[icol] += del_
    T = heat1dnt(
        kw["k"],
        kw["kA"],
        kw["kB"],
        kw["h"],
        kw["r"],
        kw["c"],
        kw["p"],
        kw["qb"],
        kw["dz"],
        kw["ip"],
        dt[istart:],
        it[istart:],
        dgst,
        Tc[:, istart],
        kw["theta"],
        kw["maxitnl"],
        kw["tolnl"],
        kw["freeze"],
        0,
        kw["props"],
        **kw["extra"],
    )
    return icol, (T - Tc[:, -1]) / del_


def sensfdt_pal(
    k,
    kA,
    kB,
    h,
    r,
    c,
    p,
    qb,
    dz,
    ip,
    dt,
    it,
    gst,
    T0,
    theta,
    maxitnl,
    tolnl,
    dp,
    freeze,
    props="full",
    n_jobs=1,
    **kw,
):
    """Finite-difference Jacobian of the final temperature profile with
    respect to the GST parameters and qb (sensfdt_pals.m / sensfdt_palp.m).

    Column i (0..ngst-1) is the response to a perturbation dp [K] of gst[i],
    computed by restarting the transient run at the first time step that
    uses gst[i] from the unperturbed state (an exact shortcut, because gst[i]
    cannot influence earlier times). Column ngst is the response to the
    relative perturbation dp*qb of the basal heat flow (full run from T0).
    GST parameters without any time step have zero columns.

    n_jobs = 1 : serial (sensfdt_pals);  n_jobs > 1 : process pool (palp).

    Returns (J (nz, ngst+1), Tc (nz, nt) = unperturbed solution).

    Differences to the MATLAB code: pals put the columns of used parameters
    into consecutive positions (misaligned when a parameter has no time
    step) and computed the qb column from a truncated time range with a full
    initial state; palp left the qb column at zero. Both are replaced by the
    straightforward definition above (callers drop the qb column anyway).
    """
    gst = np.asarray(gst, dtype=float).ravel().copy()
    dt = np.asarray(dt, dtype=float).ravel()
    it = np.asarray(it, dtype=int).ravel()
    dz = np.asarray(dz, dtype=float).ravel()
    nz = dz.size + 1
    ngst = gst.size
    if T0 is None:
        T0 = heat1dns(
            k,
            kA,
            kB,
            h,
            r,
            p,
            qb,
            gst[0],
            dz,
            ip,
            maxitnl,
            tolnl,
            freeze,
            props,
            **_ns_kw(kw),
        )
    Tc = heat1dnt(
        k,
        kA,
        kB,
        h,
        r,
        c,
        p,
        qb,
        dz,
        ip,
        dt,
        it,
        gst,
        T0,
        theta,
        maxitnl,
        tolnl,
        freeze,
        1,
        props,
        **kw,
    )
    kwd = dict(
        k=k,
        kA=kA,
        kB=kB,
        h=h,
        r=r,
        c=c,
        p=p,
        qb=qb,
        dz=dz,
        ip=ip,
        theta=theta,
        maxitnl=maxitnl,
        tolnl=tolnl,
        freeze=freeze,
        props=props,
        extra=kw,
    )
    del_ = dp * 1.0  # dTref = 1 K
    J = np.zeros((nz, ngst + 1))
    tasks = [(i, kwd, gst, it, dt, Tc, del_) for i in range(ngst)]
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            res = list(ex.map(_sens_column, tasks))
    else:
        res = [_sens_column(a) for a in tasks]
    for icol, col in res:
        if col is not None:
            J[:, icol] = col
    # basal heat flow
    delq = dp * qb if qb != 0 else dp
    Tq = heat1dnt(
        k,
        kA,
        kB,
        h,
        r,
        c,
        p,
        qb + delq,
        dz,
        ip,
        dt,
        it,
        gst,
        T0,
        theta,
        maxitnl,
        tolnl,
        freeze,
        0,
        props,
        **kw,
    )
    J[:, ngst] = (Tq - Tc[:, -1]) / delq
    return J, Tc


# ===========================================================================
# 6. Regularisation and (linearised) inversion helpers
# ===========================================================================


def reg1d(k, flag="l0"):
    """Regularisation operator L (sparse, k x k) (reg1d.m).

    'l0'/'marq' : identity (smallest model)
    'l1'/'grad' : first differences, last row zero (flattest model)
    'l2'/'lap'  : second differences, first and last row zero (smoothest)
    """
    k = int(k)
    e = np.ones(k)
    f = str(flag).lower()
    if f in ("l0", "marq"):
        return sp.identity(k, format="csr")
    if f in ("l1", "grad"):
        Op = sp.diags([-e, e[:-1]], [0, 1], shape=(k, k), format="lil")
        Op[k - 1, k - 1] = 0.0
        return Op.tocsr()
    if f in ("l2", "lap"):
        Op = sp.diags(
            [e[:-1], -2 * e, e[:-1]], [-1, 0, 1], shape=(k, k), format="lil"
        )
        Op[0, 0] = 0.0
        Op[0, 1] = 0.0
        Op[k - 1, k - 2] = 0.0
        Op[k - 1, k - 1] = 0.0
        return Op.tocsr()
    raise ValueError("reg1d: unknown regularisation '%s'" % flag)


def setregpar(regvec0, reg0vec, reg1vec, reg2vec):
    """Grid of regularisation parameter triples (setregpar.m): all
    combinations reg0vec x reg1vec x reg2vec, each scaled by regvec0.
    Returns array (n, 3); reg0 varies slowest."""
    regvec0 = np.asarray(regvec0, dtype=float).ravel()
    rows = [
        (a * regvec0[0], b * regvec0[1], c * regvec0[2])
        for a, b, c in itertools.product(
            np.atleast_1d(reg0vec),
            np.atleast_1d(reg1vec),
            np.atleast_1d(reg2vec),
        )
    ]
    return np.array(rows, dtype=float)


setregvec = setregpar  # setregvec.m is an identical copy


def pvar(p0, pfac=None):
    """Test regularisation parameters p0 .* pfac (pvar.m)."""
    p0 = np.asarray(p0, dtype=float).ravel()
    pfac = (
        np.ones_like(p0) if pfac is None else np.asarray(pfac, float).ravel()
    )
    return p0 * pfac


def lsqr_solve(A, b, tol=1e-6, maxiter=None):
    """MATLAB lsqr(A,b,tol,maxit) via scipy. atol = btol = tol. The two
    implementations use slightly different stopping tests, so iterates are
    not bit-identical to MATLAB."""
    return _sp_lsqr(A, b, atol=tol, btol=tol, conlim=1e12, iter_lim=maxiter)[0]


def _stack_operator(Jw, L0, L1, L2, regpar):
    s0, s1, s2 = np.sqrt(np.asarray(regpar, dtype=float))
    return sp.vstack(
        [sp.csr_matrix(Jw), s2 * L2, s1 * L1, s0 * L0], format="csr"
    ), (s0, s1, s2)


def gauss_newton_step(
    Jw, L0, L1, L2, m, m_apr, res, regpar, tol, maxiter, Wd=None
):
    """Regularised Gauss-Newton update dm by LSQR, as in Tikh_gsth.m:

        [Jw; sqrt(t3)*L2; sqrt(t2)*L1; sqrt(t1)*L0] dm =
        [res; -sqrt(t3)*L2*(m-m_apr); -sqrt(t2)*L1*(...); -sqrt(t1)*L0*(...)]

    `res` enters UNWEIGHTED, exactly as in the MATLAB code, although Jw is
    weighted with Wd. This is only consistent for Terr == const == 1. Pass a
    sparse diagonal Wd (=diag(1/Terr)) to use Wd@res instead.
    """
    m = np.asarray(m, dtype=float).ravel()
    m_apr = np.asarray(m_apr, dtype=float).ravel()
    res = np.asarray(res, dtype=float).ravel()
    A, (s0, s1, s2) = _stack_operator(Jw, L0, L1, L2, regpar)
    dm = m - m_apr
    d_part = res if Wd is None else Wd @ res
    rhs = np.concatenate(
        [d_part, -s2 * (L2 @ dm), -s1 * (L1 @ dm), -s0 * (L0 @ dm)]
    )
    return lsqr_solve(A, rhs, tol, maxiter)


def solvereg(
    Jw,
    L0,
    L1,
    L2,
    m,
    m_apr,
    res,
    regpar,
    tol,
    maxiter,
    forward,
    Tobs,
    id,
    Terr,
    weight_residual=False,
):
    """Solve one regularised linearised step for a trial regularisation
    parameter triple and evaluate GCV / UPRE (solvereg.m).

    forward : callable m -> temperature profile at final time (nz,)
              (e.g. ForwardModel.final)
    Returns (m_loc, T_loc, r_loc, theta_d, theta_m, r_norm, m_norm, gcv, upr)

    GCV and UPR are implemented literally as in MATLAB, including
    upr = |Wd r|^2/nd - 2|Terr|^2 + 2|Terr|^2 tr(Jw*Jdag)/nd with
    |Terr| the 2-norm of the vector Terr (not a per-datum variance).
    """
    m = np.asarray(m, dtype=float).ravel()
    m_apr = np.asarray(m_apr, dtype=float).ravel()
    Terr = np.asarray(Terr, dtype=float).ravel()
    id = np.asarray(id, dtype=int).ravel()
    nd = id.size
    Wd = sp.diags(1.0 / Terr)
    delta_m = gauss_newton_step(
        Jw,
        L0,
        L1,
        L2,
        m,
        m_apr,
        res,
        regpar,
        tol,
        maxiter,
        Wd if weight_residual else None,
    )
    m_loc = m + delta_m
    T_loc = forward(m_loc)
    r_loc = np.asarray(Tobs, dtype=float).ravel() - T_loc[id]
    s0, s1, s2 = np.sqrt(np.asarray(regpar, dtype=float))
    Wm = s0 * L0 + s1 * L1 + s2 * L2
    theta_m = float(np.linalg.norm(Wm @ (m_loc - m_apr)) ** 2)
    wr = Wd @ r_loc
    theta_d = float(np.linalg.norm(wr) ** 2)
    r_norm = float(np.linalg.norm(wr))
    m_norm = float(np.linalg.norm(Wm @ (m_loc - m_apr)))
    reg = np.asarray(regpar, dtype=float)
    AtA = (
        Jw.T @ Jw
        + reg[0] * (L0.T @ L0).toarray()
        + reg[1] * (L1.T @ L1).toarray()
        + reg[2] * (L2.T @ L2).toarray()
    )
    try:
        Jdag = np.linalg.solve(AtA, Jw.T)
    except np.linalg.LinAlgError:
        warnings.warn("solvereg: singular normal matrix, using pinv")
        Jdag = np.linalg.pinv(AtA) @ Jw.T
    M = Jw @ Jdag
    tr1 = np.trace(np.eye(M.shape[0]) - M)
    gcv = nd * theta_d / tr1**2
    nT2 = float(np.linalg.norm(Terr) ** 2)
    upr = theta_d / nd - 2.0 * nT2 + 2.0 * nT2 * np.trace(M) / nd
    return m_loc, T_loc, r_loc, theta_d, theta_m, r_norm, m_norm, gcv, upr


def geninv(Jw, reg):
    """Generalised inverse G = inv(A'A) Jw' with the stacked operator
    A = [Jw; sqrt(reg3) L2; sqrt(reg2) L1; sqrt(reg1) L0] (GI.m/geninv.m)."""
    Jw = np.asarray(Jw, dtype=float)
    npar = Jw.shape[1]
    L0, L1, L2 = reg1d(npar, "l0"), reg1d(npar, "l1"), reg1d(npar, "l2")
    A, _ = _stack_operator(Jw, L0, L1, L2, reg)
    A = A.toarray()
    return np.linalg.solve(A.T @ A, Jw.T)


def resmat(Jw, reg):
    """(Rm, Rd) = (G G', G' G) with G = geninv(Jw, reg) (resmat.m / Rm.m).
    NOTE: as coded in MATLAB, 'Rm' is G*G' (which propagates unit data
    errors into the model), not the model resolution matrix G*Jw."""
    G = geninv(Jw, reg)
    return G @ G.T, G.T @ G


# ===========================================================================
# 7. Averaging, filters, statistics, covariance and scaling helpers
# ===========================================================================


def vavg(v, w=None, flag="a"):
    """Weighted arithmetic ('a'), geometric ('g'), harmonic ('h') or
    square-root ('s') mean of a vector (vavg.m)."""
    v = np.asarray(v, dtype=float).ravel()
    w = np.ones_like(v) if w is None else np.asarray(w, dtype=float).ravel()
    w = w / w.sum()
    f = str(flag).lower()
    if f in ("a", "arithmetic"):
        return float(np.sum(v * w))
    if f in ("g", "geometric"):
        return float(np.exp(np.sum(np.log(v) * w)))
    if f in ("h", "harmonic"):
        return float(1.0 / np.sum(w / v))
    if f in ("s", "sqrt"):
        return float(np.sum(w * np.sqrt(v)) ** 2)
    warnings.warn("vavg: unknown average type, arithmetic used")
    return float(np.sum(v * w))


def avg(w, v, mode="a"):
    """Row-wise weighted mean of matrices w (weights) and v (values)
    (avg.m). The 'h' branch of the MATLAB file used matrix division `/`
    (a bug that returns an n x n matrix); the element-wise formula is used."""
    w = np.atleast_2d(np.asarray(w, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    wn = w.sum(axis=1)
    if not np.allclose(wn, 1.0):
        warnings.warn("avg: sum of volumes not 1!")
    m = (mode or "a").lower()
    if m in ("a", "ari", "arithmetic"):
        return np.sum(w * v, axis=1) / wn
    if m in ("g", "geo", "geometric"):
        return np.exp(np.sum(np.log(v) * w, axis=1) / wn)
    if m in ("h", "har", "harmonic"):
        return wn / np.sum(w / v, axis=1)
    if m in ("s", "sqr", "square"):
        return (np.sum(w * np.sqrt(v), axis=1) / wn) ** 2
    warnings.warn("avg: mode '%s' not implemented, arithmetic assumed" % mode)
    return np.sum(w * v, axis=1) / wn


def wfilt(x, M, N=1, fmode=("b", "mir")):
    """Weighted moving-average filter of length M (odd; even -> M+1), applied
    N times (wfilt.m). fmode = (filter, padding): filter 'b' boxcar or 't'
    triangular; padding 'a' average or 'm' mirror (edge sample repeated)."""
    x = np.asarray(x, dtype=float).ravel()
    nx = x.size
    M = int(M)
    if M % 2 == 0:
        M += 1
    L = (M - 1) // 2
    if L == 0:
        return x.copy()
    fil, pad = str(fmode[0]).lower(), str(fmode[1]).lower()
    w = tri(2 * L + 1) if fil in ("t", "tri", "triangular") else box(2 * L + 1)
    if pad in ("m", "mir", "mirror"):
        x1, x2 = x[:L][::-1], x[nx - L :][::-1]
    else:
        x1 = np.full(L, x[:L].mean())
        x2 = np.full(L, x[nx - L :].mean())
    xa = np.concatenate([x1, x, x2])
    nxa = xa.size
    for _ in range(int(N)):
        xb = xa.copy()
        xb[L : nxa - L] = np.convolve(xa, w, mode="valid")
        xa = xb
    return xa[L : nxa - L]


def smooth1(f):
    """Smoothing with weights (1/4, 1/2, 1/4), end values repeated
    (smooth1.m)."""
    f = np.asarray(f, dtype=float).ravel()
    fp = np.concatenate([[f[0]], f, [f[-1]]])
    return np.convolve(fp, [0.25, 0.5, 0.25], mode="valid")


def rms(res, err=None):
    """Root mean square of the finite residuals, divided by (n-1) (RMS.m).
    err may be a scalar or a vector; the finite mask of res is applied to
    both (MATLAB applied it to err after res was already filtered)."""
    res = np.asarray(res, dtype=float).ravel()
    ok = np.isfinite(res)
    if err is None:
        e = np.ones(ok.sum())
    else:
        err = np.asarray(err, dtype=float).ravel()
        e = np.full(ok.sum(), err[0]) if err.size == 1 else err[ok]
    r = res[ok] / e
    return float(np.sqrt(np.sum(r * r) / (r.size - 1)))


def mad(val):
    """Scaled median absolute deviation, 1.4826*median(|x-median(x)|)
    (mad.m)."""
    val = np.asarray(val, dtype=float)
    return 1.4826 * np.median(np.abs(val - np.median(val)))


def covar_gauss(s, L, thresh=None):
    """Gaussian covariance operator (CovarGauss.m).

    C = exp(-|i-j|^2/(2 L^2)) * diag(s^2), eigenvalues clipped at `thresh`
    (default 1e3*eps) and re-assembled. Kept as in MATLAB: C*diag(s^2) is
    NOT symmetric (a symmetric version would be S*C*S)."""
    s = np.asarray(s, dtype=float).ravel()
    thresh = 1e3 * np.finfo(float).eps if thresh is None else thresh
    n = s.size
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    C = np.exp(-(d**2) / (2.0 * L**2)) * (s * s)[None, :]
    lam, V = np.linalg.eig(C)
    lam = np.maximum(lam.real, thresh)
    C = np.linalg.solve(V.T, (V * lam[None, :]).T).T
    return C.real


def covar_expnl(s, L, thresh=0.0):
    """Markovian (exponential) covariance (CovarExpnl.m):
    exp(-|i-j|/L) * diag(s^2), plus thresh*eps if thresh != 0."""
    s = np.asarray(s, dtype=float).ravel()
    n = s.size
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    C = np.exp(-d / L) * (s * s)[None, :]
    if abs(thresh) > 0:
        C = C + thresh * np.finfo(float).eps * np.ones((n, n))
    return C


def f_reg(Dm):
    """Edge-preserving regularisation weights (f_reg.m; Youzwishen & Sacchi
    2006). Returns (J, Q, S): functional value, diag(1/(1+Dm^2)^2), and its
    square root."""
    Dm = np.asarray(Dm, dtype=float).ravel()
    Dm2 = np.abs(Dm) ** 2
    J = float(np.sum(Dm2 / (1.0 + Dm2)))
    d = 1.0 / (1.0 + Dm2) ** 2
    return J, sp.diags(d), sp.diags(np.sqrt(d))


def f_scal(J, mode="col"):
    """Norm scaling of a Jacobian (f_scal.m; Mehanee & Zhdanov 2002):
    diag of the column ('col') or row ('row') 2-norms."""
    J = np.asarray(J, dtype=float)
    V = np.sqrt(np.sum(J**2, axis=0 if str(mode).lower() == "col" else 1))
    return sp.diags(V)


def scale_jacobian(J, mode="col"):
    """Sum-normalising scale of a Jacobian (mscalsum.m; scaljac.m was
    unrunnable and is superseded by it). Returns (S, JS).

    'col': S = diag(1/rowsum(J)), JS = S*J;  'row': S = diag(1/colsum(J)),
    JS = J*S. (MATLAB names are historical: the mode refers to the
    parameter axis being scaled.)"""
    J = np.asarray(J, dtype=float)
    if str(mode).lower() == "col":
        S = sp.diags(1.0 / J.sum(axis=1))
        return S, S @ J
    S = sp.diags(1.0 / J.sum(axis=0))
    return S, J @ S


def splinefit(x, y, breaks, beta=0.5, order=4):
    """Least-squares spline fit with the robust re-weighting of Lundgren's
    splinefit.m (3 iterations, weights exp(-alpha*r^2/mean(r^2)),
    alpha = 0.5*beta/(1-beta)); beta=0 -> ordinary least squares.
    Returns a scipy BSpline (call it like ppval). Periodic boundary
    conditions and linear constraints of splinefit.m are not supported."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    br = np.asarray(breaks, dtype=float).ravel()
    kdeg = int(order) - 1
    t = np.concatenate([[br[0]] * (kdeg + 1), br[1:-1], [br[-1]] * (kdeg + 1)])
    spl = make_lsq_spline(x, y, t, k=kdeg)
    if beta > 0:
        alpha = 0.5 * beta / (1.0 - beta)
        for _ in range(3):
            rr = (spl(x) - y) ** 2
            rrmean = rr.mean() or 1.0
            w = np.exp(-(alpha / rrmean) * rr)
            spl = make_lsq_spline(x, y, t, k=kdeg, w=w)
    return spl


def zgrad(T, z, k, smooth=0):
    """Vertical heat-flow density Qz = -k dT/dz at cell centres (zgrad.m).
    smooth > 0: robust spline smoothing with `smooth` break points.
    Returns (Qz, zq, dTdz)."""
    T = np.asarray(T, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    zq = 0.5 * (z[:-1] + z[1:])
    dTdz = np.diff(T) / np.diff(z)
    Qz = -np.asarray(k, dtype=float) * dTdz
    if smooth > 0:
        spoints = np.linspace(zq.min(), zq.max(), int(smooth))
        Qz = splinefit(zq, Qz, spoints, beta=0.5)(zq)
    return Qz, zq, dTdz
