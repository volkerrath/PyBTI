"""
gsth_drivers.py -- Data containers, forward drivers and the Tikhonov GSTH
inversion (translation of the MATLAB driver scripts).

Translated from: Tikh_gsth.m (current version, March 2019), GSTH_TikhSX.m
(Aug 2014) and L1_gsth.m (both are variants of Tikh_gsth.m and are covered by
the options of tikhonov_gsth), Fwd_gsth.m, Fwd_stat.m, Fwd_tran.m.
All numerical work is done by numeric.py; physical properties come from
phys.py through numeric.PropModel.

MATLAB .mat "structure" files (name_SitePar.mat, name_FwdPar.mat,
name_InvPar.mat, name_Init.mat) are replaced by the dataclasses SitePar,
FwdPar, InvPar; load_matlab_site() reads them from .mat files (index arrays
are converted from 1-based to 0-based). That loader could only be tested on
synthetic .mat files - no real site files were provided.

Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-21
Status         : AI-generated translation of the user's MATLAB code; tested
                 on synthetic problems only, NOT against MATLAB output.
                 Review before production use.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

import numeric as nm

__all__ = [
    "SitePar",
    "FwdPar",
    "InvPar",
    "make_forward_model",
    "fwd_gsth",
    "fwd_stat",
    "fwd_tran",
    "tikhonov_gsth",
    "load_matlab_site",
]


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class SitePar:
    """Site model and observations (MATLAB struct `sitepar`).

    z, t      : depth nodes [m] (z[0]=0) and time nodes [s] (past < 0, last=0)
    ip        : cell -> unit pointer, len(z)-1, 0-BASED
    k,kA,kB,h,r,c,p : per-unit conductivity, its coefficients A and B, heat
                production [W/m^3], density, specific heat, porosity
    qb        : basal heat flow [W/m^2], signed (normal flow < 0)
    gts       : GST shift added to the GST parameters
    id        : observation node indices, 0-BASED; Tobs, Terr observed
                temperatures and errors; zobs optional observation depths
    props     : property set of phys.SITE_PROPS (e.g. "full")
    """

    z: np.ndarray
    t: np.ndarray
    ip: np.ndarray
    k: np.ndarray
    kA: np.ndarray
    kB: np.ndarray
    h: np.ndarray
    r: np.ndarray
    c: np.ndarray
    p: np.ndarray
    qb: float
    gts: float
    id: np.ndarray
    Tobs: np.ndarray
    Terr: np.ndarray
    zobs: Optional[np.ndarray] = None
    props: Any = "full"
    name: str = "site"

    def __post_init__(self):
        for a in (
            "z",
            "t",
            "k",
            "kA",
            "kB",
            "h",
            "r",
            "c",
            "p",
            "Tobs",
            "Terr",
        ):
            setattr(
                self,
                a,
                np.atleast_1d(
                    np.asarray(getattr(self, a), dtype=float)
                ).ravel(),
            )
        self.ip = np.asarray(self.ip, dtype=int).ravel()
        self.id = np.asarray(self.id, dtype=int).ravel()
        if self.ip.size != self.z.size - 1:
            raise ValueError("ip must have len(z)-1 entries")
        if self.Tobs.size != self.id.size or self.Terr.size != self.id.size:
            raise ValueError("id, Tobs, Terr must have equal length")

    @property
    def dz(self):
        return np.diff(self.z)

    @property
    def dt(self):
        return np.diff(self.t)


@dataclass
class FwdPar:
    """Numerical control of the forward model (MATLAB struct `fwdpar`).
    The defaults are placeholders: the MATLAB values came from the user's
    .mat files. theta: 0.5 Crank-Nicolson, 1 backward Euler."""

    theta: float = 1.0
    maxitnl: int = 10
    tolnl: float = 1.0e-5
    freeze: int = 1
    kw: dict = field(default_factory=dict)  # freeze_model, freeze_kw, rfl


@dataclass
class InvPar:
    """Control of the Tikhonov inversion (MATLAB struct `invpar`).

    it            : 0-based GST index for every time node (MATLAB: pt)
    nsteps        : number of GST parameters (default max(it)+1)
    m_apr_set / m_ini_set : scalar, or (mean, std) -> mean + std*randn
    tol_inv       : (rms tolerance, minimum rms improvement)
    reg_opt       : 'gcv', 'upr' ('upre'), or 'fix' (first grid entry)
    Defaults are placeholders; set them explicitly for real work.
    """

    it: np.ndarray
    nsteps: Optional[int] = None
    m_apr_set: Any = 0.0
    m_ini_set: Any = 0.0
    maxiter_inv: int = 10
    tol_inv: tuple = (0.0, 1.0e-4)
    start_regpar: int = 1
    modul_regpar: int = 1
    regpar0: tuple = (1.0, 1.0, 1.0)
    regbase: tuple = (1.0, 1.0, 1.0)
    reg0par: Any = (1.0,)
    reg1par: Any = (1.0,)
    reg2par: Any = (1.0,)
    reg_opt: str = "gcv"
    reg_shift: int = 0
    tol_solve: float = 1.0e-6
    maxiter_solve: int = 1000
    dp: float = 1.0e-3
    outsteps: bool = False
    stop_needs_start_regpar: bool = True  # True: Tikh_gsth; False: SX / L1
    seed: Optional[int] = None


def make_forward_model(site: SitePar, fwd: FwdPar, it, T0=None):
    """ForwardModel (callable m -> T field) for a site."""
    return nm.ForwardModel(
        k=site.k,
        kA=site.kA,
        kB=site.kB,
        h=site.h,
        r=site.r,
        c=site.c,
        p=site.p,
        ip=site.ip,
        dz=site.dz,
        dt=site.dt,
        it=np.asarray(it, dtype=int),
        qb=site.qb,
        gts=site.gts,
        T0=T0,
        theta=fwd.theta,
        maxitnl=fwd.maxitnl,
        tolnl=fwd.tolnl,
        freeze=fwd.freeze,
        props=site.props,
        kw=dict(fwd.kw),
    )


def _equilibrium(site, fwd, Ts, h=None, qb=None):
    return nm.heat1dns(
        site.k,
        site.kA,
        site.kB,
        site.h if h is None else h,
        site.r,
        site.p,
        site.qb if qb is None else qb,
        Ts,
        site.dz,
        site.ip,
        fwd.maxitnl,
        fwd.tolnl,
        fwd.freeze,
        site.props,
        **nm._ns_kw(fwd.kw),
    )


def _append_info(outdir, line):
    with open(os.path.join(outdir, "INFO.dat"), "a+") as fh:
        fh.write(line + " \n")


# ---------------------------------------------------------------------------
# forward drivers (Fwd_gsth.m, Fwd_stat.m, Fwd_tran.m)
# ---------------------------------------------------------------------------
def fwd_gsth(
    site: SitePar, fwd: FwdPar, GST, it, Tinit=None, outdir=None, verbose=True
):
    """Transient forward model for a given GST series and residuals
    (Fwd_gsth.m). Initial state: Tinit, else stationary for GST[0].
    Returns dict(Tcalc, z, Tobs, zobs, r, resid, id, rms, mae)."""
    GST = np.asarray(GST, dtype=float).ravel()
    T0 = Tinit if Tinit is not None else _equilibrium(site, fwd, GST[0])
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
        np.asarray(it, int),
        GST,
        T0,
        fwd.theta,
        fwd.maxitnl,
        fwd.tolnl,
        fwd.freeze,
        1,
        site.props,
        **fwd.kw,
    )
    r = site.Tobs - Tcalc[site.id, -1]
    resid = r / site.Terr
    rms = float(np.sqrt(np.sum(resid**2) / resid.size))
    mae = float(np.sum(np.abs(resid)) / resid.size)
    if verbose:
        print(" ...... rmse for this model = %g" % rms)
        print(" ...... mae for this model  = %g" % mae)
    if outdir:
        _append_info(outdir, " %10.6g %10.6g %s" % (rms, mae, site.name))
    return dict(
        Tcalc=Tcalc,
        z=site.z,
        Tobs=site.Tobs,
        zobs=site.zobs,
        r=r,
        resid=resid,
        id=site.id,
        rms=rms,
        mae=mae,
    )


def fwd_stat(site: SitePar, fwd: FwdPar):
    """Stationary forward model with surface temperature gts (Fwd_stat.m).
    Returns dict(Tcalc, z)."""
    return dict(Tcalc=_equilibrium(site, fwd, site.gts), z=site.z)


def fwd_tran(
    site: SitePar, fwd: FwdPar, Tgst, it, Tinit=None, pom=None, out=0
):
    """Transient forward model, prescribed surface series Tgst (Fwd_tran.m).
    Initial state: Tinit, else stationary for pom (default Tgst[0]).
    out=0 -> final profile; out=n -> (nz, nt) array. Returns dict(Tcalc, z)."""
    Tgst = np.asarray(Tgst, dtype=float).ravel()
    T0 = (
        Tinit
        if Tinit is not None
        else _equilibrium(site, fwd, Tgst[0] if pom is None else pom)
    )
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
        np.asarray(it, int),
        Tgst,
        T0,
        fwd.theta,
        fwd.maxitnl,
        fwd.tolnl,
        fwd.freeze,
        out,
        site.props,
        **fwd.kw,
    )
    return dict(Tcalc=Tcalc, z=site.z)


# ---------------------------------------------------------------------------
# Tikhonov inversion (Tikh_gsth.m)
# ---------------------------------------------------------------------------
def _prior_vector(setting, n, rng):
    if isinstance(setting, (tuple, list)) and len(setting) == 2:
        return setting[0] + setting[1] * rng.standard_normal(n)
    return np.ones(n) * float(setting)


def _solvereg_task(a):
    (
        Jw,
        L0,
        L1,
        L2,
        m,
        m_apr,
        res,
        regpar,
        tol,
        maxit,
        fm,
        Tobs,
        id_,
        Terr,
        wres,
    ) = a
    return nm.solvereg(
        Jw,
        L0,
        L1,
        L2,
        m,
        m_apr,
        res,
        regpar,
        tol,
        maxit,
        fm.final,
        Tobs,
        id_,
        Terr,
        wres,
    )


def tikhonov_gsth(
    site: SitePar,
    fwd: FwdPar,
    inv: InvPar,
    T0=None,
    name=None,
    n_jobs=1,
    weight_residual=False,
    verbose=True,
    outdir=None,
):
    """Nonlinear Tikhonov (Gauss-Newton) GSTH inversion for one site.

    Follows Tikh_gsth.m: forward run, residual/objective bookkeeping,
    stopping tests, FD Jacobian (n_jobs>1: parallel), regularised update
    with fixed regpar0 for iter <= start_regpar, afterwards a grid search of
    the regularisation parameters (GCV/UPR) every modul_regpar-th iteration,
    then a-posteriori quantities from the generalised inverse.

    weight_residual : False = as MATLAB (unweighted residual on the rhs
        with a weighted Jacobian; identical only if Terr == 1); True = use
        Wd*res on the rhs (statistically consistent).
    outdir : if given, write <name>_results.npz, <name>_initial_out.npz,
        <name>_prior_out.npz and append a line to INFO.dat.

    Returns a dict with (MATLAB names): m, m_iter, r_iter, rms_iter,
    theta_iter, theta_m_iter, theta_d_iter, regpar_iter, Jw, Rmm, Rdd, Cmm,
    Tcalc (final forward run), niter and the last regparameter search
    (mL, GCV, UPR, ...).
    """
    name = name or site.name
    it = np.asarray(inv.it, dtype=int).ravel()
    nsteps = inv.nsteps or int(it.max()) + 1
    rng = np.random.default_rng(inv.seed)
    m_apr = _prior_vector(inv.m_apr_set, nsteps, rng)
    m = _prior_vector(inv.m_ini_set, nsteps, rng)
    npar = m.size
    id_, Tobs, Terr = site.id, site.Tobs, site.Terr
    nobs = id_.size
    if verbose:
        print("=" * 60)
        print("Nonlinear Tikhonov GSTH Inversion (Single Site): %s" % name)
        print("=" * 60)
        print(" ...... no of parameters   = %d" % npar)
        print(" ...... no of observations = %d" % nobs)

    L0, L1, L2 = (nm.reg1d(npar, f) for f in ("l0", "l1", "l2"))
    Wd = sp.diags(1.0 / Terr)
    fm = make_forward_model(site, fwd, it, T0)
    dp = inv.dp
    regpar0 = np.asarray(inv.regpar0, dtype=float)

    theold = 1.0e6
    regpar_iter = np.zeros((inv.maxiter_inv, 3))
    hist = dict(m=[], r=[], rms=[], theta=[], theta_m=[], theta_d=[])
    Jw = None
    search = {}
    iiter = 0
    Tcalc = None
    for iiter in range(1, inv.maxiter_inv + 1):
        GST = m + site.gts
        if fm.T0 is None:  # computed once, then reused
            fm.T0 = _equilibrium(site, fwd, GST[0])
        Tcalc = fm(m)
        res = Tobs - Tcalc[id_, -1]
        resid = res / Terr
        rms_ = float(np.sqrt(np.sum(resid**2) / resid.size))
        mae_ = float(np.sum(np.abs(resid)) / resid.size)
        if verbose:
            print(
                " ...... iteration %d  rms= %g  mae= %g   (%s)"
                % (iiter - 1, rms_, mae_, name)
            )
        hist["m"].append(m.copy())
        hist["r"].append(res.copy())
        rp = regpar_iter[iiter - 2] if iiter > 1 else regpar0
        s0, s1, s2 = np.sqrt(rp)
        Wm = s0 * L0 + s1 * L1 + s2 * L2  # lumped operator, as MATLAB
        th_m = float(np.linalg.norm(Wm @ (m - m_apr)) ** 2)
        th_d = float(np.linalg.norm(Wd @ res) ** 2)
        thenew = float(np.sqrt(th_d / res.size))
        hist["theta_m"].append(th_m)
        hist["theta_d"].append(th_d)
        hist["rms"].append(thenew)
        hist["theta"].append(th_d + th_m)
        if verbose and iiter > 1:
            print(
                " ... theta = %g  theta_d = %g  theta_m = %g  tau = "
                "(%5.2g %5.2g %5.2g)" % (th_d + th_m, th_d, th_m, *rp)
            )
        if outdir and inv.outsteps and thenew < theold:
            np.savez(
                os.path.join(
                    outdir, "%s_iter%d_rms%g.npz" % (name, iiter, thenew)
                ),
                m=m,
                theta_m_iter=hist["theta_m"],
                theta_d_iter=hist["theta_d"],
                rms_iter=hist["rms"],
            )
        ok_start = (
            (iiter >= inv.start_regpar)
            if inv.stop_needs_start_regpar
            else True
        )
        if (theold - thenew <= inv.tol_inv[1]) and ok_start:
            break
        if (thenew < inv.tol_inv[0]) and ok_start:
            break
        theold = thenew

        J, _ = nm.sensfdt_pal(
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
            fm.T0,
            fwd.theta,
            fwd.maxitnl,
            fwd.tolnl,
            dp,
            fwd.freeze,
            site.props,
            n_jobs=n_jobs,
            **fwd.kw,
        )
        Jw = Wd @ J[id_, :-1]  # drop the qb column

        if iiter <= inv.start_regpar:
            rl = regpar0
            regpar_iter[iiter - 1] = rl
            dm = nm.gauss_newton_step(
                Jw,
                L0,
                L1,
                L2,
                m,
                m_apr,
                res,
                rl,
                inv.tol_solve,
                inv.maxiter_solve,
                Wd if weight_residual else None,
            )
            m = m + dm
        elif iiter % inv.modul_regpar == 0:
            regpar = nm.setregpar(
                inv.regbase, inv.reg0par, inv.reg1par, inv.reg2par
            )
            tasks = [
                (
                    Jw,
                    L0,
                    L1,
                    L2,
                    m,
                    m_apr,
                    res,
                    regpar[i],
                    inv.tol_solve,
                    inv.maxiter_solve,
                    fm,
                    Tobs,
                    id_,
                    Terr,
                    weight_residual,
                )
                for i in range(regpar.shape[0])
            ]
            if n_jobs and n_jobs > 1:
                with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
                    out = list(ex.map(_solvereg_task, tasks))
            else:
                out = [_solvereg_task(a) for a in tasks]
            mL = np.array([o[0] for o in out])
            GCV = np.array([o[7] for o in out])
            UPR = np.array([o[8] for o in out])
            r_norm = np.array([o[5] for o in out])
            search = dict(
                regpar=regpar,
                mL=mL,
                GCV=GCV,
                UPR=UPR,
                theta_dL=np.array([o[3] for o in out]),
                theta_mL=np.array([o[4] for o in out]),
                norm_r=r_norm,
                norm_m=np.array([o[6] for o in out]),
                rms_L=r_norm / np.sqrt(nobs),
            )
            ro = str(inv.reg_opt).lower()
            if ro in ("g", "gcv"):
                val = np.abs(GCV - GCV.min())
            elif ro in ("u", "upr", "upre"):
                val = np.abs(UPR - UPR.min())
            elif ro in ("f", "fix"):
                val = np.zeros(regpar.shape[0])
            else:
                raise ValueError("reg_opt '%s' not implemented" % inv.reg_opt)
            index = int(np.argmin(val))
            if inv.reg_shift:
                index = min(max(index + int(inv.reg_shift), 0), val.size - 1)
            m = mL[index].copy()
            regpar_iter[iiter - 1] = regpar[index]
            if verbose:
                print(
                    " min(%s) = %g at index: %d  RegPar = %s"
                    % (ro.upper(), val[index], index + 1, regpar[index])
                )
        else:  # MATLAB: nothing happens
            regpar_iter[iiter - 1] = regpar_iter[iiter - 2]

    # a-posteriori quantities (Nolet-style generalised inverse)
    Rmm = Rdd = Cmm = None
    regs = regpar_iter[max(iiter - 2, 0)]
    if Jw is not None:
        A, _ = nm._stack_operator(Jw, L0, L1, L2, regs)
        A = A.toarray()
        AtA = A.T @ A
        Jdag = np.linalg.solve(AtA, Jw.T)
        Rmm, Rdd, Cmm = Jdag @ Jdag.T, Jdag.T @ Jdag, np.linalg.inv(AtA)

    m_iter = np.array(hist["m"])
    result = dict(
        m=m,
        m_apr=m_apr,
        m_iter=m_iter,
        r_iter=np.array(hist["r"]),
        rms_iter=np.array(hist["rms"]),
        theta_iter=np.array(hist["theta"]),
        theta_m_iter=np.array(hist["theta_m"]),
        theta_d_iter=np.array(hist["theta_d"]),
        regpar_iter=regpar_iter[:iiter],
        Jw=Jw,
        Rmm=Rmm,
        Rdd=Rdd,
        Cmm=Cmm,
        Tcalc=Tcalc,
        niter=iiter,
        it=it,
        search=search,
    )
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        flat = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        np.savez(os.path.join(outdir, "%s_results.npz" % name), **flat)
        m_last = m_iter[-1]
        for tag, key in (("initial", "m_ini"), ("prior", "m_apr")):
            np.savez(
                os.path.join(outdir, "%s_%s_out.npz" % (name, tag)),
                **{key: m_last, "it": it, "dt": site.dt, "t": site.t},
            )
        _append_info(
            outdir,
            " %10.6g %10.6g %10.6g %10.6g %6d %10.6g %10.6g "
            "%10.6g  Tikh%s %s"
            % (
                hist["rms"][-1],
                hist["theta_m"][-1],
                hist["theta_d"][-1],
                hist["theta"][-1],
                iiter,
                *regs,
                inv.reg_opt,
                name,
            ),
        )
    return result


# ---------------------------------------------------------------------------
# .mat loader (UNTESTED on real files)
# ---------------------------------------------------------------------------
def _get(s, key, default=None):
    if s is None:
        return default
    if isinstance(s, dict):
        return s.get(key, default)
    return getattr(s, key, default)


def _cell_or_scalar(v):
    if isinstance(v, np.ndarray) and v.dtype == object and v.size == 2:
        return (float(v[0]), float(v[1]))
    return v


def load_matlab_site(
    sitepar_file, fwdpar_file=None, invpar_file=None, name=None
):
    """Read MATLAB files <name>_SitePar.mat (struct `sitepar`), optionally
    _FwdPar.mat (`fwdpar`) and _InvPar.mat (`invpar`).

    ip, id (and invpar.pt, used as the time-node -> GST pointer `it`) are
    converted from 1-based to 0-based. Returns (SitePar, FwdPar, InvPar or
    None). Field names are those used in the MATLAB scripts; missing
    optional fields fall back to the dataclass defaults.
    """
    from scipy.io import loadmat

    def load(path, var):
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        return d[var] if var in d else d

    sp_ = load(sitepar_file, "sitepar")
    site = SitePar(
        z=_get(sp_, "z"),
        t=_get(sp_, "t"),
        ip=nm.from_matlab_index(_get(sp_, "ip")),
        k=_get(sp_, "k"),
        kA=_get(sp_, "kA"),
        kB=_get(sp_, "kB"),
        h=_get(sp_, "h"),
        r=_get(sp_, "r"),
        c=_get(sp_, "c"),
        p=_get(sp_, "p"),
        qb=float(_get(sp_, "qb")),
        gts=float(_get(sp_, "gts", 0.0)),
        id=nm.from_matlab_index(_get(sp_, "id")),
        Tobs=_get(sp_, "Tobs"),
        Terr=_get(sp_, "Terr"),
        zobs=_get(sp_, "zobs"),
        props=str(_get(sp_, "props", "full")),
        name=name or os.path.basename(str(sitepar_file)).split("_")[0],
    )
    fwd = FwdPar()
    if fwdpar_file:
        fp = load(fwdpar_file, "fwdpar")
        fwd = FwdPar(
            theta=float(_get(fp, "theta", fwd.theta)),
            maxitnl=int(_get(fp, "maxitnl", fwd.maxitnl)),
            tolnl=float(_get(fp, "tolnl", fwd.tolnl)),
            freeze=int(_get(fp, "freeze", fwd.freeze)),
        )
    inv = None
    if invpar_file:
        ip_ = load(invpar_file, "invpar")
        d = InvPar(it=nm.from_matlab_index(_get(ip_, "pt")))
        for f in (
            "nsteps",
            "maxiter_inv",
            "start_regpar",
            "modul_regpar",
            "reg_shift",
            "maxiter_solve",
        ):
            v = _get(ip_, f)
            if v is not None:
                setattr(d, f, int(v))
        for f in ("tol_solve", "dp"):
            v = _get(ip_, f)
            if v is not None:
                setattr(d, f, float(v))
        for f in (
            "tol_inv",
            "regpar0",
            "regbase",
            "reg0par",
            "reg1par",
            "reg2par",
        ):
            v = _get(ip_, f)
            if v is not None:
                setattr(d, f, tuple(np.atleast_1d(v).astype(float)))
        for f in ("m_apr_set", "m_ini_set"):
            v = _get(ip_, f)
            if v is not None:
                setattr(d, f, _cell_or_scalar(v))
        v = _get(ip_, "reg_opt")
        if v is not None:
            d.reg_opt = str(v)
        inv = d
    return site, fwd, inv
