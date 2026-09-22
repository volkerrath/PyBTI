"""
init.py -- Ground surface forcing and initial temperatures (SITE_Init.m).

Public inputs/outputs are plain dictionaries emulating MATLAB structs.
build_initial supports a stationary initial profile, or repetition of a
prescribed GST history with each final profile becoming the next initial
profile. The numerical solvers in numeric.py are reused unchanged.

GST0 is an explicit absolute surface temperature in degrees C. Unlike the
inconsistent POM offsets in the MATLAB templates, no implicit shift is
applied. History times are converted explicitly to seconds, negative in
the past; the model reference year is not assumed. A step history needs
an explicit prehistory temperature if the mesh starts before its first
transition. The known synthetic history is not an inversion prior.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 implementation based on SITE_Init.m, with
                 explicit dictionary inputs and initialisation choices.
                 Tested on analytical and synthetic cases, not MATLAB output.
"""

from __future__ import annotations

import numpy as np

import numeric as nm

YEAR2SEC = 31557600.0
__all__ = ["build_gsth", "build_initial"]


def _vector(value, name):
    """Finite nonempty one-dimensional numerical input."""
    array = np.asarray(value, dtype=float).ravel()
    if not array.size or not np.isfinite(array).all():
        raise ValueError("%s must be nonempty and finite" % name)
    return array


def build_gsth(mesh: dict, gstpar: dict):
    """Build absolute ground surface temperatures on the time mesh.

    Parameters
    ----------
    mesh : dict with t (strictly increasing seconds relative to reference).
    gstpar : dict with either file (two-column CSV) or time and temperature.
        delimiter (default ',') and skiprows (0) configure file reading.
        time_unit: 'yr' (default) or 's'; time_convention: 'before_reference'
        (default, nonnegative ages) or 'relative' (signed times).
        form: 'steps' (default) or 'points'. Steps take effect at each
        supplied time and persist until the next transition. prehistory
        is required when the mesh predates the first step.
        Points use linear interpolation in signed time, including zero;
        endpoint extrapolation is rejected. This differs deliberately
        from legacy set_pntgst's log-time interpolation at age zero.

    Returns
    -------
    dict(GST, it, t_history, T_history). it is a 0-based pointer for each
    time node. The final pointer is unused by the current heat1dnt.
    Source rows are sorted chronologically; duplicate times are rejected.
    """
    t = _vector(mesh["t"], "mesh.t")
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("mesh.t must be strictly increasing")
    if "file" in gstpar:
        data = np.loadtxt(gstpar["file"], delimiter=gstpar.get("delimiter", ","),
                          skiprows=gstpar.get("skiprows", 0), ndmin=2)
        if data.shape[1] != 2:
            raise ValueError("GST history file must have two columns")
        times, temperatures = data.T
    else:
        times, temperatures = gstpar["time"], gstpar["temperature"]
    times = _vector(times, "history time")
    temperatures = _vector(temperatures, "history temperature")
    if times.size != temperatures.size:
        raise ValueError("History time and temperature lengths differ")
    unit = gstpar.get("time_unit", "yr")
    convention = gstpar.get("time_convention", "before_reference")
    if unit not in ("yr", "s") or convention not in ("before_reference", "relative"):
        raise ValueError("Specify time_unit 'yr'/'s' and a valid time_convention")
    times = times * (YEAR2SEC if unit == "yr" else 1.0)
    if convention == "before_reference":
        if np.any(times < 0):
            raise ValueError("Ages before reference must be nonnegative")
        times = -times
    order = np.argsort(times)
    times, temperatures = times[order], temperatures[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError("History times must be distinct")
    form = gstpar.get("form", "steps")
    if form == "steps":
        indices = np.searchsorted(times, t, side="right") - 1
        GST = temperatures[np.maximum(indices, 0)].copy()
        if np.any(indices < 0):
            if "prehistory" not in gstpar:
                raise ValueError("Specify prehistory before the first GST transition")
            GST[indices < 0] = float(gstpar["prehistory"])
    elif form == "points":
        if times.size < 2 or t[0] < times[0] or t[-1] > times[-1]:
            raise ValueError("Point history must cover the complete time mesh")
        GST = np.interp(t, times, temperatures)
    else:
        raise ValueError("History form must be 'steps' or 'points'")
    GST = _vector(GST, "GST")
    return dict(GST=GST, it=np.arange(t.size),
                t_history=times, T_history=temperatures)


def _numerics(fwdpar):
    """Copy numerical controls; relaxnl is retained but unused as in run_fwd."""
    allowed = {"theta", "maxitnl", "tolnl", "freeze", "kw", "relaxnl"}
    unknown = set(fwdpar) - allowed
    if unknown:
        raise ValueError("Unknown fwdpar fields: %s" % sorted(unknown))
    settings = dict(theta=1.0, maxitnl=4, tolnl=1e-5, freeze=1, kw={})
    settings.update(fwdpar)
    return settings


def _forcing(sitepar, initpar):
    """Validate supplied GST parameters and the interval index mapping."""
    t = _vector(sitepar["t"], "sitepar.t")
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("sitepar.t must contain increasing time nodes")
    GST = _vector(initpar.get("GST", [initpar["GST0"]]), "GST")
    default = np.zeros(t.size, dtype=int) if GST.size == 1 else np.arange(t.size)
    it = np.asarray(initpar.get("it", default)).ravel()
    if (it.size < t.size - 1 or not np.isfinite(it).all()
            or np.any(it != np.floor(it)) or np.any(it < 0)
            or np.any(it >= GST.size)):
        raise ValueError("it must contain valid 0-based GST indices for all intervals")
    return GST.copy(), it.astype(int)


def _transient(sitepar, fwdpar, GST, it, T0, out=0):
    """Call the translated solver with the dictionaries' physical fields."""
    s, f = sitepar, _numerics(fwdpar)
    return nm.heat1dnt(
        s["k"], s["kA"], s["kB"], s["h"], s["r"], s["c"], s["p"], s["qb"],
        np.diff(s["z"]), s["ip"], np.diff(s["t"]), it, GST, T0,
        f["theta"], f["maxitnl"], f["tolnl"], f["freeze"], out,
        s["props"], **f["kw"],
    )


def build_initial(sitepar: dict, fwdpar: dict, initpar: dict):
    """Initialise a borehole from a surface temperature or repeated GSTH.

    Parameters
    ----------
    sitepar : physical model dict (k, kA, kB, h, r, c, p, qb, ip, z, t,
        props), normally returned by prep.prepare_site. SI units.
        Observation fields, if present, are not used by initialisation.
    fwdpar : solver control dict: theta, maxitnl, tolnl, freeze, optional kw.
    initpar : dict with GST0 (absolute starting surface temperature, deg C).
        init_type: 'equilibrium' (default; also 'e'/'equi') or 'periodic'
        (also 'p'/'prior'). GST and it may be supplied from build_gsth.
        Equilibrium mode computes Tinit at GST0 without running a history.
        Periodic mode requires GST and repeats it initial_iter times
        (positive integer, default 30). Each final profile becomes the
        next initial profile, as in SITE_Init.m.
        initial_tol: optional positive maximum absolute change [K] between
        successive final profiles for early stopping; None runs all cycles.
        verbose (False) prints the L2 and maximum change after each cycle.
        plotpar: optional dict enabling a plot after EACH completed cycle.
        See init_plot.plot_initial_cycles for outdir, show, pause, plotfmt,
        dpi, figsize. One figure is updated in place; outdir saves a
        snapshot per cycle and a CSV of convergence norms.

    Returns
    -------
    dict(zinit, Tinit, Tsteady, GST, it, Tit, changes, change_l2, niter,
         converged, init_type, GST0, initial_tol, max_cycles, plot_files,
         monitor_plot).

    Tit has shape (nz, niter) and stores each cycle's final temperature.
    changes is max(abs(Tnew-Told)); change_l2 is the Euclidean norm of
    that difference over ALL depth nodes. Cycle 1 is compared with the
    starting steady-state profile. Norms are absolute in K, not divided
    by Celsius temperatures; L2 depends on the numerical mesh.
    Observation residuals and goodness-of-fit metrics are not calculated.
    Equilibrium mode has zero cycles. converged is None unless a periodic
    initial_tol was supplied; the stopping criterion is the maximum norm.
    plot_files lists saved cycle figures. monitor_plot is the last plot
    result or None; its figure is closed on return if show=False.
    Input dictionaries and arrays are not modified.
    """
    s, f = sitepar, _numerics(fwdpar)
    surface = float(initpar["GST0"])
    if not np.isfinite(surface):
        raise ValueError("GST0 must be finite")
    GST, it = _forcing(s, initpar)
    mode = str(initpar.get("init_type", "equilibrium")).lower()
    if mode not in ("e", "equi", "equilibrium", "p", "prior", "periodic"):
        raise ValueError("init_type must be 'equilibrium' or 'periodic'")
    periodic = mode in ("p", "prior", "periodic")
    cycles, tolerance = initpar.get("initial_iter", 30), initpar.get("initial_tol")
    plotpar = initpar.get("plotpar")
    if plotpar is not None and not isinstance(plotpar, dict):
        raise ValueError("plotpar must be a dictionary or None")
    if periodic:
        if "GST" not in initpar:
            raise ValueError("Periodic initialisation requires an explicit GST history")
        if isinstance(cycles, bool) or int(cycles) != cycles or cycles < 1:
            raise ValueError("initial_iter must be a positive integer")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("initial_tol must be positive and finite")
    Tsteady = nm.heat1dns(
        s["k"], s["kA"], s["kB"], s["h"], s["r"], s["p"], s["qb"],
        surface, np.diff(s["z"]), s["ip"], f["maxitnl"], f["tolnl"],
        f["freeze"], s["props"], **nm._ns_kw(f["kw"]),
    )
    if not np.isfinite(Tsteady).all():
        raise ValueError("Nonfinite steady-state initial profile")
    T0, profiles, changes, change_l2 = Tsteady.copy(), [], [], []
    plot_files = []
    converged, monitor_plot = None, None

    def snapshot():
        """Numerical dictionary for the completed cycles so far."""
        return dict(
            zinit=np.asarray(s["z"]).copy(), Tinit=T0.copy(), Tsteady=Tsteady.copy(),
            GST=GST, it=it, Tit=np.column_stack(profiles) if profiles else
            np.empty((len(T0), 0)), changes=np.asarray(changes),
            change_l2=np.asarray(change_l2), niter=len(profiles),
            converged=converged, init_type="periodic" if periodic else "equilibrium",
            GST0=surface, initial_tol=tolerance, max_cycles=int(cycles) if periodic else 0,
        )

    if periodic:
        converged = None if tolerance is None else False
        for cycle in range(int(cycles)):
            Tnew = _transient(s, f, GST, it, T0)
            if not np.isfinite(Tnew).all():
                raise ValueError("Nonfinite temperature during initialisation")
            delta = Tnew - T0
            change = float(np.max(np.abs(delta)))
            profiles.append(Tnew.copy())
            changes.append(change)
            change_l2.append(float(np.linalg.norm(delta)))
            T0 = Tnew
            if tolerance is not None and change <= tolerance:
                converged = True
            if initpar.get("verbose", False):
                print(" initial cycle %d: ||dT||2=%.6g K, max|dT|=%.6g K" %
                      (cycle + 1, change_l2[-1], change))
            if plotpar is not None:
                from init_plot import plot_initial_cycles
                options = dict(plotpar)
                if monitor_plot is not None:
                    options["figure"] = monitor_plot["figure"]
                monitor_plot = plot_initial_cycles(s, snapshot(), options)
                if monitor_plot["filename"] is not None:
                    plot_files.append(monitor_plot["filename"])
            if converged:
                break
    if monitor_plot is not None and not plotpar.get("show", False):
        import matplotlib.pyplot as plt
        plt.close(monitor_plot["figure"])
    result = snapshot()
    result.update(plot_files=plot_files, monitor_plot=monitor_plot)
    return result
