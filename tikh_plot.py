"""
tikh_plot.py -- Tikhonov inversion diagnostics from SITE_TikhPlot.m.

Uses returned inversion dictionaries rather than scanning MATLAB result files.
Plots recovered GSTH, temperature fit, observation residuals and apparent
conductive heat flow, plus iteration and regularisation-search diagnostics.
The arbitrary 13.5-year time shift and site-specific axis limits are omitted.
Initialisation convergence remains separate from these data-fit diagnostics.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 translation/adaptation; synthetic validation.
                 Not compared with MATLAB output.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numeric as nm

__all__ = ["plot_tikhonov"]


def _heat_flow(result, window):
    """Apparent upward heat flow between data depths, in W/m2.

    Use a thickness-weighted harmonic conductivity across every intervening
    numerical cell. This replaces the template's first-cell approximation
    when two observations span multiple mesh cells. Smooth data-derived flow
    only, using the template's 21-point mirrored boxcar (one pass).
    """
    z = np.asarray(result["z"])
    ids = np.asarray(result["id"], dtype=int)
    k = np.asarray(result["Tcon"])
    if ids.size < 2 or np.any(np.diff(ids) <= 0):
        raise ValueError("Heat-flow diagnostics require at least two increasing data depths")
    if k.shape != (len(z) - 1,) or np.any(k <= 0) or not np.isfinite(k).all():
        raise ValueError("Tcon must contain positive effective cell conductivities")
    zd = z[ids]
    conductivity = np.array([
        (z[b] - z[a]) / np.sum(np.diff(z)[a:b] / k[a:b])
        for a, b in zip(ids[:-1], ids[1:])
    ])
    qobs = conductivity * np.diff(result["Tobs"]) / np.diff(zd)
    qcalc = conductivity * np.diff(result["Tcalc"][ids, -1]) / np.diff(zd)
    # wfilt needs enough points for its mirrored padding.
    window = min(window, 2 * len(qobs) - 1)
    return dict(depth=(zd[:-1] + zd[1:]) / 2, conductivity=conductivity,
                observed_raw=qobs, observed=nm.wfilt(qobs, window, 1, ("box", "mir")),
                calculated=qcalc, smooth_window=window)


def plot_tikhonov(runs: dict | list[dict], plotpar: dict | None = None):
    """Plot SITE_TikhPlot diagnostics for one inversion or a heat-flow sweep.

    runs : result dictionary from workflow.run_inversion, or a list thereof.
    plotpar : dictionary with name, outdir (None), formats (('png',)),
        dpi (150), show (False), figsize ((15, 10)), time_scale ('symlog'),
        smooth_window (21, positive odd integer), optional reference
        dict(t, GST, it) for a known synthetic forcing.
        Time scale is 'symlog' (10-year linear region including the present)
        or 'linear'. No calendar reference or time offset is assumed.

    Returns dict(figure, axes, filenames, heat_flow). Residuals are observed
    minus calculated in K. RMS is uncertainty-normalised and dimensionless.
    Apparent heat flow is +lambda*dT/dz [mW/m2], with depth positive downward;
    this is the positive upward-flow convention, opposite to signed basal qb.
    It uses final effective conductivity, not the initial-profile gradient
    returned by the legacy heat solver. No covariance confidence band is
    inferred: the reported Cmm is conditional on fixed initial state and qb.
    """
    options = dict(plotpar or {})
    results = [runs] if isinstance(runs, dict) else list(runs)
    if not results:
        raise ValueError("At least one inversion result is required")
    scale = options.get("time_scale", "symlog")
    if scale not in ("linear", "symlog"):
        raise ValueError("Use linear or symlog time to retain age zero")
    window = options.get("smooth_window", 21)
    if isinstance(window, bool) or int(window) != window or window < 1 or window % 2 == 0:
        raise ValueError("smooth_window must be a positive odd integer")
    fig, panels = plt.subplots(2, 3, figsize=options.get("figsize", (15, 10)),
                              layout="constrained")
    axes = dict(zip(("gsth", "temperature", "residuals", "heat_flow", "rms", "gcv"),
                    panels.ravel()))
    name = options.get("name", results[0].get("name", "site"))
    fig.suptitle(f"{name}: Tikhonov GSTH inversion")
    diagnostics = []
    colours = plt.get_cmap("tab10")
    for i, result in enumerate(results):
        color = colours(i % 10)
        z = np.asarray(result["z"])
        t = np.asarray(result["t"])
        ids = np.asarray(result["id"], dtype=int)
        m = np.asarray(result["m"])
        pointer = np.asarray(result["it"], dtype=int)
        age = -t / nm.YEAR2SEC
        applied = (m + result["gts"])[pointer[:-1]]
        label = (f"qb={result['qb'] * 1000:.3g} mW/m²; "
                 f"RMS={result['rms_iter'][-1]:.3g}")
        axes["gsth"].step(age, np.r_[applied, applied[-1]], where="post",
                          color=color, label=label)
        axes["temperature"].plot(result["Tcalc"][:, -1], z, color=color, label=label)
        if i == 0:
            axes["temperature"].errorbar(result["Tobs"], z[ids], xerr=result["Terr"],
                                         fmt="k.", markersize=4, label="Input data")
        residual = np.asarray(result["Tobs"]) - result["Tcalc"][ids, -1]
        axes["residuals"].plot(residual, z[ids], color=color, label=label)
        heat = _heat_flow(result, int(window))
        diagnostics.append(heat)
        axes["heat_flow"].plot(heat["calculated"] * 1000, heat["depth"],
                               color=color, label=label)
        axes["heat_flow"].plot(heat["observed"] * 1000, heat["depth"], "--",
                               color=color, alpha=.65,
                               label=f"Data-derived, boxcar {heat['smooth_window']} (run {i+1})")
        axes["rms"].plot(np.arange(result["niter"]), result["rms_iter"], "o-",
                         color=color, markersize=3, label=label)
        search = result["search"]
        if search:
            reg = np.asarray(search["regpar"])
            variable = np.flatnonzero(np.ptp(reg, axis=0) > 0)
            if variable.size == 1:
                j = int(variable[0])
                x = reg[:, j]
                axes["gcv"].set_xlabel(f"Regularisation weight tau{j}")
            else:
                x = np.arange(len(reg))
                axes["gcv"].set_xlabel("Regularisation candidate index")
            axes["gcv"].plot(x, search["GCV"], ".-", color=color, label=label)
            chosen = search["selected_index"]
            axes["gcv"].scatter(x[chosen], search["GCV"][chosen], color=color,
                                 edgecolors="black", zorder=4)
            if variable.size == 1 and np.all(x > 0):
                axes["gcv"].set_xscale("log")
    reference = options.get("reference")
    if reference is not None:
        age_ref = -np.asarray(reference["t"]) / nm.YEAR2SEC
        values = np.asarray(reference["GST"])[np.asarray(reference["it"], int)[:-1]]
        axes["gsth"].step(age_ref, np.r_[values, values[-1]], where="post",
                          color="black", linestyle="--", label="Known forcing")
    if scale == "symlog":
        axes["gsth"].set_xscale("symlog", linthresh=10)
    oldest = max(-np.asarray(r["t"])[0] / nm.YEAR2SEC for r in results)
    axes["gsth"].set(xlim=(oldest, 0), xlabel="Time before reference (yr)",
                     ylabel="Ground surface temperature (°C)", title="Recovered GSTH")
    axes["temperature"].set(xlabel="Temperature (°C)", ylabel="Depth (m)",
                            title="Temperature fit",
                            ylim=(max(np.max(r["z"]) for r in results), 0))
    data_bottom = max(np.max(r["z"][r["id"]]) for r in results)
    axes["residuals"].axvline(0, color="0.5", linestyle=":")
    axes["residuals"].set(xlabel="Observed − calculated (K)", ylabel="Depth (m)",
                          ylim=(data_bottom, 0), title="Temperature residuals")
    axes["heat_flow"].set(xlabel="Apparent upward heat flow (mW/m²)", ylabel="Depth (m)",
                          ylim=(data_bottom, 0), title="Conductivity × temperature gradient")
    axes["rms"].set(xlabel="Evaluated iteration (0 = initial guess)",
                    ylabel="Uncertainty-normalised RMS", title="Inversion convergence")
    axes["rms"].set_yscale("symlog", linthresh=1e-8)
    axes["rms"].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes["gcv"].set(ylabel="GCV score", title="Last regularisation search")
    scores = np.concatenate([np.asarray(r["search"]["GCV"]) for r in results
                             if r["search"]]) if any(r["search"] for r in results) else np.array([])
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size and np.all(finite_scores > 0):
        axes["gcv"].set_yscale("log")
        axes["gcv"].set_ylim(finite_scores.min() / 1.3, finite_scores.max() * 1.3)
    else:
        axes["gcv"].set_yscale("symlog", linthresh=1e-12)
        if finite_scores.size:
            axes["gcv"].set_ylim(0, max(1e-12, finite_scores.max() * 1.3))
    if not any(r["search"] for r in results):
        axes["gcv"].text(.5, .5, "No regularisation search performed",
                          ha="center", transform=axes["gcv"].transAxes)
    for ax in axes.values():
        ax.grid(True, alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    # Freeze the calculated margins before PNG/PDF/inline renderers switch.
    # Otherwise the PDF renderer can leave stale text bounds in notebook output.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    filenames = []
    if options.get("outdir") is not None:
        directory = Path(options["outdir"])
        directory.mkdir(parents=True, exist_ok=True)
        formats = options.get("formats", ("png",))
        if isinstance(formats, str):
            formats = [formats]
        for fmt in formats:
            path = directory / f"{name}_Tikh.{fmt}"
            fig.savefig(path, dpi=options.get("dpi", 150))
            filenames.append(path)
    if options.get("show", False):
        plt.show()
    return dict(figure=fig, axes=axes, filenames=filenames, heat_flow=diagnostics)
