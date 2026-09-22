"""
fwd_plot.py -- Forward-model temperatures and their paleoclimate input.

Companion to fwd_driver.run_fwd: takes its returned dict and plots the
ground surface temperature history (GSTH), the final borehole temperature
profile with observations, and optionally the observation residuals.
The initial profile can be overlaid for comparison. No model is rerun by
plot_fwd(); all plotted values come from the supplied forward run.

This is a new forward-model plotting module, not a translation of
SITE_Plot.m (which plots MCMC posterior results and is a separate task).
It follows the existing modules' function/dict interface and docstring
structure. Requires Python 3.12, NumPy and Matplotlib; the demonstration
also uses the existing forward-model modules and SciPy.

Time follows the solver's convention: t is in seconds relative to the
model reference, normally negative in the past. The plot uses -t/YEAR2SEC
with older ages on the left. It does not assume a calendar reference year.
The GSTH is shown as the boundary value applied over each integration
interval: GST[it[j]] on t[j] .. t[j+1]. In the current heat1dnt, a final
node entry in it is unused. No site.gts offset is added, matching fwd_gsth.

Example
-------
    from fwd_plot import plot_fwd
    # out = run_fwd(name, props, prep_fn, init_fn, ...)
    plots = plot_fwd(out, outdir="figures", show=True)
    # plots['figure'] and plots['axes'] remain available for editing.

Run `python fwd_plot.py --outdir figures` for a synthetic demonstration.
The demonstration creates synthetic observations; it uses no site data.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : AI-generated companion to the user's translated MATLAB
                 forward model; validated on synthetic data, not against
                 real borehole data or MATLAB output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

YEAR2SEC = 365.25 * 24.0 * 3600.0

__all__ = ["plot_fwd"]


# ---------------------------------------------------------------------------
# forward-model overview (input GSTH, temperatures, residuals)
# ---------------------------------------------------------------------------
def plot_fwd(
    run: dict,
    outdir: Optional[str] = None,
    plotfmt: str = "png",
    time_scale: str = "linear",
    show_initial: bool = True,
    show_residuals: bool = True,
    show: bool = False,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
):
    """Plot one forward run and its prescribed paleoclimate history.

    Parameters
    ----------
    run : dict returned by fwd_driver.run_fwd or the dictionary workflow.
        site may be a MATLAB-style dictionary or an existing SitePar.
        Uses name, site, mesh, init and result. result['Tcalc'] must have
        shape (number of depth nodes, number of time nodes), as returned
        by gsth_drivers.fwd_gsth. init['it'] (or mesh['it'] if omitted)
        maps integration intervals to init['GST']; indices are 0-based.
        Observations use result['zobs'], falling back to z[id] when None.
        Error bars use site.Terr. Temperatures are in degrees C, depth in
        metres and unweighted residuals in K (observed minus calculated).
    outdir : optional directory for <name>_Fwd.<plotfmt>.
        Created if necessary. None leaves the figure in memory only.
    plotfmt : Matplotlib output format, e.g. 'png', 'pdf' or 'svg'.
    time_scale : 'linear', 'log' or 'symlog'.
        'symlog' includes age zero, with a linear region within one year
        of the reference. 'log' requires strictly positive ages; no time
        nodes are silently dropped or shifted.
    show_initial : overlay the stored initial temperature profile.
    show_residuals : include a residual panel if observations exist.
        The annotation is the dimensionless, error-normalised RMS from
        fwd_gsth, whereas the plotted residuals are unweighted.
    show : call matplotlib.pyplot.show() after plotting/saving.
        False is convenient for notebooks, tests and further editing.
    figsize : optional (width, height) in inches.
    dpi : resolution of the saved figure.

    Returns
    -------
    dict(figure, axes, filename) -- axes is a dict with keys 'gsth',
    'temperature' and, if drawn, 'residuals'. filename is a Path or None.
    The caller owns the figure and may close it with plt.close(). Inputs
    and Matplotlib's global style settings are not modified.
    """
    result, site = run["result"], run["site"]
    z = np.asarray(result["z"], dtype=float).ravel()
    t = np.asarray(site["t"] if isinstance(site, dict) else site.t, dtype=float).ravel()
    temperature = np.asarray(result["Tcalc"], dtype=float)
    gst = np.asarray(run["init"]["GST"], dtype=float).ravel()
    pointer = np.asarray(run["init"].get("it", run["mesh"]["it"])).ravel()

    if (z.size < 2 or t.size < 2 or not np.isfinite(z).all()
            or not np.isfinite(t).all() or np.any(np.diff(z) <= 0)
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Depth and time nodes must be finite and strictly increasing.")
    if temperature.shape != (z.size, t.size):
        raise ValueError("Tcalc must have shape (len(z), len(site.t)).")
    if (pointer.size < t.size - 1 or not np.isfinite(pointer).all()
            or np.any(pointer != np.floor(pointer))):
        raise ValueError("it must contain integer GST indices for every time interval.")
    pointer = pointer[:t.size - 1].astype(int)
    if np.any(pointer < 0) or np.any(pointer >= gst.size):
        raise ValueError("it contains an out-of-range GST index.")
    forcing = gst[pointer]
    if not np.isfinite(forcing).all():
        raise ValueError("The applied GST history must be finite.")
    age = -t / YEAR2SEC
    if time_scale not in ("linear", "log", "symlog"):
        raise ValueError("time_scale must be 'linear', 'log' or 'symlog'.")
    if time_scale == "log" and np.any(age <= 0):
        raise ValueError("Log time needs positive ages; use 'symlog' to include zero.")

    observed = np.asarray(result["Tobs"], dtype=float).ravel()
    zobs = result.get("zobs")
    if zobs is None:
        zobs = z[np.asarray(result["id"], dtype=int).ravel()]
    zobs = np.asarray(zobs, dtype=float).ravel()
    errors = np.asarray(site["Terr"] if isinstance(site, dict) else site.Terr, dtype=float).ravel()
    residual = np.asarray(result["r"], dtype=float).ravel()
    if not (observed.size == zobs.size == errors.size == residual.size):
        raise ValueError("Tobs, zobs, Terr and r must have equal lengths.")
    if np.any(errors < 0):
        raise ValueError("Temperature error bars (Terr) must be nonnegative.")

    # ---- layout: age increases to the left, depth increases downward ----
    draw_residuals = show_residuals and observed.size > 0
    ncols = 3 if draw_residuals else 2
    fig, axs = plt.subplots(
        1, ncols, figsize=figsize or ((13, 5) if draw_residuals else (10, 5)),
        gridspec_kw={"width_ratios": [1.5, 1, 0.85] if draw_residuals else [1.5, 1]},
        layout="constrained",
    )
    axes = dict(gsth=axs[0], temperature=axs[1])
    fig.suptitle("%s: forward model" % run["name"])

    # ---- GST boundary condition actually applied by heat1dnt ----
    ax = axes["gsth"]
    ax.step(age, np.r_[forcing, forcing[-1]], where="post",
            color="tab:blue", linewidth=1.8, label="Applied GSTH")
    if time_scale == "symlog":
        ax.set_xscale("symlog", linthresh=1.0)
    else:
        ax.set_xscale(time_scale)
    ax.set_xlim(age[0], age[-1])
    ax.set_xlabel("Time before model reference (yr)")
    ax.set_ylabel("Ground surface temperature (°C)")
    ax.set_title("Paleoclimate input")
    ax.legend(loc="best")

    # ---- stored initial and final profiles; observations and errors ----
    ax = axes["temperature"]
    if show_initial:
        ax.plot(temperature[:, 0], z, "--", color="0.5",
                linewidth=1.2, label="Initial profile")
    ax.plot(temperature[:, -1], z, color="tab:red", linewidth=2,
            label="Final model")
    if observed.size:
        ax.errorbar(observed, zobs, xerr=errors, fmt="o", markersize=3,
                    color="black", ecolor="0.6", elinewidth=0.8,
                    label="Observed ± error")
    ax.set_ylim(z[-1], z[0])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Borehole temperature")
    ax.legend(loc="best")

    # ---- unweighted residuals; the solver reports a weighted RMS ----
    if draw_residuals:
        ax = axes["residuals"] = axs[2]
        ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
        ax.plot(residual, zobs, "o-", color="tab:purple",
                markersize=3, linewidth=1)
        ax.set_ylim(z[-1], z[0])
        ax.set_xlabel("Observed − calculated (K)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Temperature residuals")
        ax.text(0.04, 0.97, "Normalised RMS = %.3g" % result["rms"],
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    for ax in axes.values():
        ax.grid(True, alpha=0.25)

    # ---- optional file output / interactive display ----
    filename = None
    if outdir is not None:
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / ("%s_Fwd.%s" % (run["name"], plotfmt))
        fig.savefig(filename, format=plotfmt, dpi=dpi)
    if show:
        plt.show()
    return dict(figure=fig, axes=axes, filename=filename)


# ---------------------------------------------------------------------------
# demonstration: synthetic forward model and synthetic observations
# ---------------------------------------------------------------------------
def _demo(outdir="figures", show=True):
    """Exercise plot_fwd with the existing synthetic prep/init stand-ins.

    Observations are the calculated final temperatures plus seeded noise;
    they are not measurements or a validation against real borehole data.
    """
    from tempfile import TemporaryDirectory

    from fwd_driver import run_fwd, _demo_prep_fn, _demo_init_fn

    # Keep provisional INFO.dat (Tobs=0 in the stand-in) out of the output.
    with TemporaryDirectory() as workdir:
        run = run_fwd(
            "DEMO", "full", _demo_prep_fn, _demo_init_fn,
            mesh_kw=dict(
                depth_kw=dict(zend=1500.0),
                time_kw=dict(
                    ttype="read",
                    tmesh_in=np.linspace(-20000, 0, 121) * YEAR2SEC,
                ),
            ),
            outdir=workdir, verbose=False,
        )
    result, site = run["result"], run["site"]
    noise = np.random.default_rng(42).normal(0, site.Terr)
    site.Tobs = result["Tcalc"][result["id"], -1] + noise
    result.update(Tobs=site.Tobs, r=noise, resid=noise / site.Terr,
                  rms=float(np.sqrt(np.mean((noise / site.Terr) ** 2))),
                  mae=float(np.mean(np.abs(noise / site.Terr))))
    return plot_fwd(run, outdir=outdir, show=show)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a synthetic borehole forward run.")
    parser.add_argument("--outdir", default="figures", help="figure output directory")
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = parser.parse_args()
    plots = _demo(outdir=args.outdir, show=not args.no_show)
    print("Saved %s" % plots["filename"])
    if args.no_show:
        plt.close(plots["figure"])
