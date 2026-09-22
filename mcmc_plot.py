"""
mcmc_plot.py -- Posterior diagnostics adapted from SITE_MCMCPlot/SITE_Plot.

Plots GST posterior intervals, posterior temperature and residual envelopes,
QB, H, RMS and likelihood-sigma distributions, and compact chain traces. The MATLAB
template's arbitrary 13.5-year time shift and swapped Q/H histogram blocks are
removed. Ages are relative to the model reference time and heat flow is positive
upward in mW/m2.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 translation/adaptation; synthetic validation.
                 Not compared with MATLAB MCMC output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import numeric as nm

__all__ = ["plot_mcmc_summary"]


def _band(ax, x, quantiles, color="tab:blue", label="Posterior median"):
    """Draw 95/68 percent envelopes and their median."""
    ax.fill_between(x, quantiles[0], quantiles[4], color=color, alpha=.16,
                    label="95% credible interval")
    ax.fill_between(x, quantiles[1], quantiles[3], color=color, alpha=.30,
                    label="68% credible interval")
    ax.plot(x, quantiles[2], color=color, linewidth=1.7, label=label)


def plot_mcmc_summary(summary: dict, plotpar: dict | None = None):
    """Plot MCMC posterior diagnostics from summarize_mcmc.

    Parameters
    ----------
    summary : dictionary returned by mcmc_workflow.summarize_mcmc.
    plotpar : optional dictionary with name, outdir, formats (('png',)), dpi
        (150), show (False), figsize ((18, 9)), time_scale ('symlog'), and
        reference (optional dict with t, GST, it for a known synthetic GSTH).

    Returns
    -------
    dict(figure, axes, filenames). Residuals are observed minus calculated.
    Intervals are empirical quantiles of the retained parameter or prediction
    samples. The temperature envelope is a parameter-posterior prediction and
    does not add a new draw of measurement noise. A short validation chain is
    insufficient for posterior inference; inspect independent production chains
    and convergence diagnostics before interpreting the intervals.
    """
    options = dict(plotpar or {})
    unknown = set(options) - {
        "name", "outdir", "formats", "dpi", "show", "figsize",
        "time_scale", "reference",
    }
    if unknown:
        raise ValueError("Unknown MCMC plot fields: %s" % sorted(unknown))
    scale = options.get("time_scale", "symlog")
    if scale not in ("linear", "symlog"):
        raise ValueError("time_scale must be 'linear' or 'symlog'")
    quantiles = np.asarray(summary["quantiles"], dtype=float)
    if quantiles.shape != (5,):
        raise ValueError("summary must contain five ordered quantiles")
    config = summary["config"]
    nsteps = int(config["nsteps"])
    full = np.asarray(summary["chain_full"], dtype=float)
    if full.ndim != 2 or full.shape[1] != nsteps + 2:
        raise ValueError("chain_full does not match the GST/Q/H layout")

    fig, panels = plt.subplots(
        2, 4, figsize=options.get("figsize", (18, 9)), layout="constrained")
    axes = dict(zip(
        ("gsth", "temperature", "residuals", "heat_flow",
         "heat_production", "trace", "rms", "variance"),
        panels.ravel(),
    ))
    name = options.get("name", summary.get("name", "site_MCMC"))
    fig.suptitle(f"{name}: DRAM posterior diagnostics")

    t = np.asarray(summary["t"], dtype=float)
    it = np.asarray(config["it"], dtype=int)
    age = -t / nm.YEAR2SEC
    gts = float(summary.get("gts", 0.0))
    qgst = np.asarray(summary["gst_quantiles"]) + gts
    applied = qgst[:, it[:-1]]
    applied = np.column_stack((applied, applied[:, -1]))
    _band(axes["gsth"], age, applied)
    prior = np.asarray(config["prior_mean"])[:nsteps] + gts
    prior_applied = np.r_[prior[it[:-1]], prior[it[-2]]]
    axes["gsth"].step(age, prior_applied, where="post", color="0.4",
                      linestyle=":", label="Prior mean")
    reference = options.get("reference")
    if reference is not None:
        rt = np.asarray(reference["t"], dtype=float)
        ri = np.asarray(reference["it"], dtype=int)
        rgst = np.asarray(reference["GST"], dtype=float)
        values = rgst[ri[:-1]]
        axes["gsth"].step(-rt / nm.YEAR2SEC, np.r_[values, values[-1]],
                          where="post", color="black", linestyle="--",
                          label="Known forcing")
    if scale == "symlog":
        axes["gsth"].set_xscale("symlog", linthresh=10)
    axes["gsth"].set(
        xlim=(age[0], 0), xlabel="Time before reference (yr)",
        ylabel="Ground surface temperature (°C)", title="GST posterior")

    zobs = np.asarray(summary["zobs"], dtype=float)
    Tobs = np.asarray(summary["Tobs"], dtype=float)
    Terr = np.asarray(summary["Terr"], dtype=float)
    qp = np.asarray(summary["prediction_quantiles"])
    qr = np.asarray(summary["residual_quantiles"])
    axes["temperature"].fill_betweenx(
        zobs, qp[0], qp[4], color="tab:blue", alpha=.16, label="95% interval")
    axes["temperature"].fill_betweenx(
        zobs, qp[1], qp[3], color="tab:blue", alpha=.30, label="68% interval")
    axes["temperature"].plot(qp[2], zobs, color="tab:blue", label="Median model")
    axes["temperature"].errorbar(
        Tobs, zobs, xerr=Terr, fmt="k.", markersize=4, capsize=2,
        label="Input data")
    axes["temperature"].set(
        xlabel="Temperature (°C)", ylabel="Depth (m)",
        ylim=(zobs.max(), 0), title="Posterior temperature")

    axes["residuals"].fill_betweenx(
        zobs, qr[0], qr[4], color="tab:orange", alpha=.16,
        label="95% interval")
    axes["residuals"].fill_betweenx(
        zobs, qr[1], qr[3], color="tab:orange", alpha=.30,
        label="68% interval")
    axes["residuals"].plot(qr[2], zobs, color="tab:orange", label="Median")
    axes["residuals"].axvline(0, color="0.4", linestyle=":")
    axes["residuals"].set(
        xlabel="Observed − calculated (K)", ylabel="Depth (m)",
        ylim=(zobs.max(), 0), title="Posterior residuals")

    qb = np.asarray(summary["posterior"]["qb"], dtype=float)
    axes["heat_flow"].hist(qb, bins="auto", density=True, color="tab:red",
                           alpha=.55, label="Retained samples")
    axes["heat_flow"].axvline(summary["qb_quantiles"][2], color="tab:red",
                              label="Posterior median")
    axes["heat_flow"].axvline(config["prior_mean"][nsteps], color="0.3",
                              linestyle=":", label="Prior mean")
    axes["heat_flow"].set(
        xlabel="Upward basal heat flow (mW/m²)", ylabel="Density",
        title="Basal heat-flow posterior")

    h = np.asarray(summary["posterior"]["h"], dtype=float)
    h_active = bool(np.asarray(config["active"])[nsteps + 1])
    h_sampled = bool(np.asarray(config.get("sampled", config["active"]))[nsteps + 1])
    if h_sampled and np.ptp(h) > 0:
        axes["heat_production"].hist(
            h, bins="auto", density=True, color="tab:purple", alpha=.55,
            label="Retained samples")
        axes["heat_production"].axvline(
            summary["h_quantiles"][2], color="tab:purple", label="Median")
    else:
        axes["heat_production"].axvline(
            h[0], color="tab:purple", label="Fixed chain value")
    axes["heat_production"].axvline(
        config["prior_mean"][nsteps + 1], color="0.3", linestyle=":",
        label="Prior mean")
    if not h_active:
        axes["heat_production"].text(
            .03, .95,
            "Inactive in forward model; prior only" if h_sampled
            else "Inactive and not sampled",
            va="top", transform=axes["heat_production"].transAxes)
    axes["heat_production"].set(
        xlabel="Heat production H (µW/m³)", ylabel="Density",
        title="Heat-production samples")

    x = np.arange(len(full))
    trace_idx = np.unique([0, max(0, nsteps // 2), nsteps - 1])
    for j in trace_idx:
        axes["trace"].plot(x, full[:, j], linewidth=.6, alpha=.75,
                           label=f"GST {j + 1}")
    qax = axes["trace"].twinx()
    qax.plot(x, full[:, nsteps], color="black", linewidth=.7, alpha=.65,
             label="QB")
    axes["trace"].set(xlabel="Retained sample", ylabel="GST anomaly (K)",
                      title="Compact chain traces")
    qax.set_ylabel("QB (mW/m²)")
    handles, labels = axes["trace"].get_legend_handles_labels()
    h2, l2 = qax.get_legend_handles_labels()
    axes["trace"].legend(handles + h2, labels + l2, fontsize=7)

    rms = np.asarray(summary["weighted_rms"], dtype=float)
    axes["rms"].hist(rms, bins="auto", density=True, color="tab:green", alpha=.6)
    axes["rms"].axvline(summary["rms_quantiles"][2], color="tab:green",
                         label="Median")
    axes["rms"].text(
        .03, .95,
        f"acceptance ≈ {summary['acceptance_fraction']:.3f}\n"
        f"chains = {summary['nchain']}; retained = {summary['nsample']}",
        va="top", transform=axes["rms"].transAxes)
    axes["rms"].set(
        xlabel="Uncertainty-normalised RMS", ylabel="Density",
        title="Posterior fit distribution")

    sigma = np.sqrt(np.asarray(summary["s2chain"], dtype=float).ravel())
    sigma = sigma[np.isfinite(sigma)]
    axes["variance"].hist(
        sigma, bins="auto", density=True, color="tab:cyan", alpha=.6,
        label="Retained samples")
    axes["variance"].axvline(
        np.sqrt(float(config["sigma2"])), color="0.3", linestyle=":",
        label="Initial sigma")
    axes["variance"].axvline(
        np.median(sigma), color="tab:cyan", label="Median")
    axes["variance"].set(
        xlabel="Likelihood sigma (K)", ylabel="Density",
        title="Sampled error scale")

    for key, ax in axes.items():
        ax.grid(True, alpha=.25)
        if key != "trace" and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
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
            path = directory / f"{name}_MCMC.{fmt}"
            fig.savefig(path, dpi=options.get("dpi", 150))
            filenames.append(path)
    if options.get("show", False):
        plt.show()
    return dict(figure=fig, axes=axes, filenames=filenames)
