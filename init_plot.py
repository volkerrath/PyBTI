"""
init_plot.py -- Repeated-history initial profiles and convergence.

Plots depend only on the numerical temperature profiles, not observations.
build_initial updates one figure after each cycle when initpar['plotpar']
is supplied. Each cycle can be saved separately. A completed result can
also be plotted without rerunning the model.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 implementation; synthetic validation only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np

__all__ = ["plot_initial_cycles"]


def plot_initial_cycles(sitepar: dict, initial: dict, plotpar: dict | None = None):
    """Plot end-of-cycle profiles and changes over the complete model depth.

    Parameters
    ----------
    sitepar : dictionary; name supplies the figure title/file prefix.
        Observation fields are ignored.
    initial : result or in-progress snapshot from init.build_initial.
        At least one completed periodic cycle is required. Tit[:, j]
        contains actual cycle-end profiles; no extra forward run is made.
    plotpar : optional dict.
        outdir: save <name>_Init_cycleNNN.<plotfmt> and update
        <name>_Init_diagnostics.csv after this cycle.
        show: False; True displays/refreshes without blocking.
        pause: GUI event-processing interval [s], default 0.01, positive.
        plotfmt: 'png'; dpi: 150; figsize: (14, 5.5).
        figure: optional existing figure to clear and reuse.
        display_callback: optional callable receiving the figure after each
        cycle plot, e.g. an IPython display handle's update method. Keeps
        notebook displays current without requiring a GUI backend.

    Returns
    -------
    dict(figure, axes, filename, diagnostics_file). axes contains
    temperature, difference and changes. The caller owns the figure.

    The panels show all cycle-end profiles, the latest signed difference
    T_j - T_(j-1) versus depth, and L2/maximum norms of successive changes.
    T_0 is the starting steady state. Both depth axes span zinit's FULL
    extent. Norms are in K; symlog includes exact zero (linear below
    1e-10 K). CSV contains cycle, profile_change_l2_K, profile_change_max_K.
    No observed temperatures, residuals or goodness-of-fit metrics appear.
    """
    options = dict(plotpar or {})
    callback = options.get("display_callback")
    if callback is not None and not callable(callback):
        raise ValueError("display_callback must be callable")
    n = int(initial["niter"])
    if n < 1:
        raise ValueError("At least one completed periodic cycle is required")
    if options.get("show", False) and options.get("pause", 0.01) <= 0:
        raise ValueError("pause must be positive when show=True")
    fig = options.get("figure")
    if fig is None:
        fig = plt.figure(figsize=options.get("figsize", (14, 5.5)), layout="constrained")
    else:
        fig.clear()
    panels = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    axes = dict(zip(("temperature", "difference", "changes"), panels))
    name = sitepar.get("name", "site")
    z = np.asarray(initial["zinit"])
    fig.suptitle("%s: initial-condition spin-up — cycle %d, model depth %.0f m" %
                 (name, n, z[-1]))
    colours = plt.get_cmap("viridis")
    norm = Normalize(vmin=1, vmax=max(2, initial.get("max_cycles", n)))
    ax = axes["temperature"]
    ax.plot(initial["Tsteady"], z, "--", color="0.5", label="Starting equilibrium")
    for j in range(n):
        latest = j == n - 1
        ax.plot(initial["Tit"][:, j], z, color=colours(norm(j + 1)),
                linewidth=2.4 if latest else 0.8, alpha=1 if latest else 0.55,
                label="Cycle %d" % n if latest else None)
    ax.set(title="End-of-cycle temperature", xlabel="Temperature (°C)", ylabel="Depth (m)",
           ylim=(z[-1], z[0]))
    ax.legend(fontsize=8)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colours),
                            ax=ax, label="Cycle", shrink=.85, pad=.02)
    colorbar.locator = MaxNLocator(integer=True)
    colorbar.update_ticks()

    previous = initial["Tsteady"] if n == 1 else initial["Tit"][:, n - 2]
    difference = initial["Tit"][:, n - 1] - previous
    ax = axes["difference"]
    ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
    ax.plot(difference, z, color="tab:blue", linewidth=2)
    ax.set(title="Change: cycle %d − %d" % (n, n - 1),
           xlabel="Temperature change (K)", ylabel="Depth (m)", ylim=(z[-1], z[0]))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=False)

    cycles = np.arange(1, n + 1)
    ax = axes["changes"]
    ax.plot(cycles, initial["change_l2"], "o-", markersize=3, label="L2 norm")
    ax.plot(cycles, initial["changes"], "s-", markersize=3, label="Maximum norm")
    tolerance = initial.get("initial_tol")
    if tolerance is not None:
        ax.axhline(tolerance, color="0.5", linestyle=":", label="Maximum-change tolerance")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set(title="Successive-profile convergence", xlabel="Cycle",
           ylabel="Norm of temperature change (K)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    for ax in axes.values():
        ax.grid(True, alpha=.25)
    filename, csv_file = None, None
    if options.get("outdir") is not None:
        directory = Path(options["outdir"])
        directory.mkdir(parents=True, exist_ok=True)
        fmt = options.get("plotfmt", "png")
        filename = directory / ("%s_Init_cycle%03d.%s" % (name, n, fmt))
        fig.savefig(filename, format=fmt, dpi=options.get("dpi", 150))
        csv_file = directory / ("%s_Init_diagnostics.csv" % name)
        np.savetxt(csv_file, np.column_stack([cycles, initial["change_l2"], initial["changes"]]),
                   delimiter=",", header="cycle,profile_change_l2_K,profile_change_max_K",
                   comments="", fmt=["%d", "%.12g", "%.12g"])
    if callback is not None:
        callback(fig)
    if options.get("show", False):
        plt.show(block=False)
        fig.canvas.draw_idle()
        plt.pause(options.get("pause", .01))
    return dict(figure=fig, axes=axes, filename=filename, diagnostics_file=csv_file)
