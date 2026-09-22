# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3.12 (BTI)
#     language: python
#     name: bti
# ---

# # PyBTI — borehole temperature workflow
# **Synthetic case A · Python 3.12**
#
# Read input data → generate the mesh → assign material properties → build the ground
# surface temperature history (GSTH) → generate initial conditions → optionally run a
# separate forward simulation → save results.
#
# All modelling steps call the existing Python modules and pass MATLAB-style
# dictionaries. The mesh is the first **generated model input**; raw data are read
# first so their depths can be included without interpolating temperatures.
#
# **Start here:** activate the `BTI` Conda environment, open this notebook with its
# kernel, and run all cells from the top. Change the **initialisation dictionary in
# step 1** to choose `"equilibrium"` or `"periodic"`. Restart and run all cells after
# changing parameters. MCMC and its result plots remain a later task.
#
# Provenance: Codex (OpenAI), 2026-09-22. Uses the existing MATLAB translations;
# validated on synthetic cases, not against MATLAB output or measured boreholes.
#

# +
from pathlib import Path
import sys
import json
import platform
from html import escape

# Locate the repository from either its root or a subfolder.
ROOT = next(
    (p for p in (Path.cwd(), *Path.cwd().parents)
     if (p / "workflow.py").is_file() and (p / "examples" / "A").is_dir()),
    None,
)
if ROOT is None:
    raise RuntimeError("Start Jupyter in the PyBTI repository or one of its subfolders.")
if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Select the Python 3.12 (BTI) kernel, then restart this notebook.")
sys.path.insert(0, str(ROOT))

# %matplotlib inline
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from prep import read_data, prepare_site
from workflow import build_mesh, run_forward, plot_forward
from init import YEAR2SEC, build_gsth, build_initial

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
print(f"Python {platform.python_version()} | NumPy {np.__version__} | "
      f"SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__}")
print(f"Project: {ROOT}")

# -

# ## 1. Choose initialisation and edit the parameter dictionaries
# - **`equilibrium`**: calculate a steady-state profile from `GST0`, material
#   properties and basal heat flow, then run the prescribed transient history.
# - **`periodic`**: start from that equilibrium and repeat the history
#   `initial_iter` times. Each cycle starts at the previous cycle's final profile.
#   By default, stop there; `Tinit` is the last cycle's result.
# - **`initial_tol`**: `None` runs all cycles; a positive value stops when the
#   maximum absolute successive-profile change is at most that value in K.
#
# The notebook uses **periodic** mode by default. It does not use the command-line
# overrides in `run_forward.py`. No data residuals are calculated during initialisation.
#

# +
initpar = dict(
    init_type="periodic",  # CHANGE HERE: "equilibrium" or "periodic"
    GST0=6.0,             # absolute surface temperature of the starting equilibrium [°C]
    initial_iter=30,
    initial_tol=None,     # optional maximum-profile-change tolerance [K], e.g. 1e-3
    verbose=False,
)

# A separate forward simulation is automatic for equilibrium mode.
# In periodic mode, True deliberately applies the GST history ONE MORE TIME.
forward_options = dict(run_after_periodic=False, show_residuals=True)

DATA = ROOT / "examples" / "A"
OUTPUT = DATA / "output" / "notebook" / initpar["init_type"]
datapar = dict(
    file=DATA / "DataBallingA.csv", zcol=0, Tcol=3, Terr=0.05,
    depth_range=(10.0, 2000.0), synthetic=True,
)
# Optional artificial noise; only allowed for synthetic data:
# datapar["noise"] = dict(kind="gaussian", length=150.0, seed=0)

meshpar = dict(
    name="SYNA", min_depth=5000.0, depth=dict(zend=5000.0),
    time=dict(tstart=110000 * YEAR2SEC, tend=30 * YEAR2SEC, nt=401),
    time_nodes=[-70000 * YEAR2SEC, -10000 * YEAR2SEC],
)
site_inputs = dict(
    name="SYNA", props="syn",
    k=2.3253, r=1000.0, c=2500.0, h=0.0001e-6, p=0.0001,
    kA=0.0, kB=0.0, qb=-0.069759,
)
fwdpar = dict(theta=1.0, maxitnl=4, tolnl=1e-5, freeze=1)
gstpar = dict(
    file=DATA / "GSTHBallingA.csv", form="steps",
    time_unit="yr", time_convention="before_reference", prehistory=6.0,
)
print(f"Initialisation: {initpar['init_type']}; output folder: {OUTPUT}")

# -

# ## 2. Read and inspect the data
# In `DataBallingA.csv`, zero-based columns are depth (0), climate anomaly (1),
# background profile (2), and total synthetic temperature (3). The full file extends
# to 7,000 m; this example retains input rows between 10 and 2,000 m for later
# forward/data comparisons. Their temperatures are not resampled onto the mesh.
#
# Uncertainty `Terr=0.05 K` does not add noise by itself. Noise is added only when
# the explicit `datapar["noise"]` setting is present. The input data are shown here
# for inspection; they are not a convergence target for repeated initialisation.
#

# +
observations = read_data(datapar)
preview = np.column_stack((
    observations["zobs"], observations["Tobs"], observations["Terr"]
))
print(f"File: {observations['file'].name}")
print(f"Raw rows: {len(observations['raw'])}; retained rows: {len(preview)}")
print("First five selected rows: depth [m], temperature [°C], uncertainty [K]")
print(preview[:5])

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.errorbar(observations["Tobs"], observations["zobs"],
            xerr=observations["Terr"], fmt=".", capsize=2, label="Selected input")
ax.set(xlabel="Temperature (°C)", ylabel="Depth (m)", title="Synthetic input data")
ax.invert_yaxis()
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# -

# ## 3. Generate the spatial and temporal meshes
# The model extends to **at least 5,000 m**, including when a smaller `zend` is
# entered. Observation depths are inserted as nodes; known GST transitions are
# inserted as time nodes. Material properties are assigned after this step.
#
# Internal time is in seconds relative to the reference, increasing from negative
# values to zero. The displayed age is `-t / YEAR2SEC`. It is not a calendar year.
#

# +
mesh_settings = dict(meshpar)
mesh_settings["min_depth"] = max(5000.0, meshpar.get("min_depth", 0.0))
mesh = build_mesh(mesh_settings, observations)
assert mesh["z"][-1] >= 5000.0
print(f"Depth: {mesh['z'][-1]:.0f} m; {mesh['nz']} depth nodes; "
      f"{mesh['nt']} time nodes")
print(f"Time span: {-mesh['t'][0] / YEAR2SEC:,.0f} years before reference to 0")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
axes[0].plot(mesh["dz"], mesh["zm"], ".-")
axes[0].set(xlabel="Cell thickness (m)", ylabel="Depth (m)",
            title="Depth mesh", ylim=(mesh["z"][-1], 0))
axes[1].plot(-mesh["tm"] / YEAR2SEC, mesh["dt"] / YEAR2SEC, ".-")
axes[1].set(xlabel="Time before reference (yr)", ylabel="Time step (yr)",
            title="Time mesh", xlim=(-mesh["t"][0] / YEAR2SEC, 0))
for ax in axes:
    ax.grid(alpha=0.25)
plt.show()

# -

# ## 4. Assign material properties and prepare SitePar
# Depth increases downward. Basal heat flow `qb` follows the translated solver's
# sign convention: normal upward terrestrial heat flow is **negative**.
# `props="syn"` selects the synthetic material laws.
#
# Inputs can be scalar or per-unit arrays with an explicit, zero-based `ip`
# mapping. Here the synthetic properties are uniform. The resulting `sitepar`
# also contains observation indices for optional later comparisons.
#

sitepar = prepare_site(site_inputs, mesh, observations)
property_rows = [
    ("Thermal conductivity", "k", "W/(m K)"),
    ("Density", "r", "kg/m³"),
    ("Specific heat capacity", "c", "J/(kg K)"),
    ("Heat production", "h", "W/m³"),
    ("Porosity", "p", "fraction"),
]
rows = "".join(
    f"<tr><td>{escape(label)}</td><td>{sitepar[key].min():.6g}</td>"
    f"<td>{sitepar[key].max():.6g}</td><td>{escape(unit)}</td></tr>"
    for label, key, unit in property_rows
)
display(HTML("<table><tr><th>Property</th><th>Minimum</th><th>Maximum</th>"
             "<th>Unit</th></tr>" + rows + "</table>"))
print(f"Basal heat flow: {sitepar['qb']:.6g} W/m²")
print(f"All {len(sitepar['id'])} retained data depths map to mesh nodes.")


# ## 5. Build and inspect the paleoclimate forcing
# The two-column GST file supplies age before reference (yr) and absolute surface
# temperature (°C). Step values apply from their transition onward. The explicit
# `prehistory=6 °C` covers the model period before the oldest supplied transition.
#
# The plot below shows the actual boundary values used on solver intervals:
# `GST[it[j]]` on `t[j] … t[j+1]`. The final node pointer is unused by the solver.
#

# +
forcing = build_gsth(mesh, gstpar)
applied_gst = forcing["GST"][forcing["it"][:-1]]
age = -mesh["t"] / YEAR2SEC

fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
ax.step(age, np.r_[applied_gst, applied_gst[-1]], where="post", label="Applied GSTH")
ax.scatter(-forcing["t_history"] / YEAR2SEC, forcing["T_history"],
           color="tab:red", zorder=3, label="Source transitions")
ax.set(xlabel="Time before reference (yr)", ylabel="Surface temperature (°C)",
       title="Prescribed paleoclimate", xlim=(age[0], age[-1]))
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# -

# ## 6. Generate initial conditions and monitor every cycle
# In periodic mode, one inline figure updates after **each completed cycle**:
# temperature profiles, the latest signed profile change, and both convergence
# norms. A PNG snapshot is also saved after each cycle, and the diagnostic CSV is
# updated immediately.
#
# The norms use **all depth nodes**: Euclidean L2 and maximum absolute change.
# Cycle 1 is compared with the starting equilibrium. No observations, residuals or
# data-fit statistics enter this calculation. L2 depends on the chosen mesh.
#

# +
cycle_display = None

def display_cycle(figure):
    """Refresh one notebook output after each completed initialisation cycle."""
    global cycle_display
    if cycle_display is None:
        cycle_display = display(figure, display_id=True)
    else:
        cycle_display.update(figure)

initial_settings = dict(initpar, GST=forcing["GST"], it=forcing["it"])
if initpar["init_type"] == "periodic":
    initial_settings["plotpar"] = dict(
        outdir=OUTPUT / "cycles", show=False, dpi=120,
        display_callback=display_cycle,
    )
initial = build_initial(sitepar, fwdpar, initial_settings)
print(f"Initialisation completed: {initial['init_type']}, {initial['niter']} cycles")
if initial["niter"]:
    print(f"Last L2 change: {initial['change_l2'][-1]:.8g} K")
    print(f"Last maximum change: {initial['changes'][-1]:.8g} K")
    print("Convergence:", "fixed cycle count; no stopping tolerance requested"
          if initial["converged"] is None else initial["converged"])
    print(f"Saved {len(initial['plot_files'])} cycle plots.")
    print(f"Diagnostics: {initial['monitor_plot']['diagnostics_file']}")

# -

# ## 7. Inspect the resulting initial profile
# `initial["Tinit"]` is the profile to pass to a subsequent model.
# `Tsteady` retains the starting equilibrium; `Tit[:, j]` retains each completed
# cycle's final profile. The following figure always shows the entire model depth.
#

# +
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
axes[0].step(age, np.r_[applied_gst, applied_gst[-1]], where="post")
axes[0].set(xlabel="Time before reference (yr)", ylabel="Surface temperature (°C)",
            title="Prescribed GSTH", xlim=(age[0], age[-1]))
axes[1].plot(initial["Tsteady"], initial["zinit"], "--", color="0.5",
             label="Starting equilibrium")
axes[1].plot(initial["Tinit"], initial["zinit"], color="tab:red",
             label="Generated initial profile")
axes[1].set(xlabel="Temperature (°C)", ylabel="Depth (m)",
            title=f"Initial state: {initial['init_type']}",
            ylim=(mesh["z"][-1], 0))
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.25)
OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / "SYNA_initial_profile.png", dpi=150)
plt.show()

if initial["niter"]:
    print("Cycle      L2 change [K]      Maximum change [K]")
    for cycle, l2, maximum in zip(
        range(1, initial["niter"] + 1), initial["change_l2"], initial["changes"]
    ):
        print(f"{cycle:5d} {l2:18.8g} {maximum:23.8g}")

# -

# ## 8. Optional separate forward simulation
# With equilibrium initialisation, this applies the prescribed GST history once.
# After periodic initialisation it is **skipped by default**; set
# `forward_options["run_after_periodic"] = True` in step 1 to deliberately run an
# additional history.
#
# Only this separate forward/data-comparison step uses observed-minus-calculated
# residuals. Those residuals do not measure initialisation convergence.
#

run_separate_forward = (
    initial["init_type"] == "equilibrium" or forward_options["run_after_periodic"]
)
result = None
run = dict(name=sitepar["name"], mesh=mesh, site=sitepar,
           fwd=fwdpar, init=initial, result=None)
if run_separate_forward:
    result = run_forward(sitepar, fwdpar, initial)
    run["result"] = result
    forward_plot = plot_forward(
        run, dict(outdir=OUTPUT / "forward", show=False,
                  show_residuals=forward_options["show_residuals"])
    )
    display(forward_plot["figure"])
    plt.close(forward_plot["figure"])
    print("Separate forward simulation completed.")
else:
    print("Periodic initialisation only: no extra forward simulation was run.")


# ## 9. Save reusable numerical results and the input settings
# The NPZ file stores arrays without Python objects, and can be loaded with
# `np.load(path, allow_pickle=False)`. The JSON file records the parameter
# dictionaries, units, and package versions. Output folders are separated by
# initialisation mode; rerunning a mode replaces files with the same names.
# If you reduce the cycle count, older PNGs may remain; use
# `initial["plot_files"]` for the current run's exact list.
#

# +
arrays = dict(
    z=mesh["z"], t=mesh["t"], GST=initial["GST"], it=initial["it"],
    Tsteady=initial["Tsteady"], Tinit=initial["Tinit"], Tit=initial["Tit"],
    change_l2=initial["change_l2"], changes=initial["changes"],
)
if result is not None:
    arrays["Tcalc"] = result["Tcalc"]
archive = OUTPUT / "SYNA_workflow.npz"
np.savez_compressed(archive, **arrays)

def json_value(value):
    """Convert paths and NumPy values for a readable parameter record."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")

record = dict(
    datapar=datapar, meshpar=mesh_settings, sitepar=site_inputs,
    fwdpar=fwdpar, gstpar=gstpar, initpar=initpar,
    forward_options=forward_options,
    units=dict(depth="m", time="s relative to reference",
               temperature="deg C", profile_change="K", heat_flow="W/m2"),
    versions=dict(python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, matplotlib=matplotlib.__version__),
    completed_cycles=initial["niter"], converged=initial["converged"],
    forward_run=result is not None, cycle_plot_files=initial["plot_files"],
)
settings_file = OUTPUT / "SYNA_parameters.json"
settings_file.write_text(json.dumps(record, indent=2, default=json_value),
                         encoding="utf-8")
with np.load(archive, allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved["Tinit"], initial["Tinit"])
print(f"Saved and checked: {archive}")
print(f"Parameter record: {settings_file}")

# -

# ## 10. Later stages: measured data and inversion
# For measured borehole data, change `datapar["file"]`, the column mapping and
# depth range; set `synthetic=False`, omit `noise`, and provide measured
# uncertainties through `Terr` or `errcol`. Change material properties, basal
# heat flow and the GST history to suit the site. Extend the depth mesh as needed
# and keep it at least 5,000 m deep.
#
# The existing Tikhonov adapter uses the same dictionary interface:
#
# ```python
# from workflow import run_inversion
#
# # Supply a complete, independently chosen invpar dictionary first.
# # Its GST parameter mapping is distinct from the forcing's interval mapping.
# inversion = run_inversion(sitepar, fwdpar, invpar, initial)
# ```
#
# This cell is documentation, not an automatic inversion: the synthetic folder
# does not supply the missing SYNA_InvPar configuration. The current inversion
# solver requires diagonal observational covariance; correlated-noise covariance
# is rejected. MCMC configuration and SITE_Plot remain deferred.
#
# See [WORKFLOW.md](WORKFLOW.md), [init.py](init.py) and [BTI.yaml](BTI.yaml)
# for the interfaces and environment.
#
