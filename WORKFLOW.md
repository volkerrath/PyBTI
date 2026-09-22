# Borehole workflow — Python 3.12

Run [PyBTI_workflow.ipynb](PyBTI_workflow.ipynb) for a step-by-step notebook;
see README.md for the BTI kernel setup. The initialisation mode is selected
in its first parameter cell.

All new workflow inputs and outputs are ordinary dictionaries, mirroring
MATLAB structures. Existing low-level numerical functions retain their
argument lists. The older dataclass interfaces and callback-based
`fwd_driver.run_fwd` remain available for compatibility.

| Routine | Dictionary inputs | Output |
| --- | --- | --- |
| `prep.read_data` | `datapar` | observations, raw data, uncertainties and optional synthetic noise |
| `workflow.build_mesh` | `meshpar`, optional observations | spatial and temporal mesh |
| `prep.prepare_site` | `sitepar`, mesh, observations | complete `sitepar` |
| `init.build_gsth` | mesh, `gstpar` | GST and time-to-GST indices |
| `init.build_initial` | `sitepar`, `fwdpar`, `initpar` | initial profile and initialisation diagnostics |
| `workflow.run_forward` | `sitepar`, `fwdpar`, initialisation result | temperatures and residuals |
| `workflow.build_inversion` | mesh, `gridpar`, `invpar` | validated inversion GST mapping and controls |
| `workflow.run_inversion` | `sitepar`, `fwdpar`, `invpar`, initialisation result, optional `runpar` | Tikhonov result and diagnostics |
| `workflow.plot_inversion` | inversion result or list, `plotpar` | GST, temperature, residual, heat-flow, RMS and GCV plots |
| `workflow.build_mcmc` | `sitepar`, `fwdpar`, inversion grid, initialisation, optional `mcmcpar` | validated GST/Q/H priors, bounds and DRAM controls |
| `workflow.run_mcmc` | dictionaries above plus MCMC configuration and optional `runpar` | raw sampled/full chains and variance chain |
| `workflow.summarize_mcmc` | chain(s), model dictionaries, optional `summarypar` | retained posterior, predictions, residuals and quantiles |
| `workflow.plot_mcmc` | posterior summary, optional `plotpar` | SITE_MCMCPlot/SITE_Plot diagnostic figure |
| `workflow.plot_forward` | run result, `plotpar` | figure, axes, optional filename |

Reading raw data first lets the mesh include observation depths. The mesh
is still the first generated numerical model input. Assign physical
properties only after constructing the final mesh.

## Initial temperature profile

The two modes use the same function and return the same dictionary fields.

```python
from init import build_gsth, build_initial
from workflow import run_forward

# Supply an absolute surface temperature; calculate a steady-state profile.
initpar = dict(init_type="equilibrium", GST0=6.0)
initial = build_initial(sitepar, fwdpar, initpar)

# Optional: supply a GST history for the subsequent forward simulation.
forcing = build_gsth(mesh, gstpar)
initpar.update(GST=forcing["GST"], it=forcing["it"])
initial = build_initial(sitepar, fwdpar, initpar)
result = run_forward(sitepar, fwdpar, initial)

# Alternatively initialise by repeating that history, as in SITE_Init.m.
initpar.update(init_type="periodic", initial_iter=30)
initial = build_initial(sitepar, fwdpar, initpar)
result = run_forward(sitepar, fwdpar, initial)
```

- `GST0` is the surface temperature for the starting steady-state profile.
  Both modes require it; no implicit POM or site temperature offset is added.
- Equilibrium mode does not simulate the GST history during initialisation.
  Without a supplied history, the later forward run uses constant GST0.
- Periodic mode starts from the steady-state profile and replays the entire
  supplied history. The end of each cycle becomes the start of the next.
  The final cycle's end profile is `Tinit`; the forward run then applies
  the history once more, following the MATLAB separation of Init and Fwd.
- `initial_iter` defaults to 30. Omit `initial_tol` to run exactly that
  number; set it to a positive absolute temperature tolerance [K] to allow
  early termination.
- Returned `Tsteady` is the original steady-state profile. `Tit[:, j]`
  is the final profile of cycle j. `changes` contains the maximum absolute
  change per cycle. `niter` records completed cycles; `converged` is
  None unless a periodic tolerance was requested.

## Input conventions

`datapar` explicitly identifies the file columns. Indices are zero-based;
depth is in metres, temperature in degrees C and uncertainties in K.
Preserved `raw` data and `source_rows` make the selected values traceable.

The default is to preserve the original observation depths and insert them
as mesh nodes. The original temperatures are neither smoothed nor expanded
into additional observations. Physical property inputs can be scalar,
per cell or per unit with an explicit `ip` mapping.

`gstpar` accepts a two-column CSV or `time`/`temperature` arrays. Set
`time_unit` to `"yr"` or `"s"` and `time_convention` to
`"before_reference"` (positive ages) or `"relative"` (signed times).
Internally time is seconds, negative in the past. Step histories require
an explicit `prehistory` temperature if the mesh begins before their
first transition. Point histories use linear interpolation in signed time
and must cover the complete mesh; this differs from the legacy log-time
interpolator, which is undefined at age zero.

As in the existing solver, `GST[it[j]]` is applied on the integration
interval `t[j] .. t[j+1]`. Put important GST transitions in
`meshpar["time_nodes"]` (signed seconds) to avoid moving them to nearby
time steps. A final node pointer is unused by the existing solver.

## Synthetic versus measured observations

For real data, leave `synthetic=False` (the default) and omit `noise`.
Provide measured uncertainties through `Terr` or `errcol`.

Synthetic noise is explicitly enabled, for example:

```python
datapar.update(
    synthetic=True,
    noise=dict(kind="gaussian", length=150.0, seed=0),
)
```

`length` is in metres; `Terr` sets noise standard deviations.
Independent and exponential noise are also supported. A fixed seed makes
the realisation repeatable. The covariance and realised noise are returned.

The forward RMS uses marginal uncertainties, as the existing driver does.
The dictionary inversion adapters reject off-diagonal covariance rather
than silently ignoring it: the current Tikhonov and MCMC paths require diagonal
observation covariance. Tikhonov enables consistent residual weighting. The notebook now
provides explicit `invpar` settings from `templates/SITE_Tikh.m`, mapped by
`build_inversion`; it does not claim to reconstruct the missing SYNA_InvPar file.

## Example A

From the repository root, create and activate the Python 3.12 environment
using BTI.yaml (see README.md):

```text
conda env create --file BTI.yaml
conda activate BTI
python examples/A/run_forward.py --outdir examples/A/output --no-show
python examples/A/run_forward.py --periodic --cycles 30 --outdir examples/A/output --no-show
python examples/A/run_forward.py --noise --outdir examples/A/output --no-show
python -m pytest -q tests/test_dictionary_workflow.py tests/test_fwd_plot.py tests/test_fwd_driver.py
```

Edit `example_parameters()` in the example to change its dictionaries.
It retains 40 input observations between 10 and 2,000 m for subsequent
forward/inversion work; initialisation ignores them. The numerical model
is at least 5,000 m deep, with constant
conductivity 2.3253 W/(m K), and basal heat flow -0.069759 W/m2. The 6 deg C
initial and prehistory temperatures are explicit assumptions consistent
with the file's background profile. Equilibrium is the default; periodic
initialisation and artificial noise are optional.

This command-line example remains focused on forward modelling. The notebook
adds explicit Tikhonov and MCMC settings instead of inferring a missing SYNA
inversion file.


## Monitor repeated-history initial conditions

Repeated-history runs generate a thermal initial state. Their diagnostics
use only successive simulated profiles over the complete numerical depth.
They do not use observations or calculate observed-minus-calculated
residuals, data RMS, or goodness of fit.

```text
python examples/A/run_forward.py --periodic --cycles 30 --outdir examples/A/output/repeated_history --no-show
```

Omit `--no-show` to watch the figure update. In notebooks,
`plotpar["display_callback"]` can receive an IPython display handle update
function; the supplied notebook sets this up for every cycle. Direct dictionary usage:

```python
initpar.update(
    init_type="periodic",
    initial_iter=30,
    verbose=True,
    plotpar=dict(outdir="examples/A/output/repeated_history", show=True),
)
initial = build_initial(sitepar, fwdpar, initpar)
```

Each cycle plot contains:
- All simulated end-of-cycle profiles and the starting equilibrium profile.
- The latest signed change T_j - T_(j-1) versus depth.
- The L2 and maximum norms of successive profile changes versus cycle.

Both depth axes span the complete mesh. T_0 is the starting equilibrium.
The two stored convergence arrays are `change_l2` (Euclidean norm of
T_j - T_(j-1), K) and `changes` (maximum absolute change, K).
The optional `initial_tol` stopping criterion uses the maximum norm.
L2 norms depend on the mesh and are compared between cycles on that mesh.

`<name>_Init_cycle001.png`, etc. are saved immediately after each cycle.
`<name>_Init_diagnostics.csv` contains only cycle number and the two
profile-change norms. `plot_files` contains the figure paths and
`monitor_plot` the last figure, axes and CSV path. No observations are
required; observation fields in sitepar are ignored by initialisation.

Example A enforces a model depth of at least **5,000 m**. Larger requested
depths are retained. `workflow.build_mesh` supports `min_depth` in its
meshpar dictionary; lower-level mesh routines remain usable for small
analytical tests.

In periodic mode, example A stops after the final repetition and returns
`result=None`; the generated initial condition is `run["init"]["Tinit"]`.
A subsequent forward run is a separate operation, not part of this
initial-condition check.

To plot an existing initialisation without rerunning:
```python
from init_plot import plot_initial_cycles
plots = plot_initial_cycles(sitepar, initial, dict(show=True))
```


## Tikhonov inversion in the notebook

Steps 10–13 of PyBTI_workflow.ipynb use SITE_Tikh.m's 21 logarithmic GST
parameters (110,000 to 30 yr), prior/start values 1, FD perturbation 0.001 K,
LSQR tolerance 1e-5 and 32 iterations, and at most 100 evaluated models.
Five warm-up updates use (tau0, tau1, tau2)=(1, 0, 0); subsequent GCV searches
use tau0=0.01, 48 logarithmic tau1 values from 1e-3 to 1e3, and tau2=0.
The synthetic site's gts=0 makes the GST parameters absolute temperatures.
The initial state, properties and basal heat flow are fixed during inversion.

The template contains OKU-specific heat-flow, site and history settings; these
are not transferred to A. The optional step 14 exposes the original signed
heat flows -36, -38, -40, -42 and -44 mW/m2 and rebuilds initial conditions
for each selected value. Serial execution is the notebook default.

```python
invpar = build_inversion(mesh, invgridpar, inv_settings)
inversion = run_inversion(sitepar, fwdpar, invpar, initial,
                          dict(outdir="results", verbose=True, n_jobs=1))
plots = plot_inversion(inversion, dict(outdir="results", formats=("png", "pdf")))
```

The tolerance tuple follows the existing driver: RMS target first, minimum
RMS improvement second. Full Gauss–Newton updates correspond to relax=1;
unused MATLAB relaxation/adaptation controls are not silently accepted.
All residuals are uncertainty-weighted consistently. Penalty bookkeeping
uses the sum of squared zeroth/first/second-difference terms, matching the
stacked least-squares system.

The solver now reaches a requested GCV search before convergence stopping,
continues with the last weights between searches, and stops at an evaluated
model at the iteration limit. Final temperatures, residuals, model vector,
Jacobian and active weights are consistent. The initial/prior output files
contain the actual initial/prior vectors. stop_reason reports RMS target,
stagnation/increase, or iteration limit; a low data misfit alone does not
establish that the recovered history is unique.

SITE_TikhPlot.m is adapted in tikh_plot.py. It plots the prescribed/recovered
histories, temperature fit, residuals, apparent upward conductive heat flow,
weighted RMS and GCV selection. Data-derived heat flow has the template's
21-point mirrored boxcar; temperatures/residuals remain unsmoothed. The
heat-flow unit is mW/m2. Effective final conductivity is averaged harmonically
over every cell between data depths. Time has no arbitrary calendar shift;
a symlog axis retains age zero. Returned Cmm is an inverse regularised local
Hessian, not a complete uncertainty estimate; legacy Rmm/Rdd definitions are
retained and are not labelled as model/data resolution matrices.


## MCMC inversion in the notebook

Steps 15–18 use the same 21-bin GST grid as Tikhonov and the Gaussian layout
from `SITE_MCMC.m`: `[GST_1 ... GST_21, QB, H]`. QB is positive upward in
mW/m²; H is in µW/m³. The configuration retains the template's GST/H priors,
three-sigma bounds, Gaussian proposal correlation length 3, DRAM scale 2,
`updatesigma=True`, likelihood starting sigma 0.1 K and `pom=-4 K`. Its QB
prior mean comes from the current site's signed `qb` instead of the template's
OKU-specific value.

```python
mcmcpar = build_mcmc(sitepar, fwdpar, invpar, initial, dict(
    method="dram", nsimu=1000, adaptint=200,
    sample_qb=True, sample_h=True,
    activate_qb=True, activate_h=False,
))
chain = run_mcmc(sitepar, fwdpar, initial, mcmcpar,
                 dict(job=1, name="site_validation", outdir="results/chains"))
posterior = summarize_mcmc(chain, sitepar, fwdpar, initial,
                           dict(burnin=.25, nsample=100, outdir="results"))
figure = plot_mcmc(posterior, dict(outdir="results"))
```

`sample_h=True` and `activate_h=False` deliberately reproduce MATLAB `Pact`:
H moves in the chain but does not enter the temperature objective, so its
posterior is not data identified. The Gaussian objective recalculates a steady
initial profile at `GST_1 + pom` for each proposal; it records but does not use
the workflow's repeated-history `Tinit`.

The notebook executes only 120 samples as an integration check. The retained
`production_nsimu=250000` and DRAM adaptation interval 10,000 come from the
MATLAB template. Use several independent job seeds and convergence/ESS checks
before interpreting posterior intervals. The plotting adapter removes the
13.5-year shift, corrects QB/H/RMS labels and retains age zero with a symlog
axis.
