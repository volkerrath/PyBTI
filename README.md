# PyBTI
Borehole Temperature Inversion

Python target: **3.12**. The numerical routines retain their existing
interfaces and dictionary inputs; module and function docstrings keep
the translation's established structure.

## Conda environment

From the repository root:

```text
conda env create --file BTI.yaml
conda activate BTI
python --version
python -m pytest -q tests
python examples/A/run_forward.py --periodic --cycles 30 --outdir examples/A/output/repeated_history --no-show
```

[BTI.yaml](BTI.yaml) selects Python 3.12 and pins NumPy, SciPy, Matplotlib,
pytest, and notebook tools. Conda selects the appropriate builds for the operating system;
this is an environment specification, not a complete dependency lock file.
To create another independent environment from the same specification:

```text
conda env create --file BTI.yaml --name BTI-test
conda activate BTI-test
```

The environment covers forward modelling, steady-state and repeated-history
initialisation, Tikhonov inversion, and DRAM MCMC. `pymcmcstat==1.9.1` is
installed through pip inside the Conda environment; the adapter supplies the
small SciPy/NumPy compatibility shims required by that older release.
Historical compatibility notes are in [README.txt](README.txt).

See [WORKFLOW.md](WORKFLOW.md) for the dictionary-based interfaces and
[examples/A/run_forward.py](examples/A/run_forward.py) for a runnable
synthetic example.

Repeated-history initialisation uses a model at least 5,000 m deep and
saves per-cycle temperature profiles and convergence-norm CSV output.
Its diagnostics compare successive simulations without using observations.

## Interactive workflow notebook

Open [PyBTI_workflow.ipynb](PyBTI_workflow.ipynb) for the complete synthetic A
workflow. The initialisation dictionary at the top selects steady-state
(`equilibrium`) or repeated-history (`periodic`) initialisation.
The default is periodic, with 30 cycles and no additional forward run.
Each completed cycle refreshes one inline plot and saves a PNG snapshot.

For an existing BTI environment, install the notebook dependencies:

```text
conda env update --name BTI --file BTI.yaml
conda activate BTI
python -m ipykernel install --sys-prefix --name bti --display-name "Python 3.12 (BTI)"
python -m jupyterlab PyBTI_workflow.ipynb
```

For a fresh environment, use `conda env create --file BTI.yaml` instead
of the update command. Launch Jupyter from the repository root and select
**Python 3.12 (BTI)**. If using a different environment name, activate it
and use a distinct kernel name/display name when registering it.

Run cells in order; restart and run all cells after changing parameters.
The notebook saves plots, convergence CSV, numerical NPZ arrays and a
JSON parameter record under `examples/A/output/notebook/<mode>/`.
The optional forward step is separate from initialisation. Steps 10–13 run
Tikhonov inversion using the settings in `templates/SITE_Tikh.m`, and
[tikh_plot.py](tikh_plot.py) supplies the `SITE_TikhPlot.m` diagnostics.
Step 14 provides an optional basal heat-flow sweep, rebuilding the initial
profile for each value. Steps 15–18 configure, run, summarize, plot and save a
short DRAM validation chain using `SITE_MCMC.m`, `SITE_MCMCPlot.m` and
`SITE_Plot.m` as the source templates.

The default inversion uses 21 logarithmic GST parameters and searches 48
first-difference regularisation weights with GCV. It keeps the synthetic
site's basal heat flow and the generated initial state fixed during inversion.
The notebook saves inversion arrays, priors, parameter JSON, iteration/GST
CSVs, the GCV search and PNG/PDF figures in its `tikhonov` output subfolder.
All settings and documented adaptations are visible in step 10.

## Python 3.12 validation

Validated on Windows with Python 3.12.14 and the dependencies in BTI.yaml:
35 tests passed. All Python sources compile with syntax warnings treated as
errors. The full notebook also executes successfully with the BTI kernel.

Example A's forward solution and 30-cycle initialisation at 5,000 m were
compared with Python 3.11.15 using the same direct scientific package
versions. The largest absolute difference across temperatures and
convergence norms was 3.2e-11 K. Per-cycle PNG and CSV output also passed
a two-cycle command-line check. No numerical algorithm changes were needed
for this migration; MATLAB-output validation remains outstanding.

## Tikhonov validation

The Tikhonov checks cover analytical forward solutions, synthetic signal recovery,
mandatory GCV execution after warm-up, iteration-limit consistency, saved
starting/prior vectors, final Jacobians, and heat-flow signs/units.

With the notebook's default periodic initialisation and SITE_Tikh.m controls,
the synthetic A inversion evaluates eight models and reduces weighted RMS
from 42.6689 to 0.235069 (temperature RMSE 0.0117534 K). GCV selects
(tau0, tau1, tau2)=(0.01, 0.001, 0), at the lower tau1 search boundary.
The recovered GSTH remains oscillatory: the small temperature residual does
not establish accurate or unique history recovery. The notebook retains the
template range and makes this limitation visible.


## MCMC workflow and validation

`workflow.build_mcmc`, `run_mcmc`, `summarize_mcmc` and `plot_mcmc` provide
plain-dictionary interfaces around the translated Gaussian GST/Q/H objective.
The default parameterization has 21 GST values, positive-upward basal heat flow
in mW/m², and heat production in µW/m³. The synthetic site's heat-flow prior
mean is derived from `abs(sitepar["qb"])`; the OKU-specific 31 mW/m² value is
not reused. H is sampled by default but physically inactive, matching `Pact` in
`SITE_MCMC.m`, so its distribution is prior-driven.

The executed notebook uses one 120-sample chain only to validate the complete
path. After 25% burn-in it retains 90 states, with successive-state acceptance
about 0.202. Its QB median is 67.302 mW/m² and median uncertainty-normalised
RMS is 8.613. These values are diagnostics of a short, non-converged run. The
MATLAB production setting of 250,000 DRAM samples and adaptation interval
10,000 remains in the configuration; production inference requires several
independent chains plus mixing, effective-sample-size and split-R-hat checks.

Posterior output is saved below the notebook's `mcmc` subfolder. The combined
figure shows GST intervals, posterior temperatures, observed-minus-calculated
residuals, QB/H densities, chain traces, RMS and sampled likelihood sigma. The arbitrary 13.5-year shift and
swapped histogram labels in the MATLAB plotting scripts are corrected.
