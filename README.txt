CURRENT DICTIONARY WORKFLOW
---------------------------
See WORKFLOW.md: prep.py reads and prepares data; init.py implements
steady-state and repeated-GSTH initialisation; workflow.py passes MATLAB-
style dictionaries between all stages. examples/A/run_forward.py is the
complete synthetic forward example. Earlier translation notes follow.

GSTH / borehole temperature modelling - Python translation of src_for_claude
============================================================================
Author: Claude (Anthropic), generated 2026-09-21. AI-generated; tested against
analytical solutions and synthetic problems only, NOT against MATLAB output.
Review before production use.

FILES
-----
tikh_plot.py      SITE_TikhPlot diagnostics; notebook steps 10-14 run inversion
fwd_plot.py       plot_fwd(run_fwd output): paleoclimate, temperatures, residuals
numeric.py        callable numerical core (forward models, Jacobians,
                  regularisation, grids, GST builders, filters, statistics)
gsth_drivers.py   SitePar / FwdPar / InvPar containers, fwd_gsth / fwd_stat /
                  fwd_tran, tikhonov_gsth (= Tikh_gsth, GSTH_TikhSX, L1_gsth),
                  load_matlab_site (.mat reader)
gsth_mcmc.py      MCMC objective functions + pymcmcstat runner
tests/test_gsth.py   python tests/test_gsth.py   (or pytest -q tests)
phys.py           NOT included - your file; must sit next to numeric.py.

QUICK START
-----------
Python target for ongoing work: 3.12.
Create the dedicated environment with:
    conda env create --file BTI.yaml
    conda activate BTI
    python -m pytest -q tests
BTI.yaml pins the direct scientific dependencies; it excludes the optional
pymcmcstat sampler pending the later MCMC work. See README.md.

Forward-model plotting (NumPy and Matplotlib):
    from fwd_plot import plot_fwd
    # out = run_fwd(name, props, prep_fn, init_fn, ...)
    plots = plot_fwd(out, outdir="figures", show=True)
    # plots['figure'], plots['axes']: customise or close after use.
    # Optional: time_scale="symlog" for long histories including age zero.

Synthetic example: python fwd_plot.py --outdir figures --no-show
Plot tests: python -m pytest -q tests/test_fwd_plot.py

The paleoclimate panel shows GST[it[j]] applied over each model time
interval, matching heat1dnt; time is years before the model reference,
not an assumed calendar epoch. Temperatures are in degrees C; residuals
are observed minus calculated in K. The annotated RMS is normalised by
the observation errors. SITE_Plot.m is for MCMC and remains untranslated.

    from gsth_drivers import SitePar, FwdPar, InvPar, tikhonov_gsth
    import numeric as nm
    T = nm.heat1dns(k,kA,kB,h,r,por,qb,Ts,dz,ip,10,1e-5,1,"full")
    Tc = nm.heat1dnt(k,kA,kB,h,r,cp,por,qb,dz,ip,dt,it,GST,T0,1.0,10,1e-5,1,1,"full")
    res = tikhonov_gsth(site, fwd, inv, weight_residual=True)
The last argument "full" is the phys.SITE_PROPS key (MATLAB: addpath props/full).

CONVENTIONS (differences to MATLAB)
-----------------------------------
- Indices ip, it, id are 0-based. from_matlab_index() converts.
- qb signed for z down: normal terrestrial heat flow is NEGATIVE (as MATLAB).
- Argument order follows the MATLAB function DEFINITIONS:
    heat1dns(k,kA,kB,h,r,por,qb,Ts,dz,ip,maxiter,tol,freeze)
    heat1dnt(k,kA,kB,h,r,cp,por,qb,dz,ip,dt,it,GST,T0,theta,maxiter,tol,freeze,out)
- nargout-dependent outputs -> full_output=True.
- mstruct -> dict / dataclass.

HOW phys.py IS USED (numeric.PropModel)
---------------------------------------
phys.py is unchanged. Four mismatches with the MATLAB call sites are handled
in the adapter; please confirm they match your props/full/*.m:
1. kfT: MATLAB calls kfT(Tc,Pch); phys.kfT_phillips reads its 2nd slot as
   SALINITY. Pressure is NOT passed; salinity=0 (option: PropModel(salinity=)).
2. rcmT: for site "full", phys expects the VOLUMETRIC heat capacity (rcm), the
   MATLAB call passes density rm. The adapter passes rcm = rm*cpm (and rm for
   the 'direct' sites), so rho*cp(T) results for every site.
3. kmT: kmT_clauser fails on per-cell A/B arrays (ambiguous truth value).
   Cells are grouped by unique (kA,kB); kA/kB are passed as A/B. NaN = phys
   default coefficients.
4. ftheta: the ftheta.m in src_for_claude has a floor rfl=0.025 (Theta<rfl ->
   rfl, dTheta=0 there) and default w=1; phys.ftheta has neither (w=0.5).
   ftheta_apparent() adds the floor and uses w=1. The other phys models
   (gal, nic, hin) are selectable with freeze_model= / freeze_kw=.

MATLAB FILE -> PYTHON
---------------------
heat1dns, heat1dnt, forward_solve          numeric.heat1dns / heat1dnt / forward_solve
heat1dnt_with_p, heat1dns_with_p           same functions (heat1dnt_with_p only adds an
                                           unused arg rcl; heat1dns_with_p differs in comments)
ftheta                                     numeric.ftheta_apparent (via phys.ftheta)
n2c, c2n                                   numeric.n2c, c2n
sensfdt_pals / sensfdt_palp                numeric.sensfdt_pal (n_jobs=1 / >1)
reg1d, setregpar, setregvec, pvar          numeric.reg1d, setregpar (=setregvec), pvar
solvereg, GI, geninv, resmat, Rm           numeric.solvereg, geninv, resmat
set_mesh, set_mgsth                        numeric.set_mesh, set_mgsth
set_gst/lingst/pntgst/stpgst, set_prior    numeric.set_*
heat1dat, msetup                           numeric.heat1dat, msetup
prop2cell, vavg, avg                       numeric.prop2cell, vavg, avg
wfilt, smooth1, tri, box                   numeric.wfilt, smooth1, tri, box
RMS, mad                                   numeric.rms, mad
CovarGauss, CovarExpnl                     numeric.covar_gauss, covar_expnl
f_reg, f_scal, mscalsum (scaljac)          numeric.f_reg, f_scal, scale_jacobian
zgrad, splinefit (robust option)           numeric.zgrad, splinefit
Tikh_gsth, GSTH_TikhSX, L1_gsth            gsth_drivers.tikhonov_gsth
Fwd_gsth / Fwd_stat / Fwd_tran             gsth_drivers.fwd_gsth / fwd_stat / fwd_tran
GSTH_MCObjFun / MCFwdFun                   gsth_mcmc.objfun_basic / fwdfun
GSTH_MCObjFunGauss / ...GaussJoint         gsth_mcmc.objfun_gauss / objfun_joint
RunMCMC / mcmc_wrapper                     gsth_mcmc.RunMCMC / mcmc_wrapper (+ run_jobs)
mcmcrun / mcmcrun2 (MCMCSTAT)              pymcmcstat (Miles 2019); calchain -> predict_chain

NOT TRANSLATED
--------------
Replaced by numpy/scipy: lsqr, lsmr (scipy.sparse.linalg), nanmean, nanmedian,
prctile, quantile (numpy), splfit/splval/ppdiff (scipy.interpolate), mstruct,
str2strs, mdir. Third-party / not project-specific and not translated:
smoothn, idctn (Garcia), hmf (image filter), lbas (Hansen), slidefun,
vtkwrite, textloc, spread. MATLAB dev tools: depfun, mydepfun, package,
archive. SITE_TikhPlot.m is now adapted in tikh_plot.py. Other plot scripts
TikhPlot, Tikh_plot, TikhPlotOrig, set_graphpars
(TikhPlot/Tikh_plot contain syntax errors, e.g. "tauval =" and "100000Plot").
GSTH_MCPrior.m contains only a path string (priors are the mu/sigma of the
parameters, see make_params).

SOURCE ISSUES
-------------
Deviations (code could not run as written, or the intent is unambiguous):
 1. Inconsistent call signatures: r and h are swapped in Tikh_gsth.m and
    Fwd_tran.m (heat1dns call); MC objective files and Fwd_gsth.m use an
    older heat1dnt signature with lumped rc (Fwd_gsth.m also contains the
    stray text "kBgit poush"). Unified to the definitions. Sites given only
    as rc = rho*cp must supply r and c separately.
 2. sensfdt_pals: Jacobian columns misaligned if a GST parameter has no time
    step; qb column computed from a truncated time range; sensfdt_palp left
    it zero. Now: one consistent definition (column i = parameter i).
 3. set_gst/set_stpgst: node t == steptime(end) (e.g. the present) stayed 0.
 4. avg 'h': matrix division (n x n result); RMS: NaN mask applied to err after
    res was filtered; f_scal 'row': wrong matrix size; scaljac.m unrunnable
    (superseded by mscalsum.m). reg_opt 'fix' and ties in min(GCV) returned
    several indices - first one is used.
Kept as in MATLAB but questionable (please review):
 a. Tikh_gsth / solvereg: right-hand side uses the UNWEIGHTED residual with the
    WEIGHTED Jacobian (Wd*J). Consistent only if Terr == 1. Test: synthetic
    problem, Terr=0.01, rms 116 -> 108 (literal) vs 116 -> 0.5 and GST
    recovered to 0.05 K (weight_residual=True). Default is the literal
    behaviour; weight_residual=True is recommended.
 b. Corrected for the notebook inversion: theta_m now sums the three
    separate squared penalties, matching the minimised stacked system.
 c. UPR: 2*|Terr|^2*tr(M)/nd with |Terr| the 2-norm of the whole vector.
 d. heat1dnt outputs dT, Q are computed from the INITIAL profile T0.
 e. set_prior writes the smoothed value to index k-L (shift by L samples).
 f. heat1dnt sets Tlast(1)=GST(1) at the start (also in restarted Jacobian
    runs); irrelevant for theta = 1, O(tolnl) effect otherwise.
 g. CovarGauss returns C*diag(s^2) (not symmetric); resmat 'Rm' is G*G'.
 h. Corrected: iterations between regularisation searches update with the
    last selected weights; convergence waits for the first requested search.
 i. rms_L in the regpar loop used length(scalar); now r_norm/sqrt(nobs).

ENVIRONMENT NOTES
-----------------
Historical notes for the optional pymcmcstat 1.9.1 (not in BTI.yaml):
 - plotting/utilities.py: "from scipy import pi, sin, cos" fails on recent
   SciPy. Change to "from numpy import pi, sin, cos" (or use older SciPy).
 - updatesigma=True fails with NumPy >= 2 (array assigned to scalar).
   gsth_mcmc applies a runtime shim (same distribution, different random
   stream); the pymcmcstat sources are not modified.
 - The tests here used stand-ins for h5py / mcmcplot / statsmodels (not
   installable offline); with a normal "pip install pymcmcstat" they exist.
Runtime: ~0.3 s per forward run (400 time steps, 120 cells, permafrost on).
MCMC cost = nsimu x that.

NOT TESTED
----------
Comparison with MATLAB output; load_matlab_site on real .mat files (only
synthetic ones); sensfdt_pal(n_jobs>1) and run_jobs on a cluster.

REFERENCES
----------
Miles, P. R., pymcmcstat: A Python Package for Bayesian Inference Using
  Delayed Rejection Adaptive Metropolis, J. Open Source Softw., 2019, 4(38),
  1417, doi:10.21105/joss.01417 (from the PDF; the file is named "2029").
Haario, H., Laine, M., Mira, A. & Saksman, E., DRAM: Efficient adaptive MCMC,
  Stat. Comput., 2006, 16(4), 339-354, doi:10.1007/s11222-006-9438-0 (from the
  reference list of the PDF).
Haario, H., Saksman, E. & Tamminen, J., An adaptive Metropolis algorithm,
  Bernoulli, 2001, 7(2), 223-242 (no DOI in the PDF - NOT VERIFIED).
Physics/method references of the MATLAB sources (Beltrami & Mareschal 1991;
Beltrami, Jessop & Mareschal 1992; Mareschal & Beltrami 1992; Youzwishen &
Sacchi 2006; Mehanee & Zhdanov 2002) are listed in numeric.py; only the last
carries a DOI (from the source); the others are NOT VERIFIED.
