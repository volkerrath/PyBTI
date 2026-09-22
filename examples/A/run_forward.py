"""
run_forward.py -- Dictionary-based forward workflow for synthetic case A.

Data columns and units are explicit. example_parameters selects periodic
initialisation at GST0=6 deg C, replaying the known history initial_iter
times as in SITE_Init.m. The command line selects equilibrium unless
--periodic is supplied.
The older-than-70-ka temperature is explicitly set to 6 deg C, consistent
with the background profile in the synthetic file. It is an example
assumption, not a hidden default of the history reader.

Provenance notice
Author         : Codex (OpenAI)
Date generated : 2026-09-22
Status         : Python 3.12 synthetic example; no MATLAB comparison.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from init import YEAR2SEC, build_gsth, build_initial
from prep import read_data, prepare_site
from workflow import build_mesh, run_forward, plot_forward


def example_parameters():
    """Return editable MATLAB-style dictionaries for synthetic example A."""
    folder = Path(__file__).resolve().parent
    return dict(
        datapar=dict(
            file=folder / "DataBallingA.csv",
            zcol=0,
            Tcol=3,
            Terr=0.05,
            depth_range=(10, 2000),
            synthetic=True,
        ),
        meshpar=dict(
            name="SYNA",
            min_depth=5000.0,
            depth=dict(zend=5000),
            time=dict(tstart=110000 * YEAR2SEC, tend=30 * YEAR2SEC, nt=401),
            time_nodes=[-70000 * YEAR2SEC, -10000 * YEAR2SEC],
        ),
        sitepar=dict(
            name="SYNA",
            props="syn",
            k=2.3253,
            r=1000.0,
            c=2500.0,
            h=0.0001e-6,
            p=0.0001,
            kA=0.0,
            kB=0.0,
            qb=-0.069759,
        ),
        fwdpar=dict(theta=1.0, maxitnl=4, tolnl=1e-5, freeze=1),
        gstpar=dict(
            file=folder / "GSTHBallingA.csv",
            form="steps",
            time_unit="yr",
            time_convention="before_reference",
            prehistory=6.0,
        ),
        initpar=dict(init_type="periodic", GST0=6.0, initial_iter=30),
    )
    

def run_example(parameters: dict):
    """Prepare the model and initialise; periodic mode stops at Tinit.

    parameters contains datapar, meshpar, sitepar, fwdpar, gstpar, initpar.
    Every step passes/returns dictionaries. Optional initpar['plotpar']
    enables per-cycle figures and convergence diagnostics. The model is
    at least 5000 m deep. Periodic mode returns result=None and performs
    no additional forward run or observation comparison; Tinit is the
    result of the final repetition. Equilibrium mode retains the separate
    forward-example calculation.
    """
    observations = read_data(parameters["datapar"])
    meshpar = dict(parameters["meshpar"])
    meshpar["min_depth"] = max(5000.0, meshpar.get("min_depth", 0.0))
    mesh = build_mesh(meshpar, observations)
    sitepar = prepare_site(parameters["sitepar"], mesh, observations)
    forcing = build_gsth(mesh, parameters["gstpar"])
    initpar = dict(parameters["initpar"], GST=forcing["GST"], it=forcing["it"])
    initial = build_initial(sitepar, parameters["fwdpar"], initpar)
    result = (
        None
        if initial["init_type"] == "periodic"
        else run_forward(sitepar, parameters["fwdpar"], initial)
    )
    return dict(
        name=sitepar["name"],
        mesh=mesh,
        site=sitepar,
        fwd=dict(parameters["fwdpar"]),
        init=initial,
        result=result,
        observations=observations,
        forcing=forcing,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic A forward model")
    parser.add_argument(
        "--periodic", action="store_true", help="repeat the GST history"
    )
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument(
        "--noise", action="store_true", help="add seeded correlated noise"
    )
    parser.add_argument(
        "--outdir", default=None, help="optional figure output directory"
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    parameters = example_parameters()
    parameters["initpar"].update(
        init_type="periodic" if args.periodic else "equilibrium",
        initial_iter=args.cycles,
    )
    if args.periodic:
        parameters["initpar"].update(
            verbose=True,
            plotpar=dict(outdir=args.outdir, show=not args.no_show),
        )
    if args.noise:
        parameters["datapar"]["noise"] = dict(
            kind="gaussian", length=150.0, seed=0
        )
    run = run_example(parameters)
    print(
        "SYNA: %d observations, %d depth nodes, %d time nodes"
        % (len(run["site"]["id"]), run["mesh"]["nz"], run["mesh"]["nt"])
    )
    print(
        "Model depth: %.0f m; initialisation: %s, %d cycles"
        % (
            run["mesh"]["z"][-1],
            run["init"]["init_type"],
            run["init"]["niter"],
        )
    )
    if args.periodic:
        # The monitor shows the actual cycle-end initial conditions.
        if not args.no_show:
            import matplotlib.pyplot as plt

            plt.show()
    else:
        plot = plot_forward(
            run, dict(outdir=args.outdir, show=not args.no_show)
        )
        if args.no_show:
            import matplotlib.pyplot as plt

            plt.close(plot["figure"])
