"""
test_fwd_driver.py -- self-tests for mesh.py and fwd_driver.py.

Run from the package directory (phys.py, numeric.py, gsth_drivers.py must
be present):
    python tests/test_fwd_driver.py

Provenance notice
Author         : Claude (Anthropic)
Date generated : 2026-09-22
Status         : AI-generated test code; review before production use.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 ".."))
import mesh as gmesh                                    # noqa: E402
from fwd_driver import run_fwd, _demo_prep_fn, _demo_init_fn  # noqa: E402
import gsth_drivers as gd                                # noqa: E402

Y = gmesh.YEAR2SEC


def test_depth_mesh_log():
    m = gmesh.build_depth_mesh(ztype="log", zend=1000.0, dzstart=5.0,
                               gfac=1.05, ngen=200)
    assert m["z"][0] == 0.0 and m["z"][-1] == 1000.0
    assert np.all(np.diff(m["z"]) > 0)
    assert m["ip"].size == m["nz"] - 1
    assert np.array_equal(m["ip"], np.arange(m["nz"] - 1))


def test_depth_mesh_special_overrides_args():
    # 'special' ignores the dzstart/gfac/ngen arguments (MATLAB hard-codes
    # its own 8.0/1.02/500 in this branch).
    m1 = gmesh.build_depth_mesh(ztype="special", zend=3000.0, zlmax=1000.0,
                                nz=11, dzstart=999.0, gfac=999.0, ngen=1)
    m2 = gmesh.build_depth_mesh(ztype="special", zend=3000.0, zlmax=1000.0,
                                nz=11, dzstart=1.0, gfac=1.001, ngen=2)
    assert np.array_equal(m1["z"], m2["z"])


def test_time_mesh_log_matches_set_mesh():
    import numeric as nm
    m = gmesh.build_time_mesh(ttype="log", tstart=1000 * Y, tend=10 * Y,
                              nt=51, direction=-1)
    t_ref, _ = nm.set_mesh(1000 * Y, 10 * Y, 51, "log", -1)
    assert np.allclose(m["t"], t_ref)
    assert m["it"].size == m["nt"] and m["it"][0] == 0
    assert m["tm"].size == m["nt"] - 1


def test_run_fwd_end_to_end_and_self_consistency():
    out = run_fwd("TESTBORE", "full", _demo_prep_fn, _demo_init_fn,
                  mesh_kw=dict(depth_kw=dict(zend=1500.0, nz=121),
                              time_kw=dict(tstart=5000 * Y, tend=1.0 * Y,
                                          nt=61)),
                  verbose=False)
    assert out["name"] == "TESTBORE"
    r = out["result"]
    assert r["Tcalc"].shape[0] == out["mesh"]["nz"]
    assert np.isfinite(r["rms"])

    # self-consistency: forward model reproduces its own output exactly
    site = out["site"]
    site.Tobs = r["Tcalc"][r["id"], -1].copy()
    r2 = gd.fwd_gsth(site, out["fwd"], out["init"]["GST"],
                     out["init"]["it"], Tinit=out["init"]["Tinit"],
                     verbose=False)
    assert r2["rms"] < 1e-8 and r2["mae"] < 1e-8


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("%-48s" % name, end="", flush=True)
            fn()
            print(" ok")
    print("all tests passed")
