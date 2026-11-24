#!/usr/bin/env python3
"""Compare a newly built WRF-AeroMELD run dir against a reference run dir.

    compare_inputs.py NEW_RUN REF_RUN [--ignore VAR[,VAR...]]

Checks wrfinput_d01, wrfbdy_d01 and every wrfchemi_d01_* of REF_RUN (following
links). Each differing variable is classed as

  LATENT     ltnt_var*, ltnt_var*_B*, E_LTNT_VAR* -- expected to differ between
             VAE models; a max |diff| of ~1e-6 or less is float32 noise (e.g.
             encoding on CPU instead of GPU), not a different model.
  DIFFERENT  anything else: met fields, ntot, E_NTOT, grid, global attributes.
             These are identical for every model built for the standard
             scenario, so a difference means the scenario, the template or the
             emission rate changed. --ignore E_NTOT accepts a deliberately
             different emission rate.

The wrfchemi `History` attribute (a creation timestamp) is never compared.
Exit status 0 if nothing outside the latent fields (and --ignore) differs.
Needs netCDF4 + numpy (the xwrf env).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from netCDF4 import Dataset

LATENT = re.compile(r"^(ltnt_var\d+(_B[A-Z]+)?|E_LTNT_VAR\d+)$", re.I)
SKIP_ATTRS = {"History"}


def max_diff(a, b):
    """None if equal (NaN == NaN), else the max |a - b| (inf for shape/type mismatch)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.dtype.kind in "fc":
        if np.array_equal(a, b, equal_nan=True):
            return None
        return float(np.nanmax(np.abs(a.astype("f8") - b.astype("f8"))))
    return None if np.array_equal(a, b) else float("inf")


def compare_file(new, ref, ignore):
    bad, latent = [], []
    with Dataset(new) as n, Dataset(ref) as r:
        vn, vr = set(n.variables), set(r.variables)
        bad += ["%s: missing from the new file" % v for v in sorted(vr - vn)]
        bad += ["%s: not in the reference file" % v for v in sorted(vn - vr)]
        for v in sorted(vn & vr):
            d = max_diff(n[v][:], r[v][:])
            if d is None or v in ignore:
                continue
            if LATENT.match(v):
                latent.append(d)
            else:
                bad.append("%s (max |diff| %.3g)" % (v, d))
        for a in sorted((set(r.ncattrs()) | set(n.ncattrs())) - SKIP_ATTRS):
            if a not in n.ncattrs() or a not in r.ncattrs():
                bad.append("global attribute %s is in only one file" % a)
            elif max_diff(np.atleast_1d(n.getncattr(a)), np.atleast_1d(r.getncattr(a))) is not None:
                bad.append("global attribute %s" % a)
    return bad, latent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_run")
    ap.add_argument("ref_run")
    ap.add_argument("--ignore", default="",
                    help="comma-separated variables not to compare (e.g. E_NTOT)")
    args = ap.parse_args()
    ignore = {v for v in args.ignore.split(",") if v}

    names = ["wrfinput_d01", "wrfbdy_d01"] + sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.ref_run, "wrfchemi_d01_*")))
    failed, latent_max, n_latent = False, 0.0, 0
    for name in names:
        new, ref = os.path.join(args.new_run, name), os.path.join(args.ref_run, name)
        if not os.path.exists(new):
            print("%-34s MISSING from %s" % (name, args.new_run))
            failed = True
            continue
        bad, latent = compare_file(new, ref, ignore)
        if bad:
            print("%-34s DIFFERENT" % name)
            for b in bad:
                print("    " + b)
            failed = True
        elif latent:
            print("%-34s ok (%d latent fields differ, max |diff| %.3g)"
                  % (name, len(latent), max(latent)))
        else:
            print("%-34s ok (identical)" % name)
        if latent:
            n_latent += len(latent)
            latent_max = max(latent_max, max(latent))
    for e in sorted(set(os.path.basename(p) for p in
                        glob.glob(os.path.join(args.new_run, "wrfchemi_d01_*"))) - set(names)):
        print("%-34s only in the new run" % e)

    if failed:
        print("RESULT: FAIL -- non-latent fields differ from the reference")
    elif n_latent == 0:
        print("RESULT: PASS -- bit-identical to the reference, latents included")
    else:
        kind = ("float32 noise (same model)" if latent_max <= 1e-5
                else "a different model's latents")
        print("RESULT: PASS -- only latent fields differ (max |diff| %.3g: %s)"
              % (latent_max, kind))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
