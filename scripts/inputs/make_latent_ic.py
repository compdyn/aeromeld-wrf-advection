#!/usr/bin/env python3
"""Build the latent-space WRF inputs (wrfinput / wrfbdy) for latent advection.

Latent-space twin of make_ic_control.py (WRF-PartMC, particles_from_file). Both
live in this input_builders/ directory and describe the SAME scenario -- a base
population everywhere plus an optional box "region" that swaps in a second
population -- so a latent-advection run can be compared apples-to-apples with a
real-particle run. Keep the ``--region`` box and grid identical between the two.

Where make_ic_control.py writes a per-cell ``num_conc`` control file pointing at
saved PartMC particle files, this tool writes the encoder's latent variables
(``ltnt_var1 .. ltnt_varN`` plus ``ntot``) straight into a wrfinput template and,
optionally, a wrfbdy template. A "population" here is a latent vector (the
encoding of a particle population) rather than a particle file.

Scenario (mirrors make_ic_control.py):
  * base population    -> ``--base-latent``   applied to every cell / boundary.
  * base num_conc      -> ``--base-value``    ntot (1/m^3) in every cell.
  * region box         -> ``--region i1 i2 j1 j2 k1 k2`` (1-based, inclusive).
  * region population  -> ``--region-latent`` swapped in inside the box.
  * region num_conc    -> ``--region-value`` shaped by ``--region-profile``
                          (shared with make_ic_control.py; see region_profiles.py).

Because latent variables can be negative, do NOT advect them with a monotonic
scheme (set chem_adv_opt accordingly in namelist.input).

Grid mapping: wrfinput latent vars have dims (Time, bottom_top, south_north,
west_east), so a 1-based box i(=west_east) j(=south_north) k(=bottom_top) is
written to [0, k1-1:k2, j1-1:j2, i1-1:i2]. Grid sizes come from the template.

Importable: write_wrf_inputs.ipynb imports build_latent_ic / build_latent_bc and
only handles plotting.

Needs a Python with xarray + netCDF4 + numpy (e.g. the `wrf` or `xwrf` conda env).

Example (CARES em_real -- same grid/box as run_latent_ics.sh / run_ics.sh):
  python make_latent_ic.py \
      --ic-template .../wrfinput_d01_cares --ic-outfile .../wrfinput_d01_advection_test \
      --bc-template .../wrfbdy_d01_cares   --bc-outfile .../wrfbdy_d01_advection_test \
      --base-value 1e5 \
      --base-latent="-0.1027,-2.2868,6.1429,1.4282,-1.0323,0.3196,-1.6883,-1.4831,0.0577" \
      --region 50 80 40 70 1 10 --region-value 1e6 --region-profile gaussian \
      --region-latent="0.0667,0.1501,-0.6538,1.8972,-0.3033,1.8006,0.8581,0.9799,-0.0596"

  (Pass the latent vectors with =VALUE so a leading "-" isn't read as a flag.)
"""

import argparse
import os
import sys

import numpy as np

# Make the shared profile module importable however this script is launched
# (directly, via a forwarding shim, or imported by the notebook).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_profiles as rp

try:
    import xarray as xr
except ImportError:
    sys.exit("xarray not found -- run under a conda env that has it, "
             "e.g.  conda activate wrf")


# ---------------------------------------------------------------------------
# Builders (operate in place on an opened dataset; also imported by the notebook)
# ---------------------------------------------------------------------------
def latent_var_names(n_latent):
    return ["ltnt_var%i" % (i + 1) for i in range(n_latent)]


def build_latent_ic(ds, *, ml_vars, base_latent, base_value,
                    region=None, region_latent=None, region_value=0.0,
                    profile="constant", sigma=None, vscale=None, normalized=True):
    """Fill an opened wrfinput dataset in place with the latent IC.

    ``ntot`` is base_value everywhere; inside ``region`` it is overwritten by
    the chosen ``profile`` (see region_profiles.py). Each latent variable is
    base_latent everywhere and, inside the region, is swapped to region_latent
    (``normalized``) or weighted by ntot (not ``normalized``). ``region`` is
    1-based inclusive (i1, i2, j1, j2, k1, k2) or None.
    """
    ds["ntot"].values[:] = base_value

    sl = None
    if region is not None:
        i1, i2, j1, j2, k1, k2 = region
        sl = (0, slice(k1 - 1, k2), slice(j1 - 1, j2), slice(i1 - 1, i2))
        ni, nj, nk = i2 - i1 + 1, j2 - j1 + 1, k2 - k1 + 1
        ds["ntot"].values[sl] = rp.region_profile(
            profile, ni, nj, nk, base_value, region_value, sigma, vscale,
            order="kji")

    for idx, name in enumerate(ml_vars):
        if normalized:
            ds[name].values[:] = base_latent[idx]
            if sl is not None:
                ds[name].values[sl] = region_latent[idx]
        else:
            ds[name].values[:] = ds["ntot"].values * base_latent[idx]
            if sl is not None:
                ds[name].values[sl] = ds["ntot"].values[sl] * region_latent[idx]
    return ds


def build_latent_bc(ds, *, ml_vars, base_latent, base_value, normalized=True):
    """Fill an opened wrfbdy dataset in place: base inflow, zero tendencies.

    All four edges are set to the base population/num_conc and all tendencies to
    zero (piecewise-constant boundaries, mirroring WRF-PartMC). The region box
    is interior-only and never touches the boundaries.
    """
    edges = ["BXS", "BXE", "BYS", "BYE"]
    tends = ["BTXS", "BTXE", "BTYS", "BTYE"]

    for e in edges:
        ds["ntot_%s" % e].values[:] = base_value
    for t in tends:
        ds["ntot_%s" % t].values[:] = 0.0

    for idx, name in enumerate(ml_vars):
        if normalized:
            for e in edges:
                ds["%s_%s" % (name, e)].values[:] = base_latent[idx]
        for t in tends:
            ds["%s_%s" % (name, t)].values[:] = 0.0
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_vector(text, n, what):
    parts = [p for p in text.replace(",", " ").split() if p]
    try:
        vec = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("%s: could not parse floats from %r"
                                         % (what, text))
    if len(vec) != n:
        raise argparse.ArgumentTypeError(
            "%s: expected %d values, got %d" % (what, n, len(vec)))
    return vec


def main():
    p = argparse.ArgumentParser(
        description="Build latent-space wrfinput (and optional wrfbdy) inputs.")
    p.add_argument("--ic-template", required=True, metavar="FILE",
                   help="wrfinput template with latent vars present (zeroed).")
    p.add_argument("--ic-outfile", default="./latent_inputs/wrfinput_d01_latent",
                   metavar="FILE", help="wrfinput to write.")
    p.add_argument("--bc-template", metavar="FILE",
                   help="wrfbdy template. If given, boundaries are written too.")
    p.add_argument("--bc-outfile", default="./latent_inputs/wrfbdy_d01_latent",
                   metavar="FILE", help="wrfbdy to write (with --bc-template).")

    p.add_argument("--n-latent", type=int, default=9,
                   help="Number of latent variables ltnt_var1..N (default 9).")
    p.add_argument("--base-value", type=float, default=0.0,
                   help="Base ntot (1/m^3) in every cell / boundary (default 0).")
    p.add_argument("--base-latent", required=True, metavar='"v1,..,vN"',
                   help="Base population: N comma/space separated latent values.")

    p.add_argument("--region", type=int, nargs=6,
                   metavar=("I1", "I2", "J1", "J2", "K1", "K2"),
                   help="1-based inclusive box that swaps in the region "
                        "population. Omit for no region.")
    p.add_argument("--region-latent", metavar='"v1,..,vN"',
                   help="Region population latent values (required with --region).")
    p.add_argument("--region-value", type=float, default=0.0,
                   help="Region ntot (1/m^3); peak value for gaussian (default 0).")
    p.add_argument("--region-profile", default="constant",
                   choices=sorted(rp.PROFILES),
                   help="Shape of ntot inside the region (default constant).")
    p.add_argument("--region-sigma", type=float, default=None,
                   help="Horizontal Gaussian width in cells (default min(box)/6).")
    p.add_argument("--region-vscale", type=float, default=None,
                   help="Vertical exponential scale height in levels for the "
                        "gaussian's decrease with height (default box_height/2).")

    p.add_argument("--ntot-weighted", action="store_true",
                   help="Store latent * ntot instead of raw normalized latent.")
    args = p.parse_args()

    ml_vars = latent_var_names(args.n_latent)
    base_latent = _parse_vector(args.base_latent, args.n_latent, "--base-latent")
    normalized = not args.ntot_weighted

    region = region_latent = None
    if args.region is not None:
        if args.region_latent is None:
            p.error("--region requires --region-latent")
        region = args.region
        region_latent = _parse_vector(args.region_latent, args.n_latent,
                                      "--region-latent")

    # ---- Initial condition ----
    ds_ic = xr.open_dataset(args.ic_template).load()
    nz = ds_ic.sizes["bottom_top"]
    ny = ds_ic.sizes["south_north"]
    nx = ds_ic.sizes["west_east"]

    if region is not None:
        i1, i2, j1, j2, k1, k2 = region
        for name, lo, hi, n in (("i", i1, i2, nx), ("j", j1, j2, ny),
                                ("k", k1, k2, nz)):
            if not (1 <= lo <= hi <= n):
                p.error("region %s range %d..%d out of bounds 1..%d"
                        % (name, lo, hi, n))

    build_latent_ic(ds_ic, ml_vars=ml_vars, base_latent=base_latent,
                    base_value=args.base_value, region=region,
                    region_latent=region_latent, region_value=args.region_value,
                    profile=args.region_profile, sigma=args.region_sigma,
                    vscale=args.region_vscale, normalized=normalized)

    os.makedirs(os.path.dirname(args.ic_outfile) or ".", exist_ok=True)
    ds_ic.to_netcdf(args.ic_outfile)

    print("wrote %s" % args.ic_outfile)
    print("  grid = %d x %d x %d  (nx x ny x nz)" % (nx, ny, nz))
    print("  ntot: base %.3g, region %.3g (%s), min/max %.3g/%.3g"
          % (args.base_value, args.region_value, args.region_profile,
             float(ds_ic["ntot"].values.min()), float(ds_ic["ntot"].values.max())))
    if region is not None:
        print("  region box (1-based incl) i %d..%d  j %d..%d  k %d..%d"
              % (i1, i2, j1, j2, k1, k2))

    # ---- Boundary conditions ----
    if args.bc_template is not None:
        ds_bc = xr.open_dataset(args.bc_template).load()
        build_latent_bc(ds_bc, ml_vars=ml_vars, base_latent=base_latent,
                        base_value=args.base_value, normalized=normalized)
        os.makedirs(os.path.dirname(args.bc_outfile) or ".", exist_ok=True)
        ds_bc.to_netcdf(args.bc_outfile)
        print("wrote %s  (base inflow, zero tendencies)" % args.bc_outfile)


if __name__ == "__main__":
    sys.exit(main())
