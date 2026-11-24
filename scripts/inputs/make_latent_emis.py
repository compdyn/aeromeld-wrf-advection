#!/usr/bin/env python3
"""Build the latent-space WRF emission inputs (wrfchemi) for latent advection.

Emission twin of make_latent_ic.py. Where that tool fills a wrfinput/wrfbdy with
the state latent variables (``ltnt_var1..N``, ``ntot``), this one writes the
hourly ``wrfchemi_d<NN>_<date>`` files holding the emission counterparts
(``E_LTNT_VAR1..N``, ``E_NTOT``) that WRF reads through auxinput5 under
``emiss_opt = 21``.

The files are built FROM SCRATCH on the grid of a wrfinput template, so no
pre-existing wrfchemi is needed -- the ones shipped with the CARES case are on a
different (335 x 443) domain and cannot be reused. Grid sizes, XLAT/XLONG and
the WRF global attributes all come from ``--grid-template``.

Scenario (mirrors make_latent_ic.py):
  * base population    -> ``--base-latent``   emitted in every cell / level.
  * base flux          -> ``--base-value``    E_NTOT everywhere (default 0).
  * region box         -> ``--region i1 i2 j1 j2 k1 k2`` (1-based, inclusive;
                          k runs over the kemit emission levels, not e_vert).
  * region population  -> ``--region-latent`` emitted inside the box.
  * region flux        -> ``--region-value`` shaped by ``--region-profile``
                          (shared with make_latent_ic.py; see region_profiles.py).
  * time variation     -> ``--time-profile`` scales E_NTOT (and with it every
                          E_LTNT_VAR) record by record, leaving the emitted
                          latent VECTOR unchanged.

INTENSIVE STORAGE. ``E_LTNT_VARi`` holds the raw latent vector of the emitted
population, exactly as ``ltnt_vari`` does in the wrfinput/wrfbdy written by
make_latent_ic.py -- the same convention as the number-weighted advection remap
in dyn_em/solve_em.F, where storage is intensive and the extensive product
ltnt*N is formed inside the operator and never leaves it.
module_emissions_anthropogenics.F does the same for emissions:

    dN       = conv_rho_aer * E_NTOT
    ltnt_new = (N_old * ltnt_old + dN * E_LTNT) / (N_old + dN)

Consequences worth knowing:
  * Rescaling the emission strength means editing E_NTOT alone; the latent
    fields are untouched and stay correct.
  * Where E_NTOT is 0, dN is 0, so E_LTNT cannot contribute. This builder still
    writes the base latent vector across the whole domain there -- the field is
    "what would be emitted", not "what is emitted". Read E_NTOT to see where
    emissions actually are; a nonzero E_LTNT_VARi does NOT mean emissions.
  * Merging sectors is a build-time job (see build_latent_emis): accumulate
    sum_s rate_s and sum_s rate_s*latent_s, then divide. That is exactly
    equivalent to letting WRF apply the sectors one after another.

Units: ``conv_rho_aer = alt * dt / dz8w``, so E_NTOT is a number flux per unit
area -- number/m^2/s in whatever number unit ``ntot`` carries (the same unit as
``--base-value`` in run_latent_ics.sh, per m^2 per s instead of per m^3).
E_LTNT_VARi is dimensionless.

Namelist requirements on the WRF side:
  * ``emiss_opt = 21``        selects the epartml emis_ant package.
  * ``io_style_emissions = 2`` dated hourly files. The CARES namelist currently
                              ships with 0, which reads NO emissions at all.
  * ``io_form_auxinput5 = 2``  or auxinput5 is opened but never actually read.
  * ``kemit``                 must equal ``--kemit`` here.
  * ``auxinput5_interval_m``  must equal ``--interval-m`` here.

Needs a Python with netCDF4 + numpy (e.g. the `wrf` or `xwrf` conda env).

Example (CARES em_real -- same grid/box as run_latent_emis.sh):
  python make_latent_emis.py \
      --grid-template .../wrfinput_d01_cares --outdir .../latent_inputs/emis \
      --start 2010-06-17_00:00:00 --n-times 13 --kemit 10 \
      --base-value 0 \
      --base-latent="-0.1027,-2.2868,6.1429,1.4282,-1.0323,0.3196,-1.6883,-1.4831,0.0577" \
      --region 50 80 40 70 1 3 --region-value 1e4 --region-profile gaussian \
      --region-latent="0.0667,0.1501,-0.6538,1.8972,-0.3033,1.8006,0.8581,0.9799,-0.0596"

  (Pass the latent vectors with =VALUE so a leading "-" isn't read as a flag.)
"""

import argparse
import datetime
import os
import sys

import numpy as np

# Make the shared profile module importable however this script is launched
# (directly, via a wrapper, or imported by the notebook).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_profiles as rp

try:
    from netCDF4 import Dataset
except ImportError:
    sys.exit("netCDF4 not found -- run under a conda env that has it, "
             "e.g.  conda activate wrf")


WRF_DATE_FMT = "%Y-%m-%d_%H:%M:%S"


def emis_var_names(n_latent):
    """E_LTNT_VAR1..N, E_NTOT -- the emis_ant members of the epartml package."""
    return ["E_LTNT_VAR%i" % (i + 1) for i in range(n_latent)] + ["E_NTOT"]


# ---------------------------------------------------------------------------
# Time profiles (emission analogue of region_profiles.PROFILES)
# ---------------------------------------------------------------------------
def _time_constant(hour, amplitude=None, peak_hour=None):
    """No diurnal variation."""
    return 1.0


def _time_diurnal(hour, amplitude=0.75, peak_hour=15.0):
    """Cosine diurnal cycle, 1 + A*cos(2*pi*(h - peak)/24), floored at 0.

    Mean over the day is 1, so ``--base-value`` / ``--region-value`` stay the
    DAILY MEAN flux; the peak is (1 + A) times that at ``peak_hour``. A = 0.75
    gives a 1.75 / 0.25 peak-to-trough swing, roughly traffic-like.
    """
    return max(0.0, 1.0 + amplitude * np.cos(2.0 * np.pi * (hour - peak_hour) / 24.0))


TIME_PROFILES = {
    "constant": _time_constant,
    "diurnal": _time_diurnal,
}


def time_factor(name, hour, amplitude=None, peak_hour=None):
    """Scalar multiplier applied to E_NTOT at local ``hour`` (0-24)."""
    try:
        fn = TIME_PROFILES[name]
    except KeyError:
        raise ValueError("unknown time profile %r (have: %s)"
                         % (name, ", ".join(sorted(TIME_PROFILES))))
    kwargs = {}
    if amplitude is not None:
        kwargs["amplitude"] = amplitude
    if peak_hour is not None:
        kwargs["peak_hour"] = peak_hour
    return float(fn(hour, **kwargs))


# ---------------------------------------------------------------------------
# Builder (pure numpy; also imported by the notebook)
# ---------------------------------------------------------------------------
def build_latent_emis(nx, ny, kemit, *, n_latent, base_latent, base_value,
                      region=None, region_latent=None, region_value=0.0,
                      profile="constant", sigma=None, vscale=None, scale=1.0):
    """Build one emission time record.

    Returns a dict {var name -> (kemit, ny, nx) f4 array} holding E_NTOT and
    E_LTNT_VAR1..n_latent, in the (bottom_top, south_north, west_east) order the
    wrfchemi file uses.

    ``E_NTOT`` is ``base_value`` in every cell and level; inside ``region`` it is
    overwritten by the chosen ``profile`` (see region_profiles.py). Each latent
    variable is written INTENSIVELY as the raw vector -- ``base_latent`` outside
    the box, ``region_latent`` inside -- so it carries no flux information at
    all. ``region`` is 1-based inclusive (i1, i2, j1, j2, k1, k2) with k in
    1..kemit, or None.

    ``scale`` multiplies E_NTOT only (see TIME_PROFILES): it changes how MANY
    particles are emitted, never WHICH population, and under intensive storage
    that is true by construction rather than by keeping 10 fields in step.

    To merge emission sectors, accumulate the extensive sums yourself and divide
    once at the end -- with ``rate`` and ``latent`` per sector s:
        ntot = sum_s rate_s
        ltnt = np.where(ntot > 0, sum_s rate_s * latent_s / ntot, fallback)
    The guard matters: summed sector fluxes can land on a nonzero float where
    every sector is off.
    """
    ntot = np.full((kemit, ny, nx), base_value, dtype="f8")

    sl = None
    if region is not None:
        i1, i2, j1, j2, k1, k2 = region
        sl = (slice(k1 - 1, k2), slice(j1 - 1, j2), slice(i1 - 1, i2))
        ni, nj, nk = i2 - i1 + 1, j2 - j1 + 1, k2 - k1 + 1
        ntot[sl] = rp.region_profile(
            profile, ni, nj, nk, base_value, region_value, sigma, vscale,
            order="kji")

    ntot *= scale

    fields = {"E_NTOT": ntot.astype("f4")}
    for idx in range(n_latent):
        # Intensive: the raw latent vector, independent of the flux (see the
        # module docstring). WRF multiplies by dN when it applies the emission.
        arr = np.full((kemit, ny, nx), base_latent[idx], dtype="f8")
        if sl is not None:
            arr[sl] = region_latent[idx]
        fields["E_LTNT_VAR%i" % (idx + 1)] = arr.astype("f4")
    return fields


# ---------------------------------------------------------------------------
# Template / file I/O
# ---------------------------------------------------------------------------
def read_grid_template(path):
    """Pull nx, ny, XLAT, XLONG and the WRF global attributes off a template.

    Accepts a wrfinput (XLAT/XLONG are (Time, south_north, west_east)) or an
    existing wrfchemi (already (south_north, west_east)).
    """
    with Dataset(path, "r") as nc:
        nx = len(nc.dimensions["west_east"])
        ny = len(nc.dimensions["south_north"])
        xlat = np.asarray(nc.variables["XLAT"][:], dtype="f4")
        xlong = np.asarray(nc.variables["XLONG"][:], dtype="f4")
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
    if xlat.ndim == 3:      # drop the leading Time axis of a wrfinput
        xlat, xlong = xlat[0], xlong[0]
    return nx, ny, xlat, xlong, attrs


def write_emis_file(path, fields, time_str, xlat, xlong, global_attrs=None):
    """Write one wrfchemi file (a single Time frame) in WRF's netCDF layout."""
    ny, nx = xlat.shape
    kemit = next(iter(fields.values())).shape[0]

    with Dataset(path, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("south_north", ny)
        nc.createDimension("west_east", nx)
        nc.createDimension("emissions_zdim_stag", kemit)

        v = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        v[0] = np.array(list(time_str.ljust(19)[:19]), dtype="S1")

        for name, data, desc, units in (
                ("XLONG", xlong, "LONGITUDE, WEST IS NEGATIVE", "degree east"),
                ("XLAT", xlat, "LATITUDE, SOUTH IS NEGATIVE", "degree north")):
            v = nc.createVariable(name, "f4", ("south_north", "west_east"))
            v.MemoryOrder = "XY "
            v.description = desc
            v.units = units
            v.stagger = ""
            v.FieldType = 104
            v[:] = data

        # E_LTNT_VAR1..N then E_NTOT, matching the epartml package order.
        for name in sorted(fields, key=lambda s: (s == "E_NTOT", s)):
            v = nc.createVariable(
                name, "f4",
                ("Time", "emissions_zdim_stag", "south_north", "west_east"))
            v.MemoryOrder = "XYZ"
            v.description = ("EMISSIONS TOTAL NUMBER" if name == "E_NTOT"
                             else "EMISSIONS LATENT VAR %s" % name[len("E_LTNT_VAR"):])
            # WRF's netCDF reader ignores this attribute; it documents the
            # conv_rho_aer = alt*dt/dz convention used by emiss_opt = 21. The
            # latent vector is stored intensively, hence dimensionless.
            v.units = "number m^-2 s^-1" if name == "E_NTOT" else "1"
            v.stagger = "Z"
            v.FieldType = 104
            v[0] = fields[name]

        for k, val in (global_attrs or {}).items():
            nc.setncattr(k, val)
        nc.Title = "Latent-variable anthropogenic emissions (emiss_opt = 21)"
        nc.Author = "make_latent_emis.py"
        nc.History = "Created on %s" % datetime.datetime.now().strftime(
            "%Y-%m-%d at %H:%M")


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
        description="Build latent-space wrfchemi emission inputs.")
    p.add_argument("--grid-template", required=True, metavar="FILE",
                   help="wrfinput (or wrfchemi) giving the grid, XLAT/XLONG "
                        "and the WRF global attributes to copy.")
    p.add_argument("--outdir", default="./latent_inputs/emis", metavar="DIR",
                   help="Directory to write the wrfchemi files into.")
    p.add_argument("--domain", type=int, default=1,
                   help="Domain number for the file name (default 1).")

    p.add_argument("--start", required=True, metavar="YYYY-MM-DD_HH:MM:SS",
                   help="First emission time; use the run's start_* namelist "
                        "time so the very first read finds a file.")
    p.add_argument("--n-times", type=int, default=25,
                   help="Number of emission records/files. Must cover the whole "
                        "run: run_length / interval + 1 (default 25 = 24 h).")
    p.add_argument("--interval-m", type=int, default=60,
                   help="Minutes between records; match auxinput5_interval_m "
                        "(default 60).")
    p.add_argument("--kemit", type=int, default=10,
                   help="Number of emission levels; match kemit (default 10).")

    p.add_argument("--n-latent", type=int, default=9,
                   help="Number of latent variables E_LTNT_VAR1..N (default 9).")
    p.add_argument("--base-value", type=float, default=0.0,
                   help="Base E_NTOT number flux (number/m^2/s) in every cell "
                        "and level (default 0 = no background emissions).")
    p.add_argument("--base-latent", required=True, metavar='"v1,..,vN"',
                   help="Base emitted population: N comma/space separated "
                        "latent values.")

    p.add_argument("--region", type=int, nargs=6,
                   metavar=("I1", "I2", "J1", "J2", "K1", "K2"),
                   help="1-based inclusive box emitting the region population. "
                        "K runs over the kemit emission levels. Omit for none.")
    p.add_argument("--region-latent", metavar='"v1,..,vN"',
                   help="Region emitted population (required with --region).")
    p.add_argument("--region-value", type=float, default=0.0,
                   help="Region E_NTOT flux; peak value for gaussian (default 0).")
    p.add_argument("--region-profile", default="constant",
                   choices=sorted(rp.PROFILES),
                   help="Shape of E_NTOT inside the region (default constant).")
    p.add_argument("--region-sigma", type=float, default=None,
                   help="Horizontal Gaussian width in cells (default min(box)/6).")
    p.add_argument("--region-vscale", type=float, default=None,
                   help="Vertical exponential scale height in emission levels "
                        "for the gaussian's decrease with height (default "
                        "box_height/2).")

    p.add_argument("--time-profile", default="constant",
                   choices=sorted(TIME_PROFILES),
                   help="Diurnal scaling of the whole emission field "
                        "(default constant).")
    p.add_argument("--diurnal-amplitude", type=float, default=None,
                   help="Amplitude A of the diurnal cosine (default 0.75); the "
                        "daily mean stays at --base-value / --region-value.")
    p.add_argument("--diurnal-peak", type=float, default=None, metavar="HOUR",
                   help="Local hour of the diurnal peak (default 15).")
    p.add_argument("--utc-offset", type=float, default=0.0, metavar="HOURS",
                   help="Local time minus UTC, used only by --time-profile "
                        "diurnal (CARES / California: -7). Default 0.")
    args = p.parse_args()

    base_latent = _parse_vector(args.base_latent, args.n_latent, "--base-latent")

    region = region_latent = None
    if args.region is not None:
        if args.region_latent is None:
            p.error("--region requires --region-latent")
        region = args.region
        region_latent = _parse_vector(args.region_latent, args.n_latent,
                                      "--region-latent")

    if args.n_times < 1:
        p.error("--n-times must be >= 1")
    if args.kemit < 1:
        p.error("--kemit must be >= 1")

    try:
        start = datetime.datetime.strptime(args.start, WRF_DATE_FMT)
    except ValueError:
        p.error("--start must look like YYYY-MM-DD_HH:MM:SS, got %r" % args.start)
    step = datetime.timedelta(minutes=args.interval_m)

    nx, ny, xlat, xlong, attrs = read_grid_template(args.grid_template)

    if region is not None:
        i1, i2, j1, j2, k1, k2 = region
        for name, lo, hi, n in (("i", i1, i2, nx), ("j", j1, j2, ny),
                                ("k", k1, k2, args.kemit)):
            if not (1 <= lo <= hi <= n):
                p.error("region %s range %d..%d out of bounds 1..%d"
                        % (name, lo, hi, n))

    os.makedirs(args.outdir, exist_ok=True)

    print("grid = %d x %d x %d  (nx x ny x kemit), from %s"
          % (nx, ny, args.kemit, args.grid_template))
    print("E_NTOT: base %.3g, region %.3g (%s), time profile %s"
          % (args.base_value, args.region_value, args.region_profile,
             args.time_profile))
    if region is not None:
        print("region box (1-based incl) i %d..%d  j %d..%d  k %d..%d"
              % (i1, i2, j1, j2, k1, k2))

    for i_time in range(args.n_times):
        when = start + i_time * step
        time_str = when.strftime(WRF_DATE_FMT)
        local_hour = (when.hour + when.minute / 60.0 + args.utc_offset) % 24.0
        scale = time_factor(args.time_profile, local_hour,
                            args.diurnal_amplitude, args.diurnal_peak)

        fields = build_latent_emis(
            nx, ny, args.kemit, n_latent=args.n_latent,
            base_latent=base_latent, base_value=args.base_value,
            region=region, region_latent=region_latent,
            region_value=args.region_value, profile=args.region_profile,
            sigma=args.region_sigma, vscale=args.region_vscale, scale=scale)

        out = os.path.join(args.outdir,
                           "wrfchemi_d%02d_%s" % (args.domain, time_str))
        write_emis_file(out, fields, time_str, xlat, xlong, attrs)
        print("  wrote %s  (x%.3f, E_NTOT max %.3g)"
              % (os.path.basename(out), scale, float(fields["E_NTOT"].max())))

    print("wrote %d file(s) to %s" % (args.n_times, args.outdir))
    print("  covering %s .. %s every %d min"
          % (start.strftime(WRF_DATE_FMT),
             (start + (args.n_times - 1) * step).strftime(WRF_DATE_FMT),
             args.interval_m))
    # WRF's PartMC-ML emission options: epartml (21) carries 9 latents,
    # epartml49 (22) carries 49; chem_opt must be 900 / 901 to match.
    emiss_opt = {9: "21 (chem_opt=900)", 49: "22 (chem_opt=901)"}.get(
        args.n_latent, "?? (WRF has no %d-latent option)" % args.n_latent)
    print("  namelist must have emiss_opt=%s, io_style_emissions=2, "
          "io_form_auxinput5=2, kemit=%d, auxinput5_interval_m=%d"
          % (emiss_opt, args.kemit, args.interval_m))


if __name__ == "__main__":
    sys.exit(main())
