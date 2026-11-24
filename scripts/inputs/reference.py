#!/usr/bin/env python
"""Reference checksums for WRF-AeroMELD inputs, so a build can be checked without the reference runs.

Usage (from scripts/, partnn env):

    python inputs/reference.py check RUN_DIR --n-latent 9 [--ignore E_NTOT]
    python inputs/reference.py write            # (re)make the JSON from the reference runs

The JSON (config.INPUT_FINGERPRINTS, committed) holds two things:

  fields   For each latent size (9, 49) and each input file (wrfinput_d01,
           wrfbdy_d01, the 13 wrfchemi_d01_*), a SHA-256 of every NON-latent
           variable's values, taken from the reference runs
           latent_advection_cares_v3match_05_nonlinvlong.{0.0,1.0}_5e7.
           Non-latent means everything except ltnt_var*, ltnt_var*_B* and
           E_LTNT_VAR*: the meteorology, grid, times, ntot and E_NTOT. Those are
           the same for every model, so any build of the standard scenario at
           rate 5e7 must reproduce them bit for bit. Global attributes are not
           checksummed: they are copied from the templates.
  latents  The latents (z, without ntot) every model was encoded to by
           input_builders' dump_populations.py, per scenario. A fresh encoding
           of a listed model must match them to LATENT_TOL; that checks the
           encoder and its environment.

check exits 0 if every non-latent field matches (latent fields are only checked
for being present and finite).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import config  # noqa: E402

LATENT = re.compile(r'^(ltnt_var\d+(_B[A-Z]+)?|E_LTNT_VAR\d+)$', re.I)
#: Encoding the particle files reproduces dump_populations' latents to <= 9.2e-7.
LATENT_TOL = 1e-5
REFERENCE_RUNS = {9: '05_nonlinvlong.0.0_5e7', 49: '05_nonlinvlong.1.0_5e7'}
STORED_LATENTS = '/path/to/input_builders/input_data'


def input_files(run):
    names = ['wrfinput_d01', 'wrfbdy_d01']
    names += sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, 'wrfchemi_d01_*')))
    return names


def digest(var):
    """SHA-256 of a netCDF variable's values: dtype (little-endian), shape and bytes."""
    a = np.asarray(var[:])
    if a.dtype.kind in 'iuf':
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(('%s|%s|' % (a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def file_digests(path):
    """({non-latent var: sha256}, [latent var names]) for one input file."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        fields, latents = {}, []
        for name, var in ds.variables.items():
            if LATENT.match(name):
                latents.append(name)
            else:
                fields[name] = digest(var)
    return fields, latents


def load(path=None):
    with open(path or config.INPUT_FINGERPRINTS) as f:
        return json.load(f)


def check(run, n_latent, ignore=(), ref=None, verbose=True):
    """True if every non-latent field of RUN matches the reference checksums."""
    from netCDF4 import Dataset
    ref = ref or load()
    ignore = set(ignore)
    expect = ref['fields'][str(n_latent)]
    ok = True
    for name, want in expect.items():
        path = os.path.join(run, name)
        if not os.path.exists(path):
            print('    %-34s MISSING' % name)
            ok = False
            continue
        got, latents = file_digests(path)
        bad = []
        for v in sorted((set(want) | set(got)) - ignore):
            if v not in got:
                bad.append('%s missing' % v)
            elif v not in want:
                bad.append('%s not in the reference' % v)
            elif want[v] != got[v]:
                bad.append(v)
        n_ltnt = len({re.match(r'(?i)(?:E_)?ltnt_var(\d+)', v).group(1) for v in latents})
        if n_ltnt != n_latent:
            bad.append('%d latent variables, expected %d' % (n_ltnt, n_latent))
        with Dataset(path) as ds:
            bad += ['%s not finite everywhere' % v for v in latents
                    if not np.all(np.isfinite(ds[v][:]))]
        if bad:
            ok = False
            print('    %-34s DIFFERENT: %s' % (name, ', '.join(bad)))
        elif verbose and not name.startswith('wrfchemi_d01_'):
            print('    %-34s ok (%d fields match the reference)' % (name, len(set(want) - ignore)))
    extra = sorted(set(input_files(run)) - set(expect))
    for name in extra:
        print('    %-34s not in the reference (extra file)' % name)
    if verbose:
        print('    %d wrfchemi files %s' % (sum(n.startswith('wrfchemi_d01_') for n in expect),
                                            'checked' if ok else 'checked (see above)'))
    return ok


def check_latents(fpidx, latents, ref=None):
    """None if FPIDX has no stored latents, else (ok, max |diff|) against them."""
    ref = ref or load()
    stored = ref['latents'].get(fpidx)
    if stored is None:
        return None
    worst = 0.0
    for s, z in stored.items():
        got = np.asarray(latents[int(s)][:len(z)])
        worst = max(worst, float(np.abs(got - np.asarray(z)).max()))
    return worst <= LATENT_TOL, worst


def write(out, stored_latents=STORED_LATENTS):
    fields = {}
    for n, key in REFERENCE_RUNS.items():
        run = config.WRF_ML_TEST / ('%s_%s' % (config.WRF_ML_RUN_PREFIX, key))
        fields[str(n)] = {name: file_digests(run / name)[0] for name in input_files(run)}
        print('fields for %d latents from %s: %d files' % (n, run, len(fields[str(n)])))
    latents = {}
    base = Path(stored_latents)
    dirs = {'02_mlpprt.0.0': base}           # the top-level dumps are 02_mlpprt's
    dirs.update({d.name: d for d in sorted(base.iterdir())
                 if d.is_dir() and re.match(r'^\w+\.\d+\.\d+$', d.name)})
    for fpidx, d in dirs.items():
        per = {}
        for s in config.POP_SCENARIOS:
            f = d / ('latent_scenario_%d_0001_%08d.txt' % (s, config.PARTICLE_TIME))
            if f.exists():
                per[str(s)] = [float(v) for v in np.loadtxt(f)[:-1]]     # z only, ntot dropped
        if len(per) == len(config.POP_SCENARIOS):
            latents[fpidx] = per
    print('stored latents for %d models: %s' % (len(latents), ', '.join(latents)))
    doc = {
        'about': 'Reference checksums for WRF-AeroMELD inputs; see inputs/reference.py.',
        'created': time.strftime('%Y-%m-%d'),
        'reference_runs': {str(n): '%s_%s' % (config.WRF_ML_RUN_PREFIX, k)
                           for n, k in REFERENCE_RUNS.items()},
        'emission_rate': config.EMISSION_RATE,
        'fields': fields,
        'latent_tolerance': LATENT_TOL,
        'latents': latents,
    }
    with open(out, 'w') as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write('\n')
    print('wrote %s (%.0f KB)' % (out, os.path.getsize(out) / 1e3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('run')
    c.add_argument('--n-latent', type=int, required=True, choices=(9, 49))
    c.add_argument('--ignore', default='', help='comma-separated fields to skip (e.g. E_NTOT)')
    w = sub.add_parser('write')
    w.add_argument('--out', default=str(config.INPUT_FINGERPRINTS))
    w.add_argument('--stored-latents', default=STORED_LATENTS)
    args = ap.parse_args()
    if args.cmd == 'write':
        write(args.out, args.stored_latents)
        return 0
    ok = check(args.run, args.n_latent, {v for v in args.ignore.split(',') if v})
    print('RESULT: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
