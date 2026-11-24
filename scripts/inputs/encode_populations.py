#!/usr/bin/env python
"""Encode the three seed populations with one VAE model, for WRF-AeroMELD inputs.

Usage (from scripts/, partnn env):

    python inputs/encode_populations.py FPIDX --outdir DIR [--overwrite]

Writes DIR/latent_scenario_<s>_0001_<t>.txt for each s in config.POP_SCENARIOS:
the model's N latents z (N = 9 or 49), then the ntot latent, one value per line.
That is the format run_latent_ics.sh / run_latent_emis.sh read (they take the
first N values and drop ntot), and the one input_builders' dump_populations.py
writes, so the two are interchangeable.

Encoding is extract.populations.encode, the code behind the latent_populations
figure: every particle of config.PARTICLE_DIR's scenario_<s>_filtered files (the
1000 particles WRF-PartMC was initialised from), in file order. No random draw,
no training-set file, no full PartMC state. The latents match
dump_populations.py's to float32 rounding: <= 9.2e-7 for 05_nonlinvlong.0.0,
05_nonlinvlong.1.0, 11_linpvlong.5.0 and 06_nonlinregvlong.3.0, on GPU and on
CPU (checked 2026-11-10). dump_populations draws the same particles out of the
full PartMC state and encodes them in a different batch.

ntot. The particle files keep the sample's own weights, so they total ~1/159 of
the populations they came from. The ntot entry is therefore the model's own
transform of config.POP_N_FULL, the full populations' totals; that reproduces
dump_populations' value to 4e-7. WRF never reads it: the input builders set the
number concentrations from the scenario values (BASE_VALUE, REGION_VALUE, rate).

Models are looked up by fpidx in config.MODEL_RESDIR (environment variable
PAPERFIGS_MODEL_RESDIR). Uses the GPU if there is one; on CPU it takes ~1 min.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import config  # noqa: E402


def latent_path(outdir, scenario):
    return Path(outdir) / ('latent_scenario_%d_0001_%08d.txt' % (scenario, config.PARTICLE_TIME))


def encode(fpidx, resdir=None, seed_idx=0, verbose=True):
    """{scenario: [z_1..z_N, ntot latent]} (float64) for one model."""
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')    # others' archives, read-only
    import torch

    from extract.aeromeld import vae_utils
    from extract.populations import encode as encode_files, particle_file

    torch.set_num_threads(1)          # intra-op threads spin against the CPU quota
    vpu = vae_utils()
    resdir = Path(resdir or config.MODEL_RESDIR)
    files = [particle_file(s) for s in config.POP_SCENARIOS]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
    t0 = time.perf_counter()
    comp = vpu.load_vae_components(fpidx=fpidx, resdir=resdir, device_name=vpu.default_device(),
                                   dtype_name='float32', seed_idx=seed_idx, verbose=False)
    _, _, z, _ = encode_files(vpu, comp, files, fpidx)
    if z.shape[1] + 1 != comp['ltnt_dim']:
        raise ValueError('%s encodes %d latents but its ltnt_dim is %d (z + ntot)'
                         % (fpidx, z.shape[1], comp['ltnt_dim']))
    out = {}
    for i, s in enumerate(config.POP_SCENARIOS):
        ntot = float(np.asarray(vpu.ntot_to_latent(config.POP_N_FULL[s], comp)).ravel()[0])
        out[s] = np.concatenate([z[i].astype(np.float64), [ntot]])
    if verbose:
        print('encoded %s with %s (%d latents + ntot) on %s in %.1f s'
              % (', '.join(str(s) for s in config.POP_SCENARIOS), fpidx, z.shape[1],
                 comp['device'], time.perf_counter() - t0))
    return out


def write(latents, outdir, overwrite=False):
    """Write encode()'s result, one value per line (Fortran list-directed safe)."""
    outdir = Path(outdir)
    paths = {s: latent_path(outdir, s) for s in latents}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError('refusing to overwrite (pass --overwrite): ' + ', '.join(existing))
    outdir.mkdir(parents=True, exist_ok=True)
    for s, vec in latents.items():
        with open(paths[s], 'w') as f:
            f.writelines('%.17g\n' % v for v in vec)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fpidx', help='model, <name>.<file>.<part> (e.g. 05_nonlinvlong.0.0)')
    ap.add_argument('--outdir', required=True, help='directory for the latent files')
    ap.add_argument('--resdir', default=None,
                    help='model archive directory (default config.MODEL_RESDIR = %s)'
                         % config.MODEL_RESDIR)
    ap.add_argument('--seed-idx', type=int, default=0, help='model seed (default 0)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    paths = write(encode(args.fpidx, args.resdir, args.seed_idx), args.outdir, args.overwrite)
    for s, p in paths.items():
        print('  wrote %s' % p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
