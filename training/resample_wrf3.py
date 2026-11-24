#!/usr/bin/env python
"""
New 1000-particle samples of the three WRF source populations (scenarios 561,
635, 812 at PartMC output index 13), chosen to look like the files the WRF
runs were started from (the original files), and their
diagnostics computed the way the dataset's were.

The original files are different 1000-particle subsamples of the same PartMC
states as our dataset's. The paper's `populations` product (exported here as
jeff_pmc.npz) holds their per-species mass size distributions. For each
scenario this script reads the full PartMC state from the raw tarball, draws
--draws uniform 1000-particle subsamples (without replacement, n_prt scaled by
n_partog / n_part, as build_prtcls_h5.rsmpl_partdata), and keeps the one
whose species x bin mass distribution, normalized by its total, is closest
to the original's in L1 (wet diameter, the dataset's 15 bins of 0.25 decade from
10^-8.75 m, diameters from the 15 kept species' volumes).

Diagnostics with partnn's AeroMeasures and the dataset family's config
(05_bwchi with the PM10 overrides n_bins 15, diam_high 1e-5, as
clean_coarse.py), for the chosen samples and, as a check, for our dataset's
samples of the same states (compared with the stored values).

Output npz: m (3, 1000, 15) kg, n_prt (3, 1000), X (3, 456) per-particle
diagnostics in affx.py's layout, ntot (3,), the same for the dataset's samples
(m_ds, X_ds, ntot_ds), the L1 distances, and the seed.

--files DIR: no resampling; use the three files in DIR
(scenario_<N>_filtered_0001_00000013.nc, the original files, which the WRF runs started
from), their weights scaled up to the full state's number (our dataset's
n_tot of the same state), and compute their diagnostics the same way.

Usage (GPU node, code_partnn venv):
    python resample_wrf3.py OUT.npz [--draws 20000] [--seed 1] [--files DIR]
"""
import argparse
import io
import os
import sys
import tarfile
import time

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--draws', type=int, default=20000)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--jeff', default='/pscratch/sd/m/mwest/aeromeld/affx/jeff_pmc.npz')
ap.add_argument('--raw', default='/pscratch/sd/m/mwest/aeromeld/raw/01_bwchisamp')
ap.add_argument('--data', default='/pscratch/sd/m/mwest/aeromeld/data/04_bwchisamp_pm10.h5')
ap.add_argument('--code-dir', default='/pscratch/sd/m/mwest/aeromeld/code_partnn')
ap.add_argument('--files', default=None, help='use these 1000-particle files instead of resampling')
args = ap.parse_args()
out_path = os.path.abspath(args.out)
files_dir = os.path.abspath(args.files) if args.files else None
data_path = os.path.abspath(args.data)

os.chdir(f'{args.code_dir}/notebooks')            # partnn.io_cfg and the yml paths need this
sys.path.insert(0, args.code_dir)
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from ruamel import yaml as ruyaml  # noqa: E402
from partnn.aerometrics import AeroMeasures  # noqa: E402
from partnn.io_utils import eval_formula, parse_refs, get_subdict  # noqa: E402

SCEN, POPS, T_FILE = [561, 635, 812], [14037, 15887, 20312], 13
N_PART = 1000

# --- the dataset family's config (build_prtcls_h5.py, n21 cells 8 and 10) -------------
with open('./21_data/01_datasets.yml') as fp:
    info_fams1 = dict(ruyaml.safe_load(fp))
info_fams2 = parse_refs(info_fams1, trnsfrmtn='hie', pathsep=' -> ', cfg_path='21_data/01_datasets.yml')
info_fams = {k: eval_formula(v, {'all': 'all'}, catch=True) for k, v in info_fams2.items()}
info_fam = get_subdict(info_fams, '05_bwchi')
species = info_fam['chem_species'].split(',')
assert info_fam['n_part'] == N_PART

# --- the full PartMC states ------------------------------------------------------------
def read_state(scen):
    name = f'scenario_{scen}/scenario_{scen}_0001_{T_FILE:08d}.nc'
    with tarfile.open(f'{args.raw}/scenario_{scen}.tar.gz', 'r:gz') as tf:
        buf = tf.extractfile(name).read()
    nc = Dataset('mem', memory=buf)
    og = nc.variables['aero_species'].names.split(',')
    keep = [og.index(s) for s in species]
    m = np.asarray(nc.variables['aero_particle_mass'][keep, :], np.float64).T      # (n_partog, 15)
    n = np.asarray(nc.variables['aero_num_conc'][:], np.float64)
    rho = np.asarray(nc.variables['aero_density'][keep], np.float64)
    nc.close()
    return m, n, rho

jeff = np.load(args.jeff)
assert list(jeff['species']) == species
EDGES = -8.75 + 0.25 * np.arange(16)

def mass_hist(m, n, rho):
    """(15 species, 15 bins) mass concentration by wet diameter bin."""
    D = (6 / np.pi * (m / rho).sum(1)) ** (1 / 3)
    b = np.floor((np.log10(D) - EDGES[0]) / 0.25).astype(int)
    out = np.zeros((15, 15))
    ok = (b >= 0) & (b < 15)
    for j in range(15):
        out[j] = np.bincount(b[ok], weights=(m[ok, j] * n[ok]), minlength=15)
    return out, D

rng = np.random.default_rng(args.seed)
fds = h5py.File(data_path, 'r')
res = dict(seed=args.seed, draws=args.draws)
M, NP, best_d = [], [], []
for k, scen in enumerate(SCEN):
    if files_dir:
        nc = Dataset(f'{files_dir}/scenario_{scen}_filtered_0001_{T_FILE:08d}.nc')
        og = nc.variables['aero_species'].names.split(',')
        keep = [og.index(s) for s in species]
        m = np.asarray(nc.variables['aero_particle_mass'][:], np.float64)
        m = (m[keep, :].T if m.shape[0] == len(og) else m[:, keep])                 # (1000, 15)
        n = np.asarray(nc.variables['aero_num_conc'][:], np.float64)
        rho = np.asarray(nc.variables['aero_density'][keep], np.float64)
        nc.close()
        assert m.shape == (N_PART, 15)
        nt = float(fds['n_tot'][POPS[k]].reshape(-1)[0])
        print(f'{scen}: file weights sum {n.sum():.4g} m^-3, scaled x{nt / n.sum():.2f} to the full state ({nt:.4g}); '
              f'largest D {((6 / np.pi * (m / rho).sum(1)) ** (1 / 3)).max() * 1e6:.2f} um', flush=True)
        M.append(m); NP.append(n * nt / n.sum()); best_d.append(np.nan)
        continue
    t0 = time.time()
    m, n, rho = read_state(scen)
    assert np.unique(n).size == 1, 'the uniform subsample below assumes equal weights'
    target = jeff['dM'][k] / jeff['dM'][k].sum()
    _, D = mass_hist(m, n, rho)
    b = np.clip(np.floor((np.log10(D) - EDGES[0]) / 0.25).astype(int), 0, 14)
    mw = m * n[:, None]
    # all draws' histograms at once: (draws, 15 species, 15 bins)
    dist = np.empty(args.draws)
    idx_all = np.empty((args.draws, N_PART), np.int64)
    eye = np.eye(15)
    for i in range(args.draws):
        idx = rng.choice(len(n), N_PART, replace=False)
        h = (eye[b[idx]].T @ mw[idx]).T                              # (species, bins)
        dist[i] = np.abs(h / h.sum() - target).sum()
        idx_all[i] = idx
    ib = int(np.argmin(dist))
    idx = idx_all[ib]
    hds, _ = mass_hist(fds['m_prtchm'][POPS[k]], fds['n_prt'][POPS[k]][:, 0], rho)
    d_ds = np.abs(hds / hds.sum() - target).sum()
    print(f'{scen}: {len(n)} particles; L1 to the original mass distribution: best {dist[ib]:.3f} (draw {ib}), '
          f'median {np.median(dist):.3f}, 5% {np.percentile(dist, 5):.3f}; our dataset\'s sample {d_ds:.3f}; '
          f'largest D in the best sample {D[idx].max() * 1e6:.2f} um ({time.time() - t0:.0f} s)', flush=True)
    M.append(m[idx]); NP.append(n[idx] * len(n) / N_PART); best_d.append(dist[ib])
    res[f'dist_{scen}'] = dist
res['m'], res['n_prt'], res['best_dist'] = np.stack(M), np.stack(NP), np.array(best_d)

# --- diagnostics (build_prtcls_h5.build_diag) -------------------------------------------
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hp1 = get_subdict(info_fam, 'diag')
hp1.pop('n_mb')
hp1['hparams/aerohst/n_bins'], hp1['hparams/aerohst/diam_high'] = 15, 1e-5       # PM10 (clean_coarse.py)
hp2 = {k: eval_formula(v, dict(n_chem=15, n_part=N_PART, chem_species=species), catch=True) for k, v in hp1.items()}
calc = AeroMeasures(hp2, dev, torch.float64, torch.complex128)

def diagnose(m, n):
    with torch.no_grad():
        d, _ = calc.forward({'m_prtchm': torch.from_numpy(m[None]).to(dev, torch.float64),
                             'n_prt': torch.from_numpy(n[None, ..., None]).to(dev, torch.float64)})
    return {k: v.squeeze(0).cpu().numpy() for k, v in d.items()}

BL = [('m_chmprthst', True), ('n_prthst', True), ('ccn_cdf', False), ('qs_pop', True), ('qa_pop', True),
      ('frznfrac_tmp', False)]

def to_X(d, n):
    nt = d['n_tot'].reshape(-1) if 'n_tot' in d else n.sum(-1)
    X = np.concatenate([d[k].reshape(len(nt), -1) / (nt[:, None] if ext else 1) for k, ext in BL], 1)
    return X, nt

with h5py.File(data_path, 'r') as f:
    m_ds, n_ds = f['m_prtchm'][POPS], f['n_prt'][POPS][..., 0]
    stored = {k: f[k][POPS] for k, _ in BL + [('n_tot', 0)]}
d_new, d_ds = diagnose(res['m'], res['n_prt']), diagnose(m_ds, n_ds)
print('diagnostic keys:', sorted(d_new))
for k in [k for k, _ in BL] + (['n_tot'] if 'n_tot' in d_ds else []):
    e = np.abs(d_ds[k] - stored[k]).max() / np.abs(stored[k]).max()
    print(f'  check vs the stored dataset values, {k}: max rel. difference {e:.1e}')
res['X'], res['ntot'] = to_X(d_new, res['n_prt'])
res['X_ds'], res['ntot_ds'] = to_X(d_ds, n_ds)
res['m_ds'] = m_ds
np.savez(out_path, **res)
print('wrote', out_path)
