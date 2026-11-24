#!/usr/bin/env python
"""Why do the paper's PartMC diagnostics differ from the training data's?

The models are trained on diagnostics computed by code_partnn's AeroMeasures
(the dataset build). The paper computes PartMC's side with other code:

  populations product (Fig. latent_populations, the attribution's sources):
      vendor/vae_partmc_utils.py -- CCN from PyPartMC's crit_rel_humids with
      its own AeroData subset at 290 K; optics and frozen fraction from
      vendor/extract_diagnostics_part.py (per particle)
  WRF-PartMC reference (partmc_diag_domain / _points): files made on the authors' cluster
      by an extractor of the vendor/extract_diagnostics.py family

This script computes the diagnostics of the same particles (the three
1000-particle files, the WRF runs' sources) every way, one setting changed at
a time, so each difference can be attributed:

  T     training code (AeroMeasures, the dataset's constants): wrf3_jeff.npz X,
        made by training/resample_wrf3.py --files (checked against the stored
        dataset values to 1e-15)
  P     the populations product's pmc_* (the paper code, as it was run)
  Vp    vendor extract_diagnostics, per particle ("prt"), its defaults
        (T = 287 K, kappas and refractive indices as the training data)
  Vh    the same on a 20-bin histogram (0.2 decade from 1 nm, "hst"), as
        extract_diagnostics.py's main() computes it
  plus CCN variants: kappa-Koehler (vendor calc_crh) at 290 K and with the
  paper's AeroData kappas (Cl, Na 1.28, OIN 0.1), to reproduce P.

And, for the reference: point B is pure background, so its PartMC diagnostics
(partmc_diag_points, every output time) should match the background
population's under whichever code made them.

Usage (from scripts/, an env with torch and netCDF4):
    python checks/pmc_diagnostics_compare.py [--out DIR]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / 'vendor'))
import extract_diagnostics as ED  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, default=REPO / 'notes' / 'pmc_diag_check')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)

ML_SPECIES = ['SO4', 'NO3', 'Cl', 'NH4', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'Na', 'OIN', 'OC', 'BC',
              'MOC', 'H2O']
SCEN = (561, 635, 812)
NAMES = ('Background', 'Urban', 'Emissions')
EPS = np.exp(np.linspace(np.log(1e-3), np.log(1e-1), 100))
TEMP = np.linspace(-40, 0, 100)
WL_T = np.array([300, 400, 500, 600, 700, 800, 900, 1000]) * 1e-9        # training / decoder grid
WL_V = np.array(ED.WAVELENGTHS_M)                                         # vendor grid (300, 400, 550, 600, 1000)
#: The paper's AeroData subset (vendor/vae_partmc_utils.make_aero_data_subset): kappa
PAPER_KAPPA = dict(SO4=.65, NO3=.65, Cl=1.28, NH4=.65, ARO1=.1, ARO2=.1, ALK1=.1, OLE1=.1, API1=.1, Na=1.28,
                   OIN=.1, OC=.001, BC=0., MOC=.001, H2O=0.)


def read(scen):
    f = REPO / 'wrf-aeromeld-inputs' / 'particles' / ('scenario_%d_filtered_0001_00000013.nc' % scen)
    with Dataset(f) as d:
        og = d.variables['aero_species'].names.split(',')
        keep = [og.index(s) for s in ML_SPECIES]
        m = np.asarray(d['aero_particle_mass'][:], np.float64)
        m = m[keep, :].T if m.shape[0] == len(og) else m[:, keep]
        n = np.asarray(d['aero_num_conc'][:], np.float64)
        kap = np.asarray(d['aero_kappa'][keep], np.float64)
        rho = np.asarray(d['aero_density'][keep], np.float64)
    return m, n, kap, rho


def t64(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def ccn_prt(m, n, tK, kappa, rho):
    """Vendor kappa-Koehler per particle (calc_crh + calc_ccncdf): CCN/CN on EPS."""
    r = ED.get_apprxccncdf(t64(m[None, None]), t64(n[None, None, :, None]), 1e-3, 1e-1, 100, tK, t64(rho),
                           t64(kappa), ML_SPECIES, 0.0, 100, 1e-12, 1, 1, len(ML_SPECIES), len(n),
                           torch.device('cpu'), torch.float64, input_type='prt', strict=False)
    cdf = r['ccn_cdf'].reshape(-1).numpy()
    return cdf / n.sum() if cdf.max() > 1.5 else cdf


def hist20(m, n, rho):
    """20 bins of 0.2 decade from 1e-9 m by WET diameter (all 15 species), as the
    WRF-PartMC binned output: (species, bins) mass and (bins,) number, per m^3."""
    D = (6 / np.pi * (m / rho).sum(1)) ** (1 / 3)
    b = np.floor((np.log10(D) + 9) / 0.2).astype(int)
    assert ((b >= 0) & (b < 20)).all()
    M = np.zeros((len(ML_SPECIES), 20))
    for j in range(len(ML_SPECIES)):
        M[j] = np.bincount(b, weights=m[:, j] * n, minlength=20)
    N = np.bincount(b, weights=n, minlength=20)
    return M, N


def vendor(m, n, kind, rho):
    """extract_diagnostics with its defaults, per particle or on the 20-bin histogram."""
    if kind == 'prt':
        import extract_diagnostics_part as EP
        mt, nt = t64(m[None]), t64(n[None, :, None])
        ccn = EP.compute_ccn_spectrum_part(mt, nt, ML_SPECIES)['ccn_cdf'].reshape(-1).numpy()
        opt = EP.compute_optical_spectrum_part(mt, nt, ML_SPECIES, wavelengths=list(WL_V))
        ff = EP.compute_frozen_fraction_spectrum_part(mt, nt, ML_SPECIES)['frznfrac_tmp'].reshape(-1).numpy()
    else:
        M, N = hist20(m, n, rho)
        Mt, Nt = t64(M[None]), t64(N[None, None])
        ccn = ED.compute_ccn_spectrum(Mt, Nt, ML_SPECIES)['ccn_cdf'].reshape(-1).numpy()
        refr = torch.tensor([[ED.DEFAULT_CHEM_REFRACTIVE.get(c, 1.5 + 0j) for c in ML_SPECIES] for _ in WL_V],
                            dtype=torch.complex128)                  # (wave, species), as main() passes it
        opt = ED.compute_optical_spectrum(Mt, Nt, ML_SPECIES, wavelengths=list(WL_V), refr_wvchm=refr)
        ff = ED.compute_frozen_fraction_spectrum(Mt, Nt, ML_SPECIES)['frznfrac_tmp'].reshape(-1).numpy()
    qs = opt['qs_pop'].reshape(-1).numpy()
    qa = opt['qa_pop'].reshape(-1).numpy()
    if ccn.max() > 1.5:
        ccn = ccn / n.sum()
    return dict(ccn=ccn, scat=qs, abs=qa, ff=ff)


def at(x_grid, y, x_new):
    return np.interp(np.log(x_new), np.log(x_grid), y)


# --- the training code's values (resample_wrf3.py --files, on Perlmutter) ---------------
J = np.load(REPO / 'data' / 'affx' / 'wrf3_jeff.npz')
blk = dict(C=(240, 100), S=(340, 8), A=(348, 8), F=(356, 100))
T = {}
for k, s in enumerate(SCEN):
    x = J['X'][k]
    nt = float(J['ntot'][k])
    T[s] = dict(ccn=x[240:340], scat=x[340:348] * nt, abs=x[348:356] * nt, ff=x[356:456], ntot=nt)

import xarray as xr  # noqa: E402
P = xr.open_dataset(REPO / 'data' / 'processed' / 'populations_05_nonlinvlong.0.0.nc')
assert [int(v) for v in P['scenario'].values] == list(SCEN)

rows, out = [], {}
for k, s in enumerate(SCEN):
    m, n, kap_file, rho = read(s)
    nt = T[s]['ntot']
    scale = nt / n.sum()                               # file weights -> full state, as the training values
    np.testing.assert_allclose(J['m'][k], m, rtol=1e-12)
    kap_tr = np.array([ED.DEFAULT_CHEM_KAPPA[c] for c in ML_SPECIES])
    kap_pa = np.array([PAPER_KAPPA[c] for c in ML_SPECIES])
    r = {}
    # CCN: the training code, then the vendor Koehler under each setting
    r['ccn T (training, 287 K, file kappas)'] = T[s]['ccn']
    r['ccn P (paper populations product)'] = P['pmc_ccn_cdf'].values[k]
    r['ccn Vp 287 K, training kappas'] = ccn_prt(m, n, 287.0, kap_tr, rho)
    r['ccn Vp 290 K, training kappas'] = ccn_prt(m, n, 290.0, kap_tr, rho)
    r['ccn Vp 287 K, paper kappas'] = ccn_prt(m, n, 287.0, kap_pa, rho)
    r['ccn Vp 290 K, paper kappas'] = ccn_prt(m, n, 290.0, kap_pa, rho)
    assert np.allclose(kap_file, kap_tr), (kap_file, kap_tr)
    vp, vh = vendor(m, n, 'prt', rho), vendor(m, n, 'hst', rho)
    r['ccn Vh 287 K, histogram'] = vh['ccn']
    # optics on the vendor grid (training interpolated in log wavelength; 300/400/600/1000 are on both)
    for q, key in (('scat', 'scattering'), ('abs', 'absorption')):
        r[q + ' T'] = at(WL_T, T[s][q], WL_V)
        r[q + ' P'] = at(WL_T, P['pmc_' + key].values[k], WL_V) * scale
        r[q + ' Vp'] = vp[q] * scale
        r[q + ' Vh'] = vh[q] * scale
    r['ff T'], r['ff P'], r['ff Vp'], r['ff Vh'] = T[s]['ff'], P['pmc_frozen_fraction'].values[k], vp['ff'], vh['ff']
    out[s] = {k2: np.asarray(v, float).tolist() for k2, v in r.items()}

    def cmp(a, b, kind):
        a, b = np.asarray(r[a], float), np.asarray(r[b], float)
        if kind == 'ccn':
            sel = (EPS >= 1e-3) & (EPS <= 1e-2)        # s_env 0.1-1 %
            return 'max |diff| %.4f' % np.abs(a[sel] - b[sel]).max()
        if kind == 'ff':
            sel = TEMP <= -10
            ok = (a[sel] > 0) & (b[sel] > 0)
            return 'ratio %.3f-%.3f' % ((a[sel][ok] / b[sel][ok]).min(), (a[sel][ok] / b[sel][ok]).max())
        return 'ratio ' + ' '.join('%.3f' % v for v in a / b)
    print('\n== %d %s' % (s, NAMES[k]))
    for a, b in [('ccn P (paper populations product)', 'ccn T (training, 287 K, file kappas)'),
                 ('ccn Vp 287 K, training kappas', 'ccn T (training, 287 K, file kappas)'),
                 ('ccn Vp 290 K, paper kappas', 'ccn P (paper populations product)'),
                 ('ccn Vp 290 K, training kappas', 'ccn T (training, 287 K, file kappas)'),
                 ('ccn Vp 287 K, paper kappas', 'ccn T (training, 287 K, file kappas)'),
                 ('ccn Vh 287 K, histogram', 'ccn T (training, 287 K, file kappas)')]:
        print('  %-36s vs %-36s %s' % (a, b, cmp(a, b, 'ccn')))
    for q in ('scat', 'abs'):
        for a, b in [(q + ' P', q + ' T'), (q + ' Vp', q + ' T'), (q + ' Vp', q + ' P'), (q + ' Vh', q + ' T')]:
            print('  %-36s vs %-36s %s  (300 400 550 600 1000 nm)' % (a, b, cmp(a, b, q)))
    for a, b in [('ff P', 'ff T'), ('ff Vp', 'ff T'), ('ff Vh', 'ff T')]:
        print('  %-36s vs %-36s %s  (-40 to -10 C)' % (a, b, cmp(a, b, 'ff')))

(args.out / 'populations_compare.json').write_text(json.dumps(out))

# --- the WRF-PartMC reference at point B (pure background) --------------------------------
D = xr.open_dataset(REPO / 'data' / 'processed' / 'partmc_diag_points.nc')
F = xr.open_dataset(REPO / 'data' / 'processed' / 'partmc_fields.nc')
iB = list(D['point'].values).index('B')
r = out[561]
print('\n== WRF-PartMC point B (background) at each diagnostics time vs the background population')
for it, t in enumerate(D['time'].values):
    ccn = D['ccn_cdf'].values[iB, it]
    if not np.isfinite(ccn).all():
        continue
    sel = (EPS >= 1e-3) & (EPS <= 1e-2)
    d = {k: np.abs(ccn[sel] - np.asarray(r[k])[sel]).max() for k in
         ('ccn T (training, 287 K, file kappas)', 'ccn Vh 287 K, histogram', 'ccn P (paper populations product)')}
    ff = D['frozen_fraction'].values[iB, it]
    selT = TEMP <= -10
    fr = {k: np.median(ff[selT] / np.asarray(r[k])[selT]) for k in ('ff T', 'ff Vh', 'ff Vp')}
    # optics: intensive per particle = per kg / ntot per kg
    ntB = D['ntot'].values[iB, it]
    sc = D['scattering'].values[iB, it] / ntB
    ab = D['absorption'].values[iB, it] / ntB
    so = {k: (sc / (np.asarray(r[k]) / T[561]['ntot'])) for k in ('scat T', 'scat Vh', 'scat Vp')}
    ao = {k: (ab / (np.asarray(r[k]) / T[561]['ntot'])) for k in ('abs T', 'abs Vh', 'abs Vp')}
    print('  %s CCN max|diff| T %.4f Vh %.4f P %.4f | FF median ratio T %.3f Vh %.3f Vp %.3f | '
          'scat/particle ratio T %s Vh %s | abs ratio T %s Vh %s'
          % (t, d['ccn T (training, 287 K, file kappas)'], d['ccn Vh 287 K, histogram'],
             d['ccn P (paper populations product)'], fr['ff T'], fr['ff Vh'], fr['ff Vp'],
             np.round(so['scat T'], 3), np.round(so['scat Vh'], 3), np.round(ao['abs T'], 3),
             np.round(ao['abs Vh'], 3)))
