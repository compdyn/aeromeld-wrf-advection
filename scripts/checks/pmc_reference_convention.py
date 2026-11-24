#!/usr/bin/env python
"""Which convention made the WRF-PartMC reference diagnostics (partmc_diag_domain)?

WRF-PartMC has no particle processes in this case, so every cell holds particles
of the three source populations: its diagnostics should be the number-weighted
mixture of the populations' own, sum_s n_s q_s (per-particle q_s) for optics and
sum_s f_s q_s for CCN/CN and frozen fraction, up to the sampling of particles
into cells. This compares the reference at 04 UTC over the mixed cells with that
mixture, for the population diagnostics computed each way (pmc_diagnostics_compare.py):

  T    training code (code_partnn): 287 K, shell-diameter cross sections
  P    the paper's populations product: 290 K, full-diameter cross sections
  Vp   vendor per particle: 287 K, full-diameter cross sections
  Vh   vendor on 20-bin histograms (extract_diagnostics.py main())

Usage (from scripts/): python checks/pmc_reference_convention.py JSON
   (JSON: populations_compare.json from pmc_diagnostics_compare.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from paperfigs import analysis, config  # noqa: E402

pc = json.loads(Path(sys.argv[1]).read_text())
SCEN = ('561', '635', '812')
D = xr.open_dataset(REPO / 'data' / 'processed' / 'partmc_diag_domain.nc')
F = xr.open_dataset(REPO / 'data' / 'processed' / 'partmc_fields.nc')
ts = D.attrs['timestamp']
it = list(F['instant'].values).index(ts)
src = F['source_scenario'].values
N = np.stack([F['n_source'].values[src == int(s), it].sum(0) for s in SCEN])        # (3, y, x) per kg
f = N / N.sum(0)
mixed, _ = analysis.mixed_cells(F, ts)
J = np.load(REPO / 'data' / 'affx' / 'wrf3_jeff.npz')
nt = {s: float(J['ntot'][k]) for k, s in enumerate(SCEN)}             # per-particle = values / ntot
EPS = D['eps'].values
TEMP = D['temperature'].values
sel_e = (EPS >= 1e-3) & (EPS <= 1e-2)
sel_t = TEMP <= -10
print('mixed cells: %d; cells with >1%% urban: %d' % (mixed.sum(), (mixed & (f[1] > .01)).sum()))
for v in ('T', 'P', 'Vp', 'Vh'):
    ccn_key = {'T': 'ccn T (training, 287 K, file kappas)', 'P': 'ccn P (paper populations product)',
               'Vp': 'ccn Vp 287 K, training kappas', 'Vh': 'ccn Vh 287 K, histogram'}[v]
    q = {k: np.stack([np.asarray(pc[s][k], float) for s in SCEN]) for k in
         (ccn_key, 'ff ' + ('Vp' if v == 'Vp' else v), 'scat ' + v, 'abs ' + v)}
    ccn = np.einsum('syx,se->yxe', f, q[ccn_key])
    ff = np.einsum('syx,st->yxt', f, q['ff ' + ('Vp' if v == 'Vp' else v)])
    per = lambda a: a / np.array([nt[s] for s in SCEN])[:, None]       # per particle
    sc = np.einsum('syx,sw->yxw', N, per(q['scat ' + v]))
    ab = np.einsum('syx,sw->yxw', N, per(q['abs ' + v]))
    ref = dict(ccn=D['ccn_cdf'].values, ff=D['frozen_fraction'].values, sc=D['scattering'].values,
               ab=D['absorption'].values)
    m = mixed
    d_ccn = np.abs(ref['ccn'][m][:, sel_e] - ccn[m][:, sel_e]).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = lambda a, b: np.nanmedian(np.log10(a / b), axis=0)
        r_ff = 10 ** np.nanmedian(lr(ref['ff'][m][:, sel_t], ff[m][:, sel_t]))
        r_sc = 10 ** lr(ref['sc'][m], sc[m])
        r_ab = 10 ** lr(ref['ab'][m], ab[m])
        u = m & (f[1] > .3)                                             # urban-heavy cells
        r_sc_u = 10 ** lr(ref['sc'][u], sc[u])
        r_ab_u = 10 ** lr(ref['ab'][u], ab[u])
    print('%-3s CCN/CN mean |ref - mix| %.4f | FF median ratio %.3f | scattering median ratio %s (urban > 30%%: %s) '
          '| absorption %s (urban > 30%%: %s)'
          % (v, d_ccn, r_ff, np.round(r_sc, 3), np.round(r_sc_u, 3), np.round(r_ab, 3), np.round(r_ab_u, 3)))

# --- CCN temperature: signed differences (CCN does not depend on water, so only the
# temperature and the kappas can shift the reference against the mixture) -----------------
print('\nCCN/CN, mean signed (reference - mixture) over mixed cells, at s_env = 0.1, 0.2, 0.3, 0.6, 1.0 %:')
s_pts = [1e-3, 2e-3, 3e-3, 6e-3, 1e-2]
j = [int(np.argmin(np.abs(EPS - s))) for s in s_pts]
for name, key in (('287 K (T)', 'ccn Vp 287 K, training kappas'), ('290 K', 'ccn Vp 290 K, training kappas')):
    q = np.stack([np.asarray(pc[s][key], float) for s in SCEN])
    mix = np.einsum('syx,se->yxe', f, q)
    d = (D['ccn_cdf'].values - mix)[mixed][:, j]
    print('  %-10s %s   (cells with urban > 30%%: %s)' % (name, np.round(d.mean(0), 4),
                                                       np.round((D['ccn_cdf'].values - mix)[mixed & (f[1] > .3)][:, j].mean(0), 4)))
