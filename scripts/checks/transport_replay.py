#!/usr/bin/env python
"""Does the offline transport replay (transport/run.py) reproduce the two online models?

At 04 UTC, over the mixed cells:
  1. offline donor-cell against WRF-AeroMELD's own run (same scheme online):
     source fractions (barycentric coordinates of the latent) and number;
  2. offline per-source scheme against WRF-PartMC's particle tags (same scheme
     in expectation, plus WRF-PartMC's particle noise);
  3. the replay's air mass against WRF's mu (consistency of the averaged fluxes).
If 1 and 2 agree much better than the schemes differ from each other, the
replay is a fair testbed for the schemes.

Usage (from scripts/): python checks/transport_replay.py [--run r3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import mixing, products, registry  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

#: The online WRF-AeroMELD run the replay reproduces (the paper's main run until 2026-11-12).
ONLINE = 'twin_k16_of.0.0_5e7'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='r3')
    args = ap.parse_args()
    s = T.load(args.run)
    it = T.time_index(s, 4)
    run = registry.load().run(ONLINE)
    rm = T.RUN_MODEL[run.model.key]
    pop = products.load('populations', model=run.model.key)
    z_star = pop['z'].values
    f_pr, mixed = T.partmc_fractions()
    wf = products.load('wrfml_fields', run=run.key)
    f_online, _ = mixing.implied_fractions(wf['z'].values, z_star)
    n_online = products.load('aeromeld_domain', run=run.key)['ntot'].values.astype(float)
    n_pmc = products.load('partmc_fields')['ntot'].isel(instant=-1).values.astype(float)
    rms = lambda a: float(np.sqrt(np.nanmean(a[mixed] ** 2)))
    print('run %s, %d mixed cells, offline time %.0f s' % (args.run, mixed.sum(), s['time_s'][it]))

    print('\nSource fractions, RMS difference over the mixed cells (background, urban, emissions):')
    f = {k: T.fractions(s, k, it) for k in ('oracle', 'donor', 'donorW')}
    rows = [('offline donorW (as coded) - online', f['donorW'] - f_online),
            ('offline donor - online WRF-AeroMELD', f['donor'] - f_online),
            ('offline per-source - WRF-PartMC', f['oracle'] - f_pr),
            ('offline donor - WRF-PartMC', f['donor'] - f_pr),
            ('online WRF-AeroMELD - WRF-PartMC', f_online - f_pr),
            ('offline donor - offline per-source', f['donor'] - f['oracle'])]
    for name, d in rows:
        print('  %-40s %s' % (name, '  '.join('%.3f' % rms(d[k]) for k in range(3))))

    print('\nNumber, RMS log10 ratio over the mixed cells:')
    n_off = s['N'][it].astype(float)
    n_orc = T.number(s, 'oracle', it)
    for name, a, b in (('offline (WRF mono) / online WRF-AeroMELD', n_off, n_online),
                       ('offline per-source / WRF-PartMC', n_orc, n_pmc),
                       ('online WRF-AeroMELD / WRF-PartMC', n_online, n_pmc)):
        print('  %-42s %.4f' % (name, rms(np.log10(a / b))))

    m = s['M_offline_over_wrf']
    print('\nAir mass, offline / WRF mu, lowest level: %s'
          % ', '.join('%gh %.5f..%.5f' % (s['time_s'][k] / 3600, np.nanmin(m[k]), np.nanmax(m[k]))
                      for k in range(0, len(m), 4)))


if __name__ == '__main__':
    main()
