#!/usr/bin/env python
"""Does WRF-AeroMELD with per-tracer latent advection, run in WRF, give what the offline replay gives?

The online run (WRF branch latent_per_tracer, training/wrf/latent_per_tracer.patch;
its extract data/wrfml_runs/<model>_pertracer.npz, latents divided by ntot) against
the replay of the same model and seed with separate limiters (transport/,
data/transport/<replay>), over the mixed cells, at every 15-minute output:

  - source fractions (barycentric coordinates of the latent): RMS difference;
  - number: RMS log10 ratio;
  - distance of the latent from the mixtures (/ source spacing), both;
  - at 04 UTC, the spectrum error against the exact answer, both.

The two differ only by the replay's 1-minute averaged fluxes (the same flux for
the three RK stages), so they should agree far better than the schemes differ.

Online and replay outputs are matched by time (hours since 00 UTC), so a replay
with other output times works; the spectrum errors are reported at 4 h and at the
online run's last output (both 04 UTC for the 4-hour run).

Usage (from scripts/): python checks/pertracer_online.py [--model twin_k16_of.6.0] [--replay rep6]
                       [--online pertracer]
  12 hours: --replay rep6_12h --online pertracer12h
            (data/wrfml_runs/twin_k16_of.6.0_pertracer12h.npz, training/wrf/extract12h.sbatch)
  final rebuild (2026-11-19): --model twin_k16_dd.5.0 --replay dd5_12h_15min --online pertracer12h
            (training/wrf/online_dd.sh; the replay's 15-minute surface.npz from Perlmutter's
            transport/runs/dd5_12h, since the local dd5_12h holds the 30-minute outputs)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import config, mixing, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

G = ('ccn', 'scattering', 'absorption', 'frozen_fraction')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='twin_k16_of.6.0')
    ap.add_argument('--replay', default='rep6')
    ap.add_argument('--online', default='pertracer', help='extract <model>_<online>.npz in data/wrfml_runs')
    args = ap.parse_args()
    rm = T.run_model(args.model)
    key = 'sep:' + rm
    with np.load(config.DATA_DIR / 'wrfml_runs' / ('%s_%s.npz' % (args.model, args.online))) as f:
        z_on, n_on, times = f['z_surf'].astype(float), f['ntot_surf'].astype(float), f['times']
    s = T.load(args.replay)
    pop = products.load('populations', model=args.model)
    z_star = pop['z'].values
    edge = mixing.edge_length(z_star)
    rms = lambda a, m: float(np.sqrt(np.nanmean(a[m] ** 2)))
    print('%s: online per-tracer WRF against replay %s (%s); mixed cells from the per-source fractions'
          % (args.model, args.replay, key))
    print('time       mixed  RMS df (bg, urban, emis)   RMS log10 N   distance from mixtures (online, replay)')
    hours = [(np.datetime64(str(t).replace('_', 'T')) - np.datetime64(str(times[0])[:10] + 'T00:00:00'))
             / np.timedelta64(1, 'h') for t in times]
    for i_on, (t, h) in enumerate(zip(times, hours)):
        if i_on == 0:
            continue
        it = T.time_index(s, h)                          # the replay's output at the same time
        f_ref = T.fractions(s, 'oracle', it)
        mixed = (f_ref >= 0.05).sum(0) >= 2
        z_off = T.latent(s, key, z_star, rm, it)
        f_on, _ = mixing.implied_fractions(z_on[i_on], z_star)
        f_off, _ = mixing.implied_fractions(z_off, z_star)
        dist = []
        for z, f in ((z_on[i_on], f_on), (z_off, f_off)):
            zb = np.tensordot(z_star.T, mixing.f_bar(f), axes=1)
            dist.append(rms(np.linalg.norm(z - zb, axis=0) / edge, mixed))
        print('%s  %5d  %s        %.5f       %.4f  %.4f' % (
            str(t)[11:16], mixed.sum(), ' '.join('%.4f' % rms(f_on[k] - f_off[k], mixed) for k in range(3)),
            rms(np.log10(n_on[i_on] / T.number(s, key, it)), mixed), *dist))
    # errors against the exact answer at 4 h and at the last output
    from extract import affx
    name = args.model.rsplit('.', 1)[0]
    for i_on in sorted({hours.index(4.0) if 4.0 in hours else len(times) - 1, len(times) - 1}):
        h = hours[i_on]
        it = T.time_index(s, h)
        stamp = 't+%gh' % h
        f_ref = T.fractions(s, 'oracle', it)
        ref = T.exact_reference(pop, f_ref, T.number(s, 'oracle', it), stamp)
        mixed = (f_ref >= 0.05).sum(0) >= 2
        for label, z, n in (('online per-tracer', z_on[i_on], n_on[i_on]),
                            ('replay, separate limiters', T.latent(s, key, z_star, rm, it), T.number(s, key, it))):
            ml = affx.decode_domain(name, z, n, {'timestamp': stamp})
            e = [float(np.nanmean(sp_err[mixed])) for sp_err in
                 (mixing.analysis.spectrum_error(sp['ml'], sp['pmc']) for sp in mixing.analysis.spectra(ml, ref))]
            print('%5.2f h, %-26s error against the exact answer: %s'
                  % (h, label, ' / '.join('%.1f%%' % (100 * x) for x in e)))


if __name__ == '__main__':
    main()
