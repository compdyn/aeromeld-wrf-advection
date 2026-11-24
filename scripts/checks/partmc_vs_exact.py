#!/usr/bin/env python
"""Numbers of the paper's Sections 4.2-4.3 that compare WRF-PartMC, the exact
answer and WRF-AeroMELD beyond the figures (2026-11-12).

Since the paper dropped WRF-PartMC (2026-11-12), items 1, 2 and the
WRF-PartMC column of 4 are a record of why; items 3 and 5 are in the paper.

For the main model at its representative seed (replay rep6, separate
limiters) at 04 UTC on the lowest level:

  1. The noise floor: the exact answer itself against WRF-PartMC, spectrum
     SMAPEs over WRF-PartMC's mixed cells (its tags), as for the model.
  2. WRF-PartMC's number in pure-background cells (its tags, background
     fraction > 0.999, 5 cells from the boundary), whose exact value is the
     uniform background mixing ratio: mean deviation, cell-to-cell scatter,
     and the mean deviation by terrain height (data/sources/terrain.npz).
  3. Size distributions against the exact answer, in the test-set measure
     |a - b| / (|a| + |b|) (number, total mass, per-species mass; decoder
     bins, wet diameter): the sources alone, the three points, and the mean
     over the mixed cells.
  4. Median diameters at the points: the exact mixture binned by wet and by
     dry diameter (from the source particles, data/sources), WRF-PartMC's
     diagnostics files, and WRF-AeroMELD. And each source's water mass share.
  5. Each scheme's distance of the latent vector from the triangle of the
     source mixtures (/ the source spacing): RMS, 99th percentile and maximum
     over the mixed cells, and the lowest implied fraction.

Usage (from scripts/): python checks/partmc_vs_exact.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import analysis, common, config, mixing, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402
from extract import affx  # noqa: E402

MODEL, RUN = 'twin_k16_of.6.0', 'rep6'
SOURCES = Path(__file__).resolve().parents[2] / 'data' / 'sources'
BG_VALUE = 1e9               # the background population's number mixing ratio (kg^-1)


def rel(a, b, axis=-1):
    return np.linalg.norm(a - b, axis=axis) / (np.linalg.norm(a, axis=axis) + np.linalg.norm(b, axis=axis))


def quantile_d(edges, m, p=.5):
    """Diameter (um) below which a fraction p of a binned distribution lies, linear in log D."""
    c = np.concatenate([[0], np.cumsum(m)]) / m.sum()
    return 10 ** np.interp(p, c, edges) * 1e6


def main():
    pop = products.load('populations', model=MODEL)
    pmc = products.load('partmc_diag_domain')
    fields = products.load('partmc_fields')
    s = T.load(RUN)
    it = T.time_index(s, 4.0)
    rm = T.run_model(MODEL)
    name = MODEL.rsplit('.', 1)[0]
    z_star = pop['z'].values
    f_ref = T.fractions(s, 'oracle', it)
    n_ref = T.number(s, 'oracle', it)
    mixed_pmc, _ = analysis.mixed_cells(fields, config.TIMESTAMP_FINAL)
    mixed = (f_ref >= analysis.MIX_MIN_FRAC).sum(0) >= 2

    # 1. noise floor
    ref = T.exact_reference(pop, f_ref, n_ref, config.TIMESTAMP_FINAL)
    e = [np.nanmean(analysis.spectrum_error(sp['ml'], sp['pmc'])[mixed_pmc]) for sp in analysis.spectra(ref, pmc)]
    print('1. exact answer vs WRF-PartMC, %d mixed cells: CCN/CN, scattering, absorption, frozen fraction %s'
          % (mixed_pmc.sum(), ' / '.join('%.1f%%' % (100 * x) for x in e)))

    # 2. background number
    ns = fields['n_source'].values[:, 1]
    fbg = (ns[0] + ns[2]) / ns.sum(0)                    # classes 0 and 2 are the background (scenario 561)
    inner = np.zeros(fbg.shape, bool)
    inner[5:-5, 5:-5] = True
    bg = (fbg > .999) & inner
    d = fields['ntot'].values[1] / BG_VALUE - 1
    t = np.load(SOURCES / 'terrain.npz')
    hgt = t['HGT']
    print('2. WRF-PartMC number in %d pure-background cells: mean %+.1f%%, cell-to-cell SD %.1f%%' % (
        bg.sum(), 100 * d[bg].mean(), 100 * d[bg].std()))
    for lo, hi, lab in ((-50, 1, 'ocean (HGT < 1 m)'), (1, 500, '1-500 m'), (500, 1000, '500-1000 m'),
                        (1000, 2000, '1000-2000 m'), (2000, 5000, 'above 2000 m')):
        sel = bg & (hgt >= lo) & (hgt < hi)
        print('   %-18s %5d cells, mean %+.1f%%' % (lab, sel.sum(), 100 * d[sel].mean()))

    # 3. size distributions against the exact answer (per particle, decoder bins)
    n_src = pop['pmc_ntot'].values.astype(float)
    w = .25
    Nsrc = pop['pmc_dNdlogDp'].values / n_src[:, None] * w
    Msrc = pop['pmc_dMdlogDp'].values / n_src[:, None, None] * w
    dec = affx.model(name).decode
    blocks = lambda x: (np.clip(affx.block(x, 'N'), 0, None),
                        np.clip(affx.block(x, 'M'), 0, None).reshape(*x.shape[:-1], 15, 15))
    Ns, Ms = blocks(dec(z_star))
    for k, sc in enumerate(pop['scenario'].values):
        print('3. source %d: number %.1f%%, total mass %.1f%%, per-species mass %.1f%%' % (
            sc, 100 * rel(Ns[k], Nsrc[k]), 100 * rel(Ms[k].sum(0), Msrc[k].sum(0)),
            100 * rel(Ms[k].ravel(), Msrc[k].ravel())))
    z = T.latent(s, 'sep:' + rm, z_star, rm, it)
    Nml, Mml = blocks(dec(np.moveaxis(z, 0, -1).reshape(-1, z.shape[0])).reshape(*z.shape[1:], -1))
    Nex = np.einsum('sb,syx->yxb', Nsrc, f_ref)
    Mex = np.einsum('scb,syx->yxcb', Msrc, f_ref)
    print('3. mixed cells (%d), mean: number %.1f%%, total mass %.1f%%, per-species mass %.1f%%' % (
        mixed.sum(), 100 * rel(Nml, Nex)[mixed].mean(), 100 * rel(Mml.sum(-2), Mex.sum(-2))[mixed].mean(),
        100 * rel(Mml.reshape(*Mml.shape[:2], -1), Mex.reshape(*Mex.shape[:2], -1))[mixed].mean()))
    pts = [common.point_index(p) for p in config.LOCATIONS]
    for lab, (j, i) in zip(config.LOC_LABELS, pts):
        print('3. point %s: number %.1f%%, total mass %.1f%%, per-species mass %.1f%%' % (
            lab, 100 * rel(Nml[j, i], Nex[j, i]), 100 * rel(Mml[j, i].sum(0), Mex[j, i].sum(0)),
            100 * rel(Mml[j, i].ravel(), Mex[j, i].ravel())))

    # 4. median diameters and water
    import netCDF4
    src = {}
    for sc in pop['scenario'].values:
        with netCDF4.Dataset(SOURCES / ('scenario_%d_filtered_0001_00000013.nc' % sc)) as ds:
            m = np.asarray(ds['aero_particle_mass'][:], float)
            rho = np.asarray(ds['aero_density'][:], float)
            nc = np.asarray(ds['aero_num_conc'][:], float)
            names = ds['aero_species'].names.split(',')
        if m.shape[0] != rho.size:
            m = m.T
        iw = names.index('H2O')
        vol = (m / rho[:, None]).sum(0)
        src[sc] = dict(mass=m.sum(0), nc=nc, Dw=(6 * vol / np.pi) ** (1 / 3),
                       Dd=(6 * (vol - m[iw] / rho[iw]) / np.pi) ** (1 / 3))
        print('4. source %d: water %.0f%% of the mass' % (sc, 100 * (m[iw] * nc).sum() / (m.sum(0) * nc).sum()))
    pmcp = common.at_time(products.load('partmc_diag_points'), config.TIMESTAMP_FINAL)
    edges_p = np.append(pmcp['log10_d_lo'].values, pmcp['log10_d_hi'].values[-1])
    edges = np.append(pop['log10_d_lo'].values, pop['log10_d_hi'].values[-1])

    def wq(x, wt):
        o = np.argsort(x)
        return x[o][np.searchsorted(np.cumsum(wt[o]) / wt.sum(), .5)] * 1e6

    for p, (lab, (j, i)) in enumerate(zip(config.LOC_LABELS, pts)):
        f = f_ref[:, j, i]
        wn = np.concatenate([src[sc]['nc'] * f[k] / src[sc]['nc'].sum() for k, sc in enumerate(pop['scenario'].values)])
        cat = lambda key: np.concatenate([src[sc][key] for sc in pop['scenario'].values])
        iw15 = list(config.ML_SPECIES).index('H2O')
        print('4. point %s: exact water %.0f%% of the mass; mass-median D (um): exact wet %.2f, exact dry %.2f, WRF-PartMC %.2f, WRF-AeroMELD %.2f' % (
            lab, 100 * Mex[j, i][iw15].sum() / Mex[j, i].sum(), wq(cat('Dw'), wn * cat('mass')), wq(cat('Dd'), wn * cat('mass')),
            quantile_d(edges_p, pmcp['dMdlogDp'].values[p].sum(0)), quantile_d(edges, Mml[j, i].sum(0))))

    # 5. distance from the mixtures
    edge = mixing.edge_length(z_star)
    for key in ('donor', 'shN', 'shB:' + rm, 'sep:' + rm):
        zz = T.latent(s, key, z_star, rm, it)
        fr, _ = mixing.implied_fractions(zz, z_star)
        dist = np.linalg.norm(zz - np.tensordot(z_star.T, mixing.f_bar(fr), axes=1), axis=0)[mixed] / edge
        print('5. %-14s distance / spacing: RMS %.2f%%, 99th pct %.1f%%, max %.1f%%; lowest implied fraction %.3f' % (
            key, 100 * np.sqrt(np.mean(dist ** 2)), 100 * np.percentile(dist, 99), 100 * dist.max(),
            fr[:, mixed].min()))


if __name__ == '__main__':
    main()
