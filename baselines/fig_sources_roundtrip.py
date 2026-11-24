#!/usr/bin/env python
"""baselines/figures/sources_roundtrip{,_mosaic_mam4}.{pdf,png}: the three source populations,
particle-resolved and as each representation reconstructs them: AeroMELD (decoder 3, twin_k16_of
seed 1), one sectional and one modal variant (VARIANTS): sources_roundtrip shows each family's
most favourable setup (sec20w, mam4), sources_roundtrip_mosaic_mam4 the MOSAIC-like sec8 and
mam4.

Columns: Background, Urban, Emission (config.POP_LABELS, colours fig_palettes.POP_COLORS).
Rows: normalized number size distribution (wet diameter), CCN/CN, frozen fraction, scattering,
absorption. Optics at the full population's number concentration (config.POP_N_FULL), as in the
paper's Figure 3 (scripts/plot/latent_populations.py). Each panel's tag carries the
representations' errors for that diagnostic (errors.errors, %, in the legend's order).

Data: baselines/data/zsrc_baselines.npz (<model>_xsrc), populations_twin_k16_of.1.0.
Usage: python baselines/fig_sources_roundtrip.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import config, errors, products  # noqa: E402
from paperfigs.fig_palettes import (ABSORPTION_RANGE, CCN_LABEL, DIAMETER_LABEL,  # noqa: E402
                                    FROZEN_FRACTION_LABEL, FROZEN_FRACTION_RANGE,
                                    FROZEN_FRACTION_TICKS, POP_COLORS, SCATTERING_RANGE,
                                    SUPERSATURATION_LABEL, TEMPERATURE_LABEL, TWO_COLUMN,
                                    WAVELENGTH_LABEL, WAVELENGTH_TICKS, create_label_box,
                                    decade_ticks, use_paper_style)

MODEL = 'twin_k16_of.1.0'
OUT = HERE / 'figures'
#: (key, legend label, line style kwargs). The particle-resolved reference is solid and widest.
LABELS = {'sec8': 'Sectional, 8 dry bins (sec8, MOSAIC)', 'sec20w': 'Sectional, 20 wet bins (sec20w)',
          'sec4': 'Sectional, 4 dry bins (sec4)', 'sec20': 'Sectional, 20 dry bins (sec20)',
          'mam4': 'Modal, MAM4 (mam4)', 'mam4a': 'Modal, MAM4, large fresh C to acc. (mam4a)',
          'mam4l': 'Modal, MAM4 species (mam4l)'}
#: Which sectional and modal variant each figure shows. sources_roundtrip: each family's most
#: favourable setup by the static representation error at 12 h (sec20w 12.8 %; mam4 17.3 % vs
#: mam4a 18.3 %; on the sources alone mam4 and mam4a are indistinguishable, 11.6/16.2/8.2 vs
#: 11.5/16.2/8.2 %). sources_roundtrip_mosaic_mam4: the MOSAIC-like sec8 and mam4.
VARIANTS = {'sources_roundtrip': ('sec20w', 'mam4'), 'sources_roundtrip_mosaic_mam4': ('sec8', 'mam4')}


def styles(sec, mod):
    return (('pmc', 'Particle-resolved', dict(ls='-', lw=1.8, alpha=0.55)),
            ('aeromeld', 'AeroMELD (decoder 3)', dict(ls=':', lw=1.3, marker='o', ms=3.0, mfc='white', mew=.8)),
            (sec, LABELS[sec], dict(ls='--', lw=1.1)),
            (mod, LABELS[mod], dict(ls='-.', lw=1.1, marker='^', ms=2.6, mew=.6, mfc='white')))
DIAG = ('n_prthst', 'ccn_cdf', 'logfrznfrac_tmp', 'logqs_pop', 'logqa_pop')
BLK = {'N': (225, 15), 'C': (240, 100), 'S': (340, 8), 'A': (348, 8), 'F': (356, 100)}


def blk(x, k):
    o, n = BLK[k]
    return x[..., o:o + n]


def vectors():
    from extract import affx
    pop = products.load('populations', model=MODEL)
    with np.load(HERE / 'data' / 'zsrc_baselines.npz') as f:
        v = {'pmc': errors.source_vectors(pop, 'pmc'),
             'aeromeld': affx.model(MODEL.rsplit('.', 1)[0]).decode(pop['z'].values.astype(float)),
             **{k[:-5]: f[k] for k in f.files if k.endswith('_xsrc') and k != 'pmc_xsrc'}}
    return pop, v


def main():
    for name, (sec, mod) in VARIANTS.items():
        draw(name, sec, mod)


def draw(name, sec, mod):
    STYLES = styles(sec, mod)
    use_paper_style()
    pop, v = vectors()
    n_full = np.array([config.POP_N_FULL[s] for s in config.POP_SCENARIOS])
    err = {k: errors.errors(v[k], n_full, v['pmc'], n_full) for k in ('aeromeld', sec, mod)}
    d_um = 10 ** (0.5 * (pop['log10_d_lo'].values + pop['log10_d_hi'].values)) * 1e6
    ss = 100 * pop['eps'].values
    temp = pop['temperature'].values
    wl = pop['wavelength'].values * 1e9

    fig, axes = plt.subplots(5, 3, figsize=(TWO_COLUMN, 7.6),
                             gridspec_kw={'left': .12, 'right': .965, 'top': .955, 'bottom': .135,
                                          'wspace': .38, 'hspace': .78})
    for j, label in enumerate(config.POP_LABELS):
        c = POP_COLORS[j]
        for key, _, kw in STYLES:
            x = v[key][j]
            n = blk(x, 'N')
            rows = ((d_um, n / max(n.sum(), 1e-300)), (ss, blk(x, 'C')), (temp, blk(x, 'F')),
                    (wl, 1e6 * n_full[j] * blk(x, 'S')), (wl, 1e6 * n_full[j] * blk(x, 'A')))
            for i, (xx, yy) in enumerate(rows):
                kk = dict(kw)
                if 'marker' in kk:
                    kk['markevery'] = {0: 1, 1: 12, 2: 12, 3: 1, 4: 1}[i]
                    kk['mec'] = c
                axes[i][j].plot(xx, yy, color=c, **kk)
        axes[0][j].set_title(label, fontsize=9, color=c, pad=14)
        for i, dname in enumerate(DIAG):
            ax = axes[i][j]
            tag = '%s' % ' / '.join('%.0f' % (100 * err[k][dname][j]) for k in ('aeromeld', sec, mod))
            create_label_box(ax, '(%s) e: %s %%' % ('abcdefghijklmno'[3 * i + j], tag))
            ax.grid('on', lw=.4)
        a = axes[0][j]
        a.set_xscale('log'); a.set_xlim(1e-3, 10); a.set_ylim(0, .45)
        a = axes[1][j]
        a.set_xscale('log'); a.set_xlim(.1, 10); a.set_ylim(0, 1)
        a = axes[2][j]
        a.set_yscale('log'); a.set_xlim(-40, 0); a.set_ylim(FROZEN_FRACTION_RANGE)
        a.set_yticks(FROZEN_FRACTION_TICKS)
        for i, rng in ((3, SCATTERING_RANGE), (4, ABSORPTION_RANGE)):
            a = axes[i][j]
            a.set_yscale('log'); a.set_xlim(300, 1000); a.set_xticks(WAVELENGTH_TICKS)
            a.set_ylim(rng); a.set_yticks(decade_ticks(*rng))
        axes[0][j].set_xlabel(DIAMETER_LABEL)
        axes[1][j].set_xlabel(SUPERSATURATION_LABEL)
        axes[2][j].set_xlabel(TEMPERATURE_LABEL)
        axes[3][j].set_xlabel(WAVELENGTH_LABEL)
        axes[4][j].set_xlabel(WAVELENGTH_LABEL)
    for i, lab in enumerate((r'Fraction per 0.25 decade', CCN_LABEL, FROZEN_FRACTION_LABEL,
                             r'Scattering (Mm$^{-1}$)', r'Absorption (Mm$^{-1}$)')):
        axes[i][0].set_ylabel(lab)
    handles = [matplotlib.lines.Line2D([], [], color='0.25', label=lab,
                                       **{k: val for k, val in kw.items() if k != 'alpha'})
               for _, lab, kw in STYLES]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=True, fontsize=7,
               handlelength=2.6, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.062, 'Panel tags: error e (%%) of AeroMELD / %s / %s against the '
             'particle-resolved source' % (sec, mod),
             ha='center', fontsize=6.5, color='0.3')
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        p = OUT / ('%s.%s' % (name, ext))
        fig.savefig(p, dpi=300 if ext == 'png' else 600)
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    main()
