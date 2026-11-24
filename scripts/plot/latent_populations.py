#!/usr/bin/env python
"""latent_populations: the three seed populations and one model's reconstruction of them.

  (a) latent vectors (parallel coordinates)
  (b) normalized number size distribution
  (c) CCN spectrum (activated fraction)
  (d) frozen fraction
  (e) scattering, (f) absorption
  (g-i) PartMC per-species mass size distribution of each population, stacked
  (j-l) AeroMELD's reconstruction of the same, stacked

In (b-f), PartMC solid, AeroMELD dotted with open markers; colour is the population. The reference is the
1000-particle sample the encoder was handed, not the full population, so the
figure measures reconstruction alone. PartMC's extensive quantities are put on
the decoder's basis, k = ml_ntot / pmc_ntot (about 161, 161, 81; see
extract/populations.py). Optics are therefore at the number concentration the
decoder's ntot latent 0 stands for (2.9e9 m^-3 for 05_nonlinvlong.0.0).

The mass panels (g-l) are all divided by the population's AeroMELD total mass,
and each population's two panels share a y-range. So the PartMC stack's height
in (g-i) is the PartMC/AeroMELD mass ratio in each bin, and the two stacks
compare by eye.

2026-11-15: (b) and (g-l) are true normalized densities,
dN/dlog10 D / N and dM/dlog10 D / M (M the decoded total), so that the decoded
curves integrate to 1 over log10 D, as their axis labels say; until then they
were each bin's fraction of the total (the density times the 0.25-decade bin
width, so 4x smaller). (a) runs from -0.1 (the latents are a softplus); (b)'s
top follows the curves; the reference is "Particle-resolved reference" in the
legend and "reference" in the (g-i) tags (it was "sample"); (e) and (f) take
the shared optics ranges, now 1e0-1e3 and 1e-1-1e2.

AeroMELD's decoded number and species-mass distributions are clipped at zero
before anything uses them: the curves in (b), and the stacks, totals and
normalization in (g-l). The decoder returns small negative bins; for
05_nonlinvlong.0.0, 11 of 45 number bins (down to -3.8e3 against peaks of
3e9) and 199 of 675 species-mass bins (0.003-0.22 % of each population's
total mass). The populations product keeps the raw decoder output.

From read_wrf_data.ipynb cells 28-29 (build_figure and its panels). Ported as
is, it renders pixel-identical to the notebook's PDF (05_nonlinvlong.0.0).
Then these fixes:
  * the reconstruction gets its own row, (j-l), AeroMELD's species stacked.
    The notebook's docstrings describe a reconstructed total in (g-i), but its
    panel code never drew one (show_vae was unused), so it showed no
    reconstruction at all. Species-summed total lines (AeroMELD dashed in
    g-i, PartMC solid in j-l) were tried and removed on 2026-11-09: too hard
    to see against the stacks.
  * (g-i) stack on the bin edges. The notebook stacked with step='mid' on the
    bin centres, which made the end bins half-width and, on the log axis, put
    each step at the arithmetic midpoint between centres instead of at the edge.
  * (a) ticks are the latent's number alone (1..9); the labels "z1z2...z9"
    ran together.
  * population names come from config.POP_LABELS ("Emission", singular as in
    the text, since 2026-11-12; "Emissions" before), and the (g-l)
    tags name the model and population, e.g. "(g) PartMC: Background", instead
    of the scenario number (561), which means nothing to a reader.
  * (e) and (f) use fig_palettes.SCATTERING_RANGE and ABSORPTION_RANGE, the
    4-decade ranges of every optics figure (2026-11-09). These optics are per
    m^3 at the decoded number, not per kg, but their values fit the same
    ranges. Each panel used to span 6 decades up to its own data maximum.
    Both are ticked at every decade.
  * axis labels, frozen-fraction range and ticks, and wavelength ticks come
    from fig_palettes, shared with the point figures (2026-11-09).

Products: populations_<model>.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from paperfigs import config, figures, products  # noqa: E402
from paperfigs.fig_palettes import (ABSORPTION_RANGE, CCN_LABEL, DIAMETER_LABEL,  # noqa: E402
                                    FROZEN_FRACTION_LABEL, FROZEN_FRACTION_RANGE,
                                    FROZEN_FRACTION_TICKS, POP_COLORS, SCATTERING_RANGE,
                                    SPECIES_COLORS, SUPERSATURATION_LABEL,
                                    TEMPERATURE_LABEL, TWO_COLUMN, WAVELENGTH_LABEL,
                                    WAVELENGTH_TICKS, create_label_box, decade_ticks, pop_lw,
                                    species_label)

#: The legend's two lines: no WRF here, so the decoded line is "Decoded", and the reference is the
#: particle-resolved population (the 1000 particles the encoder was handed).
REF_LABEL = 'Particle-resolved reference'
DECODED_LABEL = 'Decoded'
DIAM_LABEL = DIAMETER_LABEL


def diam_centres(p):
    return 10 ** (0.5 * (p['log10_d_lo'].values + p['log10_d_hi'].values))


def diam_edges(p):
    return 10 ** np.append(p['log10_d_lo'].values, p['log10_d_hi'].values[-1])


def bin_width(p):
    """The bins' common width in log10 D (0.25 decade)."""
    w = p['log10_d_hi'].values - p['log10_d_lo'].values
    if not np.allclose(w, w[0]):
        raise ValueError('the size bins are not of equal width in log10 D')
    return float(w[0])


def decoded(ax, x, y, color, i, every=12):
    """AeroMELD's reconstruction: dotted, with open markers at every `every`-th point (staggered
    by population), so that it stays visible where it lies on the particle-resolved curve."""
    ax.plot(x, y, ls=':', color=color, lw=1.3, marker='o', ms=3.4, mfc='white', mec=color, mew=.9,
            markevery=(min(i * max(every // 3, 1), max(len(x) - 1, 0)), every), zorder=4, clip_on=False)


def panel_latent(ax, p, colors):
    z = p['z'].values
    n_latent = z.shape[1]
    dims = np.arange(n_latent)
    # Up to a dozen dims: every one ticked, full-size markers. A 49-latent model
    # gets a tick every 8th dim (z1, z9, ..., z49) and small markers.
    tick_step, ms = (1, 5) if n_latent <= 12 else (8, 2.5)
    for i, color in enumerate(colors):
        ax.plot(dims, z[i], '-o', color=color, lw=pop_lw(i), ms=ms, mec='white', mew=0.5, zorder=3,
                clip_on=False)
    ax.set_xticks(dims[::tick_step])
    ax.set_xticklabels(['%d' % (d + 1) for d in dims[::tick_step]], fontsize=8)
    ax.set_xlim(-0.6, n_latent - 0.4)
    ax.set_xlabel('Latent variable')
    ax.set_ylabel('Latent value $z$')
    if z.min() >= 0:
        # The latents are a softplus, so nonnegative: from just below 0 to the half unit at or
        # above max z + 0.1 (2.0 for twin_k16_of, max z 1.71; 2026-11-15).
        ax.set_ylim(-0.1, 0.5 * np.ceil(2 * (z.max() + 0.1)))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
    else:
        # +/-2.5 unless a marker would be clipped; then the next half unit past |z|.
        lim = max(2.5, 0.5 * np.ceil(2 * np.abs(z).max() + 1))
        ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
        ax.set_ylim([-lim, lim])
    ax.grid('on')
    create_label_box(ax, '(a) Latent vectors')


def panel_size_dist(ax, p, k, colors):
    diam_um = diam_centres(p) * 1e6
    width = bin_width(p)
    top = 0.0
    for i, color in enumerate(colors):
        pmc = k[i] * p['pmc_dNdlogDp'].values[i]
        ml = np.maximum(p['ml_dNdlogDp'].values[i], 0.0)    # clipped: see the module docstring
        # Shape only: each curve divided by its own total number (the sum over the bins of
        # dN/dlog10 D times the bin width), so that it integrates to 1 over log10 D.
        pmc, ml = pmc / (pmc.sum() * width), ml / (ml.sum() * width)
        ax.plot(diam_um, pmc, '-', color=color, lw=pop_lw(i))
        decoded(ax, diam_um, ml, color, i, every=2)
        top = max(top, pmc.max(), ml.max())
    ax.set_xlabel(DIAM_LABEL)
    ax.set_ylabel(r'$d\bar{N}/d\log_{10} D$')
    create_label_box(ax, '(b) Number size distribution')
    ax.set_xscale('log')
    ax.set_xlim([.001, 10])
    ax.grid('on')
    # the next tenth above the highest curve plus 8 % (1.3 for twin_k16_of, peak 1.18; 2026-11-15,
    # (the old 0.5 top, on the per-bin fractions, left half the panel empty). Ticks every 0.5,
    # so no tick label sits at the top, under the panel tag.
    ax.set_ylim([0, 0.1 * np.ceil(10 * top * 1.08)])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))


def panel_mass_dist(ax, p, k, i, tag, stacked='pmc'):
    """One model's species-mass distribution, stacked by species and divided by
    the AeroMELD total. stacked='pmc': PartMC (g-i); 'ml': AeroMELD (j-l)."""
    edges_um = diam_edges(p) * 1e6
    pmc = k[i] * p['pmc_dMdlogDp'].values[i]
    ml = np.maximum(p['ml_dMdlogDp'].values[i], 0.0)     # clipped: see the module docstring
    # the decoded total mass: the sum over species and bins of dM/dlog10 D times the bin width, so
    # that the decoded stack integrates to 1 over log10 D
    ml_sum = ml.sum() * bin_width(p)
    stack = pmc if stacked == 'pmc' else ml
    baseline = np.zeros(stack.shape[1])
    for name, vals in zip(p['species'].values, stack):
        top = baseline + vals / ml_sum
        # step='post' holds each value until the next edge: repeat the last
        # value at the last right edge so the last bin has its width.
        ax.fill_between(edges_um, np.append(baseline, baseline[-1]), np.append(top, top[-1]),
                        step='post', label=name, color=SPECIES_COLORS[name], ec='black', lw=0.3)
        baseline = top
    ax.set_xlabel(DIAM_LABEL)
    ax.set_ylabel(r'$d\bar{M}/d\log_{10}D$')
    create_label_box(ax, tag)
    ax.set_xscale('log')
    ax.set_xlim([.001, 10])
    ax.set_ylim(bottom=0)
    ax.grid('on')


def panel_ccn(ax, p, colors):
    ss_pct = 100 * p['eps'].values
    for i, color in enumerate(colors):
        ax.plot(ss_pct, p['pmc_ccn_cdf'].values[i], '-', color=color, lw=pop_lw(i))
        decoded(ax, ss_pct, p['ml_ccn_cdf'].values[i], color, i)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1e-1, 10)
    ax.set_xlabel(SUPERSATURATION_LABEL)
    ax.set_ylabel(CCN_LABEL)
    create_label_box(ax, '(c) CCN spectrum')
    ax.set_xscale('log')
    ax.grid('on')


def panel_frznfrac(ax, p, colors):
    temp = p['temperature'].values
    for i, color in enumerate(colors):
        ax.plot(temp, p['pmc_frozen_fraction'].values[i], '-', color=color, lw=pop_lw(i))
        decoded(ax, temp, p['ml_frozen_fraction'].values[i], color, i)
    ax.set_xlabel(TEMPERATURE_LABEL)
    ax.set_ylabel(FROZEN_FRACTION_LABEL)
    create_label_box(ax, '(d) Frozen fraction')
    ax.set_yscale('log')
    ax.set_xlim([-40, 0])
    ax.set_ylim(FROZEN_FRACTION_RANGE)
    ax.set_yticks(FROZEN_FRACTION_TICKS)
    ax.grid('on')


def panel_optics(ax, p, k, colors, var, ylabel, tag, ylim):
    wl_nm = p['wavelength'].values * 1e9
    for i, color in enumerate(colors):
        ax.plot(wl_nm, 1e6 * k[i] * p['pmc_' + var].values[i], '-', color=color, lw=pop_lw(i))
        decoded(ax, wl_nm, 1e6 * p['ml_' + var].values[i], color, 0, every=1)
    ax.set_xlabel(WAVELENGTH_LABEL)
    ax.set_ylabel(ylabel)
    create_label_box(ax, tag)
    ax.set_yscale('log')
    ax.set_xlim([300, 1000])
    ax.set_xticks(WAVELENGTH_TICKS)
    ax.set_ylim(ylim)
    # Every decade, as in the other optics figures; matplotlib's own log locator
    # ticked (f) only every second decade.
    ax.set_yticks(decade_ticks(*ylim))
    ax.grid('on')


def draw(p):
    """The figure from a populations product (an xarray.Dataset)."""
    scenarios = [int(s) for s in p['scenario'].values]
    colors = [POP_COLORS[i % len(POP_COLORS)] for i in range(len(scenarios))]
    k = (p['ml_ntot'] / p['pmc_ntot']).values.astype(float)

    # Five rows: two of spectra, the legend strip, PartMC mass (g-i), AeroMELD
    # mass (j-l). The size, margins and hspace keep the 4-row version's panel
    # height, gaps and legend room in inches (it was 5.5 in tall).
    fig, axes = plt.subplots(5, 3, figsize=(TWO_COLUMN, 6.9),
                             gridspec_kw={'height_ratios': [1, 1, 0.01, 1, 1], 'bottom': 0.16,
                                          'top': .972, 'right': .95, 'wspace': .65,
                                          'hspace': .89})
    panel_latent(axes[0][0], p, colors)
    panel_size_dist(axes[0][1], p, k, colors)
    panel_ccn(axes[0][2], p, colors)
    panel_frznfrac(axes[1][0], p, colors)
    panel_optics(axes[1][1], p, k, colors, 'scattering', r'Scattering (Mm$^{-1}$)', '(e) Scattering',
                 SCATTERING_RANGE)
    panel_optics(axes[1][2], p, k, colors, 'absorption', r'Absorption (Mm$^{-1}$)', '(f) Absorption',
                 ABSORPTION_RANGE)

    for ax in axes[2]:
        ax.axis('off')
    legend_ax = fig.add_subplot(axes[2][0].get_gridspec()[2, :])     # spans all 3 columns
    legend_ax.axis('off')
    pos = legend_ax.get_position()
    legend_ax.set_position([pos.x0, pos.y0 - 0.02, pos.width, 0.032])

    if tuple(scenarios) != tuple(config.POP_SCENARIOS):
        raise ValueError('product scenarios %s are not config.POP_SCENARIOS' % scenarios)
    for i, label in enumerate(config.POP_LABELS):
        panel_mass_dist(axes[3][i], p, k, i, '(%s) %s, reference' % ('ghi'[i], label))
        panel_mass_dist(axes[4][i], p, k, i, '(%s) %s, decoded' % ('jkl'[i], label), stacked='ml')
        # One y-range per population, so the two stacks compare by eye.
        top = max(axes[3][i].get_ylim()[1], axes[4][i].get_ylim()[1])
        axes[3][i].set_ylim(0, top)
        axes[4][i].set_ylim(0, top)

    handles = [matplotlib.lines.Line2D([], [], color=c, lw=2, label=lab)
               for lab, c in zip(config.POP_LABELS, colors)]
    spacer = matplotlib.lines.Line2D([], [], color='none', label='')
    style = [matplotlib.lines.Line2D([], [], color='0.35', lw=2, ls='-', label=REF_LABEL),
             matplotlib.lines.Line2D([], [], color='0.35', lw=1.3, ls=':', marker='o', ms=3.4, mfc='white',
                                     mew=.9, label=DECODED_LABEL)]
    all_handles = handles + [spacer] + style
    # centred on the figure, as the species legend below (the strip's axes sit right of centre)
    lpos = legend_ax.get_position()
    legend_ax.legend(handles=all_handles, frameon=True, ncol=len(all_handles), loc='center',
                     bbox_to_anchor=(0.5, lpos.y0 + lpos.height / 2), bbox_transform=fig.transFigure,
                     fontsize=7, handlelength=1.5, columnspacing=1.2)
    species_handles, species_labels = axes[3][0].get_legend_handles_labels()
    by_label = dict(zip(species_labels, species_handles))
    if config.SIZE_BASIS == 'dry':            # the dry distributions hold no water
        by_label.pop('H2O', None)
    fig.legend(by_label.values(), [species_label(k) for k in by_label], frameon=True, ncol=5,
               loc='lower center', bbox_to_anchor=(0.5, 0.0))
    return fig


def make(run):
    p = products.load('populations', model=run.model.key)
    return figures.save(draw(p), 'latent_populations')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
