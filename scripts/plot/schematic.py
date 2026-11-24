#!/usr/bin/env python
"""schematic: the WRF-AeroMELD chain, encode -> transport -> decode.

(a) Encode: the particles of a population, each mapped by the per-particle
network phi_theta to a feature vector; their number-weighted mean is the
latent vector z, carried with the number N. (b) Transport: on the WRF grid
the host model advects N and m = N z like any tracer, through the faces of
each cell; where air from two cells mixes, the latent vector of the mixture
is the number-weighted average of theirs. (c) Decode: the decoder turns (N, z)
in any cell into the diagnostics (size distribution, CCN/CN fraction, frozen
fraction, optics).

A drawing, not data: the particles, latents and curves are illustrative. The
three source colours are the populations' colours of the other figures
(POP_COLORS).

Products: none.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge  # noqa: E402

from paperfigs import figures  # noqa: E402
from paperfigs.fig_palettes import POP_COLORS, POP_MAP_COLORS, SPECIES_COLORS, TWO_COLUMN  # noqa: E402

INK = '#262626'
GREY = '#9a9a9a'
SPECIES = ('SO4', 'NH4', 'NO3', 'OC', 'BC', 'H2O')
HEIGHT = 2.35                  # inches


def species_color(name):
    c = SPECIES_COLORS.get(name, GREY)
    return c[:3] if isinstance(c, tuple) else c


def arrow(ax, p, q, text=None, dy=.07, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1.0,
                                 color=INK, shrinkA=0, shrinkB=0, **kw))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha='center', va='bottom',
                fontsize=7, color=INK)


def latent_column(ax, x, y, values, color, w=.075, h=.055, number=True):
    """A column of boxes: N on top (hatched), then the latents as bars."""
    k0 = 0
    if number:
        ax.add_patch(Rectangle((x, y), w, h, fc='white', ec=INK, lw=.6, hatch='/////'))
        k0 = 1
    for k, v in enumerate(values):
        yy = y - (k + k0) * h
        ax.add_patch(Rectangle((x, yy), w, h, fc='white', ec=INK, lw=.4))
        ax.add_patch(Rectangle((x, yy), w * v, h, fc=color, ec='none'))


def particle(ax, x, y, r, fracs):
    """A particle as a pie of its species' volume fractions."""
    a0 = 90.0
    for name, f in fracs:
        ax.add_patch(Wedge((x, y), r, a0, a0 + 360 * f, fc=species_color(name), ec='none'))
        a0 += 360 * f
    ax.add_patch(Circle((x, y), r, fc='none', ec=INK, lw=.5))


def panel_encode(ax, rng):
    ax.add_patch(Rectangle((.03, .14), .42, .72, fc='#f4f4f4', ec=GREY, lw=.6))
    ax.text(.24, .9, 'Particle population', ha='center', fontsize=7, color=INK)
    xs = rng.uniform(.08, .40, 11)
    ys = rng.uniform(.22, .79, 11)
    for x, y in zip(xs, ys):
        f = rng.dirichlet(np.ones(len(SPECIES)) * .6)
        particle(ax, x, y, rng.uniform(.018, .04), list(zip(SPECIES, f)))
    arrow(ax, (.47, .5), (.61, .5), r'$\boldsymbol{\phi}_\theta$', dy=.04)
    ax.text(.54, .46, 'mean', ha='center', va='top', fontsize=7, color=INK)
    latent_column(ax, .66, .78, rng.uniform(.2, .95, 9), POP_COLORS[1])
    ax.text(.84, .78 + .027, r'$N$', va='center', fontsize=7)
    ax.text(.84, .52, r'$\mathbf{z}$', va='center', fontsize=8)
    ax.text(.70, .13, r'$\mathbf{z}=\frac{1}{N}\sum_p w_p\boldsymbol{\phi}_\theta(x_p)$',
            ha='center', fontsize=7)


def panel_transport(ax):
    n_x, n_y = 5, 4
    x0, y0, cw, ch = .07, .17, .17, .16
    # a plume of the urban source (fraction f) into a background, mixed at its edges
    for j in range(n_y):
        for i in range(n_x):
            f = np.clip(1.25 - .32 * i - .28 * abs(j - 1.6), 0, 1)
            lin = f * np.array(to_rgb(POP_MAP_COLORS[1])) ** 2.2 + (1 - f) * np.array(to_rgb(POP_MAP_COLORS[0])) ** 2.2
            c = lin ** (1 / 2.2)                       # mixed in linear light, as in Figure 2
            ax.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw, ch, fc=c, ec='white', lw=1.0))
    # face fluxes
    for j in (1, 2):
        for i in (1, 2, 3):
            xf = x0 + i * cw
            yc = y0 + (j + .5) * ch
            ax.add_patch(FancyArrowPatch((xf - .045, yc), (xf + .045, yc), arrowstyle='-|>',
                                         mutation_scale=7, lw=.8, color='white'))
    ax.text(x0 + n_x * cw / 2, .9, r'Advect $N$ and $N\mathbf{z}$ as tracers',
            ha='center', fontsize=7, color=INK)
    ax.text(x0 + n_x * cw / 2, .045,
            r'mixing: $\mathbf{z}_{\rm mix}=\sum_k N_k\mathbf{z}_k\,/\sum_k N_k$',
            ha='center', fontsize=7, color=INK)


def panel_decode(ax):
    latent_column(ax, .03, .78, np.array([.8, .35, .6, .2, .9, .5, .3, .7, .45]),
                  POP_COLORS[1], w=.06)
    arrow(ax, (.12, .5), (.25, .5), r'$\mathrm{Dec}_\theta$', dy=.05)
    x = np.linspace(0, 1, 60)
    curves = (('Size distribution', np.exp(-((x - .45) / .16) ** 2), False),
              ('CCN/CN', 1 / (1 + np.exp(-(x - .4) * 9)), False),
              ('Frozen fraction', np.exp(-6 * x), False),
              ('Optics', .9 - .55 * x, False))
    for k, (label, y, _) in enumerate(curves):
        bx, by = .30 + (k % 2) * .35, .52 - (k // 2) * .38
        sub = ax.inset_axes([bx, by, .27, .27])
        sub.plot(x, y, color=POP_COLORS[1], lw=1.2)
        sub.set_xticks([])
        sub.set_yticks([])
        for sp in sub.spines.values():
            sp.set_linewidth(.5)
        sub.set_title(label, fontsize=6.5, pad=2)


def draw():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, HEIGHT),
                             gridspec_kw=dict(left=.005, right=.995, bottom=.01, top=.88, wspace=.06))
    for ax, title in zip(axes, ('(a) Encode', '(b) Transport', '(c) Decode')):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        ax.axis('off')
        ax.set_title(title, fontsize=8, loc='left', pad=2)
    panel_encode(axes[0], rng)
    panel_transport(axes[1])
    panel_decode(axes[2])
    for a, b in ((axes[0], axes[1]), (axes[1], axes[2])):
        p = a.transAxes.transform((1.0, .5))
        q = b.transAxes.transform((0.0, .5))
        p, q = fig.transFigure.inverted().transform(p), fig.transFigure.inverted().transform(q)
        fig.add_artist(FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=10, lw=1.2, color=GREY))
    return fig


def make(run=None):
    return figures.save(draw(), 'schematic')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    figures.setup(args)
    make()


if __name__ == '__main__':
    main()
