#!/usr/bin/env python
"""The sampling points A, B, C: at one time, from per-source transport on the lowest level, a
mixed cell in which a different source population carries most of the aerosol MASS:

  A  urban       50-75 % of the mass
  B  background  50-75 % of the mass
  C  emission    50-75 % of the mass

each with another source holding at least 20 % of the mass (so every point is a mixture), the
smoothest such cell (least 3 x 3 standard deviation of the mass fractions, summed over the three
sources), at least --margin cells (default 10) from the lateral boundaries, where the boundary
values are imposed. Mass fractions are the number fractions weighted by each source's mass per
particle (its particle-resolved speciated mass distribution, summed): urban 2.2 and emission
11.8 times the background's (the emission population is 74 % water).

Why mass (2026-11-13): chosen by number fractions (urban + background, emission +
background, all three, each >= 30 % or 20 % of the number), B and C both held about 40 % of
their number from the emission population and so about 85 % of their mass, and their speciated
mass distributions looked alike. That rule gave (65, 46), (75, 73), (122, 16) at 12 h, and at
4 h (66, 25), (108, 94), (89, 77).

Prints 1-based (i, j) = (west_east, south_north), as config.LOCATIONS.

Usage (from scripts/): python checks/pick_points.py [--replay all1_12h] [--hours 12]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import config, errors, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

#: (label, source index (background 0, urban 1, emission 2)), in the order A, B, C.
KINDS = (('A', 1), ('B', 0), ('C', 2))
DOMINANT = (.5, .75)
SECOND = .2


def window_std(a):
    """Standard deviation over each cell's 3 x 3 neighbourhood (nan on the outer ring)."""
    ny, nx = a.shape
    stack = np.stack([a[1 + dy:ny - 1 + dy, 1 + dx:nx - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    out = np.full(a.shape, np.nan)
    out[1:-1, 1:-1] = stack.std(0)
    return out


def mass_per_particle(model='twin_k16_of.1.0'):
    """Each source's mass per particle (background, urban, emission), from its particle-resolved
    speciated mass distribution (any model's populations product: the PR part is the same). Since
    the dry rebuild (2026-11-16, config.SIZE_BASIS 'dry') the dry mass, water excluded, from the
    source files binned by dry diameter (extract/affx.size_distributions): urban 2.2 and emission
    3.1 times the background's. Since 2026-11-19 (config.SIZE_BASIS 'drydiam') the mass in the
    dry-diameter bins with every species, water included, as the speciated mass distribution."""
    if config.SIZE_BASIS != 'wet':
        from extract import affx
        N, M = affx.size_distributions()[config.SIZE_BASIS]
        return M.sum((1, 2)) / N.sum(1)
    x = errors.source_vectors(products.load('populations', model=model), 'pmc')
    return x[:, :225].sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replay', default=config.REPLAY_12H % 0)
    ap.add_argument('--hours', type=float, default=12.0)
    ap.add_argument('--margin', type=int, default=10)
    args = ap.parse_args()
    s = T.load(args.replay)
    f = T.fractions(s, 'oracle', T.time_index(s, args.hours))
    w = f * mass_per_particle()[:, None, None]
    g = w / w.sum(0)                                                   # mass fractions
    rough = sum(window_std(g[k]) for k in range(3))
    m = args.margin
    inner = np.zeros(rough.shape, bool)
    inner[m:-m, m:-m] = True
    rough[~inner] = np.nan
    second = np.sort(g, 0)[1]
    names = ('background', 'urban', 'emission')
    for label, k in KINDS:
        ok = (g[k] >= DOMINANT[0]) & (g[k] <= DOMINANT[1]) & (second >= SECOND) & np.isfinite(rough)
        r = np.where(ok, rough, np.inf)
        j, i = np.unravel_index(np.argmin(r), r.shape)
        print('%s %-10s (%d, %d)  cells of its kind %4d  3x3 std %.4f  number bg/urb/emi %.2f %.2f %.2f'
              '  mass %.2f %.2f %.2f' % (label, names[k], i + 1, j + 1, ok.sum(), r[j, i], *f[:, j, i], *g[:, j, i]))


if __name__ == '__main__':
    main()
