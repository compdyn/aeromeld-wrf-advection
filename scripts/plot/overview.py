#!/usr/bin/env python
"""overview: WRF-AeroMELD at a glance, particles -> encoder -> transport -> decoder.

(a) A population of particles, each a pie of its species. The encoder maps it
to the state of one grid cell: the number concentration N and the 9 components
of the latent vector z. (b) Every cell of the host model's grid carries its own
(N, z), drawn as a bar (N) over a 3 x 3 block (z); the grey arrows centred on the
cell vertices are the wind (the same on both grids). (c) The host model transports N and N z with that wind:
the two plumes move downwind, converge and mix. (d) The decoder turns (N, z) of
any cell into the diagnostics: number and speciated mass size distributions,
CCN spectrum, frozen fraction, scattering and absorption.

A drawing, not data. The cells are mixtures of three source populations
(background, urban, emission; POP_COLORS) with number weights w_s(x, y); the
latent vector of a cell is the number-weighted mean of three fixed source
vectors, z = sum_s f_s z*_s, which is what exact mixing of the latent state
gives. Each entry is drawn in the cell's blended source colour, darker for a
larger value. The transported grid is the first one moved along the
trajectories of the drawn wind (uniform flow plus a confluence, divergence
free), so the two grids and the arrows are consistent.

Products: none.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge  # noqa: E402

from paperfigs import figures  # noqa: E402
from paperfigs.fig_palettes import POP_COLORS, SPECIES_COLORS, species_label  # noqa: E402

MM = 25.4
#: The figure is drawn at the journal's text width (agujournal2019: \textwidth 5.5 in), so it is
#: printed at 100 % and the font sizes below are the printed ones.
PAGE_WIDTH = 5.5
WIDTH = PAGE_WIDTH * MM        # 139.7 mm; the drawing's coordinates are millimetres
HEIGHT = 56.0

INK = '#262626'
GREY = '#8c8c8c'
LINE = '#c8c8c8'
SPECIES = ('SO4', 'NO3', 'NH4', 'OC', 'BC', 'H2O')

# the grid: N_X x N_Y square cells of side CELL (set by layout())
N_X, N_Y, CELL = 4, 3, 6.1
GRID_Y = 17.0
SQ = 1.2                       # side of one entry of a cell's array

# the wind, in cells per unit time: uniform flow plus a confluence on the mid row
U0, STRAIN, T_END = 1.0, 0.9, 1.0
# the sources: (x, y, radius, amplitude) of the urban and the emission plume, set by layout()
PLUMES = ()

HEAD_Y = 52.0                  # baseline of the panel headings
LABEL = 7.0                    # font size of the small labels
#: Font size of the six diagnostics' titles in (d): the paper's names, two pairs of which
#: (CCN spectrum, Frozen fraction; Number, Speciated mass) only fit side by side at 6 pt.
DIAG_TITLE = 6.0
ARROW_LABEL = 7.5
HEADING = 8.5


def layout(nx, ny):
    """Set the grid size: nx x ny cells, the plumes in the top and bottom rows of the
    first column, converging on the middle (ny odd: a middle row of mixed cells)."""
    global N_X, N_Y, Y_C, PLUMES, ENCODED, DECODED, GRID_TOP, GRID_MID
    N_X, N_Y = nx, ny
    Y_C = ny / 2
    PLUMES = ((.6, ny - .5, .3, 9.0), (.6, .5, .3, 9.0))
    ENCODED = (0, ny - 1)                  # the encoder arrow points at the urban cell
    DECODED = (nx - 1, ny // 2)            # the decoder arrow leaves a mixed cell
    GRID_TOP = GRID_Y + N_Y * CELL
    GRID_MID = GRID_Y + N_Y * CELL / 2


layout(N_X, N_Y)


def species_color(name):
    return SPECIES_COLORS[name][:3] if isinstance(SPECIES_COLORS[name], tuple) else SPECIES_COLORS[name]


def linear(c):
    return np.array(to_rgb(c)) ** 2.2


def wind(x, y):
    """The wind at cell coordinates (x, y), in cells per unit time: a uniform flow plus a
    confluence on the middle (divergence free)."""
    return U0 + STRAIN * x, -STRAIN * (y - Y_C)


def weights(x, y, t=0.0):
    """Number weights (background, urban, emission) at cell coordinates (x, y)
    and time t: the initial field at the foot of the wind's trajectory."""
    x0 = (x + U0 / STRAIN) * np.exp(-STRAIN * t) - U0 / STRAIN
    y0 = Y_C + (y - Y_C) * np.exp(STRAIN * t)
    w = [np.ones_like(x0)]
    for (px, py, r, amp) in PLUMES:
        w.append(amp * np.exp(-.5 * ((x0 - px) ** 2 + (y0 - py) ** 2) / r ** 2))
    return np.array(w)


def cell_state(i, j, t, z_sources):
    """(N, z, colour) of cell (i, j): the average over the cell of the weights."""
    s = (np.arange(7) + .5) / 7
    xx, yy = np.meshgrid(i + s, j + s)
    w = weights(xx, yy, t).mean(axis=(1, 2))
    n = w.sum()
    f = w / n
    z = f @ z_sources
    col = sum(fk * linear(c) for fk, c in zip(f, POP_COLORS))      # mixed in linear light
    return n, z, col


def shade(col, v):
    """The colour ``col`` (linear light) at strength v in [0, 1], on white."""
    return np.clip(v * col + (1 - v) * np.ones(3), 0, 1) ** (1 / 2.2)


def n_strength(n):
    return .3 + .7 * np.log(n) / np.log(8.0)


def z_strength(v):
    return .1 + .9 * v


def cell_array(ax, xc, yc, n, z, col, sq=SQ):
    """A cell's state centred on (xc, yc): a bar for N over a 3 x 3 block for z."""
    x0, y0 = xc - 1.5 * sq, yc - 2 * sq
    kw = dict(ec='white', lw=.35)
    ax.add_patch(Rectangle((x0, y0 + 3 * sq), 3 * sq, sq, fc=shade(col, n_strength(n)), **kw))
    for k, v in enumerate(z):
        ax.add_patch(Rectangle((x0 + (k % 3) * sq, y0 + (2 - k // 3) * sq), sq, sq,
                               fc=shade(col, z_strength(v)), **kw))


def grid(ax, x0, t, z_sources, mark=None):
    """The grid at time t, every cell's state, the wind at the interior vertices (the same
    steady wind on both grids) and the marked cell outlined."""
    w, h = N_X * CELL, N_Y * CELL
    ax.add_patch(Rectangle((x0, GRID_Y), w, h, fc='white', ec='none'))
    for i in range(1, N_X):
        ax.plot([x0 + i * CELL] * 2, [GRID_Y, GRID_Y + h], color=LINE, lw=.45, solid_capstyle='butt')
    for j in range(1, N_Y):
        ax.plot([x0, x0 + w], [GRID_Y + j * CELL] * 2, color=LINE, lw=.45, solid_capstyle='butt')
    ax.add_patch(Rectangle((x0, GRID_Y), w, h, fc='none', ec=GREY, lw=.7))
    for i in range(N_X):
        for j in range(N_Y):
            n, z, col = cell_state(i, j, t, z_sources)
            cell_array(ax, x0 + (i + .5) * CELL, GRID_Y + (j + .5) * CELL, n, z, col)
    for i in range(N_X + 1):                      # every vertex, the boundary included
        for j in range(N_Y + 1):
            dx, dy = wind_vector(i, j)
            xv, yv = x0 + i * CELL, GRID_Y + j * CELL
            wind_arrow(ax, xv - dx / 2, yv - dy / 2, dx, dy)                  # centred on the vertex
    if mark is not None:
        ax.add_patch(Rectangle((x0 + mark[0] * CELL, GRID_Y + mark[1] * CELL), CELL, CELL,
                               fc='none', ec=INK, lw=1.1, zorder=5))


def wind_vector(i, j):
    """The drawn vector (mm) of the wind at vertex (i, j): its direction, and a length
    that grows with the speed but over a compressed range (square root), so that the slow
    wind on the left still reads as an arrow."""
    u, v = wind(i, j)
    speed = np.hypot(u, v)
    s_max = np.hypot(*wind(N_X, 0))
    length = CELL * (.14 + .32 * np.sqrt(speed / s_max))
    return length * u / speed, length * v / speed


def wind_arrow(ax, x, y, dx, dy, color='#4d4d4d'):
    ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle='-|>,head_length=.38,head_width=.17',
                                 mutation_scale=5.0, lw=.7, color=color, shrinkA=0, shrinkB=0,
                                 zorder=4, joinstyle='miter', capstyle='butt'))


def arrow(ax, x0, x1, y, text, text_x):
    """A big arrow from x0 to x1 at height y, labelled above at text_x (the middle of the free
    space between the wind vectors that stick out of the grids and the next panel)."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=10, lw=1.2,
                                 color=INK, shrinkA=0, shrinkB=0, zorder=6))
    ax.text(text_x, y + 1.1, text, ha='center', va='bottom', fontsize=ARROW_LABEL, color=INK)


def heading(ax, x, letter, text):
    ax.text(x, HEAD_Y, f'({letter}) {text}', ha='left', va='baseline', fontsize=HEADING, color=INK)


def particle(ax, x, y, r, fracs):
    """A particle as a pie of its species' mass fractions."""
    a0 = 90.0
    for name, f in zip(SPECIES, fracs):
        ax.add_patch(Wedge((x, y), r, a0, a0 + 360 * f, fc=species_color(name), ec='none'))
        a0 += 360 * f
    ax.add_patch(Circle((x, y), r, fc='none', ec=INK, lw=.45))


def population(ax, rng, x0, x1, y0, y1, n=11):
    ax.add_patch(FancyBboxPatch((x0 + 1.5, y0 + 1.5), x1 - x0 - 3, y1 - y0 - 3,
                                boxstyle='round,pad=1.5,rounding_size=3', fc='#f5f5f5', ec=LINE, lw=.6))
    # non-overlapping discs, largest first
    placed = []
    for r in np.sort(rng.lognormal(np.log(1.4), .42, n))[::-1]:
        r = float(np.clip(r, .75, 2.8))
        for _ in range(4000):
            x, y = rng.uniform(x0 + r + 1.1, x1 - r - 1.1), rng.uniform(y0 + r + 1.1, y1 - r - 1.1)
            if all(np.hypot(x - px, y - py) > r + pr + .75 for px, py, pr in placed):
                placed.append((x, y, r))
                break
    base = np.array([2.2, 1.6, 1.2, 1.8, .7, .9])
    for x, y, r in placed:
        particle(ax, x, y, r, rng.dirichlet(base * 1.6))


def species_key(ax, x0, y0, dx=8.4, dy=3.3):
    for k, (name, label) in enumerate(zip(SPECIES, map(species_label, SPECIES))):
        x, y = x0 + (k // 3) * dx, y0 - (k % 3) * dy
        ax.add_patch(Rectangle((x, y - .8), 1.6, 1.6, fc=species_color(name), ec='none'))
        ax.text(x + 2.4, y, label, ha='left', va='center', fontsize=LABEL, color=INK)


def state_key(ax, xc, y, z_sources):
    """Below the first grid, centred on xc: what the bar, the block and the arrows are."""
    n, z, col = cell_state(*ENCODED, 0.0, z_sources)
    sq = SQ * .9
    x0 = xc - 13.0
    ax.add_patch(Rectangle((x0, y - sq / 2), 3 * sq, sq, fc=shade(col, n_strength(n)), ec='white', lw=.35))
    ax.text(x0 + 3 * sq + .8, y, r'$N$', ha='left', va='center_baseline', fontsize=LABEL, color=INK)
    x1 = x0 + 8.6
    for k, v in enumerate(z):
        ax.add_patch(Rectangle((x1 + (k % 3) * sq, y - 1.5 * sq + (2 - k // 3) * sq), sq, sq,
                               fc=shade(col, z_strength(v)), ec='white', lw=.35))
    ax.text(x1 + 3 * sq + .8, y, r'$\mathbf{z}$', ha='left', va='center_baseline', fontsize=LABEL, color=INK)
    x2 = x0 + 17.2
    wind_arrow(ax, x2, y, 3.4, 0)
    ax.text(x2 + 3.8, y, 'wind', ha='left', va='center_baseline', fontsize=LABEL, color=INK)


def diagnostics(ax, x0, col_w, gap, y_top, color):
    """Six small plots, 3 rows x 2 columns, of the decoded cell, titles centred: the paper's
    diagnostics in its order (short names), row by row."""
    x = np.linspace(0, 1, 200)
    # pitch: room under each row for its axis label (padded clear of the axis line) above the
    # next row's titles
    h, pitch = 9.0, 15.8
    panels = (('Number', r'$D$'), ('Speciated mass', r'$D$'), ('CCN spectrum', r'$s$'),
              ('Frozen fraction', r'$T$'), ('Scattering', r'$\lambda$'), ('Absorption', r'$\lambda$'))
    subs = {}
    for k, (title, x_label) in enumerate(panels):
        row, c = k // 2, k % 2
        sub = ax.inset_axes([x0 + c * (col_w + gap), y_top - h - row * pitch, col_w, h],
                            transform=ax.transData)
        sub.set_xlim(0, 1)
        sub.set_ylim(0, 1.12)
        sub.set_xticks([])
        sub.set_yticks([])
        for side in ('top', 'right'):
            sub.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            sub.spines[side].set_linewidth(.65)
            sub.spines[side].set_color(INK)
        sub.patch.set_alpha(0)
        sub.set_title(title, fontsize=DIAG_TITLE, pad=2.0, loc='center', color=INK)
        sub.set_xlabel(x_label, fontsize=LABEL, labelpad=2.6, color=INK)     # T and lambda touched the axis at 1.0
        subs[title] = sub
    kw = dict(color=color, lw=1.2, solid_capstyle='round', clip_on=False)

    def mode(mu, sig):
        return np.exp(-.5 * ((x - mu) / sig) ** 2)

    # number size distribution: two modes
    subs['Number'].plot(x, mode(.34, .11) + .42 * mode(.66, .12), **kw)
    # speciated mass size distribution: stacked by species over size bins
    edges = np.linspace(.04, .96, 11)
    mid = .5 * (edges[1:] + edges[:-1])
    shapes = ((.62, .15, 1.0), (.66, .13, .7), (.63, .14, .5), (.50, .17, .75), (.40, .12, .3), (.70, .13, .45))
    stack = np.array([a * np.exp(-.5 * ((mid - mu) / sig) ** 2) for mu, sig, a in shapes])
    stack *= 1.02 / stack.sum(axis=0).max()
    bottom = np.zeros_like(mid)
    for name, part in zip(SPECIES, stack):
        subs['Speciated mass'].bar(mid, part, width=np.diff(edges), bottom=bottom,
                                   color=species_color(name), ec='white', lw=.2)
        bottom += part
    # CCN spectrum: activated fraction against supersaturation
    subs['CCN spectrum'].plot(x, .97 / (1 + np.exp(-(x - .45) * 10)), **kw)
    # frozen fraction against temperature: rises towards the cold (left) end
    subs['Frozen fraction'].plot(x, .97 * np.exp(-((x + .02) / .34) ** 1.6), **kw)
    # scattering and absorption against wavelength
    lam = .3 + .7 * x
    subs['Scattering'].plot(x, .98 * (lam / .3) ** -1.7, **kw)
    subs['Absorption'].plot(x, .62 * (lam / .3) ** -1.05, **kw)


def draw(nx=4, ny=3):
    layout(nx, ny)
    rng = np.random.default_rng(11)
    z_sources = np.array([[.30, .55, .20, .45, .15, .60, .35, .25, .50],     # background
                          [.95, .25, .70, .15, .85, .40, .10, .90, .55],     # urban
                          [.20, .90, .35, .80, .10, .65, .95, .30, .75]])    # emission
    fig = plt.figure(figsize=(PAGE_WIDTH, HEIGHT / MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis('off')

    grid_w = N_X * CELL
    pop_x0, pop_x1 = .4, 15.8
    grid1_x = pop_x1 + 13.0
    grid2_x = grid1_x + grid_w + 15.8
    out_r, out_l = 1.4, .9                 # how far the wind vectors stick out right and left
    col_w, gap = 12.0, 6.0
    diag_x = WIDTH - 2 * col_w - gap - 2.8      # 2.8 mm on the right: room for the centred titles
    pop_y0, pop_y1 = GRID_Y - 3.0, GRID_TOP + 3.0

    # (a) the population
    population(ax, rng, pop_x0, pop_x1, pop_y0, pop_y1)
    species_key(ax, pop_x0 + .8, pop_y0 - 3.4)
    heading(ax, pop_x0, 'a', 'Particles')

    # (b) the latent state on the grid, with the wind
    y_enc = GRID_Y + (ENCODED[1] + .5) * CELL
    arrow(ax, pop_x1 + .9, grid1_x - 1.0, y_enc, 'Encoder', (pop_x1 + grid1_x - out_l) / 2)
    grid(ax, grid1_x, 0.0, z_sources, mark=ENCODED)
    state_key(ax, grid1_x + grid_w / 2, GRID_Y - 5.0, z_sources)
    heading(ax, grid1_x, 'b', r'State $(N, \mathbf{z})$ per cell')

    # (c) the same grid after transport by the same wind
    arrow(ax, grid1_x + grid_w + 1.2, grid2_x - 1.2, GRID_MID, 'Transport',
          (grid1_x + grid_w + out_r + grid2_x - out_l) / 2)
    grid(ax, grid2_x, T_END, z_sources, mark=DECODED)
    heading(ax, grid2_x, 'c', 'After transport')
    ax.text(grid2_x + grid_w / 2, GRID_Y - 3.6, 'mixing averages $\\mathbf{z}$,\nweighted by $N$',
            ha='center', va='top', fontsize=LABEL, color=INK, linespacing=1.15)

    # (d) the diagnostics of one cell
    y_dec = GRID_Y + (DECODED[1] + .5) * CELL
    arrow(ax, grid2_x + grid_w + 1.0, diag_x - 1.6, y_dec, 'Decoder', (grid2_x + grid_w + out_r + diag_x) / 2)
    diagnostics(ax, diag_x, col_w, gap, HEAD_Y - 6.0, INK)
    heading(ax, diag_x - .6, 'd', 'Diagnostics')
    return fig


def make(run=None):
    return figures.save(draw(), 'overview')


def main(argv=None):
    ap = figures.cli(__doc__)
    ap.add_argument('--grid', default='4x3', help='cells across x down, e.g. 4x3, 3x3 or 4x4')
    args = ap.parse_args(argv)
    figures.setup(args)
    nx, ny = (int(v) for v in args.grid.split('x'))
    figures.save(draw(nx, ny), 'overview' if args.grid == '4x3' else 'overview_%s' % args.grid)


if __name__ == '__main__':
    main()
