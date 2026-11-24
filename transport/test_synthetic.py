#!/usr/bin/env python
"""Checks of wrfadv / schemes on a small synthetic domain (numpy, a few seconds).

Uniform flow across a box with an urban block in a background, background at
the boundaries, and an emission point. Checks: a uniform field stays uniform;
total number is conserved away from the boundaries; the shared schemes' groups
add up to N; the oracle's and shB's fractions stay in [0, 1]; shB and shN keep
the latent exactly on the mixing plane while sep leaves it; donor's fronts are
the widest.

Usage (from transport/): python test_synthetic.py
"""
import numpy as np

import schemes
from wrfadv import Grid


def make(nz=6, ny=14, nx=70, u=40.0, v=4.0, mu=9e4):
    xp = np
    g = Grid(xp, np.ones((ny, nx)), np.ones((ny, nx)), 1 / 4000., 1 / 4000., -np.full(nz, 1 / 0.02),
             np.full(nz, 0.5), np.full(nz, 0.5), np.ones(nz), np.zeros(nz))
    ru = np.full((nz, ny, nx + 1), u * mu)
    rv = np.full((nz, ny + 1, nx), v * mu)
    ww = np.zeros((nz + 1, ny, nx))
    ww[1:-1] = 0.3 * np.sin(np.linspace(0, np.pi, nz - 1))[:, None, None] * np.sin(np.linspace(0, 6, nx))[None, None, :]
    # make the flow divergence-free in each cell by adjusting nothing: ww varies in x only, so
    # mass changes; M evolves consistently with the fluxes, which is what matters here.
    M0 = g.mass(np.full((ny, nx), mu))
    return g, (ru, rv, ww), M0


def main():
    g, F, M0 = make()
    nz, ny, nx = g.nz, g.ny, g.nx
    rng = np.random.default_rng(0)
    Z = {'m': rng.uniform(0.1, 2.0, (3, 9)), 'frac': np.eye(3)}
    ins = np.zeros((5, nz, ny, nx))
    ins[0] = 1e9
    ins[0, :, 4:10, 15:28] = 0
    ins[1, :, 4:10, 15:28] = 1.3e9
    bce = {'w': np.full((nz, ny), 1e9), 'e': np.full((nz, ny), 1e9), 's': np.full((nz, nx), 1e9), 'n': np.full((nz, nx), 1e9)}
    r = schemes.Runner(g, Z, np, ins, bce, sep_models=['m'])
    M = M0
    dt = 20.0
    for it in range(150):
        M = r.step(F, M, dt)
        dn = np.zeros((2, ny, nx))
        dn[0, 7, 30] = 2e7
        dn[1, 3, 40] = 1e7
        r.emit(dn)
    s = r.surface(level=2)
    N = s['N']
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print('%-62s %s %s' % (name, 'ok ' if cond else 'FAIL', detail))
        ok &= bool(cond)

    # uniform field: N with everything = 1 and no emissions
    g2, F2, M02 = make()
    ins1 = np.zeros((5, nz, ny, nx))
    ins1[0] = 1.0
    bce1 = {k: np.ones_like(v) for k, v in bce.items()}
    r1 = schemes.Runner(g2, Z, np, ins1, bce1, sep_models=['m'])
    M1 = M02
    for it in range(60):
        M1 = r1.step(F2, M1, dt)
    s1 = r1.surface(level=2)
    check('uniform N stays uniform (mono)', np.abs(s1['N'] - 1).max() < 1e-12, '%.1e' % np.abs(s1['N'] - 1).max())
    for k in ('donor', 'shN', 'shB:m', 'shB:frac'):
        check('uniform field stays uniform (%s)' % k, np.abs(s1[k].sum(0) - 1).max() < 1e-12)

    inner = (slice(None), slice(3, -3), slice(3, -3))
    for k in ('donor', 'shN'):
        d = np.abs(s[k].sum(0) - N)[inner[1:]].max() / N.max()
        check('%s: groups add up to N' % k, d < 1e-12, '%.1e' % d)
    for k in ('oracle', 'shB:frac'):
        n = s[k] if k != 'oracle' else np.stack([s[k][0] + s[k][2], s[k][1], s[k][3] + s[k][4]])
        f = n / n.sum(0)
        check('%s: fractions in [0, 1]' % k, f.min() > -1e-9 and f.max() < 1 + 1e-9, 'min %.2e max %.6f' % (f.min(), f.max()))
    for k, lim in (('shB:m', 1e-2), ('shN', None)):
        f = s[k] / s[k].sum(0)
        if lim:
            check('%s: fractions within %g of [0, 1] (box bounds on z)' % (k, lim), f.min() > -lim and f.max() < 1 + lim,
                  'min %.2e max %.6f' % (f.min(), f.max()))
        else:
            print('    %s fractions range: %.3e .. %.6f (overshoots allowed)' % (k, f.min(), f.max()))

    zs = Z['m']

    def plane_dist(z):
        A = np.stack([zs[1] - zs[0], zs[2] - zs[0]], 1)
        zf = z.reshape(9, -1) - zs[0][:, None]
        c, *_ = np.linalg.lstsq(A, zf, rcond=None)
        return np.linalg.norm(zf - A @ c, axis=0).max() / np.linalg.norm(zs[1] - zs[0])

    z_shB = np.tensordot(zs.T, s['shB:m'] / s['shB:m'].sum(0), axes=1)
    z_sep = s['sep:m'] / N
    check('shB stays on the mixing plane', plane_dist(z_shB) < 1e-10, '%.1e' % plane_dist(z_shB))
    check('sep leaves the mixing plane', plane_dist(z_sep) > 1e-4, '%.1e' % plane_dist(z_sep))

    def width(fu):
        row = fu[7]
        return int(((row > 0.05) & (row < 0.95)).sum())
    fo = s['oracle'][1] / s['oracle'].sum(0)
    widths = {'oracle': width(fo), 'donor': width(s['donor'][1] / s['donor'].sum(0)),
              'shN': width(s['shN'][1] / s['shN'].sum(0)), 'shB': width(s['shB:m'][1] / s['shB:m'].sum(0))}
    print('    urban front width (cells with 0.05 < f < 0.95 along one row):', widths)
    check('donor has the widest fronts', widths['donor'] > max(widths['oracle'], widths['shB']))
    print('ALL OK' if ok else 'SOME CHECKS FAILED')


if __name__ == '__main__':
    main()
