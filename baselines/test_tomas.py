#!/usr/bin/env python
"""Seconds-long checks of tomas.py (TOMAS-40) and of modal.py's mam4la:

  1. projection of the three sources: number and every PartMC species' mass conserved (bulk NH4
     and NO3 included), every bin's mean TOMAS dry mass inside its bounds, mnfix a no-op on them;
  2. representative particles reproduce each bin's number, TOMAS dry mass and species masses;
  3. mnfix (vectorized, numpy) equals a line-by-line scalar port of tomas_mod.F90's MNFIX on
     perturbed states (both water options), and:
       - every bin's mean inside its bounds afterwards, and a second call changes nothing;
       - upward shifts conserve number and dry mass (and water with water='with_dry');
       - downward shifts conserve number (TOMAS's code creates mass there: reported);
       - homogeneous of degree 1 above the floors (Q vs N x);
  4. mam4la: the layout and calcsize of mam4l (bitwise), the projection rule of mam4a.

Usage: python baselines/test_tomas.py      (about 10 s; no worker pool)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag  # noqa: E402
import modal  # noqa: E402
import tomas  # noqa: E402

NB = tomas.NBINS


def mnfix_scalar(nk, mk, water='with_dry'):
    """tomas_mod.F90 MNFIX for one grid box, transcribed loop by loop (0-based bins), on the
    TOMAS part of one state: nk (40,), mk (40, 8) [SF SS ECOB ECIL OCOB OCIL DUST AW]; floors as
    tomas.mnfix (per kg of air, bins emptied)."""
    nk, mk = nk.copy(), mk.copy()
    xk, avgm = tomas.XK, tomas.AVGMASS
    nk[nk < 0] = 0
    mk[mk < 0] = 0
    for k in range(NB):
        dry = mk[k, :7].sum()
        if (nk[k] > 0 or dry > 0) and (nk[k] <= tomas.NK_FLOOR or dry < tomas.M_FLOOR):
            nk[k] = 0
            mk[k] = 0
    for k in range(NB):
        dry = mk[k, :7].sum()
        if nk[k] == 0:
            continue
        avg = dry / nk[k]
        if avg > xk[NB]:
            mshift = nk[k] * xk[NB] / 1.2
            mk[k, :] *= mshift / dry
        if avg < xk[0]:
            nk[k] = dry / (xk[0] * 1.2)
    for k in range(NB):
        dry = mk[k, :7].sum()
        avg = dry / nk[k] if nk[k] != 0 else avgm[k]
        if avg > xk[k + 1]:
            kk = k
            xnew = xk[kk + 1] / 1.1
            top = False
            while xnew <= avg:
                if kk < NB - 1:
                    kk += 1
                    xnew = xk[kk + 1] / 1.1
                else:
                    mshift = nk[k] * xk[k + 1] / 1.1
                    mk[k, :] *= mshift / dry
                    top = True
                    break
            if not top:
                xold = avgm[k]
                nshift = (dry - xold * nk[k]) / (xnew - xold)
                mshift = xnew * nshift
                nk[k] -= nshift
                nk[kk] += nshift
                js = range(8) if water == 'with_dry' else range(7)
                for j in js:
                    fj = mk[k, j] / dry
                    mk[k, j] = xold * nk[k] * fj
                    mk[kk, j] += mshift * fj
        if avg < xk[k]:
            kk = k
            xnew = xk[kk] * 1.1
            bottom = False
            while xnew >= avg:
                if kk > 0:
                    kk -= 1
                    xnew = xk[kk] * 1.1
                else:
                    nk[k] = dry / (xk[0] * 1.2)
                    bottom = True
                    break
            if not bottom:
                xold = avgm[k]
                nshift = nk[k] - dry / xold
                mshift = xnew * nshift
                nk[k] -= nshift
                nk[kk] += nshift
                for j in range(8):
                    fj = mk[k, j] / dry
                    mk[k, j] = xold * nk[k] * fj
                    mk[kk, j] += mshift * fj
    nk[nk < 0] = 0
    mk[mk < 0] = 0
    for k in range(NB):
        dry = mk[k, :7].sum()
        if (nk[k] != 0 and dry == 0) or (nk[k] == 0 and dry != 0):
            nk[k] = 0
            mk[k] = 0
    return nk, mk


def split(Q):
    A = Q[:9 * NB].reshape(9, NB, -1)
    return A[0], np.moveaxis(A[1:], 0, 1)                    # (40, n), (40, 8, n)


def report(name, ok, detail=''):
    print('%-74s %s %s' % (name, 'ok ' if ok else 'FAIL', detail))
    return bool(ok)


def main():
    good = True
    m, w = diag.sources()
    T = tomas.Tomas()
    x = T.project(m, w)
    x0 = T.project(m, w, mnfix_once=False)
    good &= report('projection: mnfix once is a no-op on the sources', np.array_equal(x, x0))
    # 1. conservation
    wn = w / w.sum(1, keepdims=True)
    true = np.einsum('pn,pnk->pk', wn, m)
    a = x[:, :9 * NB].reshape(3, 9, NB)
    good &= report('projection: number fractions add up to 1', np.allclose(a[:, 0].sum(1), 1, rtol=1e-14))
    lump = {}
    for j, s in enumerate(tomas.TSPECIES[1:], 1):
        lump.setdefault(tomas.LUMP[s], 0.0)
        lump[tomas.LUMP[s]] = lump[tomas.LUMP[s]] + a[:, j].sum(1)
    groups = {('SO4',): None, ('Cl', 'Na'): None, ('BC',): None, ('OIN',): None, ('H2O',): None}
    rel = []
    for g in groups:
        got = lump[g]
        ref = true[:, [diag.SP[s] for s in g]].sum(1)
        rel.append(np.abs(got - ref).max() / ref.max())
    carbon = a[:, tomas.J_OCOB].sum(1) + a[:, tomas.J_OCIL].sum(1)
    ref = true[:, [diag.SP[s] for s in ('OC', 'MOC') + tomas.SOA]].sum(1)
    rel.append(np.abs(carbon - ref).max() / ref.max())
    rel.append(np.abs(x[:, tomas.I_NH4] - true[:, diag.SP['NH4']]).max() / true[:, diag.SP['NH4']].max())
    rel.append(np.abs(x[:, tomas.I_NO3] - true[:, diag.SP['NO3']]).max() / true[:, diag.SP['NO3']].max())
    good &= report('projection: every species group\'s mass conserved', max(rel) < 1e-14, 'max rel %.1e' % max(rel))
    avg = T.dry_mass_per_particle(x)
    inside = np.isnan(avg) | ((avg >= tomas.XK[:-1]) & (avg <= tomas.XK[1:]))
    good &= report('projection: every bin\'s mean TOMAS dry mass inside its bounds', inside.all())
    bc = a[:, tomas.J_ECOB].sum(1) / (a[:, tomas.J_ECOB].sum(1) + a[:, tomas.J_ECIL].sum(1))
    oc_ob = a[:, tomas.J_OCOB].sum(1)
    oc = true[:, [diag.SP[s] for s in ('OC', 'MOC')]].sum(1)
    print('    hydrophobic share of BC: %s; of OC+MOC: %s (background, urban, emission)'
          % (np.round(bc, 3), np.round(oc_ob / oc, 3)))
    print('    bins used: %s' % [list(np.flatnonzero(a[i, 0] > 0)) for i in range(3)])
    # 2. representative particles
    pm, pw = T.particles(x)
    e = T.expand(x).reshape(3, NB, 1 + diag.NS)
    pm4 = pm.reshape(3, NB, -1, diag.NS)
    pw4 = pw.reshape(3, NB, -1)
    nb_ = np.abs(pw4.sum(-1) - e[..., 0]).max()
    sm = np.abs(np.einsum('pbr,pbrk->pbk', pw4, pm4) - e[..., 1:]).max() / e[..., 1:].max()
    good &= report('particles: number and every species mass per bin reproduced', nb_ < 1e-15 and sm < 1e-12,
                   'number %.1e, mass rel %.1e; single-particle fallbacks %d' % (nb_, sm, T.last_single))
    live = pw4 > 0
    bm = T.binned_mass(pm4)
    lo, hi = tomas.XK[:-1][None, :, None], tomas.XK[1:][None, :, None]
    okn = np.all(~live | ((bm >= lo * (1 - 1e-12)) & (bm <= hi * (1 + 1e-12))))
    good &= report('particles: every node\'s TOMAS dry mass inside its bin', okn)
    tot_nh4 = (pw[..., None] * pm).sum(1)[:, diag.SP['NH4']]
    good &= report('particles: bulk NH4 placed in full', np.allclose(tot_nh4, x[:, tomas.I_NH4], rtol=1e-12))
    # 3. mnfix on perturbed states
    rng = np.random.default_rng(3)
    fr = rng.dirichlet(np.ones(3) * 0.5, 400)
    xm = fr @ x
    xm = xm * rng.lognormal(0, 0.15, xm.shape)              # separate limiters' kind of departure
    xm[:20, :NB] *= rng.lognormal(0, 1.5, (20, NB))         # a few far out of their bins
    xm[20:25, 5:9] = 0.0                                    # number without... (empty bins)
    xm[25:30, NB + 30] *= 1e9                               # mass far above the top
    xm[30:35, 9] = 1e-40                                    # floors
    A = xm[:, :9 * NB].reshape(len(xm), 9, NB)              # view
    for c, target in ((35, 1e3 * tomas.XK[-1]), (36, 0.95 * tomas.XK[-1]),       # above the range / no bin above
                      (37, 0.5 * tomas.XK[0]), (38, 1.05 * tomas.XK[0])):        # below the range / no bin below
        k = 20 if target > 1e-15 else 9
        A[c, 1:8, k] *= target / (A[c, 1:8, k].sum() / A[c, 0, k])
    Q = (xm * rng.lognormal(np.log(1e9), 1.0, (len(xm), 1))).T
    for water in ('with_dry', 'tomas'):
        st = {}
        out = tomas.mnfix(Q, stats=st, water=water)
        nk, mk = split(out)
        worst = 0.0
        n0, m0 = split(Q)
        for c in range(Q.shape[1]):
            rn, rm = mnfix_scalar(n0[:, c], m0[:, :, c], water=water)
            worst = max(worst, np.abs(rn - nk[:, c]).max() / max(np.abs(rn).max(), 1e-300),
                        np.abs(rm - mk[:, :, c]).max() / max(np.abs(rm).max(), 1e-300))
        good &= report('mnfix (water=%s) == scalar transcription of MNFIX, 400 cells' % water, worst < 1e-13,
                       'max rel %.1e; events %s' % (worst, {k: v for k, v in st.items() if v}))
        # cell 38: a bin whose mean is below 1.1 Xk(1) with no bin below to take particles; the code
        # sets NK = DRYMASS / (1.2 Xk(1)), which leaves that bin's mean below its bounds (k > 0)
        keep = np.arange(Q.shape[1]) != 38
        av = T.dry_mass_per_particle(out.T)
        ins = np.isnan(av) | ((av >= tomas.XK[:-1] * (1 - 1e-12)) & (av <= tomas.XK[1:] * (1 + 1e-12)))
        good &= report('mnfix (%s): every bin\'s mean inside its bounds afterwards (but cell 38)' % water,
                       ins[keep].all(), 'outside %d; cell 38: bin 9 mean %.2f Xk(1)'
                       % ((~ins[keep]).sum(), av[38, 9] / tomas.XK[0]))
        again = tomas.mnfix(out, water=water)
        d2 = np.abs(again - out)[:, keep].max() / np.abs(out).max()
        good &= report('mnfix (%s): idempotent (but cell 38)' % water, d2 < 1e-14, '%.1e' % d2)
        assert np.array_equal(out[9 * NB:], Q[9 * NB:])
    # conservation per event type, on single-bin cases
    k = 20
    base = np.zeros(tomas.D)
    base[k] = 1e8                                           # NK in bin k
    comp = np.array([0.4, 0.05, 0.05, 0.1, 0.1, 0.2, 0.1])
    for label, avg_mass, expect in (('upward shift', tomas.XK[k + 1] * 1.3, 'number, dry mass, water'),
                                    ('downward shift', tomas.XK[k] * 0.8, 'number')):
        q = base.copy()
        dry = 1e8 * avg_mass
        for j in range(7):
            q[(1 + j) * NB + k] = dry * comp[j]
        q[8 * NB + k] = 0.5 * dry
        st = {}
        o = tomas.mnfix(q[:, None], stats=st)[:, 0]
        A0, A1 = q[:9 * NB].reshape(9, NB), o[:9 * NB].reshape(9, NB)
        dn = A1[0].sum() / A0[0].sum() - 1
        dd = A1[1:8].sum() / A0[1:8].sum() - 1
        dw = A1[8].sum() / A0[8].sum() - 1
        print('    %s (%s): number %+.1e, dry mass %+.2e, water %+.2e; events %s'
              % (label, expect, dn, dd, dw, {kk: v for kk, v in st.items() if v}))
        if label == 'upward shift':
            good &= report('mnfix: upward shift conserves number, dry mass and water',
                           max(abs(dn), abs(dd), abs(dw)) < 1e-14)
        else:
            good &= report('mnfix: downward shift conserves number (creates mass, as the code)', abs(dn) < 1e-14,
                           'mass created %.1f %% of the bin\'s' % (100 * dd))
    # homogeneity above the floors
    s = 7.3
    h = np.abs(tomas.mnfix(Q * s) - s * tomas.mnfix(Q)).max() / np.abs(s * tomas.mnfix(Q)).max()
    good &= report('mnfix: homogeneous of degree 1 (Q scaled by 7.3)', h < 1e-14, '%.1e' % h)
    # 4. mam4la
    La, Ll, L4a = modal.Modal('mam4la'), modal.Modal('mam4l'), modal.Modal('mam4a')
    good &= report('mam4la: the layout of mam4l', La.comps == Ll.comps and La.d == 25)
    xa = La.project(m, w)
    Qa = (fr[:50] @ xa * 1e9).T
    same = all(np.array_equal(modal.calcsize(Qa, dt, 'mam4la'), modal.calcsize(Qa, dt, 'mam4l')) for dt in (20.0, 86400.0))
    good &= report('mam4la: calcsize(mam4la) == calcsize(mam4l) bitwise', same)
    good &= report('mam4la: the mode assignment of mam4a', np.array_equal(La.assign(m), L4a.assign(m)))
    nums = xa[:, La.i_num]
    dgn = La.dgn(xa) * 1e9
    print('    mam4la number per mode (acc ait coa pca): %s' % np.round(nums, 3).tolist())
    print('    mam4la dgn (nm): %s' % np.round(dgn, 1).tolist())
    raw = La.project(m, w, calcsize_dt=0)
    good &= report('mam4la: the one calcsize at projection changes the sources by < 1e-3',
                   np.abs(xa - raw).max() / np.abs(raw).max() < 1e-3)
    print('ALL OK' if good else 'SOME CHECKS FAILED')
    return good


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
