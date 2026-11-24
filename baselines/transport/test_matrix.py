#!/usr/bin/env python
"""Checks of baselines/matrix.py and of run_baselines.py's --matrix-adjust hook (numpy, ~30 s).

A. matrix.py on hand-made particles that fill every mechanism-1 population:
   1. the projection puts each particle where MATRIX's definitions say, and conserves number and every
      MATRIX species (the lumped PartMC species);
   2. adjust conserves every MATRIX species and the bulk totals, is homogeneous of degree one, moves whole
      populations (DD1 -> DS1, BC1 -> BC2 -> BC3) when their coating exceeds MIMR, sets the sea-salt
      numbers from the NaCl mass, moves at most half of AKK to ACC per call; a second call moves no
      population again (class transfers and sea-salt numbers are idempotent);
   3. mixing alone crosses a threshold: a dry source's BC1 mixed with a wet source's sulfate;
   4. the representative particles reproduce each population's number and dry volume, and every
      species' mass (NO3, NH4 by sulfate, H2O by sulfate + sea salt);
   5. the real sources (data/zsrc_baselines.npz matrix1_z): adjust is a no-op (populations it acts on
      are empty), and on random mixtures of them too.
B. run_baselines.py on transport/test_baselines.py's tiny WRF-like case:
   6. real sources, --matrix-adjust matrix1 --nocs matrix1: sep:matrix1 == sep:matrix1_nocs bitwise
      (adjust has nothing to act on), and the _nocs copy == a run without the hook, bitwise;
   7. hand-made sources that populate BC1, DD1, AKK and SSA next to a wet source: the hook acts
      (events > 0), the state stays finite and nonnegative, species are conserved, check_run.py passes.

Usage: python test_matrix.py [--dir TMPDIR] [--steps 90]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import diag  # noqa: E402
import matrix  # noqa: E402
import run_baselines as RB  # noqa: E402
import test_baselines as TB  # noqa: E402

ZSRC = HERE.parent / 'data' / 'zsrc_baselines.npz'
LAYOUT = HERE.parent / 'data' / 'state_layout.json'
SP = diag.SP


def particle(**kg):
    m = np.zeros(diag.NS)
    for k, v in kg.items():
        m[SP[k]] = v
    return m


def vol_mass(d_um, rho=1700.0):
    return rho * np.pi / 6 * (d_um * 1e-6) ** 3


def handmade():
    """Particles (n, 15) and the population mechanism 1 must give each."""
    b = vol_mass(0.1)
    P = [(particle(SO4=0.3 * vol_mass(0.03), NH4=0.1 * vol_mass(0.03)), 'AKK'),
         (particle(SO4=vol_mass(0.3), H2O=0.2 * vol_mass(0.3)), 'ACC'),
         (particle(OIN=vol_mass(0.5), SO4=0.01 * vol_mass(0.5)), 'DD1'),
         (particle(OIN=vol_mass(0.5), SO4=0.2 * vol_mass(0.5)), 'DS1'),
         (particle(OIN=vol_mass(5.0), SO4=0.01 * vol_mass(5.0)), 'DD2'),
         (particle(OIN=vol_mass(5.0), NO3=0.3 * vol_mass(5.0)), 'DS2'),
         (particle(Na=0.4 * vol_mass(0.5), Cl=0.6 * vol_mass(0.5)), 'SSA'),
         (particle(Na=0.4 * vol_mass(3.0), Cl=0.6 * vol_mass(3.0), SO4=0.1 * vol_mass(3.0)), 'SSC'),
         (particle(OC=vol_mass(0.1), API1=vol_mass(0.1), SO4=vol_mass(0.05)), 'OCC'),
         (particle(BC=b, SO4=0.02 * b), 'BC1'),
         (particle(BC=b, SO4=0.1 * b), 'BC2'),
         (particle(BC=b, SO4=0.1 * b, H2O=0.3 * b), 'BC3'),
         (particle(BC=b, OIN=b, SO4=0.1 * b), 'DBC'),
         (particle(BC=b, ARO1=b, SO4=0.1 * b), 'BOC'),
         (particle(BC=b, OC=b, OIN=b), 'MXX'),
         (particle(Na=b, OC=b), 'MXX'),
         (particle(NH4=0.3 * vol_mass(0.3), Cl=0.5 * vol_mass(0.3), SO4=0.2 * vol_mass(0.3)), 'ACC')]  # NH4Cl: no sea salt
    return np.stack([p for p, _ in P]), [q for _, q in P]


def part_a(check):
    R = matrix.Matrix('matrix1')
    m, expect = handmade()
    got = [R.pops[i] for i in R.assign(m)]
    check('1. every hand-made particle in its MATRIX population', got == expect,
          str([(g, e) for g, e in zip(got, expect) if g != e]))
    w = np.linspace(1, 2, len(m))
    x0 = R.project(m[None], w[None], calcsize_dt=0)[0]
    wn = w / w.sum()
    L = matrix.lumped_masses(m)
    err = max(abs(x0[[k for k, c in enumerate(R.comps) if c[1] == 'mass' and c[2] == s]].sum() - (wn * L[s]).sum())
              / (wn * L[s]).sum() for s in matrix.MSPC + matrix.BULK)
    check('1. projection conserves each MATRIX species (lumped PartMC masses)', err < 1e-14, '%.1e' % err)
    check('1. projection conserves number', abs(x0[R.i_num].sum() - 1) < 1e-14)
    tot_pm = (wn[:, None] * m).sum(0)
    lump_tot = sum((wn * L[s]).sum() for s in matrix.MSPC + matrix.BULK)
    check('1. lumping keeps the total mass', abs(lump_tot - tot_pm.sum()) / tot_pm.sum() < 1e-14)

    st = {}
    x1 = matrix.adjust(x0[:, None], None, R, stats=st)[:, 0]
    err = max(abs(x1[[k for k, c in enumerate(R.comps) if c[1] == 'mass' and c[2] == s]].sum()
                  - x0[[k for k, c in enumerate(R.comps) if c[1] == 'mass' and c[2] == s]].sum())
              / max(x0[[k for k, c in enumerate(R.comps) if c[1] == 'mass' and c[2] == s]].sum(), 1e-300)
              for s in matrix.MSPC + matrix.BULK)
    check('2. adjust conserves every MATRIX species', err < 1e-14, '%.1e' % err)
    n = lambda x, p: x[R.comp(p, 'number')]
    check('2. adjust moved DD1 -> DS1 and DD2 -> DS2 (OPTOT > 1 from the bulk NO3/NH4/H2O)',
          n(x1, 'DD1') == 0 and n(x1, 'DD2') == 0 and n(x1, 'DS1') > n(x0, 'DS1'), str(st))
    check('2. adjust moved BC1 -> BC2 -> BC3', n(x1, 'BC1') == 0 and n(x1, 'BC2') == 0
          and np.isclose(n(x1, 'BC3'), n(x0, 'BC1') + n(x0, 'BC2') + n(x0, 'BC3')), str(st))
    for p in ('SSA', 'SSC'):
        check('2. %s number = NaCl mass / MATRIX\'s mass per emitted particle' % p,
              np.isclose(n(x1, p), x1[R.comp(p, 'SEAS')] / R.ss_mpp[p], rtol=1e-14))
    f = 1 - n(x1, 'AKK') / n(x0, 'AKK')
    check('2. AKK -> ACC moves a fraction in [0, 0.5] of AKK', 0 <= f <= 0.5, 'fraction %.3g' % f)
    c = 7.3e9
    xc = matrix.adjust(c * x0[:, None], None, R)[:, 0]
    check('2. adjust homogeneous of degree one', np.allclose(xc, c * x1, rtol=1e-13, atol=0))
    st2 = {}
    x2 = matrix.adjust(x1[:, None], None, R, stats=st2, do_imtr=False)[:, 0]
    check('2. second call: no class transfer, sea-salt numbers unchanged', np.array_equal(x2, x1)
          and all(int(v) == 0 for k, v in st2.items()), str(st2))

    # 3. mixing alone: a dry source with BC1 (2 % sulfate on BC), a wet source with sulfate + water
    dry = R.project(particle(BC=vol_mass(0.1), SO4=0.02 * vol_mass(0.1))[None, None], np.ones((1, 1)))[0]
    wet = R.project(particle(SO4=vol_mass(0.3), H2O=10 * vol_mass(0.3))[None, None], np.ones((1, 1)))[0]
    check('3. each source alone: BC1 stays BC1', dry[R.comp('BC1', 'number')] == 1)
    mix = 0.5 * dry + 0.5 * wet
    st3 = {}
    xm = matrix.adjust(mix[:, None], None, R, stats=st3)[:, 0]
    check('3. their 50/50 mixture: BC1 moves (the cell\'s bulk water per sulfate enters the test)',
          xm[R.comp('BC1', 'number')] == 0, str({k: int(v) for k, v in st3.items()}))

    # 4. representative particles
    X = np.stack([x0, x1, mix])
    mass, num = R.population_masses(X)
    mm, ww = R.particles(X)
    ok = True
    off = 0
    for i, p in enumerate(R.pops):
        mi, wi = matrix.lognormal_nodes(mass[:, i], num[:, i], R.sigma[i], R.dg_default[i])
        k = mi.shape[1]
        sl = slice(off, off + k)
        off += k
        nn = ww[:, sl].sum(1)
        sp = np.einsum('pr,prk->pk', ww[:, sl], mm[:, sl])
        ok &= np.allclose(nn, num[:, i], rtol=1e-12, atol=0)
        ok &= np.allclose(sp, mass[:, i], rtol=1e-12, atol=1e-40)
    check('4. representative particles: number and every species mass of each population', bool(ok))
    tot = np.einsum('pr,prk->pk', ww, mm).sum(1)
    want = sum(X[:, [k for k, c in enumerate(R.comps) if c[1] == 'mass' and c[2] == s]].sum(1)
               for s in matrix.MSPC + matrix.BULK)
    check('4. diagnosed total mass = carried total mass (NO3, NH4, H2O shared out)',
          np.allclose(tot, want, rtol=1e-12), str(tot / want))

    # 5. the real sources
    with np.load(ZSRC) as f:
        z = f['matrix1_z']
        zr = f['matrix1_raw_z']
    st5 = {}
    za = matrix.adjust(z.T, None, R, stats=st5).T
    check('5. real sources: adjust is a no-op (and the stored states are after it)',
          np.array_equal(za, z) and np.array_equal(z, zr), str({k: int(v) for k, v in st5.items()}))
    rng = np.random.default_rng(0)
    Q = (rng.dirichlet(np.ones(3) * 0.5, 2000) @ z) * rng.lognormal(0, 0.05, (2000, R.d))
    st5b = {}
    Qa = matrix.adjust(Q.T, None, R, stats=st5b).T
    check('5. real sources, 2000 perturbed mixtures (5 %% noise per tracer): no-op', np.array_equal(Qa, Q),
          str({k: int(v) for k, v in st5b.items()}))
    occupied = [R.pops[i] for i in range(R.np) if (z[:, R.i_num[i]] > 0).any()]
    check('5. real sources occupy only BC3, BOC, MXX', occupied == ['BC3', 'BOC', 'MXX'], str(occupied))


def stress_sources(path):
    """Hand-made source states: background (BC1, DD1, AKK, ACC, SSA, a bit of OC), urban (wet sulfate),
    emission (OC + BC2): the hook has something to act on once they mix."""
    R = matrix.Matrix('matrix1')
    b = vol_mass(0.1)
    bg = np.stack([particle(BC=b, SO4=0.02 * b), particle(OIN=vol_mass(0.5), SO4=0.01 * vol_mass(0.5)),
                   particle(SO4=vol_mass(0.04)), particle(SO4=vol_mass(0.2)),
                   particle(Na=0.4 * vol_mass(0.5), Cl=0.6 * vol_mass(0.5)), particle(OC=b)])
    ur = np.stack([particle(SO4=vol_mass(0.2), H2O=8 * vol_mass(0.2)), particle(SO4=vol_mass(0.08), H2O=vol_mass(0.08))])
    em = np.stack([particle(OC=b, SO4=0.1 * b), particle(BC=b, SO4=0.1 * b)])
    z = []
    for parts, w in ((bg, [3, 1, 4, 2, 0.1, 1]), (ur, [1, 2]), (em, [2, 1])):
        z.append(R.project(parts[None], np.asarray(w, float)[None], calcsize_dt=0)[0])
    z = np.stack(z)
    np.savez(path, matrix1_z=z)
    return z


def part_b(check, d, steps):
    TB.write_inputs(d, steps=steps)
    T = TB.TRANSPORT
    base = ['--steps', str(steps), '--fluxes', str(d / 'fluxes.nc'), '--orig', str(d / 'orig.nc'),
            '--pmc', str(d / 'pmc'), '--transport-code', str(T), '--modal-path', str(HERE.parent)]
    RB.main(['--out', str(d / 'adj'), '--zsrc', str(ZSRC), '--models', 'matrix1', '--nocs', 'matrix1',
             '--matrix-adjust', 'matrix1', '--chunk', '20'] + base)
    RB.main(['--out', str(d / 'plain'), '--zsrc', str(ZSRC), '--models', 'matrix1', '--chunk', '1000'] + base)
    ld = lambda p: dict(np.load(p, allow_pickle=False))
    A, P = ld(d / 'adj' / 'surface.npz'), ld(d / 'plain' / 'surface.npz')
    check('6. real sources: sep:matrix1 == sep:matrix1_nocs bitwise', np.array_equal(A['sep:matrix1'], A['sep:matrix1_nocs']))
    check('6. sep:matrix1_nocs == a run without the hook, bitwise', np.array_equal(A['sep:matrix1_nocs'], P['sep:matrix1']))
    st = json.loads(str(A['settings']))
    check('6. hook called every step', st['calcsize_calls'] == steps and st['matrix_adjust'] == ['matrix1'],
          str(st.get('matrix_adjust_stats')))

    zs = d / 'zsrc_stress.npz'
    stress_sources(zs)
    RB.main(['--out', str(d / 'stress'), '--zsrc', str(zs), '--models', 'matrix1', '--nocs', 'matrix1',
             '--matrix-adjust', 'matrix1', '--chunk', '30'] + base)
    S = ld(d / 'stress' / 'surface.npz')
    st = json.loads(str(S['settings']))
    ev = st['matrix_adjust_stats']['matrix1']
    check('7. hand-made sources: the hook acts (class transfers, AKK -> ACC)',
          ev.get('BC1->BC2', 0) > 0 and ev.get('AKK->ACC', 0) > 0, str(ev))
    check('7. adjusted state differs from _nocs', not np.array_equal(S['sep:matrix1'], S['sep:matrix1_nocs']))
    for k in ('sep:matrix1', 'sep:matrix1_nocs'):
        v = S[k]
        check('7. %s finite and nonnegative' % k, np.isfinite(v).all() and v.min() >= -1e-10 * np.abs(v).max(),
              'min %.2e' % v.min())
    import check_run
    res = check_run.check(d / 'stress', layout=None, zsrc=zs, modal_path=HERE.parent)
    print('\n'.join('    ' + m for m in res['messages']))
    check('7. check_run.py passes (species conserved by the hook)', res['ok'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None)
    ap.add_argument('--steps', type=int, default=90)
    a = ap.parse_args()
    d = Path(a.dir or tempfile.mkdtemp(prefix='test_matrix_'))
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print('%-92s %s %s' % (name, 'ok ' if cond else 'FAIL', detail), flush=True)
        ok &= bool(cond)

    part_a(check)
    part_b(check, d, a.steps)
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    print('test files in', d)
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
