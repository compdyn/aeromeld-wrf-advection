#!/usr/bin/env python
"""Replay the CARES case's transport for the sectional and modal baselines (12 h, WRF's fluxes).

A wrapper of the paper's replay driver (transport/run.py, unchanged and imported
from --transport-code): the same fluxes, IC/BC/emissions, steps, boundary
conditions and limiter, for per-particle source state vectors (3, d) of the
baseline representations instead of latents. Schemes:

  oracle        each WRF-PartMC particle class its own tracer and limiter
                (per-source transport; identical to the paper's runs, a check)
  sep:<model>   N and every component of Q = N x with its own monotonic limiter,
                which is what WRF-Chem does for MOSAIC and MAM tracers

For modal models listed in --calcsize, MAM's calcsize (number adjustment into
the diameter bounds, Aitken <-> accumulation transfer) is applied to that
model's extensive tracers Q every --calcsize-every steps, after the transport
and the emissions of the step (WRF-Chem: advection in solve_em, then
emissions and aerosol chemistry in chem_driver). --nocs <models> also carries
each listed model a second time as <model>_nocs (same source vectors, no
calcsize), so the limiter and the adjustments can be told apart.

--mnfix <models> (added 2026-11-14): TOMAS models (baselines/tomas.py, e.g. tomas40) get
TOMAS's MNFIX at the same place instead (bins whose mean dry mass per particle has left the
bin's bounds are fixed; GEOS-Chem applies it at each chemistry step, after transport). Its
cumulative domain-total changes go to diag.npz as mnfix_delta:<model>, its event counts to
settings['mnfix_stats']. Without --mnfix nothing changes.

--matrix-adjust <models> (added 2026-11-14): MATRIX models (baselines/matrix.py, e.g. matrix1) get
MATRIX's state-driven operations (matrix.adjust: sea-salt number from mass, the DD1->DS1,
DD2->DS2, BC1->BC2, BC2->BC3 class transfers, the AKK->ACC transfer) at the same place, every
--calcsize-every steps. Cumulative domain-total changes go to diag.npz as
matrix_adjust_delta:<model>, the counts of cells where each operation acted to
settings['matrix_adjust_stats']. Without --matrix-adjust nothing changes.

Note: calcsize changes the modes' numbers, so after it the model's number is
the sum of its mode-number components of Q, no longer the bundle's N (index 0,
WRF's monotonic N, untouched). Without calcsize Q's mode numbers add up to N
up to the limiters.

The only change to the numerics: the independently limited tracers are advanced
in chunks of at most --chunk tracers (GPU memory). Each tracer's update depends on that
tracer alone, so chunking gives the same numbers as one bundle (checked in
test_baselines.py, bitwise on numpy).

Writes <out>/surface.npz (run.py's format: time_s, N, oracle, sep:<model>,
M_offline_over_wrf, settings), the extracts surface_30min.npz (every 30 min)
and surface_end.npz (0, 4 and 12 h) as transport/rep12_array.sbatch writes
them, and diag.npz: per output time and tracer the domain total (sum of
M q d(eta) / (msftx msfty) over the interior, all levels), the minimum, the
maximum and the number of non-finite values, and for calcsize models the cumulative domain
total of calcsize's changes.

Usage: python run_baselines.py --out DIR [--gpu] --zsrc zsrc_baselines.npz --models sec8,mam4,mam4l
           [--nocs mam4,mam4l] [--calcsize mam4,mam4l] [--calcsize-every 1] [--steps 2160]
           [--fluxes F] [--orig O] [--pmc DIR] [--transport-code DIR] [--modal-path DIR] [--chunk 72]

The calcsize is baselines/modal.py's calcsize(Q, dt, model=, xp=) (E3SM MAM4's
modal_aero_calcsize ported by the representation agent): number relaxed toward the
diameter bounds with fracadj = dt / 1 day, then the Aitken <-> accumulation transfer.
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TRANSPORT = HERE.parents[1] / 'transport'          # the paper's transport/ (local checkout)
DEFAULT_MODAL = HERE.parent                                 # baselines/ (modal.py)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def import_paper(code_dir):
    """The paper's run.py, schemes.py and wrfadv.py, imported unchanged from code_dir."""
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    R = importlib.import_module('run')
    S = importlib.import_module('schemes')
    W = importlib.import_module('wrfadv')
    for m in (R, S, W):
        if Path(m.__file__).resolve().parent != Path(code_dir):
            raise ImportError('%s imported from %s, not %s' % (m.__name__, m.__file__, code_dir))
    return R, S, W


def make_chunked_runner(S, W):
    """schemes.Runner whose independently limited tracers (N, oracle, sep) are
    advanced in chunks; only for runs without the shared schemes."""

    class ChunkedRunner(S.Runner):
        def __init__(self, *a, chunk=72, **kw):
            super().__init__(*a, **kw)
            if self.q_sh:
                raise ValueError('ChunkedRunner carries only oracle and sep (no shared schemes)')
            T = self.q_mono.shape[0]
            n = -(-T // max(1, int(chunk)))                     # chunks of at most `chunk`, balanced
            size = -(-T // n)
            self.chunks = [slice(i, min(i + size, T)) for i in range(0, T, size)]
            for i, sl in enumerate(self.chunks):            # each chunk's boundary weights, for Runner.bc
                self.bc_w['mono#%d' % i] = self.bc_w['mono'][sl]

        def step(self, F, M0, dt):
            """Runner.step's independent-limiter part, chunk by chunk, in place."""
            g = self.g
            M = [M0] + S.stage_masses(g, M0, F, dt)
            for i, sl in enumerate(self.chunks):
                key = 'mono#%d' % i
                q0 = self.q_mono[sl]
                q2 = self._stages12(q0, key, F, M, dt)
                FL = W.lo_fluxes(g, q0, *F)
                A = W.sub(W.hi_fluxes(g, q2, *F), FL)
                del q2
                coef = W.mono_coef(g, q0, M[0], M[3], dt, FL, A, *F)
                q3 = W.update(g, q0, M[0], M[3], dt, W.add(FL, W.apply_coef(A, coef)))
                del FL, A, coef
                self.bc(q3, key, F)
                self.q_mono[sl] = q3                          # chunk i reads only chunk i
                del q3, q0
            return M[3]

    return ChunkedRunner


# --- calcsize ---------------------------------------------------------------------------------------

def import_modal(modal_path):
    """baselines/modal.py, imported from modal_path (on Perlmutter: the code directory, with a
    constants-only diag.py next to it, see make_diag_shim.py)."""
    modal_path = str(Path(modal_path).resolve())
    if modal_path not in sys.path:
        sys.path.insert(0, modal_path)
    modal = importlib.import_module('modal')
    if Path(modal.__file__).resolve().parent != Path(modal_path):
        raise ImportError('modal imported from %s, not %s' % (modal.__file__, modal_path))
    return modal


def load_calcsize(model, modal_path, xp):
    """A function (Q (d, ...), dt) -> new Q for one modal model, and a description:
    modal.calcsize(Q, dt, model=model, xp=xp) (MAM4's number adjustment and Aitken <->
    accumulation transfer, E3SM modal_aero_calcsize.F90; returns a new array)."""
    modal = import_modal(modal_path)
    d = modal.Modal(model).d

    def f(Q, dt):
        if Q.shape[0] != d:
            raise ValueError('%s: calcsize expects %d tracers, got %d' % (model, d, Q.shape[0]))
        return modal.calcsize(Q, dt, model=model, xp=xp)
    return f, 'modal.calcsize(model=%s) from %s (md5 %s; diag %s md5 %s)' % (
        model, modal.__file__, md5(modal.__file__), modal.diag.__file__, md5(modal.diag.__file__))


def load_mnfix(model, modal_path, xp, stats):
    """--mnfix (added 2026-11-14 for TOMAS-40): a function (Q (d, ...), dt) -> Q applying
    baselines/tomas.py's mnfix (GEOS-Chem TOMAS's MNFIX) in place on the copy apply_calcsize
    hands it; `stats` accumulates its event counts. Imported from modal_path, like modal.py."""
    import_modal(modal_path)
    tomas = importlib.import_module('tomas')
    if Path(tomas.__file__).resolve().parent != Path(modal_path).resolve():
        raise ImportError('tomas imported from %s, not %s' % (tomas.__file__, modal_path))
    d = tomas.Tomas(model).d

    def f(Q, dt):
        if Q.shape[0] != d:
            raise ValueError('%s: mnfix expects %d tracers, got %d' % (model, d, Q.shape[0]))
        return tomas.mnfix(Q, xp=xp, inplace=True, stats=stats)
    return f, 'tomas.mnfix(model=%s) from %s (md5 %s)' % (model, tomas.__file__, md5(tomas.__file__))


def load_matrix_adjust(model, modal_path, xp, stats):
    """--matrix-adjust (added 2026-11-14 for MATRIX): a function (Q (d, ...), dt) -> Q applying
    baselines/matrix.py's adjust (MATRIX's state-driven operations); `stats` accumulates the counts of
    cells where each operation acted. Imported from modal_path, like modal.py."""
    import_modal(modal_path)
    matrix = importlib.import_module('matrix')
    if Path(matrix.__file__).resolve().parent != Path(modal_path).resolve():
        raise ImportError('matrix imported from %s, not %s' % (matrix.__file__, modal_path))
    M = matrix.Matrix(model)

    def f(Q, dt):
        if Q.shape[0] != M.d:
            raise ValueError('%s: matrix.adjust expects %d tracers, got %d' % (model, M.d, Q.shape[0]))
        return matrix.adjust(Q, dt, M, xp=xp, stats=stats)
    return f, 'matrix.adjust(model=%s) from %s (md5 %s)' % (model, matrix.__file__, md5(matrix.__file__))


# --- driver ----------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--zsrc', required=True, help="the baselines' per-particle source states (<model>_z (3, d))")
    ap.add_argument('--models', default='sec8,mam4,mam4l')
    ap.add_argument('--nocs', default='', help='models also carried as <model>_nocs, without calcsize')
    ap.add_argument('--calcsize', default='', help='models to which calcsize is applied')
    ap.add_argument('--calcsize-every', type=int, default=1, help='steps between calcsize calls (dt = this x 20 s)')
    ap.add_argument('--mnfix', default='', help='TOMAS models (baselines/tomas.py) to which mnfix is applied '
                                                'where calcsize would be (every --calcsize-every steps)')
    ap.add_argument('--matrix-adjust', default='', help='MATRIX models (baselines/matrix.py) to which matrix.adjust '
                                                        'is applied where calcsize would be (every --calcsize-every steps)')
    ap.add_argument('--steps', type=int, default=2160)
    ap.add_argument('--schemes', default='oracle,sep')
    ap.add_argument('--chunk', type=int, default=72, help='at most this many tracers advanced together (GPU memory)')
    ap.add_argument('--transport-code', default=str(DEFAULT_TRANSPORT), help="the paper's transport/ code")
    ap.add_argument('--modal-path', default=str(DEFAULT_MODAL), help='directory holding modal.py')
    ap.add_argument('--fluxes', default=None, help="default: run.py's FLUXES")
    ap.add_argument('--orig', default=None, help="default: run.py's ORIG")
    ap.add_argument('--pmc', default=None, help="WRF-PartMC control files; default: run.py's PMC")
    ap.add_argument('--levels', default='', help='extra model levels (0-based) written as level_<k>.npz')
    args = ap.parse_args(argv)

    R, S, W = import_paper(args.transport_code)
    netCDF4 = R.netCDF4
    if args.gpu:
        import cupy as xp
    else:
        xp = np
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    DT, STEPS_PER_FRAME, OUT_EVERY = R.DT, R.STEPS_PER_FRAME, R.OUT_EVERY
    fluxes = args.fluxes or str(R.FLUXES)
    orig_path = args.orig or str(R.ORIG)
    pmc = Path(args.pmc) if args.pmc else R.PMC

    sch = tuple(args.schemes.split(','))
    if not set(sch) <= {'oracle', 'sep'} or 'sep' not in sch:
        raise ValueError('schemes must be sep, optionally with oracle')
    models = [m for m in args.models.split(',') if m]
    nocs = [m for m in args.nocs.split(',') if m]
    cs_models = [m for m in args.calcsize.split(',') if m]
    mnfix_models = [m for m in args.mnfix.split(',') if m]
    madj_models = [m for m in args.matrix_adjust.split(',') if m]
    for m in madj_models:
        if m not in models:
            raise ValueError('%s is not in --models' % m)
    for m in nocs + cs_models + mnfix_models:
        if m not in models:
            raise ValueError('%s is not in --models' % m)

    fx = netCDF4.Dataset(fluxes)
    orig = netCDF4.Dataset(orig_path)
    if len(R.times_of(orig)) < args.steps // OUT_EVERY + 1:
        raise ValueError('the 15-minute output does not cover the run')
    if R.times_of(fx)[0] != '2010-06-17_00:00:00' or len(R.times_of(fx)) < args.steps // STEPS_PER_FRAME + 1:
        raise ValueError('flux file does not cover the run')
    g = R.load_static(fx, xp)
    nz, ny, nx = g.nz, g.ny, g.nx

    # inputs: IC background, IC urban, BC background, emissions 1, 2 (as run.py)
    ic = np.transpose(np.asarray(netCDF4.Dataset(pmc / 'ic_control.nc')['num_conc'][:], np.float64), (0, 3, 2, 1))
    ins = np.zeros((S.N_INPUTS, nz, ny, nx))
    ins[0:2] = ic
    bc = netCDF4.Dataset(pmc / 'bc_control.nc')
    if bc.dimensions['n_source'].size != 1 or bc.dimensions['n_time'].size != 1:
        raise ValueError('expected one time-constant BC source')
    edge = lambda name: np.asarray(bc[name][0, 0], np.float64).T          # (nz, along)
    bce = {'w': edge('num_conc_xs'), 'e': edge('num_conc_xe'), 's': edge('num_conc_ys'), 'n': edge('num_conc_ye')}
    em = netCDF4.Dataset(pmc / 'emis_control.nc')
    em_t = np.asarray(em['emis_time'][:], np.float64)
    em_flux = np.transpose(np.asarray(em['num_flux'][:, :, :, :, 0], np.float64), (0, 1, 3, 2))   # (2, nt, ny, nx)
    ef_t, ef = R.emission_factor(orig)

    with np.load(args.zsrc) as f:
        if 'pops' in f.files:
            if list(f['pops'] // 25) != [561, 635, 812]:
                raise ValueError('source states are not for scenarios 561, 635, 812')
        else:
            print('warning: %s has no pops; the source order is assumed (background, urban, emissions)' % args.zsrc)
        Z = {m: f[m + '_z'].astype(np.float64) for m in models}
    for m in nocs:
        Z[m + '_nocs'] = Z[m].copy()
    for m, z in Z.items():
        if z.ndim != 2 or z.shape[0] != 3 or not np.all(np.isfinite(z)):
            raise ValueError('%s: source states must be finite (3, d), got %s' % (m, z.shape))
        if z.min() < 0:
            print('warning: %s has negative source components (min %.3g)' % (m, z.min()))
    sep_models = list(Z)
    n_tr = 1 + (S.N_INPUTS if 'oracle' in sch else 0) + sum(Z[m].shape[1] for m in sep_models)
    print('grid nz %d ny %d nx %d; %d independently limited tracers (%s); chunks of at most %d'
          % (nz, ny, nx, n_tr, ', '.join('%s %d' % (m, Z[m].shape[1]) for m in sep_models), args.chunk), flush=True)

    Runner = make_chunked_runner(S, W)
    r = Runner(g, Z, xp, ins, bce, schemes=sch, sep_models=sep_models, shB_models=[], chunk=args.chunk)
    print('chunks:', ', '.join('%d-%d' % (c.start, c.stop - 1) for c in r.chunks), flush=True)
    mu0 = xp.asarray(np.asarray(fx['MU'][0] + fx['MUB'][0], np.float64))
    M = g.mass(mu0)

    every = max(1, args.calcsize_every)
    hooks, descs = {}, {}
    for m in cs_models:
        fn, desc = load_calcsize(m, args.modal_path, xp)
        hooks[m] = fn
        descs[m] = desc
        print('calcsize for %s every %d steps (%g s): %s' % (m, every, every * DT, desc), flush=True)
    mnfix_stats = {}
    for m in mnfix_models:
        fn, desc = load_mnfix(m, args.modal_path, xp, mnfix_stats.setdefault(m, {}))
        hooks[m] = fn
        descs[m] = desc
        print('mnfix for %s every %d steps: %s' % (m, every, desc), flush=True)

    madj_stats = {}
    for m in madj_models:
        fn, desc = load_matrix_adjust(m, args.modal_path, xp, madj_stats.setdefault(m, {}))
        hooks[m] = fn
        descs[m] = desc
        print('matrix.adjust for %s every %d steps: %s' % (m, every, desc), flush=True)

    get = (lambda a: a.get()) if xp is not np else (lambda a: np.asarray(a))
    # each cell's air mass per unit M (for domain totals): d(eta) / (msftx msfty), interior only
    deta = 1.0 / xp.abs(g.rdnw)                                             # (nz, 1, 1)
    area = xp.where(g.interior, 1.0 / (g.msftx * g.msfty), 0.0)              # (ny, nx)
    cs_delta = {m: xp.zeros(Z[m].shape[1]) for m in cs_models + mnfix_models}  # cumulative domain totals
    cs_delta.update({m: xp.zeros(Z[m].shape[1]) for m in madj_models})
    cs_calls = [0]

    def apply_calcsize(Mcur):
        wt = (Mcur * deta * area).reshape(-1)
        for m, fn in hooks.items():
            sl = r.idx['sep:' + m]
            before = r.q_mono[sl].copy()
            work = before.copy()
            res = fn(work, every * DT)
            after = work if res is None else res                    # in-place or returning APIs
            d = after.shape[0]
            cs_delta[m] += xp.tensordot((after - before).reshape(d, -1), wt, axes=1)
            r.q_mono[sl] = after
        cs_calls[0] += 1

    rec, diag = {}, {}
    levels = [int(k) for k in args.levels.split(',') if k]
    rec_lv = {k: {} for k in levels}
    t_out = []
    c1h0, c2h0 = float(fx['C1H'][0][0]), float(fx['C2H'][0][0])

    def record(step, Mcur):
        s = r.surface(0)
        for k, v in s.items():
            rec.setdefault(k, []).append(v.astype(np.float32))
        for lv in levels:
            for k, v in r.surface(lv).items():
                rec_lv[lv].setdefault(k, []).append(v.astype(np.float32))
        mu_wrf = np.asarray(fx['MU'][step // STEPS_PER_FRAME] + fx['MUB'][step // STEPS_PER_FRAME], np.float64)
        M0 = get(Mcur)[0]
        rec.setdefault('M_offline_over_wrf', []).append((M0 / (c1h0 * mu_wrf + c2h0)).astype(np.float32)[1:-1, 1:-1])
        t_out.append(step * DT)
        # diagnostics of the whole 3-D state
        q = r.q_mono.reshape(r.q_mono.shape[0], -1)
        wt = (Mcur * deta * area).reshape(-1)
        diag.setdefault('total', []).append(get(xp.tensordot(q, wt, axes=1)))
        diag.setdefault('min', []).append(get(q.min(axis=1)))
        diag.setdefault('max', []).append(get(q.max(axis=1)))
        diag.setdefault('n_nonfinite', []).append(get((~xp.isfinite(q)).sum(axis=1)))
        diag.setdefault('air_total', []).append(float(get(wt.sum())))
        for m in cs_models:
            diag.setdefault('calcsize_delta:' + m, []).append(get(cs_delta[m]).copy())
        for m in mnfix_models:
            diag.setdefault('mnfix_delta:' + m, []).append(get(cs_delta[m]).copy())
        for m in madj_models:
            diag.setdefault('matrix_adjust_delta:' + m, []).append(get(cs_delta[m]).copy())

    record(0, M)
    t0 = time.time()
    for step in range(args.steps):
        frame = step // STEPS_PER_FRAME + 1
        F = tuple(xp.asarray(np.asarray(fx[v][frame], np.float64)) for v in ('AVGFLX_RUM', 'AVGFLX_RVM', 'AVGFLX_WWM'))
        M = r.step(F, M, DT)
        t = step * DT
        irec = max(int(np.searchsorted(em_t, t, side='right')) - 1, 0)
        i0 = min(int(t // 900), len(ef_t) - 2)
        a = (t - ef_t[i0]) / (ef_t[i0 + 1] - ef_t[i0])
        conv = (1 - a) * ef[i0] + a * ef[i0 + 1]                           # (ny, nx)
        r.emit(xp.asarray(em_flux[:, irec] * conv[None]))
        if hooks and (step + 1) % every == 0:
            apply_calcsize(M)
        if (step + 1) % OUT_EVERY == 0:
            record(step + 1, M)
        if step % 45 == 0:
            mem = ''
            if xp is not np:
                pool = xp.get_default_memory_pool()
                mem = '  GPU pool %.2f GB used, %.2f GB held' % (pool.used_bytes() / 1e9, pool.total_bytes() / 1e9)
            print('step %d/%d  %.1f s elapsed%s' % (step, args.steps, time.time() - t0, mem), flush=True)
    elapsed = time.time() - t0
    print('done in %.1f s' % elapsed, flush=True)

    code = {Path(m.__file__).name: md5(m.__file__) for m in (R, S, W)}
    code[Path(__file__).name] = md5(__file__)
    settings = dict(models=models, nocs=nocs, sep_models=sep_models, calcsize=cs_models,
                    calcsize_every=every if hooks else 0, calcsize_dt=every * DT if hooks else 0.0,
                    calcsize_impl=descs, calcsize_calls=cs_calls[0],
                    schemes=sch, steps=args.steps, dt=DT, chunk=args.chunk, fluxes=str(fluxes), orig=str(orig_path),
                    pmc=str(pmc), zsrc=str(args.zsrc), zsrc_md5=md5(args.zsrc), code_md5=code,
                    dims=dict((m, int(Z[m].shape[1])) for m in sep_models), elapsed_s=elapsed,
                    note='sep:<model> holds Q = N x per kg dry air; for calcsize models the number is the '
                         'sum of the mode numbers in Q, not N')
    if mnfix_models:
        settings.update(mnfix=mnfix_models, mnfix_stats=mnfix_stats,
                        mnfix_note='mnfix (TOMAS MNFIX) every calcsize_every steps; the number of a TOMAS '
                                   'model is the sum of its bin numbers in Q (mnfix can change it)')
    if madj_models:
        settings.update(matrix_adjust=madj_models,
                        matrix_adjust_stats={m: {k: int(v) for k, v in st_.items()} for m, st_ in madj_stats.items()},
                        matrix_adjust_note='matrix.adjust (MATRIX state-driven operations) every calcsize_every '
                                           'steps; stats count (cell, call) pairs where each operation acted')
    js = json.dumps(settings)
    arrays = {k: np.stack(v) for k, v in rec.items()}
    t = np.array(t_out)
    np.savez_compressed(out / 'surface.npz', time_s=t, **arrays, settings=js)
    print('wrote', out / 'surface.npz', flush=True)
    for name, keep in (('surface_end', np.isin(t, [0.0, 4 * 3600.0, 12 * 3600.0])),
                       ('surface_30min', np.isclose(t % 1800.0, 0.0))):
        np.savez_compressed(out / ('%s.npz' % name), time_s=t[keep], **{k: v[keep] for k, v in arrays.items()},
                            settings=js)
        print(name, int(keep.sum()), 'outputs', flush=True)
    tracer_names = ['N'] + (['oracle:%s' % c for c in S.ORACLE_NAMES] if 'oracle' in sch else []) + \
        ['sep:%s:%d' % (m, j) for m in sep_models for j in range(Z[m].shape[1])]
    np.savez_compressed(out / 'diag.npz', time_s=t, tracer=np.array(tracer_names),
                        **{k: np.stack(v) if k != 'air_total' else np.array(v) for k, v in diag.items()},
                        settings=js)
    print('wrote', out / 'diag.npz', flush=True)
    for lv in levels:
        np.savez_compressed(out / ('level_%d.npz' % lv), time_s=t, **{k: np.stack(v) for k, v in rec_lv[lv].items()},
                            settings=json.dumps(dict(settings, level=lv)))
        print('wrote', out / ('level_%d.npz' % lv))


if __name__ == '__main__':
    main()
