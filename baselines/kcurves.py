#!/usr/bin/env python
"""
Test-set K-curves for the sectional and modal baselines: the quantity of the paper's Figure 7
(figures/mixing_kcurves.pdf; training/affx_mixk.py --logratio ->
notes/affx_stats/mixk_r33_lr.json) for representations that mix in a per-particle state vector
instead of a learned latent.

Mixtures (exactly affx_mixk.py's; the lines are checked against its source at start-up):
for split seed si (an index into the ten test splits of models/test_idx.npz; the decoders' seed
si was tested on the same split) and each K, np.random.RandomState(4321 + 97 si + K) draws
--nmix (2000) mixtures, each of K distinct test populations of the split with Dirichlet(1)
weights lam (shares of air), number fractions alpha = lam n_tot / sum(lam n_tot) and number
concentration n = sum lam n_tot. K = 1 is every test population once. The true mixture is
sum alpha X_k, X the dataset's per-particle diagnostics (456 values, affx.py's blocks M, N, C, S,
A, F) from --cache: cache_optfix.npz, the optics-corrected targets the round-33 models (the
paper's decoders) are scored against.

A representation R maps a population's particles to a per-particle state vector x (sectional:
the number fraction and the mass per particle of each species in each bin; modal: the same per
mode), so a mixture's state is sum alpha_k x_k, which is how these representations mix (bin and
mode numbers and masses add). Its diagnostics q_R(x) come from representative particles built
from the state, through baselines/diag.py. As in affx_mixk.py, three comparisons per mixture:

    mix    q_R(sum alpha x)      vs  sum alpha X         the mixture in the representation (Q2 vs Q0)
    dm     sum alpha q_R(x)      vs  sum alpha X         each population represented, then mixed (Q1 vs Q0)
    comm   q_R(sum alpha x)      vs  sum alpha q_R(x)    what mixing inside bins/modes does (Q2 vs Q1)

(Q0-Q2 as in baselines/PLAN.md.) When 'prdiag' is among the representations (the default, and
first), each baseline entry also has mix_pr and dm_pr: mix and dm against the mixture of diag.py's
diagnostics of the true particles instead of the dataset's X, which isolates the representation from
the difference between diag.py (the WRF reference's code) and the dataset's diagnostics code. Modal states are mixed without MAM's calcsize, which belongs to
the transport term; --calcsize-after-mix adds <name>_cs entries with it applied once after mixing.

Errors: affx_mixk.py's own units() and metrics(), executed from its source with --logratio's
paper_errors.logratio_metrics and --pct, so every number means what it means in mixk_r33_lr.json
and the output has its shape: {name: {"seeds": [...], "K": {"<K>": [per seed {mix, dm, comm}]}}}
(plus "_meta": settings, timing, the representative-particle counts and diag.py's check).

Subcommands:
  run    one split seed -> a per-seed JSON (resumable: (representation, K) pairs already in it are
         skipped; q_R of the single test populations is cached next to it)
  merge  per-seed JSONs -> one JSON with every seed (what the figure reads)
  time   diagnose --n mixtures of K = 16 per representation and extrapolate a seed's run time

Representations: prdiag (no representation: diag.py on the true particles, the floor of the
comparison, since diag.py follows the WRF reference's code, not the dataset's: water's refractive
index there is 1.33 at every wavelength, in the dataset 1.35-1.33 with a small imaginary part),
sec4, sec8, sec20 (baselines/sectional.py: MOSAIC 4 and 8 bins, 20 log bins, dry diameter),
sec20w (20 log bins in wet diameter: the coag paper's sectional), mam4a (MAM4 all species, fresh
carbon above the pca mode's dgnumhi to accumulation),
mam4, mam4l (baselines/modal.py: all 15 species per mode; MAM4's lumped species; calcsize once at
projection, as there), and, to check this script against mixk_r33_lr.json, dec=PREFIX (an affx
decoder PREFIX_s<si>.pt with latents --latents, mixed in its latent exactly as affx_mixk.py does).

Usage (Perlmutter CPU job: baselines/kcurves.sbatch):
  python kcurves.py run --seed 0 --reps sec4,sec8,sec20,mam4,mam4l --out data/kcurves_s0.json
  python kcurves.py merge data/kcurves_s*.json --out data/kcurves_baselines.json
  python kcurves.py time --seed 0 --reps sec8,mam4 --n 64
--data may also be an npz (test_idx, m, n: the split's particles) for runs without h5py.
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                      # baselines/
DEFAULT_TRAINING = HERE.parent / 'training'                 # the paper's training/ (paper_errors, affx_mixk)
SCRATCH = '/pscratch/sd/m/mwest/aeromeld'
KS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 1024, 5000]      # mixk_r33_lr.json's K
REPS = ('prdiag', 'sec4', 'sec8', 'sec20', 'mam4', 'mam4l')     # also sec20w, mam4a (added 2026-11-14)
N_SPEC = 15                                                 # SO4 NO3 Cl NH4 ARO1 ARO2 ALK1 OLE1 API1 Na OIN OC BC MOC H2O
D_X = 456


# ------------------------------------------------------------------ affx_mixk.py, from its source
def mixk_code(training_dir, pct=True):
    """units(), metrics(), GEO, SIX-compatible names of training/affx_mixk.py, executed from its
    source (it parses arguments and loads data at import) with METRICS = logratio_metrics (its
    --logratio) and args.pct (its --pct). Also checks that its mixture lines are the ones
    draw_mixtures() below copies."""
    try:                                                        # paper_errors imports h5py for its CLI only;
        import h5py  # noqa: F401                               # stub it only where it is missing (the real
    except ImportError:                                         # one reads the particles in load_particles)
        sys.modules['h5py'] = types.ModuleType('h5py')
    sys.path.insert(0, str(training_dir))
    from paper_errors import logratio_metrics, ROWS
    src = (Path(training_dir) / 'affx_mixk.py').read_text()
    body = src[src.index('BLOCKS = ['):src.index('def decoder(')]
    load = "c = np.load(args.cache)\nX, ntot = c['X'], c['ntot']\nsplits = np.load(args.splits)\n"
    assert body.count(load) == 1, 'affx_mixk.py changed (data loading); check kcurves.py'
    ns = dict(np=np, ROWS=ROWS, METRICS=logratio_metrics, args=types.SimpleNamespace(pct=pct))
    exec(body.replace(load, ''), ns)
    for line in MIXK_LINES:
        assert line in src, 'affx_mixk.py changed (mixtures): %r not found; check draw_mixtures()' % line
    dec_src = src[src.index('sp = lambda'):src.index('\n', src.index('sp = lambda'))] + '\n' + \
        src[src.index('def decoder('):src.index('rows = []')]
    return ns, dec_src


MIXK_LINES = ('rng = np.random.RandomState(4321 + 97 * si + K)',
              'pick = np.stack([rng.choice(len(tst), K, replace=False) for _ in range(args.nmix)])',
              'lam = rng.dirichlet(np.ones(K), size=args.nmix)',
              'nk = ntot[tst][pick]',
              'al = lam * nk; nm = al.sum(1); al /= nm[:, None]',
              'np.put_along_axis(Wm, pick, al, 1)')


def draw_mixtures(si, K, ntot_t, nmix):
    """affx_mixk.py's mixtures of seed si: weights Wm (nmix, n_test) (rows: number fractions) and
    the mixtures' number concentrations nm (nmix,). The same RNG calls in the same order."""
    rng = np.random.RandomState(4321 + 97 * si + K)
    pick = np.stack([rng.choice(len(ntot_t), K, replace=False) for _ in range(nmix)])
    lam = rng.dirichlet(np.ones(K), size=nmix)
    nk = ntot_t[pick]
    al = lam * nk; nm = al.sum(1); al /= nm[:, None]
    Wm = np.zeros((nmix, len(ntot_t)))
    np.put_along_axis(Wm, pick, al, 1)
    return Wm, nm


# ------------------------------------------------------------------ data
def load_truth(cache, splits, si):
    """Sorted test indices of split seed si, the true per-particle diagnostics X (n_test, 456) and
    n_tot (n_test,). --cache: affx's cache npz (all populations; X, ntot) or a test-set truth npz
    (test_idx, X, ntot; data/affx_testset_of/truth_s0.npz, for local checks)."""
    tst = np.sort(np.load(splits)['idx'][si])
    c = np.load(cache)
    if 'test_idx' in c.files:
        if not np.array_equal(c['test_idx'], tst):
            raise ValueError('%s: test_idx is not split seed %d' % (cache, si))
        X, ntot = c['X'], c['ntot']
    else:
        X, ntot = c['X'][tst], c['ntot'][tst]
    return tst, np.asarray(X, np.float64), np.asarray(ntot, np.float64)


def load_particles(data, tst, X, ntot):
    """Species masses per particle m (n_test, 1000, 15) kg and number fractions w (n_test, 1000)
    (n_prt / n_tot; 1/1000 each in this dataset) of the test populations, from the dataset h5
    (or an npz with test_idx, m, n: the same arrays for one split). Checks that they are the
    populations of X (n_tot, and the dataset's number distribution against X's N block)."""
    nh, names = None, ''
    if str(data).endswith('.npz'):
        d = np.load(data)
        if not np.array_equal(d['test_idx'], tst):
            raise ValueError('%s: test_idx is not this split' % data)
        m, n = d['m'], d['n']
    else:
        import h5py
        with h5py.File(data, 'r') as f:
            m = f['m_prtchm'][tst]                              # increasing indices: h5py selection
            n = f['n_prt'][tst][..., 0]
            nh = f['n_prthst'][tst].reshape(len(tst), -1)
            names = f['dims'][0].decode() if 'dims' in f else ''
    if names and 'SO4,NO3,Cl,NH4,ARO1,ARO2,ALK1,OLE1,API1,Na,OIN,OC,BC,MOC,H2O' not in names:
        raise ValueError('unexpected species order in %s' % data)
    if not np.allclose(n.sum(1), ntot, rtol=1e-10, atol=0):
        raise ValueError('n_prt does not add up to the cache\'s n_tot: not the same populations')
    import diag                                                # (dry targets: the h5's N is wet)
    if nh is not None and diag.SIZE_BASIS == 'wet' and \
            not np.allclose(nh / ntot[:, None], X[:, 225:240], rtol=1e-6, atol=1e-12):
        raise ValueError('the dataset\'s number distribution differs from the cache\'s N block')
    return np.asarray(m, np.float64), n / ntot[:, None]


# ------------------------------------------------------------------ representations
class Representation:
    """name; state(m, w) -> per-particle state S (n, d); diagnose(S) -> per-particle diagnostics
    (n, 456) (blocks as affx.py: extensive ones per particle); n_rep: representative particles
    per state of the last diagnose() call (mean), for the notes."""
    needs_particles = True

    def __init__(self, name):
        self.name, self.n_rep = name, []


class ParticleRep(Representation):
    """A sectional or modal representation: baselines/sectional.py (Sectional) or modal.py (Modal)
    for the state and the representative particles, baselines/common.py's evaluate() (diag.py's
    raw_vectors in worker processes) for their diagnostics:
      obj.project(m (n, P, 15) kg, w (n, P)) -> x (n, d)        [Modal: calcsize once, dt 20 s]
      common.evaluate(obj, x (n, d)) -> (n, 456) per particle    [obj.particles -> diag.raw_vectors]
    With a calcsize function (modal, --calcsize-after-mix) calcsize is applied once more after
    mixing: modal.calcsize(x.T, dt, obj).T, as Modal.project applies it."""

    def __init__(self, name, obj, common, cs_fn=None, cs_dt=None):
        super().__init__(name)
        self.obj, self.common, self.cs_fn, self.cs_dt = obj, common, cs_fn, cs_dt

    def state(self, m, w):
        return np.asarray(self.obj.project(m, w), np.float64)

    def diagnose(self, S, batch=None):
        if self.cs_fn is not None:
            S = self.cs_fn(S.T, self.cs_dt, self.obj).T
        _, wr = self.obj.particles(S[:200])                     # for the notes: live representative particles
        self.n_rep.append(float((np.asarray(wr) > 0).sum(-1).mean()))
        return self.common.evaluate(self.obj, S)


class TrueParticles(Representation):
    """'prdiag': diag.py on the test populations' own particles. Its 'state' is that per-particle
    diagnostic vector, which mixes exactly in number (every block is a number-weighted sum or
    fraction over particles), so its mix entries are the floor of the comparison: diag.py against
    the dataset's diagnostics X, with no representation in between (comm = 0)."""

    def __init__(self, common, cache):
        super().__init__('prdiag')
        self.common, self.cache = common, cache

    def state(self, m, w):
        if self.cache.exists():
            return np.load(self.cache)
        t = time.time()
        Xp = self.common.particles_vectors(m, w)
        np.save(self.cache, Xp)
        print('prdiag: diag.py on %d x %d true particles in %.0f s' % (*m.shape[:2], time.time() - t), flush=True)
        return Xp

    def diagnose(self, S, batch=None):
        return S


class DecoderRep(Representation):
    """An affx decoder (affx_mixk.py's decoder()) on its latents: the check that this script
    reproduces mixk_r33_lr.json. The state is the latent z, read from --latents."""
    needs_particles = False

    def __init__(self, name, prefix, si, latents, dec_src, tst):
        super().__init__(name)
        import torch                                            # noqa: F401  (decoder() loads the .pt)
        ns = dict(np=np, torch=torch, BLOCKS=MIXK_NS['BLOCKS'])
        exec(dec_src, ns)
        self.dec = ns['decoder'](prefix, si)
        self.Z = np.load(latents).astype(np.float64)[tst]

    def state(self, m, w):
        return self.Z

    def diagnose(self, S, batch=None):
        return self.dec(S)


def import_baselines():
    """baselines/common.py (and through it diag.py, sectional.py, modal.py). diag.py imports
    scripts/paperfigs/config.py and scripts/vendor/extract_diagnostics.py, which imports netCDF4
    for file reading only; if netCDF4 is missing, an empty stub module is put on sys.path and on
    PYTHONPATH (common.py's worker processes are spawned and import diag.py themselves).
    BASELINES_PROCS (worker processes) defaults to the job's SLURM_CPUS_PER_TASK."""
    try:
        import netCDF4  # noqa: F401
    except ImportError:                                         # a stub module the workers find too
        stub = HERE / '_stubs'
        stub.mkdir(exist_ok=True)
        (stub / 'netCDF4.py').write_text('"""Stub: extract_diagnostics imports netCDF4 for file reading only."""\n')
        sys.path.append(str(stub))
        os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [os.environ.get('PYTHONPATH'), str(stub)]))
    os.environ.setdefault('BASELINES_PROCS', os.environ.get('SLURM_CPUS_PER_TASK', '12'))
    sys.path.insert(0, str(HERE))
    import common
    import modal
    import sectional
    return common, sectional, modal


def make_reps(names, args, si, tst, out=None):
    reps = []
    for name in names:
        if name.startswith('dec='):
            reps.append(DecoderRep(Path(name[4:]).name, name[4:], si, args.latents, DEC_SRC, tst))
            continue
        common, sectional, modal = import_baselines()
        base = name[:-3] if name.endswith('_cs') else name
        if name == 'prdiag':
            reps.append(TrueParticles(common, out.with_name(out.stem + '_prdiag_Xp.npy')))
        elif base in sectional.EDGES:
            reps.append(ParticleRep(name, sectional.Sectional(base), common))
        elif base.startswith('tomas'):                          # added 2026-11-14: baselines/tomas.py
            import tomas                                        # (mnfix once at projection, as calcsize;
            obj = tomas.Tomas(base)                             #  <name>_cs: also after mixing)
            cs = (lambda Q, dt, obj: tomas.mnfix(Q)) if name.endswith('_cs') else None
            reps.append(ParticleRep(name, obj, common, *((cs, 20.0) if cs else ())))
        elif base in ('mam4', 'mam4l', 'mam4a', 'mam4la'):
            obj = modal.Modal(base)
            reps.append(ParticleRep(name, obj, common, *((modal.calcsize, obj.calcsize_dt) if name.endswith('_cs')
                                                         else ())))
        elif base.startswith('matrix'):                         # added 2026-11-14: baselines/matrix.py
            import matrix                                       # (adjust once at projection, as calcsize;
            obj = matrix.Matrix(base)                           #  <name>_cs: also after mixing)
            reps.append(ParticleRep(name, obj, common, *((matrix.adjust, obj.calcsize_dt) if name.endswith('_cs')
                                                         else ())))
        else:
            raise ValueError('unknown representation %r' % name)
    return reps


# ------------------------------------------------------------------ run
def save_json(obj, path):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as fp:
        json.dump(obj, fp)
    os.replace(tmp, path)


def cache_basis(cache):
    """The size basis of a targets cache, from its name (training/dataset/dry_cache.py's cache_<basis>.npz)."""
    name = Path(cache).name
    return 'drydiam' if 'drydiam' in name else 'dry' if 'cache_dry' in name else 'wet'


def run(args):
    si = args.seed
    import_baselines()                                          # the netCDF4 stub before diag.py's import
    import diag
    if cache_basis(args.cache) != diag.SIZE_BASIS:
        raise SystemExit('--cache %s holds %s targets, diag.py bins by %s (set AEROMELD_SIZE_BASIS)' % (
            args.cache, cache_basis(args.cache), diag.SIZE_BASIS))
    tst, X, ntot = load_truth(args.cache, args.splits, si)
    names = [r for r in args.reps.split(',') if r]
    if args.calcsize_after_mix:
        names += [r + '_cs' for r in names if r.startswith('mam4')]
    out = Path(args.out)
    reps = make_reps(names, args, si, tst, out)
    res = json.loads(out.read_text()) if out.exists() else {}
    meta = res.setdefault('_meta', {})
    meta.update(seed=si, nmix=args.nmix, cache=str(args.cache), splits=str(args.splits), data=str(args.data),
                n_test=len(tst), host=os.uname().nodename, procs=os.environ.get('BASELINES_PROCS'),
                size_basis=diag.SIZE_BASIS)
    t0 = time.time()
    m = w = None
    if any(r.needs_particles for r in reps):
        m, w = load_particles(args.data, tst, X, ntot)
        print('particles: %s in %.0f s' % (m.shape, time.time() - t0), flush=True)
    states, Xd = {}, {}
    for r in reps:
        states[r.name] = r.state(m, w)
        cache = out.with_name(out.stem + '_%s_Xd.npy' % r.name)
        if cache.exists():
            Xd[r.name] = np.load(cache)
        else:
            t = time.time()
            Xd[r.name] = r.diagnose(states[r.name])
            np.save(cache, Xd[r.name])
            meta.setdefault('seconds', {})['%s:single' % r.name] = time.time() - t
        assert Xd[r.name].shape == X.shape, (r.name, Xd[r.name].shape)
        print('%s: state %s, single populations diagnosed' % (r.name, states[r.name].shape), flush=True)
    for K in args.K:
        todo = [r for r in reps if str(K) not in res.get(r.name, {}).get('K', {})]
        if not todo:
            continue
        Xp = Xd.get('prdiag')                                     # diag.py on the true particles
        if K > 1:
            Wm, nm = draw_mixtures(si, K, ntot, args.nmix)
            mo = Wm @ X
            mp = Wm @ Xp if Xp is not None else None
        for r in todo:
            t = time.time()
            r.n_rep = []
            pr = Xp is not None and isinstance(r, ParticleRep)    # also against prdiag (mix_pr, dm_pr)
            if K == 1:
                e = MIXK_NS['metrics'](X, Xd[r.name], ntot)
                entry = dict(mix=e, dm=e, comm={k: 0.0 for k in e})
                if pr:
                    e = MIXK_NS['metrics'](Xp, Xd[r.name], ntot)
                    entry.update(mix_pr=e, dm_pr=e)
            else:
                mdm = Wm @ Xd[r.name]
                mdec = r.diagnose(Wm @ states[r.name])
                entry = dict(mix=MIXK_NS['metrics'](mo, mdec, nm), dm=MIXK_NS['metrics'](mo, mdm, nm),
                             comm=MIXK_NS['metrics'](mdm, mdec, nm))
                if pr:
                    entry.update(mix_pr=MIXK_NS['metrics'](mp, mdec, nm), dm_pr=MIXK_NS['metrics'](mp, mdm, nm))
            d = res.setdefault(r.name, dict(seeds=[si], K={}))
            d['K'][str(K)] = [entry]
            meta.setdefault('seconds', {})['%s:%d' % (r.name, K)] = time.time() - t
            if r.n_rep:
                meta.setdefault('n_rep', {})['%s:%d' % (r.name, K)] = float(np.mean(r.n_rep))
            save_json(res, out)
            six = np.mean([entry['mix'][k] for k in SIX])
            print('seed %d K=%4d %-8s mix %.2f%% (six), dm %.2f%%, comm %.2f%%, mix vs prdiag %s   %.0f s' % (
                si, K, r.name, 100 * six, 100 * np.mean([entry['dm'][k] for k in SIX]),
                100 * np.mean([entry['comm'][k] for k in SIX]),
                ('%.2f%%' % (100 * np.mean([entry['mix_pr'][k] for k in SIX]))) if 'mix_pr' in entry else '-',
                time.time() - t), flush=True)
    meta['seconds_total_last_run'] = time.time() - t0
    save_json(res, out)
    print('wrote', out)


def merge(args):
    """Per-seed JSONs (several per seed allowed: e.g. a second array with more representations)
    -> {name: {seeds: [...], K: {K: [one entry per seed, in seed order]}}, _meta: {seed: [metas]}}.
    A (name, seed) found in several files is taken from the first file listed that has all its K."""
    per = [json.loads(Path(p).read_text()) for p in args.files]
    out = {'_meta': {}}
    for p, d in zip(args.files, per):
        out['_meta'].setdefault(str(d['_meta']['seed']), []).append(dict(d['_meta'], file=str(p)))
    names = sorted({k for d in per for k in d if k != '_meta'})
    for name in names:
        Ks = sorted({int(K) for d in per if name in d for K in d[name]['K']})
        got = {}
        for d in per:
            si = d['_meta']['seed']
            if name in d and si not in got and all(str(K) in d[name]['K'] for K in Ks):
                got[si] = d[name]['K']
        missing = sorted({d['_meta']['seed'] for d in per if name in d} - set(got))
        if missing:
            print('%s: seeds %s incomplete, left out' % (name, missing))
        seeds = sorted(got)
        out[name] = dict(seeds=seeds, K={str(K): [got[si][str(K)][0] for si in seeds] for K in Ks})
        print('%s: seeds %s, K %s' % (name, seeds, Ks))
    save_json(out, args.out)
    print('wrote', args.out)


def timing(args):
    si = args.seed
    tst, X, ntot = load_truth(args.cache, args.splits, si)
    reps = make_reps([r for r in args.reps.split(',') if r and r != 'prdiag'], args, si, tst)
    m, w = load_particles(args.data, tst, X, ntot) if any(r.needs_particles for r in reps) else (None, None)
    Wm, nm = draw_mixtures(si, 16, ntot, args.n)
    n_eval = len(tst) + args.nmix * (len(args.K) - 1)
    total = 0.0
    for r in reps:
        S = r.state(m, w)
        r.n_rep = []
        t = time.time()
        r.diagnose(Wm @ S)
        dt = (time.time() - t) / args.n
        total += dt * n_eval
        print('%-8s %.1f ms per mixture (%s representative particles), %.2f h per seed (%d diagnoses)' % (
            r.name, 1e3 * dt, ('%.0f' % np.mean(r.n_rep)) if r.n_rep else '-', dt * n_eval / 3600, n_eval),
            flush=True)
    print('all: %.2f h per seed' % (total / 3600))


SIX = ('n_prthst', 'm_chmprthst', 'ccn_cdf', 'logfrznfrac_tmp', 'logqs_pop', 'logqa_pop')   # paperfigs/errors.py SIX
MIXK_NS, DEC_SRC = {}, ''


def main(argv=None):
    global MIXK_NS, DEC_SRC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('run', 'time'):
        p = sub.add_parser(cmd)
        p.add_argument('--seed', type=int, default=0, help='split seed (index into test_idx.npz)')
        p.add_argument('--reps', default=','.join(REPS))
        p.add_argument('--K', type=int, nargs='+', default=KS)
        p.add_argument('--nmix', type=int, default=2000)
        p.add_argument('--data', default=SCRATCH + '/data/04_bwchisamp_pm10.h5')
        p.add_argument('--cache', default=SCRATCH + '/affx/cache_optfix.npz')
        p.add_argument('--splits', default=SCRATCH + '/models/test_idx.npz')
        p.add_argument('--training', default=str(DEFAULT_TRAINING), help='directory with affx_mixk.py, paper_errors.py')
        p.add_argument('--latents', default=None, help='dec=PREFIX only: the latents of every population (Z.npy)')
        p.add_argument('--calcsize-after-mix', action='store_true', help='modal: also <name>_cs, calcsize after mixing')
        if cmd == 'run':
            p.add_argument('--out', required=True)
        else:
            p.add_argument('--n', type=int, default=64)
    p = sub.add_parser('merge')
    p.add_argument('files', nargs='+')
    p.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'merge':
        return merge(args)
    MIXK_NS, DEC_SRC = mixk_code(args.training)
    return run(args) if args.cmd == 'run' else timing(args)


if __name__ == '__main__':
    main()
